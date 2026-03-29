"""
main.py
=======
End-to-end pipeline orchestrator for the Google Smartphone Decimeter Challenge 2022.

Pipeline structure (per trip/device) follows Taro Suzuki's approach:
1. Load raw GNSS data
2. Carrier-phase smoothing (Hatch filter)
3. Robust WLS position + velocity (cauchy loss)
4. Outlier rejection (velocity/height thresholds)
5. Forward-backward Kalman smoother

See gnss_solver.py for full attribution. Post-processing (SavGol smoothing,
median filtering) and ML correction are original to this project.

CLI usage:
    python src/main.py --root kaggle_dataset --output submission.csv --split test
    python src/main.py --root kaggle_dataset --evaluate --n-trips 5
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

from src.data_loader import load_gnss, load_gnss_raw, load_ground_truth, load_imu, align_imu_to_gnss
from src.gnss_solver import solve_trip_robust
from src.post_processing import (
    median_filter_trajectory, savgol_smooth_trajectory, osrm_snap_to_road,
    detect_stops, apply_stop_averaging
)


# ── Single-trip processor ─────────────────────────────────────────────────────

def process_trip(
    device_dir: pathlib.Path,
    train: bool = True,
    osrm: bool = False,
    sat_weight_model=None,
    use_savgol: bool = False,
    use_stop_averaging: bool = False,
) -> pd.DataFrame:
    """
    Run the full pipeline for one (trip, device) directory.

    Returns DataFrame: UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
    """
    device_dir = pathlib.Path(device_dir)
    device_name = device_dir.name

    # ── 1. Load data ──────────────────────────────────────────────────────
    gnss_df = load_gnss_raw(device_dir)

    if len(gnss_df) == 0:
        return pd.DataFrame(
            columns=["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        )

    # ── 2-6. Robust solver pipeline ──────────────────────────────────────
    result = solve_trip_robust(gnss_df, device_name=device_name,
                               sat_weight_model=sat_weight_model)

    if len(result) == 0:
        return pd.DataFrame(
            columns=["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        )

    pos_df = pd.DataFrame({
        "UnixTimeMillis": result["epoch_ms"].values.astype(np.int64),
        "LatitudeDegrees": result["lat"].values,
        "LongitudeDegrees": result["lon"].values,
    })

    # Fill NaN epochs by interpolation
    pos_df["LatitudeDegrees"] = pos_df["LatitudeDegrees"].interpolate().ffill().bfill()
    pos_df["LongitudeDegrees"] = pos_df["LongitudeDegrees"].interpolate().ffill().bfill()

    # Light median filter for remaining spikes
    pos_df = median_filter_trajectory(pos_df, kernel_size=3)

    # Optional IMU-based stop detection: average positions within stationary segments
    if use_stop_averaging:
        try:
            imu_df = load_imu(device_dir)
            if len(imu_df) > 0:
                aligned_imu = align_imu_to_gnss(imu_df, pos_df["UnixTimeMillis"].values)
                # Attach horizontal acceleration to pos_df for stop detection
                stop_input = pos_df.copy()
                stop_input["accel_x"] = aligned_imu["accel_x"].values
                stop_input["accel_y"] = aligned_imu["accel_y"].values
                is_stop = detect_stops(stop_input)
                stop_input["is_stop"] = is_stop.values
                pos_df = apply_stop_averaging(stop_input)
        except Exception:
            pass  # IMU unavailable or malformed — silently skip

    # Optional Savitzky-Golay smoothing (preserves trends better than median)
    if use_savgol:
        pos_df = savgol_smooth_trajectory(pos_df, window_length=7, polyorder=2)

    # Optional OSRM snap-to-road (requires network access)
    if osrm:
        speeds = np.zeros(len(pos_df))
        snapped_lats, snapped_lons = osrm_snap_to_road(
            pos_df["LatitudeDegrees"].values,
            pos_df["LongitudeDegrees"].values,
            speeds + 1.0,
        )
        pos_df = pos_df.copy()
        pos_df["LatitudeDegrees"] = snapped_lats
        pos_df["LongitudeDegrees"] = snapped_lons

    return pos_df[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]


# ── Accuracy evaluation ───────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine great-circle distance in metres."""
    R = 6_371_000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def evaluate_accuracy(pred_df, gt_df):
    """Compute accuracy metrics by nearest-epoch matching."""
    gt = gt_df.sort_values("UnixTimeMillis")
    pred = pred_df.sort_values("UnixTimeMillis")

    gt_t = gt["UnixTimeMillis"].values
    gt_lat = gt["LatitudeDegrees"].values
    gt_lon = gt["LongitudeDegrees"].values

    pred_t = pred["UnixTimeMillis"].values
    pred_lat = pred["LatitudeDegrees"].values
    pred_lon = pred["LongitudeDegrees"].values

    idx = np.searchsorted(gt_t, pred_t, side="left")
    idx = np.clip(idx, 0, len(gt_t) - 1)
    idx_prev = np.maximum(idx - 1, 0)
    dt_curr = np.abs(gt_t[idx] - pred_t)
    dt_prev = np.abs(gt_t[idx_prev] - pred_t)
    idx = np.where(dt_prev < dt_curr, idx_prev, idx)

    errors = haversine_m(pred_lat, pred_lon, gt_lat[idx], gt_lon[idx])
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {"mean_m": np.nan, "median_m": np.nan, "p95_m": np.nan, "n_epochs": 0}

    return {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "n_epochs": int(len(errors)),
    }


# ── Full-split submission generator ──────────────────────────────────────────

def run_full_submission(dataset_root, output_path, split="test", osrm=False,
                        use_ml=False, ml_models=None, use_sat_weights=False,
                        use_savgol=False, use_stop_averaging=False,
                        adaptive_blend=False, ml_blend=1.0):
    """Process all trips in a dataset split and write a Kaggle submission CSV."""
    dataset_root = pathlib.Path(dataset_root)
    split_dir = dataset_root / split
    all_rows = []

    # Load sample_submission for exact timestamps
    sample_path = dataset_root / "sample_submission.csv"
    sample_ts = {}
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        for trip_id, grp in sample_df.groupby("tripId"):
            sample_ts[trip_id] = np.sort(grp["UnixTimeMillis"].values)

    # Satellite weight model
    sat_weight_model = None
    if use_sat_weights:
        try:
            import joblib
            sat_model_path = pathlib.Path("models/sat_weight_model.joblib")
            sat_weight_model = joblib.load(sat_model_path)
            print("Satellite ML weighting ENABLED")
        except Exception as exc:
            print(f"Warning: Could not load sat weight model: {exc}. Using default weights.")

    # ML models
    lat_model, lon_model = None, None
    if use_ml and ml_models is not None:
        lat_model, lon_model = ml_models
        print("ML correction ENABLED")
    elif use_ml:
        try:
            from src.ml_correction import load_models
            lat_model, lon_model = load_models()
            print("ML correction ENABLED (loaded from disk)")
        except Exception as exc:
            print(f"Warning: Could not load ML models: {exc}. Running without ML.")
            use_ml = False

    # Process each trip
    trip_dirs = sorted(split_dir.iterdir())
    for trip_dir in tqdm(trip_dirs, desc=f"Processing {split}"):
        if not trip_dir.is_dir():
            continue
        for dev_dir in sorted(trip_dir.iterdir()):
            if not dev_dir.is_dir():
                continue
            trip_id = f"{trip_dir.name}/{dev_dir.name}"
            is_train = (split == "train")
            device_name = dev_dir.name
            try:
                if use_ml and lat_model is not None:
                    from src.ml_correction import (
                        process_trip_with_features, apply_correction
                    )
                    pos_df, feat_df = process_trip_with_features(
                        dev_dir, device_name,
                        sat_weight_model=sat_weight_model)

                    if len(pos_df) > 0 and len(feat_df) > 0:
                        result = apply_correction(
                            pos_df, feat_df, lat_model, lon_model,
                            blend=ml_blend, adaptive_blend=adaptive_blend)
                        if use_stop_averaging:
                            try:
                                imu_df = load_imu(dev_dir)
                                if len(imu_df) > 0:
                                    aligned_imu = align_imu_to_gnss(
                                        imu_df, result["UnixTimeMillis"].values)
                                    stop_input = result.copy()
                                    stop_input["accel_x"] = aligned_imu["accel_x"].values
                                    stop_input["accel_y"] = aligned_imu["accel_y"].values
                                    stop_input["is_stop"] = detect_stops(stop_input).values
                                    result = apply_stop_averaging(stop_input)[
                                        ["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
                            except Exception:
                                pass
                        if use_savgol:
                            result = savgol_smooth_trajectory(result, window_length=7, polyorder=2)
                    else:
                        result = process_trip(dev_dir, train=is_train,
                                              osrm=osrm,
                                              sat_weight_model=sat_weight_model,
                                              use_savgol=use_savgol,
                                              use_stop_averaging=use_stop_averaging)
                else:
                    result = process_trip(dev_dir, train=is_train, osrm=osrm,
                                          sat_weight_model=sat_weight_model,
                                          use_savgol=use_savgol,
                                          use_stop_averaging=use_stop_averaging)

                # Re-index to exact Kaggle-required timestamps via spline
                if trip_id in sample_ts:
                    req_ts = sample_ts[trip_id]
                    our_ts = result["UnixTimeMillis"].values.astype(float)
                    our_lat = result["LatitudeDegrees"].values
                    our_lon = result["LongitudeDegrees"].values

                    if len(our_ts) >= 4:
                        lat_interp = InterpolatedUnivariateSpline(
                            our_ts, our_lat, ext=3)(req_ts.astype(float))
                        lon_interp = InterpolatedUnivariateSpline(
                            our_ts, our_lon, ext=3)(req_ts.astype(float))
                    else:
                        lat_interp = np.interp(req_ts, our_ts, our_lat)
                        lon_interp = np.interp(req_ts, our_ts, our_lon)

                    result = pd.DataFrame({
                        "UnixTimeMillis": req_ts,
                        "LatitudeDegrees": lat_interp,
                        "LongitudeDegrees": lon_interp,
                    })

                result = result.copy()
                result.insert(0, "tripId", trip_id)
                all_rows.append(result)
            except Exception as exc:
                print(f"  Error processing {trip_id}: {exc}")

    if not all_rows:
        print("No results generated.")
        return pd.DataFrame()

    submission = pd.concat(all_rows, ignore_index=True)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}  ({len(submission):,} rows)")
    return submission


# ── Accuracy report ───────────────────────────────────────────────────────────

def run_accuracy_report(dataset_root, n_trips=5, osrm=False):
    """Evaluate accuracy on training trips and print a report."""
    dataset_root = pathlib.Path(dataset_root)
    train_dir = dataset_root / "train"
    report_rows = []

    trip_dirs = sorted(train_dir.iterdir())[:n_trips]
    for trip_dir in tqdm(trip_dirs, desc="Evaluating accuracy"):
        if not trip_dir.is_dir():
            continue
        for dev_dir in sorted(trip_dir.iterdir()):
            if not dev_dir.is_dir():
                continue
            trip_id = f"{trip_dir.name}/{dev_dir.name}"
            try:
                gt = load_ground_truth(dev_dir)
                if gt is None or len(gt) == 0:
                    continue
                result = process_trip(dev_dir, train=True, osrm=osrm)
                metrics = evaluate_accuracy(result, gt)
                metrics["trip_id"] = trip_id
                report_rows.append(metrics)
                print(f"  {trip_id}: mean={metrics['mean_m']:.2f} m  "
                      f"median={metrics['median_m']:.2f} m  "
                      f"p95={metrics['p95_m']:.2f} m")
            except Exception as exc:
                print(f"  Error {trip_id}: {exc}")

    if not report_rows:
        print("No accuracy data computed.")
        return pd.DataFrame()

    report = pd.DataFrame(report_rows)
    mean_agg = float(np.nanmean(report["mean_m"].values))
    median_agg = float(np.nanmean(report["median_m"].values))
    p95_agg = float(np.nanmean(report["p95_m"].values))

    print("\n" + "=" * 60)
    print("AGGREGATE ACCURACY (across evaluated trips)")
    print(f"  Mean error   : {mean_agg:.2f} m  (target <= 2 m, hard limit 3 m)")
    print(f"  Median error : {median_agg:.2f} m  (target <= 1.5 m, hard limit 2.5 m)")
    print(f"  P95 error    : {p95_agg:.2f} m  (target <= 5 m, hard limit 8 m)")
    print("=" * 60)

    failures = []
    if mean_agg > 3.0:
        failures.append(f"Mean error {mean_agg:.2f} m > 3 m hard limit")
    if median_agg > 2.5:
        failures.append(f"Median error {median_agg:.2f} m > 2.5 m hard limit")
    if p95_agg > 8.0:
        failures.append(f"P95 error {p95_agg:.2f} m > 8 m hard limit")

    if failures:
        print("\nACCEPTANCE CRITERIA FAILED:")
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        marginal = []
        if mean_agg > 2.0:
            marginal.append(f"Mean {mean_agg:.2f} m > 2 m target (but within limit)")
        if not marginal:
            print("\nAll acceptance criteria PASSED")
        else:
            print("\nMARGINAL (within hard limits, but above targets):")
            for m in marginal:
                print(f"  ~ {m}")

    return report


# ── ML training entry point ──────────────────────────────────────────────────

def run_ml_training(dataset_root, n_trips=None):
    """Train ML correction models on training data."""
    from src.ml_correction import (
        build_training_dataset, train_correction_model, save_models
    )
    print("=" * 60)
    print("ML CORRECTION MODEL TRAINING")
    print("=" * 60)

    # Build training dataset
    train_df = build_training_dataset(dataset_root, n_trips=n_trips)
    if len(train_df) == 0:
        print("No training data. Aborting.")
        return

    # Save training data for analysis
    train_df.to_csv("ml_training_data.csv", index=False)
    print(f"Training data saved to ml_training_data.csv ({len(train_df)} rows)")

    # Train models with CV
    lat_model, lon_model, cv_df = train_correction_model(train_df)

    # Save models
    save_models(lat_model, lon_model)

    # Save CV results
    cv_df.to_csv("ml_cv_results.csv", index=False)
    print(f"CV results saved to ml_cv_results.csv")

    return lat_model, lon_model


# ── Satellite weight model training ──────────────────────────────────────────

def run_sat_weight_training(dataset_root, n_trips=None):
    """Train a satellite-level LightGBM model for WLS weighting."""
    import joblib
    from src.data_loader import load_gnss
    from src.feature_engineering import build_feature_matrix, train_lgbm_residual_model

    print("=" * 60)
    print("SATELLITE WEIGHT MODEL TRAINING")
    print("=" * 60)

    dataset_root = pathlib.Path(dataset_root)
    train_dir = dataset_root / "train"
    all_features = []

    trip_dirs = sorted(train_dir.iterdir())
    if n_trips is not None:
        trip_dirs = trip_dirs[:n_trips]

    for trip_dir in tqdm(trip_dirs, desc="Building satellite features"):
        if not trip_dir.is_dir():
            continue
        for dev_dir in sorted(trip_dir.iterdir()):
            if not dev_dir.is_dir():
                continue
            trip_id = f"{trip_dir.name}/{dev_dir.name}"
            try:
                gnss_df = load_gnss(dev_dir)
                gt_df = load_ground_truth(dev_dir)
                if gt_df is None or len(gt_df) == 0:
                    continue
                feat_df = build_feature_matrix(gnss_df, gt_df)
                all_features.append(feat_df)
            except Exception as exc:
                print(f"  Error {trip_id}: {exc}")

    if not all_features:
        print("No training data. Aborting.")
        return None

    combined = pd.concat(all_features, ignore_index=True)
    print(f"Training data: {len(combined):,} satellite observations")

    model = train_lgbm_residual_model(combined)

    model_dir = pathlib.Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "sat_weight_model.joblib"
    joblib.dump(model, model_path)
    print(f"Satellite weight model saved to {model_path}")

    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS Decimeter Challenge pipeline")
    parser.add_argument("--root", default="kaggle_dataset")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n-trips", type=int, default=5)
    parser.add_argument("--osrm", action="store_true",
                        help="Snap positions to nearest road via public OSRM API")
    parser.add_argument("--ml", action="store_true",
                        help="Apply ML error correction (requires trained models)")
    parser.add_argument("--train-ml", action="store_true",
                        help="Train ML correction models on training data")
    parser.add_argument("--sat-weights", action="store_true",
                        help="Use ML satellite weighting in WLS solver")
    parser.add_argument("--train-sat-model", action="store_true",
                        help="Train satellite-level weight model")
    parser.add_argument("--savgol", action="store_true",
                        help="Apply Savitzky-Golay smoothing after median filter")
    parser.add_argument("--stop-averaging", action="store_true",
                        help="Average positions within IMU-detected stationary segments")
    parser.add_argument("--adaptive-blend", action="store_true",
                        help="Scale ML correction blend per-epoch based on HDOP")
    parser.add_argument("--ml-blend", type=float, default=1.0,
                        help="ML correction blend strength (0=none, 1=full)")
    args = parser.parse_args()

    if args.train_sat_model:
        n = args.n_trips if args.n_trips != 5 else None
        run_sat_weight_training(pathlib.Path(args.root), n_trips=n)
    elif args.train_ml:
        n = args.n_trips if args.n_trips != 5 else None  # default=all trips
        run_ml_training(pathlib.Path(args.root), n_trips=n)
    elif args.evaluate:
        run_accuracy_report(pathlib.Path(args.root), n_trips=args.n_trips, osrm=args.osrm)
    else:
        run_full_submission(
            pathlib.Path(args.root),
            pathlib.Path(args.output),
            split=args.split,
            osrm=args.osrm,
            use_ml=args.ml,
            use_sat_weights=args.sat_weights,
            use_savgol=args.savgol,
            use_stop_averaging=args.stop_averaging,
            adaptive_blend=args.adaptive_blend,
            ml_blend=args.ml_blend,
        )
