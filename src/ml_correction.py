# ml_correction.py
# LightGBM-based per-epoch lat/lon error correction for the GNSS pipeline.
#
# Train two LightGBM regressors to predict lat/lon errors (predicted - true)
# from per-epoch features, then subtract predictions from pipeline output.
#
# Position-level ML correction was used by multiple top-10 finishers in
# the Google Smartphone Decimeter Challenge 2022. This implementation is original.
#
# Phone-specific error modelling and vehicle heading features inspired by
# J.B.O. Mitchell's phone-specific bias correction notebooks:
#   https://www.kaggle.com/code/jbomitchell/phone-specific-bias-corrections-clipped
# Built on saitodevel01's bias analysis from the 2021 competition:
#   https://www.kaggle.com/code/saitodevel01/gsdc-bias-eda
#   https://www.kaggle.com/code/saitodevel01/gsdc-bias-correction

import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.data_loader import load_gnss_raw, load_ground_truth
from src.gnss_solver import solve_trip_robust
from src.post_processing import median_filter_trajectory

# Phone model encoding
PHONE_MODELS = {
    "GooglePixel4": 0,
    "GooglePixel4XL": 1,
    "GooglePixel5": 2,
    "SamsungGalaxyS20Ultra": 3,
    "XiaomiMi8": 4,
}

# Feature columns for ML model
FEATURE_COLS = [
    "n_sats_pos", "n_sats_vel", "n_sats_raw",
    "mean_cn0", "std_cn0", "min_cn0", "max_cn0",
    "mean_elev", "std_elev", "min_elev",
    "mean_pr_unc", "std_pr_unc", "max_pr_unc",
    "hdop", "vdop", "pdop",
    "wls_residual_norm", "cov_trace",
    "speed_mps", "n_multipath",
    "frac_gps", "frac_glonass", "frac_galileo", "frac_beidou",
    "phone_model",
    "lat", "lon",
    "time_since_start",
    # Rolling features
    "speed_mean_5", "speed_std_5",
    "speed_mean_15", "speed_std_15",
    "pos_jitter_5", "pos_jitter_15",
    "cn0_mean_5", "hdop_mean_5",
]

MODEL_DIR = pathlib.Path("models")


def _compute_bearing_deg(lat1, lon1, lat2, lon2):
    # Forward bearing in degrees clockwise from north.
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.rad2deg(np.arctan2(x, y)) % 360


def haversine_m(lat1, lon1, lat2, lon2):
    # Haversine great-circle distance in metres.
    R = 6_371_000.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_rolling_features(df):
    # Add rolling statistics to feature DataFrame.
    df = df.copy()

    for w in [5, 15]:
        df[f"speed_mean_{w}"] = df["speed_mps"].rolling(w, min_periods=1, center=True).mean()
        df[f"speed_std_{w}"] = df["speed_mps"].rolling(w, min_periods=1, center=True).std().fillna(0)

    if "lat" in df.columns and "lon" in df.columns:
        lat_diff = df["lat"].diff().abs()
        lon_diff = df["lon"].diff().abs()
        jitter = np.sqrt(lat_diff**2 + lon_diff**2)
        for w in [5, 15]:
            df[f"pos_jitter_{w}"] = jitter.rolling(w, min_periods=1, center=True).mean()

        df["position_jump_m"] = haversine_m(
            df["lat"].values,
            df["lon"].values,
            df["lat"].shift(1).bfill().values,
            df["lon"].shift(1).bfill().values,
        )

        df["lat_accel"] = df["lat"].diff().diff().fillna(0)
        df["lon_accel"] = df["lon"].diff().diff().fillna(0)

        df["epoch_consistency_5"] = jitter.rolling(5, min_periods=1, center=True).std().fillna(0)

        # Vehicle heading features
        if len(df) >= 2:
            lat_v = df["lat"].values
            lon_v = df["lon"].values
            bearing = np.zeros(len(df))
            bearing[1:] = _compute_bearing_deg(lat_v[:-1], lon_v[:-1], lat_v[1:], lon_v[1:])
            bearing[0] = bearing[1]
            bearing_rad = np.deg2rad(bearing)
            df["cos_bearing"] = np.cos(bearing_rad)
            df["sin_bearing"] = np.sin(bearing_rad)
            bearing_s = pd.Series(bearing)
            # Wrap-around-safe absolute delta (e.g. 355->5 = 10 deg, not 350)
            raw_delta = bearing_s.diff().abs()
            wrapped = np.minimum(raw_delta, 360.0 - raw_delta).fillna(0)
            df["bearing_change_rate"] = wrapped.values
            df["bearing_std_5"] = bearing_s.rolling(5, min_periods=1, center=True).std().fillna(0).values
        else:
            df["cos_bearing"] = 1.0
            df["sin_bearing"] = 0.0
            df["bearing_change_rate"] = 0.0
            df["bearing_std_5"] = 0.0
    else:
        for w in [5, 15]:
            df[f"pos_jitter_{w}"] = 0.0
        df["position_jump_m"] = 0.0
        df["lat_accel"] = 0.0
        df["lon_accel"] = 0.0
        df["epoch_consistency_5"] = 0.0
        df["cos_bearing"] = 1.0
        df["sin_bearing"] = 0.0
        df["bearing_change_rate"] = 0.0
        df["bearing_std_5"] = 0.0

    df["cn0_mean_5"] = df["mean_cn0"].rolling(5, min_periods=1, center=True).mean()
    df["hdop_mean_5"] = df["hdop"].rolling(5, min_periods=1, center=True).mean()
    df["hdop_change"] = df["hdop"].diff().fillna(0)
    df["residual_mean_5"] = df["wls_residual_norm"].rolling(5, min_periods=1, center=True).mean()
    df["pr_unc_change"] = df["mean_pr_unc"].diff().fillna(0)
    df["speed_change"] = df["speed_mps"].diff().abs().fillna(0)
    df["n_sats_change"] = df["n_sats_pos"].diff().fillna(0).astype(float)

    if "utcTimeMillis" in df.columns:
        t0 = df["utcTimeMillis"].iloc[0]
        df["time_since_start"] = (df["utcTimeMillis"] - t0) / 1000.0
    else:
        df["time_since_start"] = np.arange(len(df), dtype=float)

    return df


def process_trip_with_features(device_dir, device_name, sat_weight_model=None,
                               use_dualfreq=False, sigma_mahalanobis=30.0):
    # Run pipeline on a single trip, return (pos_df, features_df).
    gnss_df = load_gnss_raw(device_dir)
    if len(gnss_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    result, feat_df = solve_trip_robust(
        gnss_df, device_name=device_name, collect_features=True,
        sat_weight_model=sat_weight_model, use_dualfreq=use_dualfreq,
        sigma_mahalanobis=sigma_mahalanobis)

    if len(result) == 0 or len(feat_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    pos_df = pd.DataFrame({
        "UnixTimeMillis": result["epoch_ms"].values.astype(np.int64),
        "LatitudeDegrees": result["lat"].values,
        "LongitudeDegrees": result["lon"].values,
    })
    pos_df["LatitudeDegrees"] = pos_df["LatitudeDegrees"].interpolate().ffill().bfill()
    pos_df["LongitudeDegrees"] = pos_df["LongitudeDegrees"].interpolate().ffill().bfill()

    pos_df = median_filter_trajectory(pos_df, kernel_size=3)

    feat_df["lat"] = pos_df["LatitudeDegrees"].values
    feat_df["lon"] = pos_df["LongitudeDegrees"].values
    feat_df["phone_model"] = PHONE_MODELS.get(device_name, -1)
    feat_df = add_rolling_features(feat_df)

    return pos_df, feat_df


def build_training_dataset(dataset_root, n_trips=None, use_dualfreq=False, sigma_mahalanobis=30.0):
    # Extract features and targets from all training trips.
    # Returns DataFrame with features + lat_error + lon_error columns.
    dataset_root = pathlib.Path(dataset_root)
    train_dir = dataset_root / "train"

    all_rows = []
    trip_dirs = sorted(train_dir.iterdir())
    if n_trips is not None:
        trip_dirs = trip_dirs[:n_trips]

    for trip_dir in tqdm(trip_dirs, desc="Extracting features"):
        if not trip_dir.is_dir():
            continue
        for dev_dir in sorted(trip_dir.iterdir()):
            if not dev_dir.is_dir():
                continue
            trip_id = f"{trip_dir.name}/{dev_dir.name}"
            device_name = dev_dir.name
            try:
                gt = load_ground_truth(dev_dir)
                if gt is None or len(gt) == 0:
                    continue

                pos_df, feat_df = process_trip_with_features(
                    dev_dir, device_name, use_dualfreq=use_dualfreq,
                    sigma_mahalanobis=sigma_mahalanobis)
                if len(pos_df) == 0 or len(feat_df) == 0:
                    continue

                gt = gt.sort_values("UnixTimeMillis")
                gt_t = gt["UnixTimeMillis"].values
                gt_lat = gt["LatitudeDegrees"].values
                gt_lon = gt["LongitudeDegrees"].values

                pred_t = pos_df["UnixTimeMillis"].values
                pred_lat = pos_df["LatitudeDegrees"].values
                pred_lon = pos_df["LongitudeDegrees"].values

                # Nearest-epoch matching
                idx = np.searchsorted(gt_t, pred_t, side="left")
                idx = np.clip(idx, 0, len(gt_t) - 1)
                idx_prev = np.maximum(idx - 1, 0)
                dt_curr = np.abs(gt_t[idx] - pred_t)
                dt_prev = np.abs(gt_t[idx_prev] - pred_t)
                idx = np.where(dt_prev < dt_curr, idx_prev, idx)

                lat_error = pred_lat - gt_lat[idx]
                lon_error = pred_lon - gt_lon[idx]

                feat_df = feat_df.copy()
                feat_df["lat_error"] = lat_error
                feat_df["lon_error"] = lon_error
                feat_df["trip_id"] = trip_id

                feat_df["haversine_error_m"] = haversine_m(
                    pred_lat, pred_lon, gt_lat[idx], gt_lon[idx])

                all_rows.append(feat_df)
                n_epochs = len(feat_df)
                mean_err = feat_df["haversine_error_m"].mean()
                print(f"  {trip_id}: {n_epochs} epochs, mean_err={mean_err:.2f}m")

            except Exception as exc:
                print(f"  Error {trip_id}: {exc}")

    if not all_rows:
        print("No training data extracted.")
        return pd.DataFrame()

    train_df = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal: {len(train_df)} epochs, "
          f"{train_df['trip_id'].nunique()} trips, "
          f"mean error: {train_df['haversine_error_m'].mean():.2f}m")
    return train_df


def train_correction_model(train_df, n_folds=5):
    # Train LightGBM lat/lon error correction models with trip-level CV.
    # Returns (lat_model, lon_model, cv_results_df).
    cols_present = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"Using {len(cols_present)} features: {cols_present}")

    mask = train_df["lat_error"].notna() & train_df["lon_error"].notna()
    df = train_df[mask].copy()

    X = df[cols_present].fillna(0.0)
    y_lat = df["lat_error"].values
    y_lon = df["lon_error"].values
    groups = df["trip_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    cv_errors = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_lat, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_lat_tr, y_lat_val = y_lat[train_idx], y_lat[val_idx]
        y_lon_tr, y_lon_val = y_lon[train_idx], y_lon[val_idx]

        lgbm_params = dict(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
        )
        lat_m = lgb.LGBMRegressor(**lgbm_params)
        lat_m.fit(X_tr, y_lat_tr,
                  eval_set=[(X_val, y_lat_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        lon_m = lgb.LGBMRegressor(**lgbm_params)
        lon_m.fit(X_tr, y_lon_tr,
                  eval_set=[(X_val, y_lon_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        val_df = df.iloc[val_idx].copy()
        val_df["corrected_lat"] = val_df["lat"].values - lat_m.predict(X_val)
        val_df["corrected_lon"] = val_df["lon"].values - lon_m.predict(X_val)

        for trip_id, trip_grp in val_df.groupby("trip_id"):
            orig_lat = trip_grp["lat"].values
            orig_lon = trip_grp["lon"].values
            corr_lat = trip_grp["corrected_lat"].values
            corr_lon = trip_grp["corrected_lon"].values

            orig_err = trip_grp["haversine_error_m"].values

            gt_lat = orig_lat - trip_grp["lat_error"].values
            gt_lon = orig_lon - trip_grp["lon_error"].values
            corr_err = haversine_m(corr_lat, corr_lon, gt_lat, gt_lon)

            device_name = trip_id.split("/")[-1]
            cv_errors.append({
                "fold": fold,
                "trip_id": trip_id,
                "device": device_name,
                "n_epochs": len(trip_grp),
                "orig_mean": float(np.mean(orig_err)),
                "orig_p50": float(np.percentile(orig_err, 50)),
                "orig_p95": float(np.percentile(orig_err, 95)),
                "corr_mean": float(np.mean(corr_err)),
                "corr_p50": float(np.percentile(corr_err, 50)),
                "corr_p95": float(np.percentile(corr_err, 95)),
            })

        print(f"  Fold {fold}: lat_estimators={lat_m.best_iteration_}, "
              f"lon_estimators={lon_m.best_iteration_}")

    cv_df = pd.DataFrame(cv_errors)

    print("\nCROSS-VALIDATION RESULTS (leave-trips-out)")
    print(f"  {'Metric':<25} {'Original':>12} {'Corrected':>12} {'Delta':>10}")
    metrics = [("Mean error (m)", "orig_mean", "corr_mean"),
               ("P50 error (m)", "orig_p50", "corr_p50"),
               ("P95 error (m)", "orig_p95", "corr_p95")]
    for name, orig_col, corr_col in metrics:
        o = cv_df[orig_col].mean()
        c = cv_df[corr_col].mean()
        print(f"  {name:<25} {o:>12.3f} {c:>12.3f} {c-o:>+10.3f}")

    # Competition metric: mean of (p50 + p95) / 2 per device
    print(f"\n  Competition metric (mean per-phone (p50+p95)/2):")
    for label, p50_col, p95_col in [("Original", "orig_p50", "orig_p95"),
                                     ("Corrected", "corr_p50", "corr_p95")]:
        per_device = cv_df.groupby("device").agg({p50_col: "mean", p95_col: "mean"})
        per_device["comp_metric"] = (per_device[p50_col] + per_device[p95_col]) / 2
        comp = per_device["comp_metric"].mean()
        print(f"    {label}: {comp:.3f} m")

    # Train final models on ALL data
    print("\nTraining final models on all data...")
    final_params = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
    )
    lat_model = lgb.LGBMRegressor(**final_params)
    lat_model.fit(X, y_lat)

    lon_model = lgb.LGBMRegressor(**final_params)
    lon_model.fit(X, y_lon)

    print("\nTop 15 feature importances (lat model):")
    imp = pd.Series(lat_model.feature_importances_, index=cols_present)
    for feat, val in imp.nlargest(15).items():
        print(f"  {feat}: {val}")

    return lat_model, lon_model, cv_df


def save_models(lat_model, lon_model, model_dir=None):
    # Save trained models to disk.
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lat_model, model_dir / "lat_model.joblib")
    joblib.dump(lon_model, model_dir / "lon_model.joblib")
    print(f"Models saved to {model_dir}/")


def load_models(model_dir=None):
    # Load trained models from disk.
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = pathlib.Path(model_dir)
    lat_model = joblib.load(model_dir / "lat_model.joblib")
    lon_model = joblib.load(model_dir / "lon_model.joblib")
    return lat_model, lon_model


def apply_correction(pos_df, feat_df, lat_model, lon_model, blend=1.0,
                     cap_degrees=None, adaptive_blend=False):
    # Apply ML error correction to positions.
    # blend: correction strength (0=none, 1=full)
    # adaptive_blend: scale blend per-epoch using HDOP (low HDOP -> 50% blend, high -> 100%)
    cols_present = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[cols_present].fillna(0.0)

    pred_lat_err = lat_model.predict(X)
    pred_lon_err = lon_model.predict(X)

    if cap_degrees is not None:
        pred_lat_err = np.clip(pred_lat_err, -cap_degrees, cap_degrees)
        pred_lon_err = np.clip(pred_lon_err, -cap_degrees, cap_degrees)

    if adaptive_blend and "hdop" in feat_df.columns:
        hdop = np.where(np.isfinite(feat_df["hdop"].values),
                        feat_df["hdop"].values, 2.5)
        blend_arr = np.clip(blend * (0.5 + 0.5 * (hdop - 1.5) / 2.5), 0.5 * blend, blend)
    else:
        blend_arr = blend

    corrected = pos_df.copy()
    corrected["LatitudeDegrees"] = pos_df["LatitudeDegrees"].values - blend_arr * pred_lat_err
    corrected["LongitudeDegrees"] = pos_df["LongitudeDegrees"].values - blend_arr * pred_lon_err

    return corrected
