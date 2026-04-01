# feature_engineering.py
# Per-satellite feature engineering and LightGBM residual prediction model.
#
# Original implementation. ML prediction of per-satellite pseudorange residuals
# inspired by Google Smartphone Decimeter Challenge 2022 community techniques.
#
# The feature matrix has one row per (epoch, satellite) observation.
# The reciprocal square of predicted |residual| is used as WLS weight:
#   weight_i = 1 / max(sigma_i, floor)^2

import numpy as np
import pandas as pd
import lightgbm as lgb
from pyproj import Transformer

# WGS-84 constants
WGS84_A  = 6_378_137.0
WGS84_E2 = 6.694_379_990_14e-3

_lla2ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=False)
_ecef2lla = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=False)

# Feature columns used by LightGBM (only columns guaranteed to exist)
FEATURE_COLS = [
    "elevation_deg",
    "azimuth_deg",
    "Cn0DbHz",
    "RawPseudorangeUncertaintyMeters",
    "AccumulatedDeltaRangeUncertaintyMeters",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "SvClockBiasMeters",
    "IonosphericDelayMeters",
    "TroposphericDelayMeters",
    "ConstellationType",
    "MultipathIndicator",
    "pr_minus_geometric_m",    # pseudorange residual after clock removal
    "adr_valid",               # bool cast to int
]


def _lla_to_ecef_np(lat_deg: np.ndarray,
                    lon_deg: np.ndarray,
                    alt_m: np.ndarray) -> np.ndarray:
    # Vectorised geodetic (degrees) to ECEF (metres). Returns (N, 3) array.
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + alt_m) * np.sin(lat)
    return np.column_stack([x, y, z])


def _add_elev_azim_from_ecef(df: pd.DataFrame,
                              ref_lat_arr: np.ndarray,
                              ref_lon_arr: np.ndarray) -> pd.DataFrame:
    # Compute per-row elevation and azimuth from ECEF positions.
    # Fallback if SvElevationDegrees is unavailable.
    # Uses ENU rotation matrix at each receiver position.
    lat = np.deg2rad(ref_lat_arr)
    lon = np.deg2rad(ref_lon_arr)
    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)

    rx_ecef = _lla_to_ecef_np(ref_lat_arr, ref_lon_arr, np.zeros(len(ref_lat_arr)))

    dx = df["SvPositionXEcefMeters"].values - rx_ecef[:, 0]
    dy = df["SvPositionYEcefMeters"].values - rx_ecef[:, 1]
    dz = df["SvPositionZEcefMeters"].values - rx_ecef[:, 2]

    e =  -sn * dx + cn * dy
    n =  -sl * cn * dx - sl * sn * dy + cl * dz
    u =   cl * cn * dx + cl * sn * dy + sl * dz

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    df = df.copy()
    df["elevation_deg"] = np.degrees(np.arcsin(np.clip(u / dist, -1.0, 1.0)))
    df["azimuth_deg"]   = np.degrees(np.arctan2(e, n)) % 360.0
    return df


def build_feature_matrix(
    gnss_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    # Construct per-observation feature matrix for LightGBM.
    # 1. Derive per-epoch receiver ECEF from WLS baseline
    # 2. Compute geometric range to each satellite
    # 3. Estimate per-epoch clock bias as median pseudorange residual
    # 4. Compute pr_minus_geometric_m = pseudorange - geometric range - clock_bias
    # 5. If ground truth available, compute target_residual_m for training
    df = gnss_df.copy()

    # 1. Per-epoch receiver position from WLS ECEF
    wls_x_col = "WlsPositionXEcefMeters"
    wls_y_col = "WlsPositionYEcefMeters"
    wls_z_col = "WlsPositionZEcefMeters"

    epoch_ref = (
        df.groupby("epoch_ms")[[wls_x_col, wls_y_col, wls_z_col]]
        .first()
        .rename(columns={wls_x_col: "rx_x", wls_y_col: "rx_y", wls_z_col: "rx_z"})
    )
    df = df.join(epoch_ref, on="epoch_ms")

    # 2. Geometric range per satellite
    sv_xyz = df[["SvPositionXEcefMeters",
                 "SvPositionYEcefMeters",
                 "SvPositionZEcefMeters"]].values
    rx_xyz = df[["rx_x", "rx_y", "rx_z"]].values
    geo_rng = np.linalg.norm(sv_xyz - rx_xyz, axis=1)

    # 3. Per-epoch clock bias (median residual)
    df["_raw_res"] = df["pseudorange_m"] - geo_rng
    df["clock_bias_m"] = df.groupby("epoch_ms")["_raw_res"].transform("median")

    # 4. Cleaned residual feature
    df["pr_minus_geometric_m"] = df["_raw_res"] - df["clock_bias_m"]
    df.drop(columns=["_raw_res"], inplace=True)

    # Ensure elevation/azimuth columns exist
    if "elevation_deg" not in df.columns or df["elevation_deg"].isna().all():
        rx_lat_v, rx_lon_v, _ = _ecef2lla.transform(
            df["rx_x"].values, df["rx_y"].values, df["rx_z"].values,
            radians=False
        )
        df = _add_elev_azim_from_ecef(df, rx_lat_v, rx_lon_v)

    # Cast bool to int for LightGBM
    df["adr_valid"] = df["adr_valid"].astype(int)

    # 5. Ground truth target (train only)
    if ground_truth_df is not None and len(ground_truth_df) > 0:
        gt = ground_truth_df.sort_values("UnixTimeMillis")
        gt_t   = gt["UnixTimeMillis"].values
        gt_lat = gt["LatitudeDegrees"].values
        gt_lon = gt["LongitudeDegrees"].values

        # Nearest-epoch join
        epoch_arr = df["epoch_ms"].values
        idx = np.searchsorted(gt_t, epoch_arr, side="left")
        idx = np.clip(idx, 0, len(gt_t) - 1)

        idx_prev = np.maximum(idx - 1, 0)
        dt_curr  = np.abs(gt_t[idx]      - epoch_arr)
        dt_prev  = np.abs(gt_t[idx_prev] - epoch_arr)
        idx      = np.where(dt_prev < dt_curr, idx_prev, idx)

        gt_lat_matched = gt_lat[idx]
        gt_lon_matched = gt_lon[idx]

        gt_ecef = _lla_to_ecef_np(gt_lat_matched, gt_lon_matched,
                                   np.zeros(len(gt_lat_matched)))
        gt_rng  = np.linalg.norm(sv_xyz - gt_ecef, axis=1)
        df["target_residual_m"] = df["pseudorange_m"] - gt_rng - df["clock_bias_m"]
    else:
        df["target_residual_m"] = np.nan

    return df.reset_index(drop=True)


def train_lgbm_residual_model(
    feature_df: pd.DataFrame,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
) -> lgb.LGBMRegressor:
    # Train LightGBM to predict |pseudorange residual| in metres.
    # Satellites with large predicted residuals are down-weighted in WLS:
    #   weight_i = 1 / max(f(x_i), floor)^2
    train_mask = (
        feature_df["target_residual_m"].notna() &
        feature_df["pseudorange_m"].notna()
    )
    cols_present = [c for c in FEATURE_COLS if c in feature_df.columns]

    X = feature_df.loc[train_mask, cols_present].fillna(0.0)
    y = feature_df.loc[train_mask, "target_residual_m"].abs()

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,
        min_child_samples=20,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def predict_satellite_weights(
    feature_df: pd.DataFrame,
    model: lgb.LGBMRegressor,
    floor_m: float = 1.0,
) -> np.ndarray:
    # Compute per-observation WLS weights: weight = 1 / max(predicted_residual, floor)^2
    cols_present = [c for c in FEATURE_COLS if c in feature_df.columns]
    X = feature_df[cols_present].fillna(0.0)
    pred_residual = model.predict(X)
    sigma = np.maximum(np.abs(pred_residual), floor_m)
    return 1.0 / sigma ** 2
