# post_processing.py
# Post-processing routines applied after the EKF to produce the final submission trajectory.
#
# Routines:
#   detect_stops         - classify stationary vs. moving epochs using IMU variance
#   apply_stop_averaging - replace stop-segment positions with the segment median
#   median_filter_trajectory - remove position spikes with a median filter
#   ensemble_submissions - weighted coordinate averaging across multiple runs

import time
import numpy as np
import pandas as pd
import requests
from scipy.signal import medfilt, savgol_filter

# Rolling window for variance computation (number of epochs, ~seconds at 1 Hz)
STOP_WINDOW_EPOCHS = 5

# Horizontal acceleration variance threshold for stationary detection.
# At rest: variance ~ 0.0025. Threshold of 0.04 separates rest from walking/driving.
STOP_VAR_THRESHOLD = 0.04


def detect_stops(
    df: pd.DataFrame,
    accel_x_col: str = "accel_x",
    accel_y_col: str = "accel_y",
    window: int = STOP_WINDOW_EPOCHS,
    threshold: float = STOP_VAR_THRESHOLD,
) -> pd.Series:
    # Classify each epoch as stationary (True) or moving (False).
    # Uses rolling variance of horizontal acceleration magnitude.
    horiz = np.sqrt(df[accel_x_col].values ** 2 + df[accel_y_col].values ** 2)
    series = pd.Series(horiz, index=df.index)
    roll_var = series.rolling(
        window=max(1, window), center=True, min_periods=1
    ).var()
    return roll_var < threshold


def apply_stop_averaging(
    pos_df: pd.DataFrame,
    lat_col: str = "LatitudeDegrees",
    lon_col: str = "LongitudeDegrees",
    is_stop_col: str = "is_stop",
) -> pd.DataFrame:
    # Replace positions within each stationary segment with the segment median.
    # Reduces random GNSS drift during stops by sqrt(N). Uses median for outlier robustness.
    df = pos_df.copy()
    stops = df[is_stop_col].astype(bool)

    segment_id = (stops != stops.shift()).cumsum()
    df["_seg"] = segment_id

    for seg_id, group in df.groupby("_seg"):
        if not group[is_stop_col].iloc[0]:
            continue
        med_lat = group[lat_col].median()
        med_lon = group[lon_col].median()
        df.loc[group.index, lat_col] = med_lat
        df.loc[group.index, lon_col] = med_lon

    return df.drop(columns=["_seg"])


def median_filter_trajectory(
    pos_df: pd.DataFrame,
    lat_col: str = "LatitudeDegrees",
    lon_col: str = "LongitudeDegrees",
    kernel_size: int = 3,
) -> pd.DataFrame:
    # Apply 1-D median filter to remove single-epoch position spikes.
    if kernel_size % 2 == 0:
        kernel_size += 1
    df = pos_df.copy()
    df[lat_col] = medfilt(df[lat_col].values.astype(float), kernel_size=kernel_size)
    df[lon_col] = medfilt(df[lon_col].values.astype(float), kernel_size=kernel_size)
    return df


def despike_trajectory(
    pos_df: pd.DataFrame,
    lat_col: str = "LatitudeDegrees",
    lon_col: str = "LongitudeDegrees",
    threshold_m: float = 30.0,
) -> pd.DataFrame:
    # Remove spikes where position jumps far from both neighbors but neighbors are consistent.
    # Replaces spikes with linear interpolation.
    df = pos_df.copy()
    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)
    n = len(lat)
    if n < 3:
        return df

    cos_lat = np.cos(np.deg2rad(np.nanmean(lat)))
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * cos_lat

    def dist_m(i, j):
        return np.sqrt(((lat[i] - lat[j]) * m_per_deg_lat) ** 2 +
                       ((lon[i] - lon[j]) * m_per_deg_lon) ** 2)

    spike_idx = []
    for i in range(1, n - 1):
        d_prev = dist_m(i, i - 1)
        d_next = dist_m(i, i + 1)
        d_neighbors = dist_m(i - 1, i + 1)
        if d_prev > threshold_m and d_next > threshold_m and d_neighbors < threshold_m:
            spike_idx.append(i)

    if spike_idx:
        for i in spike_idx:
            lat[i] = (lat[i - 1] + lat[i + 1]) / 2
            lon[i] = (lon[i - 1] + lon[i + 1]) / 2
        df[lat_col] = lat
        df[lon_col] = lon

    return df


def savgol_smooth_trajectory(
    pos_df: pd.DataFrame,
    lat_col: str = "LatitudeDegrees",
    lon_col: str = "LongitudeDegrees",
    window_length: int = 7,
    polyorder: int = 2,
) -> pd.DataFrame:
    # Apply Savitzky-Golay smoothing to trajectory coordinates.
    df = pos_df.copy()
    n = len(df)
    if n < window_length:
        return df
    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        return df
    df[lat_col] = savgol_filter(df[lat_col].values.astype(float), wl, polyorder)
    df[lon_col] = savgol_filter(df[lon_col].values.astype(float), wl, polyorder)
    return df


def ensemble_submissions(
    submissions: list[pd.DataFrame],
    weights=None,
) -> pd.DataFrame:
    # Weighted coordinate average across multiple submission DataFrames.
    # All DataFrames must have identical rows in the same order.
    n = len(submissions)
    if n == 0:
        raise ValueError("Need at least one submission")
    if n == 1:
        return submissions[0].copy()

    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    base = submissions[0][["tripId", "UnixTimeMillis"]].copy()
    lat = sum(w * df["LatitudeDegrees"].values
              for w, df in zip(weights, submissions))
    lon = sum(w * df["LongitudeDegrees"].values
              for w, df in zip(weights, submissions))

    base["LatitudeDegrees"]  = lat
    base["LongitudeDegrees"] = lon
    return base


_OSRM_BASE = "http://router.project-osrm.org/nearest/v1/driving"
_OSRM_BATCH = 50
_OSRM_SLEEP = 0.05
_STOP_SPEED = 0.5  # m/s below which vehicle is considered stopped


def osrm_snap_to_road(lats, lons, speeds, max_snap_dist_m=15.0, timeout=5.0):
    # Snap positions to nearest road using OSRM nearest service.
    # Only moving epochs (speed > 0.5 m/s) are snapped, within max_snap_dist_m.
    # Returns original coordinates unchanged on network error.
    lats   = np.asarray(lats,   dtype=float).copy()
    lons   = np.asarray(lons,   dtype=float).copy()
    speeds = np.asarray(speeds, dtype=float)
    N      = len(lats)

    moving_idx = np.where(speeds >= _STOP_SPEED)[0]
    if len(moving_idx) == 0:
        return lats, lons

    for batch_start in range(0, len(moving_idx), _OSRM_BATCH):
        batch = moving_idx[batch_start: batch_start + _OSRM_BATCH]
        for i in batch:
            try:
                url  = f"{_OSRM_BASE}/{lons[i]:.6f},{lats[i]:.6f}"
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != "Ok":
                    continue
                wp   = data["waypoints"][0]
                dist = float(wp["distance"])
                if dist < max_snap_dist_m:
                    lons[i] = float(wp["location"][0])
                    lats[i] = float(wp["location"][1])
            except Exception:
                pass
        if batch_start + _OSRM_BATCH < len(moving_idx):
            time.sleep(_OSRM_SLEEP)

    return lats, lons
