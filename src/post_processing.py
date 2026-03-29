"""
post_processing.py
==================
Post-processing routines applied after the EKF to produce the final
submission trajectory.

Routines
--------
detect_stops        — classify stationary vs. moving epochs using IMU variance
apply_stop_averaging— replace stop-segment positions with the segment median
median_filter_trajectory — remove position spikes with a median filter
ensemble_submissions — weighted coordinate averaging across multiple runs
"""

import time
import numpy as np
import pandas as pd
import requests
from scipy.signal import medfilt, savgol_filter

# ── Tuneable constants ────────────────────────────────────────────────────────

# Rolling window for variance computation (number of epochs, ≈ seconds at 1 Hz)
STOP_WINDOW_EPOCHS = 5

# Horizontal acceleration variance threshold below which the phone is stationary.
# Units: (m/s²)².  At rest the horizontal accel is dominated by noise (< 0.05 m/s²),
# giving variance ≈ 0.0025.  A threshold of 0.04 comfortably separates rest from
# walking/driving (> 0.1 m/s² variations).
STOP_VAR_THRESHOLD = 0.04


# ── Stop detection ────────────────────────────────────────────────────────────

def detect_stops(
    df: pd.DataFrame,
    accel_x_col: str = "accel_x",
    accel_y_col: str = "accel_y",
    window: int = STOP_WINDOW_EPOCHS,
    threshold: float = STOP_VAR_THRESHOLD,
) -> pd.Series:
    """
    Classify each epoch as stationary (True) or moving (False).

    Method
    ------
    Compute the rolling variance of the horizontal acceleration magnitude
    over a centred window.  If the variance drops below ``threshold`` the
    phone is considered stationary.

    Horizontal magnitude is used (not vertical) because vertical acceleration
    always contains ~9.81 m/s² of gravity; horizontal components are
    near-zero when stationary.

    Parameters
    ----------
    df          : DataFrame with accel_x and accel_y columns
    accel_x_col : name of the East (or body-X) acceleration column
    accel_y_col : name of the North (or body-Y) acceleration column
    window      : rolling window size (epochs)
    threshold   : variance threshold (m/s²)²

    Returns
    -------
    Boolean Series aligned to df's index (True = stationary).
    """
    horiz = np.sqrt(df[accel_x_col].values ** 2 + df[accel_y_col].values ** 2)
    series = pd.Series(horiz, index=df.index)
    roll_var = series.rolling(
        window=max(1, window), center=True, min_periods=1
    ).var()
    return roll_var < threshold


# ── Stop-segment position averaging ──────────────────────────────────────────

def apply_stop_averaging(
    pos_df: pd.DataFrame,
    lat_col: str = "LatitudeDegrees",
    lon_col: str = "LongitudeDegrees",
    is_stop_col: str = "is_stop",
) -> pd.DataFrame:
    """
    Replace positions within each stationary segment with the segment median.

    Rationale
    ---------
    When the vehicle/phone is stationary the true position is constant but
    individual GNSS fixes drift within ±5 m due to multipath and noise.
    Averaging all fixes in a stop segment reduces this random error
    by √N.  The median is used instead of mean to be robust against the
    occasional large outlier.

    Contiguous stop epochs form one segment; the segment median replaces all
    individual estimates within that segment.  Moving segments are unchanged.

    Parameters
    ----------
    pos_df      : DataFrame with lat/lon and is_stop columns
    lat_col     : name of latitude column
    lon_col     : name of longitude column
    is_stop_col : boolean column identifying stationary epochs

    Returns
    -------
    Copy of pos_df with lat/lon replaced inside stop segments.
    """
    df = pos_df.copy()
    stops = df[is_stop_col].astype(bool)

    # Assign a segment ID to each contiguous run of same is_stop value
    segment_id = (stops != stops.shift()).cumsum()
    df["_seg"] = segment_id

    for seg_id, group in df.groupby("_seg"):
        if not group[is_stop_col].iloc[0]:
            continue   # moving segment — leave as-is
        med_lat = group[lat_col].median()
        med_lon = group[lon_col].median()
        df.loc[group.index, lat_col] = med_lat
        df.loc[group.index, lon_col] = med_lon

    return df.drop(columns=["_seg"])


# ── Trajectory smoothing ──────────────────────────────────────────────────────

def median_filter_trajectory(
    pos_df: pd.DataFrame,
    lat_col: str = "LatitudeDegrees",
    lon_col: str = "LongitudeDegrees",
    kernel_size: int = 3,
) -> pd.DataFrame:
    """
    Apply a 1-D median filter along the trajectory to remove position spikes.

    The median filter replaces each value with the median of its kernel_size
    nearest neighbours.  Spikes (single-epoch large errors) are suppressed
    without blurring smooth motion.

    Parameters
    ----------
    pos_df      : DataFrame with lat/lon columns
    kernel_size : filter kernel width (must be odd; 3 = single-neighbour median)

    Returns
    -------
    Copy of pos_df with filtered lat/lon columns.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1   # scipy.signal.medfilt requires odd kernel
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
    """
    Remove position spikes by detecting epochs where the position jumps far
    from both neighbors and replacing them with linear interpolation.

    A spike at epoch i is detected when both:
      dist(i, i-1) > threshold AND dist(i, i+1) > threshold
    but dist(i-1, i+1) < threshold (neighbors are consistent).
    """
    df = pos_df.copy()
    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)
    n = len(lat)
    if n < 3:
        return df

    # Approximate meters per degree at this latitude
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
    """Apply Savitzky-Golay smoothing to trajectory coordinates."""
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


# ── Submission ensemble ───────────────────────────────────────────────────────

def ensemble_submissions(
    submissions: list[pd.DataFrame],
    weights=None,  # list[float] | None
) -> pd.DataFrame:
    """
    Produce a weighted coordinate average across multiple submission DataFrames.

    This is the simplest and often most effective ensemble strategy: averaging
    independent predictions reduces random error by √N while preserving
    systematic improvements in each individual model.

    All DataFrames must have identical rows in the same order with columns:
        tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees

    Parameters
    ----------
    submissions : list of submission DataFrames
    weights     : relative weights (will be normalised to sum to 1.0);
                  if None, uniform weights are used

    Returns
    -------
    Single ensembled submission DataFrame.
    """
    n = len(submissions)
    if n == 0:
        raise ValueError("Need at least one submission")
    if n == 1:
        return submissions[0].copy()

    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]   # normalise

    base = submissions[0][["tripId", "UnixTimeMillis"]].copy()
    lat = sum(w * df["LatitudeDegrees"].values
              for w, df in zip(weights, submissions))
    lon = sum(w * df["LongitudeDegrees"].values
              for w, df in zip(weights, submissions))

    base["LatitudeDegrees"]  = lat
    base["LongitudeDegrees"] = lon
    return base


# ── OSRM snap-to-road ─────────────────────────────────────────────────────────

_OSRM_BASE = "http://router.project-osrm.org/nearest/v1/driving"
_OSRM_BATCH = 50    # requests per batch
_OSRM_SLEEP = 0.05  # seconds between batches
_STOP_SPEED = 0.5   # m/s below which vehicle is considered stopped


def osrm_snap_to_road(lats, lons, speeds, max_snap_dist_m=15.0, timeout=5.0):
    """
    Snap positions to the nearest road using the OSRM nearest service.

    Only moving epochs (speed > 0.5 m/s) are snapped, and only when the
    nearest road point is within `max_snap_dist_m`.  On any network error
    the original coordinates are returned unchanged.

    Args:
        lats:            (N,) latitude array in degrees.
        lons:            (N,) longitude array in degrees.
        speeds:          (N,) vehicle speed in m/s.
        max_snap_dist_m: Maximum distance to road for snapping (metres).
        timeout:         HTTP request timeout in seconds.

    Returns:
        (snapped_lats, snapped_lons) — both (N,) arrays in degrees.
    """
    lats   = np.asarray(lats,   dtype=float).copy()
    lons   = np.asarray(lons,   dtype=float).copy()
    speeds = np.asarray(speeds, dtype=float)
    N      = len(lats)

    moving_idx = np.where(speeds >= _STOP_SPEED)[0]
    if len(moving_idx) == 0:
        return lats, lons

    # Process in batches
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
                # Network error or bad response: keep original coordinate
                pass
        if batch_start + _OSRM_BATCH < len(moving_idx):
            time.sleep(_OSRM_SLEEP)

    return lats, lons
