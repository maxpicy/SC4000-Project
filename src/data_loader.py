"""
data_loader.py
==============
Parse and synchronise all sensor streams for one (trip, device) pair.

File layout expected
--------------------
device_dir/
    device_gnss.csv       — raw GNSS measurements (43 cols)
    device_imu.csv        — IMU sensor readings (accel/gyro/mag)
    ground_truth.csv      — reference positions (train split only)
    supplemental/
        gnss_log.txt      — raw Android GNSS log (text, multi-type records)

Pseudorange notes
-----------------
The Kaggle dataset already provides ``RawPseudorangeMeters`` — a corrected
pseudorange computed by the Android GNSS API.  We use it directly and apply
the remaining satellite-clock, ionosphere, and troposphere corrections stored
in the same row so that:

    pseudorange_m = RawPseudorangeMeters
                    + SvClockBiasMeters
                    - IonosphericDelayMeters
                    - TroposphericDelayMeters

SvElevationDegrees and SvAzimuthDegrees are pre-computed in the CSV, so we
copy them directly rather than re-deriving from ECEF positions.
"""

import pathlib
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

SPEED_OF_LIGHT = 299_792_458.0  # m/s

# Accumulated Delta Range state bit flags
# (from metadata/accumulated_delta_range_state_bit_map.json)
ADR_STATE_VALID      = 1 << 0   # bit 0 — measurement is valid
ADR_STATE_RESET      = 1 << 1   # bit 1 — discontinuity occurred
ADR_STATE_CYCLE_SLIP = 1 << 2   # bit 2 — cycle slip detected


# ── GNSS CSV loader ───────────────────────────────────────────────────────────

def load_gnss(device_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load and pre-process device_gnss.csv for a single (trip, device).

    Parameters
    ----------
    device_dir : path to the device folder, e.g.
        kaggle_dataset/train/2020-05-15-US-MTV-1/GooglePixel4/

    Returns
    -------
    DataFrame with one row per (epoch, satellite) observation, enriched with:
        pseudorange_m  — corrected pseudorange in metres
        adr_valid      — bool: ADR measurement is usable (no cycle slip/reset)
        epoch_ms       — utcTimeMillis rounded to the nearest second (integer ms)
        elevation_deg  — copy of SvElevationDegrees (for feature engineering)
        azimuth_deg    — copy of SvAzimuthDegrees
    """
    path = pathlib.Path(device_dir) / "device_gnss.csv"
    df = pd.read_csv(path, low_memory=False)

    # Keep only Raw measurement rows (all rows should be Raw, but be defensive)
    df = df[df["MessageType"] == "Raw"].copy()

    # ── Corrected pseudorange ─────────────────────────────────────────────
    # The full pseudorange correction chain applied here:
    #
    #   pseudorange = RawPseudorangeMeters
    #                 + SvClockBiasMeters      (satellite clock correction)
    #                 - IonosphericDelayMeters (ionospheric path delay)
    #                 - TroposphericDelayMeters(tropospheric path delay)
    #                 - IsrbMeters             (inter-system range bias)
    #
    # IsrbMeters (ISRB) is critical for multi-constellation positioning.
    # Each constellation uses a different signal frequency / hardware path,
    # introducing a receiver-specific bias relative to GPS L1:
    #   GPS L1      → ISRB ≈    0 m  (reference)
    #   GLONASS G1  → ISRB ≈ +1158 m
    #   Galileo E1  → ISRB ≈  -216 m
    #   GPS L5/E5A  → ISRB ≈ -2350 m
    # Without this correction, GLONASS satellites appear ~1158 m further
    # than they are, causing the WLS to produce positions ~hundreds of
    # metres from the true location.
    sv_clock_m = df["SvClockBiasMeters"].fillna(0.0)
    iono_m     = df["IonosphericDelayMeters"].fillna(0.0)
    tropo_m    = df["TroposphericDelayMeters"].fillna(0.0)
    isrb_m     = df["IsrbMeters"].fillna(0.0) if "IsrbMeters" in df.columns else 0.0

    df["pseudorange_m"] = (
        df["RawPseudorangeMeters"] + sv_clock_m - iono_m - tropo_m - isrb_m
    )

    # Sanity filter: discard implausible pseudoranges (< 1 000 km or > 90 000 km)
    valid_pr = (df["pseudorange_m"] > 1e6) & (df["pseudorange_m"] < 9e7)
    df.loc[~valid_pr, "pseudorange_m"] = np.nan

    # Also invalidate rows where RawPseudorangeUncertaintyMeters is very large
    high_unc = df["RawPseudorangeUncertaintyMeters"].fillna(999) > 200.0
    df.loc[high_unc, "pseudorange_m"] = np.nan

    # ── ADR (carrier phase) validity flag ────────────────────────────────
    adr_state = df["AccumulatedDeltaRangeState"].fillna(0).astype(int)
    df["adr_valid"] = (
        ((adr_state & ADR_STATE_VALID)      != 0) &
        ((adr_state & ADR_STATE_RESET)      == 0) &
        ((adr_state & ADR_STATE_CYCLE_SLIP) == 0)
    )

    # ── Elevation / azimuth aliases ──────────────────────────────────────
    # The CSV already has pre-computed geometry columns.
    df["elevation_deg"] = df["SvElevationDegrees"]
    df["azimuth_deg"]   = df["SvAzimuthDegrees"]

    # ── Epoch alignment key ───────────────────────────────────────────────
    # Round to nearest second so GNSS and IMU share a common join key.
    df["epoch_ms"] = (
        (df["utcTimeMillis"] / 1000.0).round().astype(np.int64) * 1000
    )

    # ── One signal per satellite per epoch ───────────────────────────────
    # Multiple signal types (GPS L1 + L5, Galileo E1 + E5A) may appear for
    # the same satellite at the same epoch.  Mixing them in TDCP produces
    # huge ADR jumps because L1 and L5 have different carrier frequencies and
    # therefore different accumulated phase values.
    #
    # Strategy: keep only the PRIMARY signal for each satellite:
    #   GPS      → GPS_L1   (if available)
    #   GLONASS  → GLO_G1
    #   Galileo  → GAL_E1
    #   BeiDou   → BDS_B1I
    #   Others   → highest C/N0 row
    #
    # This is implemented by assigning a preference rank and keeping the
    # lowest-rank (highest priority) row per (epoch_ms, Svid, ConstellationType).
    PRIMARY_SIGNALS = {
        "GPS_L1": 0, "GLO_G1": 0, "GAL_E1": 0, "BDS_B1I": 0, "QZS_J1": 0,
        "GPS_L5": 1, "GAL_E5A": 1, "GPS_L2": 2,
    }
    if "SignalType" in df.columns:
        df["_sig_rank"] = df["SignalType"].map(PRIMARY_SIGNALS).fillna(3).astype(int)
        # Sort so that primary signals come first, then best C/N0 as tie-break
        df = df.sort_values(
            ["epoch_ms", "Svid", "ConstellationType", "_sig_rank",
             "Cn0DbHz"],
            ascending=[True, True, True, True, False],
        )
        df = df.drop_duplicates(
            subset=["epoch_ms", "Svid", "ConstellationType"], keep="first"
        )
        df = df.drop(columns=["_sig_rank"])

    return df.reset_index(drop=True)


def load_gnss_raw(device_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load device_gnss.csv keeping ALL signals (no primary-signal dedup).

    Unlike load_gnss(), this does NOT pre-apply pseudorange corrections and
    does NOT filter to one signal per satellite. Corrections are applied
    inline by the solver. Computes CarrierErrorHz for satellite selection.
    """
    path = pathlib.Path(device_dir) / "device_gnss.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[df["MessageType"] == "Raw"].copy()

    # No timestamp rounding — use raw utcTimeMillis as the notebook does.
    # Rounding loses epochs when timestamps straddle a second boundary.

    # Carrier frequency error: deviation from nominal per (Svid, SignalType)
    carrier_ref = df.groupby(["Svid", "SignalType"])["CarrierFrequencyHz"].median()
    df = df.merge(carrier_ref, how="left", on=["Svid", "SignalType"],
                  suffixes=("", "Ref"))
    df["CarrierErrorHz"] = np.abs(df["CarrierFrequencyHz"] - df["CarrierFrequencyHzRef"])

    return df.reset_index(drop=True)


# ── IMU loader ────────────────────────────────────────────────────────────────

def load_imu(device_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load device_imu.csv and apply bias correction.

    Bias-corrects each measurement in-place:
        calibrated_value = uncalibrated_value − bias

    Returns
    -------
    DataFrame with columns:
        MessageType, utcTimeMillis, MeasurementX, MeasurementY, MeasurementZ
    (BiasX/Y/Z are subtracted and dropped from the returned frame)
    """
    path = pathlib.Path(device_dir) / "device_imu.csv"
    df = pd.read_csv(path, low_memory=False)

    for axis in ("X", "Y", "Z"):
        raw_col  = f"Measurement{axis}"
        bias_col = f"Bias{axis}"
        df[raw_col] = df[raw_col] - df[bias_col].fillna(0.0)

    return df[["MessageType", "utcTimeMillis",
               "MeasurementX", "MeasurementY", "MeasurementZ"]]


def align_imu_to_gnss(
    imu_df: pd.DataFrame,
    gnss_epochs_ms: np.ndarray,
) -> pd.DataFrame:
    """
    Interpolate each IMU sensor stream onto the GNSS epoch grid.

    Strategy
    --------
    For each sensor type (UncalAccel, UncalGyro, UncalMag):
      1. Sort by utcTimeMillis
      2. numpy.interp each axis onto gnss_epochs_ms (linear interpolation,
         boundary values clamped)

    This fills gaps and produces exactly one IMU vector per GNSS epoch.

    Parameters
    ----------
    imu_df         : output of load_imu()
    gnss_epochs_ms : array of epoch timestamps (integer ms); need not be sorted

    Returns
    -------
    DataFrame indexed from 0, with columns:
        epoch_ms, accel_x, accel_y, accel_z,
                  gyro_x,  gyro_y,  gyro_z,
                  mag_x,   mag_y,   mag_z
    """
    epochs = np.sort(np.asarray(gnss_epochs_ms, dtype=float))
    result = pd.DataFrame({"epoch_ms": epochs.astype(int)})

    sensor_map = {
        "UncalAccel": ("accel_x", "accel_y", "accel_z"),
        "UncalGyro":  ("gyro_x",  "gyro_y",  "gyro_z"),
        "UncalMag":   ("mag_x",   "mag_y",   "mag_z"),
    }

    for msg_type, (cx, cy, cz) in sensor_map.items():
        sub = (
            imu_df[imu_df["MessageType"] == msg_type]
            .sort_values("utcTimeMillis")
        )
        if len(sub) == 0:
            result[cx] = 0.0
            result[cy] = 0.0
            result[cz] = 0.0
            continue

        t = sub["utcTimeMillis"].values.astype(float)
        for col, axis in zip([cx, cy, cz], ["X", "Y", "Z"]):
            vals = sub[f"Measurement{axis}"].values.astype(float)
            result[col] = np.interp(epochs, t, vals)

    return result.reset_index(drop=True)


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(device_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load ground_truth.csv (train split only).

    Returns an empty DataFrame if the file does not exist (test split).
    """
    path = pathlib.Path(device_dir) / "ground_truth.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── GNSS log parser ───────────────────────────────────────────────────────────

def load_gnss_log(device_dir: pathlib.Path) -> pd.DataFrame:
    """
    Parse supplemental/gnss_log.txt into a tidy DataFrame.

    File format
    -----------
    Header lines start with '#'.  Column-header comments for each record type
    look like:
        # Raw,utcTimeMillis,TimeNanos,...

    Data lines are comma-separated with the record type as the first token:
        Raw,1589510400000,123456789,...

    Returns
    -------
    DataFrame with column 'type' (Raw / Fix / UncalAccel / …) plus all
    numeric or string fields parsed from the header row for that type.
    """
    log_path = (
        pathlib.Path(device_dir) / "supplemental" / "gnss_log.txt"
    )
    headers: dict[str, list[str]] = {}
    records: list[dict] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Strip leading '#' and whitespace to get the inner content
                inner = line.lstrip("# ").strip()
                parts = inner.split(",")
                # A header comment for a record type has ≥ 2 comma-separated fields
                if len(parts) > 1:
                    rec_type = parts[0]
                    # Only overwrite if we haven't seen a longer header yet
                    if rec_type not in headers or len(parts) > len(headers[rec_type]):
                        headers[rec_type] = parts
                continue

            parts = line.split(",")
            rec_type = parts[0]
            if rec_type not in headers:
                continue

            cols = headers[rec_type]
            row: dict = {"type": rec_type}
            for col, val in zip(cols[1:], parts[1:]):
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = val
            records.append(row)

    return pd.DataFrame(records)
