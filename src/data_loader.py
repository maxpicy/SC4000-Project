# data_loader.py
# Parse and synchronise all sensor streams for one (trip, device) pair.
#
# Expected file layout:
# device_dir/
#     device_gnss.csv       - raw GNSS measurements (43 cols)
#     device_imu.csv        - IMU sensor readings (accel/gyro/mag)
#     ground_truth.csv      - reference positions (train split only)
#     supplemental/
#         gnss_log.txt      - raw Android GNSS log (text, multi-type records)
#
# Pseudorange correction applied:
#   pseudorange_m = RawPseudorangeMeters
#                   + SvClockBiasMeters
#                   - IonosphericDelayMeters
#                   - TroposphericDelayMeters
#
# SvElevationDegrees and SvAzimuthDegrees are pre-computed in the CSV.

import pathlib
import numpy as np
import pandas as pd

SPEED_OF_LIGHT = 299_792_458.0  # m/s

# Accumulated Delta Range state bit flags
ADR_STATE_VALID      = 1 << 0   # bit 0 - measurement is valid
ADR_STATE_RESET      = 1 << 1   # bit 1 - discontinuity occurred
ADR_STATE_CYCLE_SLIP = 1 << 2   # bit 2 - cycle slip detected


def load_gnss(device_dir: pathlib.Path) -> pd.DataFrame:
    # Load and pre-process device_gnss.csv for a single (trip, device).
    # Returns DataFrame with one row per (epoch, satellite) observation, enriched with:
    #   pseudorange_m, adr_valid, epoch_ms, elevation_deg, azimuth_deg
    path = pathlib.Path(device_dir) / "device_gnss.csv"
    df = pd.read_csv(path, low_memory=False)

    # Keep only Raw measurement rows
    df = df[df["MessageType"] == "Raw"].copy()

    # Corrected pseudorange:
    #   pseudorange = RawPseudorangeMeters + SvClockBias - Iono - Tropo - ISRB
    # ISRB is critical for multi-constellation positioning.
    # Each constellation has a receiver-specific bias relative to GPS L1:
    #   GPS L1 ~ 0m, GLONASS G1 ~ +1158m, Galileo E1 ~ -216m, GPS L5/E5A ~ -2350m
    sv_clock_m = df["SvClockBiasMeters"].fillna(0.0)
    iono_m     = df["IonosphericDelayMeters"].fillna(0.0)
    tropo_m    = df["TroposphericDelayMeters"].fillna(0.0)
    isrb_m     = df["IsrbMeters"].fillna(0.0) if "IsrbMeters" in df.columns else 0.0

    df["pseudorange_m"] = (
        df["RawPseudorangeMeters"] + sv_clock_m - iono_m - tropo_m - isrb_m
    )

    # Discard implausible pseudoranges (< 1000 km or > 90000 km)
    valid_pr = (df["pseudorange_m"] > 1e6) & (df["pseudorange_m"] < 9e7)
    df.loc[~valid_pr, "pseudorange_m"] = np.nan

    # Invalidate rows with very large uncertainty
    high_unc = df["RawPseudorangeUncertaintyMeters"].fillna(999) > 200.0
    df.loc[high_unc, "pseudorange_m"] = np.nan

    # ADR (carrier phase) validity flag
    adr_state = df["AccumulatedDeltaRangeState"].fillna(0).astype(int)
    df["adr_valid"] = (
        ((adr_state & ADR_STATE_VALID)      != 0) &
        ((adr_state & ADR_STATE_RESET)      == 0) &
        ((adr_state & ADR_STATE_CYCLE_SLIP) == 0)
    )

    # Elevation / azimuth aliases (pre-computed in CSV)
    df["elevation_deg"] = df["SvElevationDegrees"]
    df["azimuth_deg"]   = df["SvAzimuthDegrees"]

    # Round to nearest second so GNSS and IMU share a common join key
    df["epoch_ms"] = (
        (df["utcTimeMillis"] / 1000.0).round().astype(np.int64) * 1000
    )

    # Keep only the primary signal per satellite per epoch.
    # Mixing L1 and L5 in TDCP produces huge ADR jumps due to different carrier frequencies.
    # Priority: GPS_L1/GLO_G1/GAL_E1/BDS_B1I > GPS_L5/GAL_E5A > GPS_L2 > others
    PRIMARY_SIGNALS = {
        "GPS_L1": 0, "GLO_G1": 0, "GAL_E1": 0, "BDS_B1I": 0, "QZS_J1": 0,
        "GPS_L5": 1, "GAL_E5A": 1, "GPS_L2": 2,
    }
    if "SignalType" in df.columns:
        df["_sig_rank"] = df["SignalType"].map(PRIMARY_SIGNALS).fillna(3).astype(int)
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
    # Load device_gnss.csv keeping ALL signals (no primary-signal dedup).
    # Does NOT pre-apply pseudorange corrections. Computes CarrierErrorHz for satellite selection.
    path = pathlib.Path(device_dir) / "device_gnss.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[df["MessageType"] == "Raw"].copy()

    # No timestamp rounding - use raw utcTimeMillis.
    # Rounding loses epochs when timestamps straddle a second boundary.

    # Carrier frequency error: deviation from nominal per (Svid, SignalType)
    carrier_ref = df.groupby(["Svid", "SignalType"])["CarrierFrequencyHz"].median()
    df = df.merge(carrier_ref, how="left", on=["Svid", "SignalType"],
                  suffixes=("", "Ref"))
    df["CarrierErrorHz"] = np.abs(df["CarrierFrequencyHz"] - df["CarrierFrequencyHzRef"])

    return df.reset_index(drop=True)


def load_imu(device_dir: pathlib.Path) -> pd.DataFrame:
    # Load device_imu.csv and apply bias correction: calibrated = uncalibrated - bias.
    # Returns DataFrame with MessageType, utcTimeMillis, MeasurementX/Y/Z.
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
    # Interpolate each IMU sensor stream onto the GNSS epoch grid.
    # Linear interpolation per sensor type (UncalAccel, UncalGyro, UncalMag).
    # Returns DataFrame with epoch_ms, accel_x/y/z, gyro_x/y/z, mag_x/y/z.
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


def load_ground_truth(device_dir: pathlib.Path) -> pd.DataFrame:
    # Load ground_truth.csv (train split only). Returns empty DataFrame if missing.
    path = pathlib.Path(device_dir) / "ground_truth.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_gnss_log(device_dir: pathlib.Path) -> pd.DataFrame:
    # Parse supplemental/gnss_log.txt into a tidy DataFrame.
    # Header lines start with '#'. Data lines are comma-separated with record type as first token.
    # Returns DataFrame with column 'type' (Raw / Fix / UncalAccel / ...) plus parsed fields.
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
                inner = line.lstrip("# ").strip()
                parts = inner.split(",")
                if len(parts) > 1:
                    rec_type = parts[0]
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
