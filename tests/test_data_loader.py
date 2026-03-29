"""Tests for src/data_loader.py"""
import numpy as np
import pytest
from src.data_loader import load_gnss, load_imu, align_imu_to_gnss, load_ground_truth, load_gnss_log


# ── load_gnss ─────────────────────────────────────────────────────────────────

def test_load_gnss_columns(sample_device_dir):
    df = load_gnss(sample_device_dir)
    required = {
        "utcTimeMillis", "Svid", "ConstellationType",
        "AccumulatedDeltaRangeMeters", "Cn0DbHz",
        "SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters",
        "pseudorange_m",
    }
    assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"
    assert len(df) > 0


def test_load_gnss_pseudorange_valid(sample_device_dir):
    df = load_gnss(sample_device_dir)
    pr = df["pseudorange_m"].dropna()
    # Pseudoranges must be positive and within plausible Earth-satellite range
    assert (pr > 1e6).all() and (pr < 9e7).all()


# ── load_imu ──────────────────────────────────────────────────────────────────

def test_load_imu_message_types(sample_device_dir):
    imu = load_imu(sample_device_dir)
    types = set(imu["MessageType"].unique())
    assert {"UncalAccel", "UncalGyro", "UncalMag"}.issubset(types)


def test_align_imu_has_expected_columns(sample_device_dir):
    gnss = load_gnss(sample_device_dir)
    imu = load_imu(sample_device_dir)
    epochs = gnss["epoch_ms"].unique()
    aligned = align_imu_to_gnss(imu, epochs)
    for col in ("accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "mag_x", "mag_y", "mag_z"):
        assert col in aligned.columns, f"Missing {col}"
    # One row per epoch
    assert len(aligned) == len(epochs)


# ── load_ground_truth ─────────────────────────────────────────────────────────

def test_load_ground_truth(sample_device_dir):
    gt = load_ground_truth(sample_device_dir)
    assert {"LatitudeDegrees", "LongitudeDegrees", "UnixTimeMillis"}.issubset(gt.columns)
    assert len(gt) > 0


# ── load_gnss_log ─────────────────────────────────────────────────────────────

def test_load_gnss_log_raw(sample_device_dir):
    log = load_gnss_log(sample_device_dir)
    raw = log[log["type"] == "Raw"]
    assert "Cn0DbHz" in raw.columns
    assert len(raw) > 0
