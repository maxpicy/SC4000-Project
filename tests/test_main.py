"""Integration test for src/main.py"""
import pathlib
import numpy as np
import pytest
from src.main import process_trip


def test_process_trip_produces_output(sample_device_dir):
    result = process_trip(sample_device_dir, train=True)
    assert {"UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}.issubset(
        result.columns
    ), f"Missing columns: {set(result.columns)}"
    assert len(result) > 0


def test_process_trip_lat_lon_range(sample_device_dir):
    result = process_trip(sample_device_dir, train=True)
    assert result["LatitudeDegrees"].between(36.0, 39.0).all(), \
        f"Lat out of range: {result['LatitudeDegrees'].describe()}"
    assert result["LongitudeDegrees"].between(-123.0, -120.0).all(), \
        f"Lon out of range: {result['LongitudeDegrees'].describe()}"


def test_process_trip_no_nans(sample_device_dir):
    result = process_trip(sample_device_dir, train=True)
    assert result["LatitudeDegrees"].notna().all()
    assert result["LongitudeDegrees"].notna().all()
