# Tests for src/post_processing.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.post_processing import (
    detect_stops, apply_stop_averaging,
    median_filter_trajectory, ensemble_submissions,
    osrm_snap_to_road,
)


def _make_pos_df(n=20, lat=37.0, lon=-122.0, is_stop=True):
    return pd.DataFrame({
        "epoch_ms": np.arange(n) * 1000,
        "LatitudeDegrees":  np.full(n, lat) + np.random.normal(0, 0.0001, n),
        "LongitudeDegrees": np.full(n, lon) + np.random.normal(0, 0.0001, n),
        "accel_x": np.random.normal(0, 0.02 if is_stop else 2.0, n),
        "accel_y": np.random.normal(0, 0.02 if is_stop else 2.0, n),
        "accel_z": np.random.normal(9.81, 0.02, n),
    })


def test_detect_stops_stationary():
    df = _make_pos_df(n=30, is_stop=True)
    stops = detect_stops(df)
    assert stops.sum() == len(df), "All epochs should be stationary"


def test_detect_stops_moving():
    df = _make_pos_df(n=30, is_stop=False)
    stops = detect_stops(df)
    assert stops.sum() < len(df), "Some epochs should be detected as moving"


def test_apply_stop_averaging_reduces_variance():
    df = _make_pos_df(n=20, is_stop=True)
    df["is_stop"] = True
    orig_var = df["LatitudeDegrees"].var()
    out = apply_stop_averaging(df)
    out_var = out["LatitudeDegrees"].var()
    assert out_var < orig_var * 0.01, "Stop averaging should collapse variance"


def test_apply_stop_averaging_moving_unchanged():
    df = _make_pos_df(n=10, is_stop=False)
    df["is_stop"] = False
    orig_vals = df["LatitudeDegrees"].values.copy()
    out = apply_stop_averaging(df)
    np.testing.assert_array_equal(orig_vals, out["LatitudeDegrees"].values)


def test_median_filter_removes_spike():
    df = pd.DataFrame({
        "LatitudeDegrees":  [37.0, 37.0001, 37.5, 37.0002, 37.0003],
        "LongitudeDegrees": [-122.0] * 5,
    })
    out = median_filter_trajectory(df, kernel_size=3)
    assert abs(out["LatitudeDegrees"].iloc[2] - 37.0) < 0.1


def test_osrm_snap_returns_original_on_network_error():
    lats  = np.array([37.4000, 37.4001, 37.4002])
    lons  = np.array([-122.1000, -122.1001, -122.1002])
    speeds = np.array([10.0, 10.0, 10.0])

    with patch("src.post_processing.requests.get", side_effect=Exception("network error")):
        snapped_lats, snapped_lons = osrm_snap_to_road(lats, lons, speeds)

    np.testing.assert_array_equal(snapped_lats, lats)
    np.testing.assert_array_equal(snapped_lons, lons)


def test_osrm_snap_applies_when_distance_within_threshold():
    lats   = np.array([37.4000])
    lons   = np.array([-122.1000])
    speeds = np.array([10.0])

    snapped_lat = 37.40005
    snapped_lon = -122.10005
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "code": "Ok",
        "waypoints": [{"location": [snapped_lon, snapped_lat], "distance": 5.0}],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.post_processing.requests.get", return_value=mock_resp):
        out_lats, out_lons = osrm_snap_to_road(lats, lons, speeds, max_snap_dist_m=15.0)

    assert abs(out_lats[0] - snapped_lat) < 1e-8
    assert abs(out_lons[0] - snapped_lon) < 1e-8


def test_osrm_snap_skips_when_distance_exceeds_threshold():
    lats   = np.array([37.4000])
    lons   = np.array([-122.1000])
    speeds = np.array([10.0])

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "code": "Ok",
        "waypoints": [{"location": [-122.1050, 37.4050], "distance": 100.0}],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.post_processing.requests.get", return_value=mock_resp):
        out_lats, out_lons = osrm_snap_to_road(lats, lons, speeds, max_snap_dist_m=15.0)

    np.testing.assert_array_equal(out_lats, lats)
    np.testing.assert_array_equal(out_lons, lons)


def test_osrm_snap_skips_stopped_vehicle():
    lats   = np.array([37.4000])
    lons   = np.array([-122.1000])
    speeds = np.array([0.1])

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "code": "Ok",
        "waypoints": [{"location": [-122.1001, 37.4001], "distance": 3.0}],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.post_processing.requests.get", return_value=mock_resp) as mock_get:
        out_lats, out_lons = osrm_snap_to_road(lats, lons, speeds, max_snap_dist_m=15.0)

    mock_get.assert_not_called()
    np.testing.assert_array_equal(out_lats, lats)
    np.testing.assert_array_equal(out_lons, lons)


def test_ensemble_submissions_uniform():
    base = pd.DataFrame({
        "tripId": ["trip/A", "trip/A"],
        "UnixTimeMillis": [1000, 2000],
        "LatitudeDegrees":  [37.0, 37.1],
        "LongitudeDegrees": [-122.0, -122.1],
    })
    shifted = base.copy()
    shifted["LatitudeDegrees"]  = [37.2, 37.3]
    shifted["LongitudeDegrees"] = [-122.2, -122.3]
    ens = ensemble_submissions([base, shifted])
    np.testing.assert_allclose(ens["LatitudeDegrees"].values,  [37.1, 37.2])
    np.testing.assert_allclose(ens["LongitudeDegrees"].values, [-122.1, -122.2])
