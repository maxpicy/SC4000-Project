"""Tests for src/gnss_solver.py (Suzuki WLS+Kalman approach)"""
import numpy as np
import pandas as pd
import pytest
from src.gnss_solver import (
    satellite_selection,
    carrier_smoothing,
    los_vector,
    pr_residuals,
    jac_pr_residuals,
    point_positioning,
    exclude_interpolate_outlier,
    Kalman_filter,
    Kalman_smoothing,
    solve_trip_robust,
    CLIGHT,
    OMGE,
)


# ── satellite_selection ──────────────────────────────────────────────────────

def test_satellite_selection_filters_correctly():
    """satellite_selection removes low-quality satellites."""
    df = pd.DataFrame({
        "pr_smooth": [2.3e7, 2.3e7, np.nan, 2.3e7, 2.3e7],
        "CarrierErrorHz": [100, 3e6, 100, 100, 100],     # idx 1 fails
        "SvElevationDegrees": [45, 45, 45, 5, 45],       # idx 3 fails
        "Cn0DbHz": [35, 35, 35, 35, 10],                 # idx 4 fails
        "MultipathIndicator": [0, 0, 0, 0, 0],
    })
    result = satellite_selection(df, "pr_smooth")
    assert len(result) == 1  # only idx 0 passes all filters
    assert result.index[0] == 0


def test_satellite_selection_multipath_filter():
    """MultipathIndicator != 0 should be rejected."""
    df = pd.DataFrame({
        "pr_smooth": [2.3e7, 2.3e7],
        "CarrierErrorHz": [100, 100],
        "SvElevationDegrees": [45, 45],
        "Cn0DbHz": [35, 35],
        "MultipathIndicator": [0, 1],
    })
    result = satellite_selection(df, "pr_smooth")
    assert len(result) == 1


# ── los_vector ───────────────────────────────────────────────────────────────

def test_los_vector_unit_length():
    """LOS vectors must be unit length."""
    xusr = np.array([0.0, 0.0, 0.0])
    xsat = np.array([[1e7, 0, 0], [0, 2e7, 0], [0, 0, 3e7]])
    u, rng = los_vector(xusr, xsat)
    norms = np.linalg.norm(u, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_los_vector_range():
    """Range must match Euclidean distance."""
    xusr = np.array([1e6, 2e6, 3e6])
    xsat = np.array([[2e7, 0, 0], [0, 2e7, 0]])
    _, rng = los_vector(xusr, xsat)
    expected = np.linalg.norm(xsat - xusr, axis=1)
    np.testing.assert_allclose(rng, expected, rtol=1e-10)


# ── pr_residuals / jac_pr_residuals ─────────────────────────────────────────

def test_pr_residuals_zero_at_true_position():
    """Residuals should be near zero when x is the true position."""
    rx = np.array([-2694706.0, -4293790.0, 3857576.0])
    sv = rx + np.array([[2.4e7, 0, 0], [0, 2.4e7, 0],
                        [0, 0, 2.4e7], [-2.4e7, 0, 0]])
    pr = np.linalg.norm(sv - rx, axis=1)
    # Add Sagnac correction to pseudorange
    for j in range(len(sv)):
        pr[j] += OMGE * (sv[j, 0] * rx[1] - sv[j, 1] * rx[0]) / CLIGHT
    W = np.eye(4)
    x = np.append(rx, 0.0)  # clock bias = 0
    res = pr_residuals(x, sv, pr, W)
    assert np.max(np.abs(res)) < 1e-3


def test_jac_pr_residuals_shape():
    """Jacobian should be (n_sats, 4)."""
    rx = np.array([-2694706.0, -4293790.0, 3857576.0])
    sv = rx + np.array([[2.4e7, 0, 0], [0, 2.4e7, 0], [0, 0, 2.4e7]])
    pr = np.linalg.norm(sv - rx, axis=1)
    W = np.eye(3)
    x = np.append(rx, 0.0)
    J = jac_pr_residuals(x, sv, pr, W)
    assert J.shape == (3, 4)


# ── carrier_smoothing ───────────────────────────────────────────────────────

def test_carrier_smoothing_uses_adr():
    """With valid ADR, smoothed PR should differ from raw PR."""
    n = 10
    df = pd.DataFrame({
        "Svid": [1] * n,
        "SignalType": ["GPS_L1"] * n,
        "utcTimeMillis": np.arange(n) * 1000,
        "RawPseudorangeMeters": np.full(n, 2.3e7) + np.random.normal(0, 5, n),
        "AccumulatedDeltaRangeMeters": np.full(n, 2.3e7) + np.cumsum(np.random.normal(0, 0.01, n)),
        "AccumulatedDeltaRangeState": np.full(n, 1, dtype=int),  # VALID
        "PseudorangeRateMetersPerSecond": np.zeros(n),
    })
    result = carrier_smoothing(df)
    assert "pr_smooth" in result.columns
    # Smoothed should not be all NaN
    assert not np.all(np.isnan(result["pr_smooth"].values))


def test_carrier_smoothing_fallback_no_adr():
    """When ADR is zero/invalid, falls back to raw pseudorange."""
    n = 5
    df = pd.DataFrame({
        "Svid": [1] * n,
        "SignalType": ["GPS_L1"] * n,
        "utcTimeMillis": np.arange(n) * 1000,
        "RawPseudorangeMeters": np.full(n, 2.3e7),
        "AccumulatedDeltaRangeMeters": np.zeros(n),  # invalid
        "AccumulatedDeltaRangeState": np.full(n, 0, dtype=int),
        "PseudorangeRateMetersPerSecond": np.zeros(n),
    })
    result = carrier_smoothing(df)
    np.testing.assert_allclose(result["pr_smooth"].values, 2.3e7)


# ── exclude_interpolate_outlier ─────────────────────────────────────────────

def test_outlier_detection_height():
    """Positions with height > 200m from median should be NaN'd."""
    n = 10
    # Create ECEF positions at roughly (37.4, -122.1, ~50m)
    from pyproj import Transformer
    lla2ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=False)
    x, y, z = lla2ecef.transform(37.4, -122.1, 50.0, radians=False)
    x_wls = np.tile([x, y, z], (n, 1))
    v_wls = np.zeros((n, 3))
    cov_x = np.tile(10.0 * np.eye(3), (n, 1, 1))
    cov_v = np.tile(1.0 * np.eye(3), (n, 1, 1))

    # Make one position at wildly different height
    x_h, y_h, z_h = lla2ecef.transform(37.4, -122.1, 5000.0, radians=False)
    x_wls[5] = [x_h, y_h, z_h]

    x_out, v_out, _, _ = exclude_interpolate_outlier(
        x_wls.copy(), v_wls.copy(), cov_x.copy(), cov_v.copy())
    # The outlier epoch should have been interpolated (not the original value)
    assert not np.allclose(x_out[5], [x_h, y_h, z_h])


# ── Kalman_filter ───────────────────────────────────────────────────────────

def test_kalman_filter_shape():
    """Output shapes must match input."""
    n = 20
    zs = np.random.randn(n, 3) * 10
    us = np.random.randn(n, 3)
    cov_zs = np.tile(5.0 * np.eye(3), (n, 1, 1))
    cov_us = np.tile(1.0 * np.eye(3), (n, 1, 1))
    x_kf, P_kf = Kalman_filter(zs, us, cov_zs, cov_us)
    assert x_kf.shape == (n, 3)
    assert P_kf.shape == (n, 3, 3)


def test_kalman_filter_reduces_noise():
    """KF output should have less variance than raw measurements."""
    n = 100
    true_pos = np.cumsum(np.ones((n, 3)) * 0.1, axis=0)
    noise = np.random.randn(n, 3) * 5.0
    zs = true_pos + noise
    us = np.ones((n, 3)) * 0.1
    cov_zs = np.tile(25.0 * np.eye(3), (n, 1, 1))
    cov_us = np.tile(0.01 * np.eye(3), (n, 1, 1))
    x_kf, _ = Kalman_filter(zs, us, cov_zs, cov_us)
    raw_err = np.mean(np.linalg.norm(zs - true_pos, axis=1))
    kf_err = np.mean(np.linalg.norm(x_kf - true_pos, axis=1))
    assert kf_err < raw_err


# ── Kalman_smoothing ────────────────────────────────────────────────────────

def test_kalman_smoothing_shape():
    """Smoother output must match input shape."""
    n = 30
    x_wls = np.random.randn(n, 3) * 100
    v_wls = np.random.randn(n, 3)
    cov_x = np.tile(10.0 * np.eye(3), (n, 1, 1))
    cov_v = np.tile(1.0 * np.eye(3), (n, 1, 1))
    x_fb, x_f, x_b = Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, "GooglePixel5")
    assert x_fb.shape == (n, 3)
    assert x_f.shape == (n, 3)
    assert x_b.shape == (n, 3)


# ── solve_trip_robust (integration) ─────────────────────────────────────────

def test_solve_trip_robust_shape(sample_device_dir):
    """solve_trip_robust must return expected columns."""
    from src.data_loader import load_gnss_raw
    gnss = load_gnss_raw(sample_device_dir)
    result = solve_trip_robust(gnss)
    required = {"epoch_ms", "lat", "lon"}
    assert required.issubset(result.columns), \
        f"Missing columns: {required - set(result.columns)}"
    assert len(result) > 0


def test_solve_trip_robust_lat_lon_range(sample_device_dir):
    """Output positions must be in reasonable California area."""
    from src.data_loader import load_gnss_raw
    gnss = load_gnss_raw(sample_device_dir)
    result = solve_trip_robust(gnss)
    valid = result["lat"].notna()
    assert result.loc[valid, "lat"].between(36.0, 39.0).all()
    assert result.loc[valid, "lon"].between(-123.0, -120.0).all()
