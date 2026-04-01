# gnss_solver.py
# Robust GNSS position/velocity solver and Kalman smoother.
#
# Core algorithms adapted from Taro Suzuki's (Chiba Institute of Technology)
# public Kaggle notebook for the Google Smartphone Decimeter Challenge 2022:
#   https://www.kaggle.com/code/taroz1461/carrier-smoothing-robust-wls-kalman-smoother
#
# Adapted components:
#   - Hatch filter carrier-phase smoothing (N=1000 window)
#   - Robust WLS position/velocity (scipy.optimize.least_squares with Sagnac correction)
#   - Outlier detection thresholds (vertical velocity > 2.6 m/s, height > 200 m)
#   - Forward-backward Kalman smoother (RTS, constant-velocity state model)
#
# Modifications from the original:
#   - Loss function changed from soft_l1 to cauchy
#   - Adaptive Kalman Q/R scaling by speed and HDOP
#   - Mahalanobis distance gating for measurement rejection (sigma=17)
#   - Per-satellite ML weighting via LightGBM
#   - Numba JIT compilation of Hatch filter inner loop
#   - Per-epoch feature collection for downstream ML correction
#
# Reference:
#   Suzuki, T. (2023). "Precise Position Estimation Using Smartphone Raw GNSS
#   Data Based on Two-Step Optimization." Sensors, 23(3), 1205.
#
# Note: Suzuki's winning method used factor graph optimization (GTSAM) with TDCP
# - a more advanced technique not used here.

import numpy as np
import numba
import pandas as pd
import scipy.optimize
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance
from pyproj import Transformer

CLIGHT = 299_792_458.0        # speed of light (m/s)
RE_WGS84 = 6_378_137.0        # earth semimajor axis WGS84 (m)
OMGE = 7.2921151467e-5         # earth angular velocity (rad/s)

# Dual-frequency iono-free combination constants
F_L1 = 1_575.42e6             # L1/E1 carrier frequency (Hz)
F_L5 = 1_176.45e6             # L5/E5A carrier frequency (Hz)
ALPHA_IF = F_L1**2 / (F_L1**2 - F_L5**2)   # ~2.5457
BETA_IF  = F_L5**2 / (F_L1**2 - F_L5**2)   # ~1.5457

_ecef2lla = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=False)


def satellite_selection(df, column):
    # Filter satellites by carrier error, elevation, C/N0, and multipath.
    idx = df[column].notnull()
    idx &= df["CarrierErrorHz"].fillna(0.0) < 2.0e6
    idx &= df["SvElevationDegrees"] > 10.0
    idx &= df["Cn0DbHz"] > 15.0
    idx &= df["MultipathIndicator"] == 0
    return df[idx]


def los_vector(xusr, xsat):
    # Compute unit LOS vector and range from user to satellite.
    u = xsat - xusr
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)
    u /= rng
    return u, rng.reshape(-1)


def pr_residuals(x, xsat, pr, W):
    # Pseudorange residuals with Sagnac correction, weighted by W.
    u, rng = los_vector(x[:3], xsat)
    rng += OMGE * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0]) / CLIGHT
    residuals = rng - (pr - x[3])
    return residuals @ W


def jac_pr_residuals(x, xsat, pr, W):
    # Analytical Jacobian for pseudorange residuals.
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(pr), 1])])
    return W @ J


def prr_residuals(v, vsat, prr, x, xsat, W):
    # Pseudorange rate residuals with Sagnac correction.
    u, _ = los_vector(x[:3], xsat)
    rate = np.sum((vsat - v[:3]) * u, axis=1) \
        + OMGE / CLIGHT * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                           - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])
    residuals = rate - (prr - v[3])
    return residuals @ W


def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    # Analytical Jacobian for pseudorange rate residuals.
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])
    return W @ J


@numba.njit(cache=True)
def _hatch_filter_numba(pr_vals, adr_vals, idx_slip, max_window):
    # Numba-compiled Hatch filter inner loop.
    n = len(pr_vals)
    smoothed = np.empty(n)
    arc_count = 0
    for i in range(n):
        if idx_slip[i] or np.isnan(adr_vals[i]):
            smoothed[i] = pr_vals[i]
            arc_count = 1
        else:
            arc_count += 1
            N = min(arc_count, max_window)
            alpha = 1.0 / N
            delta_adr = adr_vals[i] - adr_vals[i - 1]
            smoothed[i] = alpha * pr_vals[i] + (1.0 - alpha) * (smoothed[i - 1] + delta_adr)
    return smoothed


def carrier_smoothing(gnss_df):
    # Hatch-filter carrier smoothing of pseudoranges.
    # Per (Svid, SignalType) arc, applies divergence-free Hatch filter.
    # Arc breaks on: ADR reset/cycle-slip flags, carrier jump > 1.5m, PR jump > 20m.
    # Falls back to raw pseudorange where smoothing is not possible.
    carr_th = 1.5
    pr_th = 20.0
    max_window = 1000

    prsmooth = np.full(len(gnss_df), np.nan)

    for (svid_sigtype, df) in gnss_df.groupby(["Svid", "SignalType"]):
        df = df.replace({"AccumulatedDeltaRangeMeters": {0: np.nan}})

        drng1 = df["AccumulatedDeltaRangeMeters"].diff() - df["PseudorangeRateMetersPerSecond"]
        drng2 = df["RawPseudorangeMeters"].diff() - df["PseudorangeRateMetersPerSecond"]

        slip1 = (df["AccumulatedDeltaRangeState"].to_numpy() & 2**1) != 0  # reset
        slip2 = (df["AccumulatedDeltaRangeState"].to_numpy() & 2**2) != 0  # cycle-slip
        slip3 = np.fabs(drng1.to_numpy()) > carr_th
        slip4 = np.fabs(drng2.to_numpy()) > pr_th

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        pr_vals = df["RawPseudorangeMeters"].values.astype(np.float64)
        adr_vals = df["AccumulatedDeltaRangeMeters"].values.astype(np.float64)
        smoothed = _hatch_filter_numba(pr_vals, adr_vals, idx_slip, max_window)

        idx = (gnss_df["Svid"] == svid_sigtype[0]) & (
            gnss_df["SignalType"] == svid_sigtype[1])
        prsmooth[idx] = smoothed

    # Fallback to raw pseudorange where smoothing failed
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df["RawPseudorangeMeters"].values[idx_nan]
    gnss_df = gnss_df.copy()
    gnss_df["pr_smooth"] = prsmooth

    return gnss_df


# Signal types eligible for IF combination
_L1_SIGNALS = {"GPS_L1", "GAL_E1"}
_L5_SIGNALS = {"GPS_L5", "GAL_E5A"}


def combine_dualfreq_pseudoranges(gnss_df):
    # Combine L1 and L5 carrier-smoothed pseudoranges into iono-free (IF) pseudoranges.
    # For each (epoch, satellite):
    #   Both L1 and L5 exist -> produce IF row (iono cancelled)
    #   L1 only -> keep L1 row unchanged
    if "SignalType" not in gnss_df.columns:
        return gnss_df

    mask_l1 = gnss_df["SignalType"].isin(_L1_SIGNALS)
    mask_l5 = gnss_df["SignalType"].isin(_L5_SIGNALS)

    df_l1 = gnss_df[mask_l1].copy()
    df_l5 = gnss_df[mask_l5].copy()
    df_other = gnss_df[~mask_l1 & ~mask_l5].copy()

    if len(df_l5) == 0:
        return pd.concat([df_l1, df_other], ignore_index=True)

    join_keys = ["utcTimeMillis", "Svid", "ConstellationType"]
    df_merged = df_l1.merge(
        df_l5[join_keys + [
            "pr_smooth", "IsrbMeters", "RawPseudorangeUncertaintyMeters",
            "SignalType",
        ]],
        on=join_keys, how="left", suffixes=("", "_L5"),
        indicator=True,
    )

    mask_both = df_merged["_merge"] == "both"
    df_if = df_merged[mask_both].copy()
    df_l1_only = df_merged[~mask_both].copy()

    if len(df_if) > 0:
        # IF pseudorange: alpha*PR_L1 - beta*PR_L5
        pr_l1 = df_if["pr_smooth"].values
        pr_l5 = df_if["pr_smooth_L5"].values
        df_if["pr_smooth"] = ALPHA_IF * pr_l1 - BETA_IF * pr_l5

        # Combined ISRB
        isrb_l1 = df_if["IsrbMeters"].fillna(0.0).values if "IsrbMeters" in df_if.columns else 0.0
        isrb_l5 = df_if["IsrbMeters_L5"].fillna(0.0).values
        df_if["IsrbMeters"] = ALPHA_IF * isrb_l1 - BETA_IF * isrb_l5

        # Iono cancelled
        df_if["IonosphericDelayMeters"] = 0.0

        # IF uncertainty: sqrt(alpha^2 * sigma_L1^2 + beta^2 * sigma_L5^2)
        sigma_l1 = df_if["RawPseudorangeUncertaintyMeters"].values
        sigma_l5 = df_if["RawPseudorangeUncertaintyMeters_L5"].values
        df_if["RawPseudorangeUncertaintyMeters"] = np.sqrt(
            ALPHA_IF**2 * sigma_l1**2 + BETA_IF**2 * sigma_l5**2
        )

        df_if["SignalType"] = df_if["SignalType"].map(
            {"GPS_L1": "IF_L1L5", "GAL_E1": "IF_E1E5A"}
        ).fillna("IF_L1L5")

    drop_cols = ["pr_smooth_L5", "IsrbMeters_L5",
                 "RawPseudorangeUncertaintyMeters_L5", "SignalType_L5", "_merge"]
    df_if = df_if.drop(columns=[c for c in drop_cols if c in df_if.columns])
    df_l1_only = df_l1_only.drop(columns=[c for c in drop_cols if c in df_l1_only.columns])

    result = pd.concat([df_if, df_l1_only, df_other], ignore_index=True)
    result = result.sort_values("utcTimeMillis").reset_index(drop=True)

    return result


def compute_tdcp_displacements(gnss_df, x_wls, utcTimeMillis):
    # Compute epoch-to-epoch position displacements using TDCP.
    # Uses carrier phase differences from common satellites to estimate 3D displacement.
    # Returns (dx_tdcp, cov_tdcp) arrays of shape (N, 3) and (N, 3, 3).
    n = len(utcTimeMillis)
    dx_tdcp = np.full((n, 3), np.nan)
    cov_tdcp = np.full((n, 3, 3), np.nan)

    epoch_data = {}
    for t_utc, df in gnss_df.groupby("utcTimeMillis"):
        mask = df["AccumulatedDeltaRangeMeters"].notna()
        if "AccumulatedDeltaRangeState" in df.columns:
            adr_state = df["AccumulatedDeltaRangeState"].fillna(0).astype(int)
            mask &= (adr_state & 1) != 0
        mask &= df["SvElevationDegrees"] > 15.0
        mask &= df["Cn0DbHz"] > 20.0
        df_valid = df[mask]
        if len(df_valid) > 0:
            epoch_data[t_utc] = df_valid

    for i in range(1, n):
        t_prev = utcTimeMillis[i - 1]
        t_curr = utcTimeMillis[i]

        if t_prev not in epoch_data or t_curr not in epoch_data:
            continue
        if np.any(np.isnan(x_wls[i - 1])) or np.any(np.isnan(x_wls[i])):
            continue

        df_prev = epoch_data[t_prev]
        df_curr = epoch_data[t_curr]

        common_svids = set(df_prev["Svid"].values) & set(df_curr["Svid"].values)
        if len(common_svids) < 5:
            continue

        # TDCP observation: d_adr - drho_sat = -e_s . dx + dclock
        H_rows = []
        delta_phi = []
        weights = []

        prev_by_svid = df_prev.set_index("Svid")
        curr_by_svid = df_curr.set_index("Svid")
        x_ref = x_wls[i - 1]

        for svid in common_svids:
            try:
                row_prev = prev_by_svid.loc[svid]
                row_curr = curr_by_svid.loc[svid]

                if isinstance(row_prev, pd.DataFrame):
                    row_prev = row_prev.iloc[0]
                if isinstance(row_curr, pd.DataFrame):
                    row_curr = row_curr.iloc[0]

                adr_prev = row_prev["AccumulatedDeltaRangeMeters"]
                adr_curr = row_curr["AccumulatedDeltaRangeMeters"]
                if np.isnan(adr_prev) or np.isnan(adr_curr):
                    continue

                d_adr = adr_curr - adr_prev

                xs_prev = np.array([row_prev["SvPositionXEcefMeters"],
                                    row_prev["SvPositionYEcefMeters"],
                                    row_prev["SvPositionZEcefMeters"]])
                xs_curr = np.array([row_curr["SvPositionXEcefMeters"],
                                    row_curr["SvPositionYEcefMeters"],
                                    row_curr["SvPositionZEcefMeters"]])

                rho_prev = np.linalg.norm(xs_prev - x_ref)
                rho_curr_ref = np.linalg.norm(xs_curr - x_ref)
                drho_sat = rho_curr_ref - rho_prev

                e_s = (xs_curr - x_ref) / rho_curr_ref

                # Sat clock, iono, tropo cancel in epoch differencing
                obs = d_adr - drho_sat
                if abs(obs) > 1000.0:
                    continue

                H_rows.append(np.append(-e_s, 1.0))
                delta_phi.append(obs)

                elev = row_curr.get("SvElevationDegrees", 30.0)
                if np.isnan(elev):
                    elev = 30.0
                cn0 = row_curr.get("Cn0DbHz", 30.0)
                if np.isnan(cn0):
                    cn0 = 30.0
                w = np.sin(np.deg2rad(max(elev, 10.0))) * (cn0 / 45.0)
                weights.append(w)
            except (KeyError, ValueError):
                continue

        if len(H_rows) < 5:
            continue

        H = np.array(H_rows)
        y = np.array(delta_phi)
        W = np.diag(np.array(weights))

        try:
            HtWH = H.T @ W @ H
            HtWy = H.T @ W @ y
            sol = np.linalg.solve(HtWH, HtWy)

            dx_norm = np.linalg.norm(sol[:3])
            if dx_norm > 100.0:
                continue

            dx_tdcp[i] = sol[:3]

            residuals = y - H @ sol
            sigma2 = np.sum(weights * residuals**2) / max(len(y) - 4, 1)
            cov = sigma2 * np.linalg.inv(HtWH)
            cov_tdcp[i] = cov[:3, :3]
        except np.linalg.LinAlgError:
            continue

    return dx_tdcp, cov_tdcp


def _compute_ml_sat_weights(df_pr, model, x0):
    # Compute per-satellite WLS weights using a trained LightGBM residual model.
    # Returns weights = 1/max(pred, floor) to match existing 1/sigma convention.
    from src.feature_engineering import FEATURE_COLS

    n_sats = len(df_pr)
    feat_df = pd.DataFrame(index=range(n_sats))

    col_map = {
        "Cn0DbHz": "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters": "RawPseudorangeUncertaintyMeters",
        "AccumulatedDeltaRangeUncertaintyMeters": "AccumulatedDeltaRangeUncertaintyMeters",
        "PseudorangeRateMetersPerSecond": "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond": "PseudorangeRateUncertaintyMetersPerSecond",
        "SvClockBiasMeters": "SvClockBiasMeters",
        "IonosphericDelayMeters": "IonosphericDelayMeters",
        "TroposphericDelayMeters": "TroposphericDelayMeters",
        "ConstellationType": "ConstellationType",
        "MultipathIndicator": "MultipathIndicator",
    }
    for feat_col, src_col in col_map.items():
        if src_col in df_pr.columns:
            feat_df[feat_col] = df_pr[src_col].values
        else:
            feat_df[feat_col] = 0.0

    if "SvElevationDegrees" in df_pr.columns:
        feat_df["elevation_deg"] = df_pr["SvElevationDegrees"].values
    else:
        feat_df["elevation_deg"] = 30.0
    if "SvAzimuthDegrees" in df_pr.columns:
        feat_df["azimuth_deg"] = df_pr["SvAzimuthDegrees"].values
    else:
        feat_df["azimuth_deg"] = 0.0

    if "AccumulatedDeltaRangeState" in df_pr.columns:
        adr_state = df_pr["AccumulatedDeltaRangeState"].fillna(0).astype(int).values
        feat_df["adr_valid"] = ((adr_state & 1) != 0).astype(int)
    else:
        feat_df["adr_valid"] = 0

    # Pseudorange residual feature
    if not np.all(x0[:3] == 0):
        xsat = df_pr[["SvPositionXEcefMeters", "SvPositionYEcefMeters",
                       "SvPositionZEcefMeters"]].to_numpy()
        geo_rng = np.linalg.norm(xsat - x0[:3], axis=1)
        isrb = df_pr["IsrbMeters"].fillna(0.0).values if "IsrbMeters" in df_pr.columns else 0.0
        pr_corrected = (df_pr["pr_smooth"].values + df_pr["SvClockBiasMeters"].values
                        - isrb - df_pr["IonosphericDelayMeters"].values
                        - df_pr["TroposphericDelayMeters"].values)
        raw_res = pr_corrected - geo_rng
        clock_bias = np.median(raw_res[np.isfinite(raw_res)]) if np.any(np.isfinite(raw_res)) else 0.0
        feat_df["pr_minus_geometric_m"] = raw_res - clock_bias
    else:
        feat_df["pr_minus_geometric_m"] = 0.0

    cols_present = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[cols_present].fillna(0.0)
    pred_residual = model.predict(X)
    sigma = np.maximum(np.abs(pred_residual), 1.0)
    return 1.0 / sigma


def point_positioning(gnss_df, collect_features=False, sat_weight_model=None,
                      use_dualfreq=False):
    # GNSS single point positioning using carrier-smoothed pseudoranges.
    # Returns (utcTimeMillis, x_wls, v_wls, cov_x, cov_v) or adds epoch_features if collect_features=True.

    gnss_df = carrier_smoothing(gnss_df)

    if use_dualfreq:
        gnss_df = combine_dualfreq_pseudoranges(gnss_df)

    utcTimeMillis = gnss_df["utcTimeMillis"].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)
    v0 = np.zeros(4)
    x_wls = np.full([nepoch, 3], np.nan)
    v_wls = np.full([nepoch, 3], np.nan)
    cov_x = np.full([nepoch, 3, 3], np.nan)
    cov_v = np.full([nepoch, 3, 3], np.nan)
    epoch_features = [] if collect_features else None

    for i, (t_utc, df) in enumerate(gnss_df.groupby("utcTimeMillis")):
        df_pr = satellite_selection(df, "pr_smooth")
        df_prr = satellite_selection(df, "PseudorangeRateMetersPerSecond")

        # Corrected pseudorange / pseudorange rate
        isrb = df_pr["IsrbMeters"].fillna(0.0) if "IsrbMeters" in df_pr.columns else 0.0
        pr = (df_pr["pr_smooth"] + df_pr["SvClockBiasMeters"] - isrb
              - df_pr["IonosphericDelayMeters"] - df_pr["TroposphericDelayMeters"]).to_numpy()
        prr = (df_prr["PseudorangeRateMetersPerSecond"]
               + df_prr["SvClockDriftMetersPerSecond"]).to_numpy()

        xsat_pr = df_pr[["SvPositionXEcefMeters", "SvPositionYEcefMeters",
                         "SvPositionZEcefMeters"]].to_numpy()
        xsat_prr = df_prr[["SvPositionXEcefMeters", "SvPositionYEcefMeters",
                           "SvPositionZEcefMeters"]].to_numpy()
        vsat = df_prr[["SvVelocityXEcefMetersPerSecond", "SvVelocityYEcefMetersPerSecond",
                       "SvVelocityZEcefMetersPerSecond"]].to_numpy()

        # Weight matrices (1/sigma, NOT 1/sigma^2)
        w_default = 1.0 / df_pr["RawPseudorangeUncertaintyMeters"].to_numpy()

        # ML satellite weighting (only after valid initial position)
        if sat_weight_model is not None and len(df_pr) >= 4 and not np.all(x0[:3] == 0):
            try:
                w_ml = _compute_ml_sat_weights(df_pr, sat_weight_model, x0)
                scale = np.mean(w_default) / (np.mean(w_ml) + 1e-12)
                w_ml_scaled = w_ml * scale
                w_blended = 0.5 * w_ml_scaled + 0.5 * w_default
                Wx = np.diag(w_blended)
            except Exception:
                Wx = np.diag(w_default)
        else:
            Wx = np.diag(w_default)

        Wv = np.diag(1.0 / df_prr["PseudorangeRateUncertaintyMetersPerSecond"].to_numpy())

        wls_residual_norm = np.nan
        wls_status = -1

        # Position estimation
        if len(df_pr) >= 4:
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals,
                    args=(xsat_pr, pr, Wx))
                x0 = opt.x

            opt = scipy.optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals,
                args=(xsat_pr, pr, Wx), loss="cauchy")
            if opt.status >= 1 and opt.status != 2:
                try:
                    cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
                    cov_x[i, :, :] = cov[:3, :3]
                except np.linalg.LinAlgError:
                    cov_x[i, :, :] = 100.0**2 * np.eye(3)
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x
                wls_residual_norm = float(np.linalg.norm(opt.fun))
                wls_status = int(opt.status)

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals,
                    args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x

            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals,
                args=(vsat, prr, x0, xsat_prr, Wv), loss="cauchy")
            if opt.status >= 1:
                try:
                    cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                    cov_v[i, :, :] = cov[:3, :3]
                except np.linalg.LinAlgError:
                    cov_v[i, :, :] = 100.0**2 * np.eye(3)
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

        # Collect per-epoch features for ML
        if collect_features:
            cn0_vals = df_pr["Cn0DbHz"].values if len(df_pr) > 0 else np.array([0.0])
            elev_vals = df_pr["SvElevationDegrees"].values if len(df_pr) > 0 else np.array([0.0])
            pr_unc = df_pr["RawPseudorangeUncertaintyMeters"].values if len(df_pr) > 0 else np.array([999.0])
            mp_count = int((df["MultipathIndicator"] != 0).sum()) if "MultipathIndicator" in df.columns else 0

            const_counts = df_pr["ConstellationType"].value_counts() if len(df_pr) > 0 else pd.Series(dtype=int)
            n_total_sats = len(df_pr)
            frac_gps = const_counts.get(1, 0) / max(n_total_sats, 1)
            frac_glonass = const_counts.get(3, 0) / max(n_total_sats, 1)
            frac_galileo = const_counts.get(6, 0) / max(n_total_sats, 1)
            frac_beidou = const_counts.get(5, 0) / max(n_total_sats, 1)

            # DOP from geometry
            hdop, vdop, pdop = np.nan, np.nan, np.nan
            if len(df_pr) >= 4 and not np.isnan(x_wls[i, 0]):
                try:
                    u_los, _ = los_vector(x_wls[i], xsat_pr)
                    G = np.hstack([-u_los, np.ones((len(u_los), 1))])
                    Q = np.linalg.inv(G.T @ G)
                    lat_r, lon_r, _ = _ecef2lla.transform(
                        x_wls[i, 0], x_wls[i, 1], x_wls[i, 2], radians=False)
                    lat_r = np.deg2rad(lat_r)
                    lon_r = np.deg2rad(lon_r)
                    sl, cl = np.sin(lat_r), np.cos(lat_r)
                    sn, cn = np.sin(lon_r), np.cos(lon_r)
                    R = np.array([[-sn, cn, 0],
                                  [-sl*cn, -sl*sn, cl],
                                  [cl*cn, cl*sn, sl]])
                    Q_enu = R @ Q[:3, :3] @ R.T
                    hdop = float(np.sqrt(Q_enu[0, 0] + Q_enu[1, 1]))
                    vdop = float(np.sqrt(Q_enu[2, 2]))
                    pdop = float(np.sqrt(Q_enu[0, 0] + Q_enu[1, 1] + Q_enu[2, 2]))
                except (np.linalg.LinAlgError, ValueError):
                    pass

            ct = float(np.trace(cov_x[i])) if not np.any(np.isnan(cov_x[i])) else np.nan

            if "AccumulatedDeltaRangeState" in df.columns:
                adr_states = df["AccumulatedDeltaRangeState"].fillna(0).astype(int)
                adr_valid_count = ((adr_states & 1) != 0).sum()
                adr_frac = adr_valid_count / max(len(df), 1)
            else:
                adr_frac = 0.0

            n_constellations = df_pr["ConstellationType"].nunique() if len(df_pr) > 0 else 0
            cn0_range = float(np.max(cn0_vals) - np.min(cn0_vals)) if len(cn0_vals) > 1 else 0.0

            feat = {
                "utcTimeMillis": t_utc,
                "n_sats_pos": len(df_pr),
                "n_sats_vel": len(df_prr),
                "n_sats_raw": len(df),
                "mean_cn0": float(np.mean(cn0_vals)),
                "std_cn0": float(np.std(cn0_vals)),
                "min_cn0": float(np.min(cn0_vals)),
                "max_cn0": float(np.max(cn0_vals)),
                "mean_elev": float(np.mean(elev_vals)),
                "std_elev": float(np.std(elev_vals)),
                "min_elev": float(np.min(elev_vals)),
                "mean_pr_unc": float(np.mean(pr_unc)),
                "std_pr_unc": float(np.std(pr_unc)),
                "max_pr_unc": float(np.max(pr_unc)),
                "hdop": hdop,
                "vdop": vdop,
                "pdop": pdop,
                "wls_residual_norm": wls_residual_norm,
                "wls_status": wls_status,
                "cov_trace": ct,
                "n_multipath": mp_count,
                "frac_gps": frac_gps,
                "frac_glonass": frac_glonass,
                "frac_galileo": frac_galileo,
                "frac_beidou": frac_beidou,
                "adr_frac": float(adr_frac),
                "n_constellations": int(n_constellations),
                "cn0_range": cn0_range,
                "max_elev": float(np.max(elev_vals)) if len(elev_vals) > 0 else 0.0,
                "elev_range": float(np.max(elev_vals) - np.min(elev_vals)) if len(elev_vals) > 1 else 0.0,
                "cn0_q25": float(np.percentile(cn0_vals, 25)) if len(cn0_vals) > 0 else 0.0,
                "pr_unc_median": float(np.median(pr_unc)) if len(pr_unc) > 0 else 999.0,
                "mean_prr_unc": float(np.mean(df_prr["PseudorangeRateUncertaintyMetersPerSecond"].values)) if len(df_prr) > 0 else 999.0,
                "wls_converged": int(wls_status >= 1),
            }
            epoch_features.append(feat)

    if collect_features:
        return utcTimeMillis, x_wls, v_wls, cov_x, cov_v, epoch_features
    return utcTimeMillis, x_wls, v_wls, cov_x, cov_v


def exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v):
    # Remove outliers based on up-velocity and height thresholds, then interpolate gaps.
    # Thresholds match Taro Suzuki's original notebook.
    v_up_th = 2.6
    height_th = 200.0
    v_out_sigma = 3.0
    x_out_sigma = 30.0

    lat, lon, alt = _ecef2lla.transform(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2],
                                         radians=False)
    x_llh = np.column_stack([lat, lon, alt])

    # ENU up component from ECEF velocity
    x_llh_mean = np.nanmean(x_llh, axis=0)
    lat_r = np.deg2rad(x_llh_mean[0])
    lon_r = np.deg2rad(x_llh_mean[1])
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)
    v_up = (cos_lat * cos_lon * v_wls[:, 0]
            + cos_lat * sin_lon * v_wls[:, 1]
            + sin_lat * v_wls[:, 2])

    # Up velocity outlier
    idx_v_out = np.abs(v_up) > v_up_th
    idx_v_out |= np.isnan(v_wls[:, 0])
    v_wls[idx_v_out, :] = np.nan
    cov_v[idx_v_out] = v_out_sigma**2 * np.eye(3)

    # Height outlier
    hmedian = np.nanmedian(x_llh[:, 2])
    idx_x_out = np.abs(x_llh[:, 2] - hmedian) > height_th
    idx_x_out |= np.isnan(x_wls[:, 0])
    x_wls[idx_x_out, :] = np.nan
    cov_x[idx_x_out] = x_out_sigma**2 * np.eye(3)

    # Interpolate NaN at edges for position
    x_df = pd.DataFrame({"x": x_wls[:, 0], "y": x_wls[:, 1], "z": x_wls[:, 2]})
    x_df = x_df.interpolate(limit_area="outside", limit_direction="both")

    # Interpolate all NaN for velocity (spline)
    v_df = pd.DataFrame({"x": v_wls[:, 0], "y": v_wls[:, 1], "z": v_wls[:, 2]})
    v_df = v_df.interpolate(limit_area="outside", limit_direction="both")
    v_df = v_df.interpolate("spline", order=3)

    return x_df.to_numpy(), v_df.to_numpy(), cov_x, cov_v


def Kalman_filter(zs, us, cov_zs, cov_us, speeds=None, hdops=None,
                  sigma_mahalanobis=30.0, speed_q_ref=5.0, hdop_r_ref=1.5):
    # 3D Kalman filter: position state, velocity as control input.
    # Supports adaptive Q scaling by speed and R scaling by HDOP.
    # Mahalanobis gating rejects outlier measurements.
    n, dim_x = zs.shape
    F = np.eye(3)
    H = np.eye(3)

    x = None
    for k in range(n):
        if not np.any(np.isnan(zs[k])):
            x = zs[k, :3].T.copy()
            break
    if x is None:
        x_kf = np.full([n, dim_x], np.nan)
        P_kf = np.full([n, dim_x, dim_x], np.nan)
        return x_kf, P_kf

    P = 5.0**2 * np.eye(3)
    I = np.eye(dim_x)

    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    for i, (u, z) in enumerate(zip(us, zs)):
        if i == 0:
            x_kf[i] = x.T
            P_kf[i] = P
            continue

        # Prediction with adaptive Q scaling
        Q = cov_us[i]
        if np.any(np.isnan(Q)):
            Q = 100.0**2 * np.eye(3)
        if speeds is not None and np.isfinite(speeds[i]):
            q_scale = max(1.0, speeds[i] / speed_q_ref)
            Q = Q * q_scale
        x = F @ x + u.T
        P = (F @ P) @ F.T + Q

        if np.any(np.isnan(z)) or np.any(np.isnan(cov_zs[i])):
            P += 10**2 * Q
            x_kf[i] = x.T
            P_kf[i] = P
            continue

        # Mahalanobis distance check
        try:
            d = distance.mahalanobis(z, H @ x, np.linalg.inv(P))
        except (np.linalg.LinAlgError, ValueError):
            d = sigma_mahalanobis + 1

        # Update with adaptive R scaling
        if d < sigma_mahalanobis:
            R = cov_zs[i]
            if hdops is not None and np.isfinite(hdops[i]):
                r_scale = max(1.0, hdops[i] / hdop_r_ref)
                R = R * r_scale
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            try:
                K = (P @ H.T) @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = np.zeros_like(P)
            x = x + K @ y
            P = (I - (K @ H)) @ P
        else:
            P += 10**2 * Q

        x_kf[i] = x.T
        P_kf[i] = P

    return x_kf, P_kf


def Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, phone, speeds=None, hdops=None,
                     dx_tdcp=None, cov_tdcp=None,
                     sigma_mahalanobis=30.0, speed_q_ref=5.0, hdop_r_ref=1.5):
    # Forward + backward Kalman filter with covariance-weighted fusion.
    # XiaomiMi8 has known velocity issues - applies special handling.
    # Uses TDCP displacements as control input when available.
    n, dim_x = x_wls.shape

    if phone == "XiaomiMi8":
        v_wls = np.vstack([(v_wls[:-1, :] + v_wls[1:, :]) / 2,
                           np.zeros([1, 3])])
        cov_v = 1000.0**2 * cov_v

    # Control input: prefer TDCP over Doppler velocity
    v_doppler = np.vstack([np.zeros([1, 3]),
                           (v_wls[:-1, :] + v_wls[1:, :]) / 2])
    v = v_doppler.copy()
    cov_ctrl = cov_v.copy()

    if dx_tdcp is not None and cov_tdcp is not None:
        for i in range(1, n):
            if not np.any(np.isnan(dx_tdcp[i])) and not np.any(np.isnan(cov_tdcp[i])):
                v[i] = dx_tdcp[i]
                cov_ctrl[i] = cov_tdcp[i]

    x_f, P_f = Kalman_filter(x_wls, v, cov_x, cov_ctrl, speeds=speeds, hdops=hdops,
                              sigma_mahalanobis=sigma_mahalanobis,
                              speed_q_ref=speed_q_ref, hdop_r_ref=hdop_r_ref)

    # Backward pass
    v_back_doppler = -np.flipud(v_wls)
    v_back = np.vstack([np.zeros([1, 3]),
                        (v_back_doppler[:-1, :] + v_back_doppler[1:, :]) / 2])

    if dx_tdcp is not None and cov_tdcp is not None:
        dx_tdcp_rev = np.flipud(dx_tdcp)
        cov_tdcp_rev = np.flipud(cov_tdcp)
        for i in range(1, n):
            j = n - i
            if j > 0 and not np.any(np.isnan(dx_tdcp[j])):
                v_back[i] = -dx_tdcp[j]

    cov_xf = np.flip(cov_x, axis=0)
    cov_vf = np.flip(cov_v, axis=0)
    if dx_tdcp is not None and cov_tdcp is not None:
        cov_vf_tdcp = np.flip(cov_ctrl, axis=0)
    else:
        cov_vf_tdcp = cov_vf

    speeds_rev = np.flipud(speeds) if speeds is not None else None
    hdops_rev = np.flipud(hdops) if hdops is not None else None
    x_b, P_b = Kalman_filter(np.flipud(x_wls), v_back, cov_xf, cov_vf_tdcp,
                              speeds=speeds_rev, hdops=hdops_rev,
                              sigma_mahalanobis=sigma_mahalanobis,
                              speed_q_ref=speed_q_ref, hdop_r_ref=hdop_r_ref)

    # Fuse forward and backward via covariance weighting
    x_fb = np.zeros_like(x_f)
    for (f, b) in zip(range(n), range(n - 1, -1, -1)):
        try:
            P_fi = np.linalg.inv(P_f[f])
            P_bi = np.linalg.inv(P_b[b])
            P_fb = np.linalg.inv(P_fi + P_bi)
            x_fb[f] = P_fb @ (P_fi @ x_f[f] + P_bi @ x_b[b])
        except np.linalg.LinAlgError:
            x_fb[f] = x_f[f]

    return x_fb, x_f, np.flipud(x_b)


def solve_trip_robust(gnss_df, device_name="", collect_features=False,
                      sat_weight_model=None,
                      sigma_mahalanobis=30.0, speed_q_ref=5.0, hdop_r_ref=1.5,
                      use_tdcp=True, use_dualfreq=False):
    # Full pipeline: WLS -> outlier removal -> Kalman smoother.
    # Returns DataFrame with: epoch_ms, lat, lon, alt, ecef_x, ecef_y, ecef_z.
    # If collect_features=True, returns (result_df, features_df).

    # 1. Point positioning
    if collect_features:
        utc, x_wls, v_wls, cov_x, cov_v, epoch_feats = point_positioning(
            gnss_df, collect_features=True, sat_weight_model=sat_weight_model,
            use_dualfreq=use_dualfreq)
    else:
        utc, x_wls, v_wls, cov_x, cov_v = point_positioning(
            gnss_df, sat_weight_model=sat_weight_model,
            use_dualfreq=use_dualfreq)
        epoch_feats = None

    empty_result = pd.DataFrame(columns=["epoch_ms", "lat", "lon", "alt",
                                         "ecef_x", "ecef_y", "ecef_z"])

    if np.all(np.isnan(x_wls)):
        if collect_features:
            return empty_result, pd.DataFrame()
        return empty_result

    # Add speed to features
    if collect_features and epoch_feats:
        for j, feat in enumerate(epoch_feats):
            if not np.any(np.isnan(v_wls[j])):
                feat["speed_mps"] = float(np.linalg.norm(v_wls[j]))
            else:
                feat["speed_mps"] = np.nan

    # Per-epoch speed and HDOP for adaptive Kalman
    n_epochs = len(x_wls)
    speeds = np.full(n_epochs, np.nan)
    hdops = np.full(n_epochs, np.nan)
    for j in range(n_epochs):
        if not np.any(np.isnan(v_wls[j])):
            speeds[j] = np.linalg.norm(v_wls[j])
        if not np.any(np.isnan(cov_x[j])):
            try:
                pos = x_wls[j]
                if not np.any(np.isnan(pos)):
                    lat_r, lon_r, _ = _ecef2lla.transform(
                        pos[0], pos[1], pos[2], radians=False)
                    lat_r = np.deg2rad(lat_r)
                    lon_r = np.deg2rad(lon_r)
                    sl, cl = np.sin(lat_r), np.cos(lat_r)
                    sn, cn = np.sin(lon_r), np.cos(lon_r)
                    R_enu = np.array([[-sn, cn, 0],
                                      [-sl*cn, -sl*sn, cl],
                                      [cl*cn, cl*sn, sl]])
                    cov_enu = R_enu @ cov_x[j] @ R_enu.T
                    hdops[j] = np.sqrt(cov_enu[0, 0] + cov_enu[1, 1])
            except (ValueError, np.linalg.LinAlgError):
                pass

    # 2. TDCP displacements
    dx_tdcp, cov_tdcp = None, None
    if use_tdcp:
        try:
            dx_tdcp, cov_tdcp = compute_tdcp_displacements(gnss_df, x_wls, utc)
            n_valid = int(np.sum(~np.any(np.isnan(dx_tdcp), axis=1)))
            if n_valid < 0.3 * len(utc):
                dx_tdcp, cov_tdcp = None, None
        except Exception:
            dx_tdcp, cov_tdcp = None, None

    # 3. Outlier detection and interpolation
    x_wls, v_wls, cov_x, cov_v = exclude_interpolate_outlier(
        x_wls, v_wls, cov_x, cov_v)

    # 4. Kalman smoothing
    x_kf, _, _ = Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, device_name,
                                   speeds=speeds, hdops=hdops,
                                   sigma_mahalanobis=sigma_mahalanobis,
                                   speed_q_ref=speed_q_ref,
                                   hdop_r_ref=hdop_r_ref,
                                   dx_tdcp=dx_tdcp,
                                   cov_tdcp=cov_tdcp)

    # 5. Convert ECEF -> lat/lon/alt
    lats, lons, alts = _ecef2lla.transform(
        x_kf[:, 0], x_kf[:, 1], x_kf[:, 2], radians=False)

    result_df = pd.DataFrame({
        "epoch_ms": utc,
        "lat": lats,
        "lon": lons,
        "alt": alts,
        "ecef_x": x_kf[:, 0],
        "ecef_y": x_kf[:, 1],
        "ecef_z": x_kf[:, 2],
    })

    if collect_features:
        feat_df = pd.DataFrame(epoch_feats) if epoch_feats else pd.DataFrame()
        return result_df, feat_df

    return result_df
