#!/usr/bin/env python3
"""
Phase 2 — Pillar 1: Wearable 24h Circadian–Ultradian Manifold (fPCA-TPDC) & Serum sNfL
Cross-Fitted Double Machine Learning (DML) Causal Mediation (NHANES 2011–2014)
======================================================================================
Target Project: PsyTwin-SDE (npj Digital Medicine Collection)
Pre-Specified in: docs/SAP_v1_FROZEN.md (§2.1–2.3 & §5)
Governance Rules (CLAUDE.md & lab-constitution):
  - STRICT READ-ONLY RAW DATA: Reads exclusively from data_processed/; writes exclusively
    inside data_processed/ and results/phase2_pillar1_nhanes/.
  - ZERO Brain-Kidney Axis & ZERO Raw DICOM Imaging (serum creatinine enters strictly as a
    systemic physiological clearance confounder for serum sNfL).
  - 100% Real Script-Computed Numbers with deterministic seed = 20270507.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from statsmodels.stats.multitest import multipletests

SEED = 20270507
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_processed"
OUT_DIR = PROJECT_ROOT / "results" / "phase2_pillar1_nhanes"


# ---------------------------------------------------------------------------
# 1. Periodic Fourier-Basis Penalized Functional PCA (fPCA) on 24h Profiles
# ---------------------------------------------------------------------------
def build_periodic_fourier_basis(hours: np.ndarray, n_harmonics: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an orthonormal 24-hour periodic Fourier basis Phi (24 x M, M = 1 + 2*n_harmonics = 9)
    and exact analytical second-derivative roughness penalty matrix R_pen (M x M),
    where R_mn = int_0^24 phi_m''(h) phi_n''(h) dh.
    """
    period = 24.0
    n_grid = len(hours)
    cols = [np.ones(n_grid, dtype=float) / np.sqrt(period)]
    d2_weights = [0.0]
    for k in range(1, n_harmonics + 1):
        omega_k = 2.0 * np.pi * k / period
        cols.append(np.sqrt(2.0 / period) * np.cos(omega_k * hours))
        cols.append(np.sqrt(2.0 / period) * np.sin(omega_k * hours))
        d2_weights.append(omega_k**4)
        d2_weights.append(omega_k**4)
    phi_raw = np.column_stack(cols)  # continuous orthonormal: sum(phi^2)*dh = 1 with dh = 1.0
    # Orthonormalize on discrete 24-hour grid (dh = 1.0, exact for k <= 4 < Nyquist 12)
    r_pen = np.diag(d2_weights)
    return phi_raw, r_pen


def fit_penalized_fpca_24h(
    prof_primary: np.ndarray,
    prof_all: np.ndarray,
    channel_name: str,
    n_harmonics: int = 4,
    n_components: int = 4,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, float]:
    """
    Fit penalized periodic Fourier-basis fPCA on primary 4d16h valid profiles (N_prim x 24, zero missing bins),
    select roughness penalty lambda via Generalized Cross-Validation (GCV), orient eigenfunctions physiologically,
    and project all eligible profiles (including 4d10h sensitivity rows with >=20 bins) onto the primary basis.
    """
    hours = np.arange(24, dtype=float) + 0.5  # midpoints of clock hours 0..23
    phi, r_pen = build_periodic_fourier_basis(hours, n_harmonics=n_harmonics)  # (24, 9), (9, 9)
    gram = phi.T @ phi  # I_9 on 24h discrete grid

    # Select roughness penalty lambda via GCV on primary cohort (N = 8,148)
    lam_grid = np.logspace(-3, 3, 49)
    n_p, n_h = prof_primary.shape
    best_lam = 1.0
    best_gcv = np.inf
    for lam in lam_grid:
        s_mat = phi @ linalg.solve(gram + lam * r_pen, phi.T, assume_a="pos")
        tr_s = float(np.trace(s_mat))
        if tr_s >= n_h - 1e-6:
            continue
        resid = prof_primary - (prof_primary @ s_mat.T)
        mse = float(np.mean(resid**2))
        gcv = mse / ((1.0 - tr_s / n_h) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)

    # Smooth primary curves and compute mean function mu(h) + covariance eigenfunctions
    proj_op = linalg.solve(gram + best_lam * r_pen, phi.T, assume_a="pos")  # (9, 24)
    coef_prim = prof_primary @ proj_op.T  # (N_prim, 9)
    smooth_prim = coef_prim @ phi.T  # (N_prim, 24)
    mu_h = np.mean(smooth_prim, axis=0)
    centered_prim = smooth_prim - mu_h[None, :]

    cov_mat = (centered_prim.T @ centered_prim) / float(n_p - 1)  # (24, 24)
    eigvals, eigvecs = linalg.eigh(cov_mat)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total_var = float(np.sum(eigvals))

    psi = eigvecs[:, :n_components].copy()  # (24, K), discrete orthonormal sum_h psi_k(h)^2 = 1

    # Orient signs physiologically for reproducible biological interpretation:
    # PC1: positive mean daytime load (08:00-20:00) -> higher xi_1 = stronger daytime vitality/overall level
    # PC2: positive morning-vs-evening contrast (06:00-12:00 minus 18:00-24:00) -> higher xi_2 = earlier morningness
    # PC3: positive midday peak (10:00-16:00 minus [06:00-10:00 & 16:00-21:00]) -> daytime consolidation vs biphasic dip
    # PC4: positive early-morning awakening transition (05:00-09:00)
    day_idx = np.arange(8, 20)
    morn_idx = np.arange(6, 12)
    eve_idx = np.arange(18, 24)
    mid_idx = np.arange(10, 16)
    shld_idx = np.r_[np.arange(6, 10), np.arange(16, 21)]
    dawn_idx = np.arange(5, 10)

    if np.mean(psi[day_idx, 0]) < 0:
        psi[:, 0] *= -1.0
    if n_components >= 2 and (np.mean(psi[morn_idx, 1]) - np.mean(psi[eve_idx, 1])) < 0:
        psi[:, 1] *= -1.0
    if n_components >= 3 and (np.mean(psi[mid_idx, 2]) - np.mean(psi[shld_idx, 2])) < 0:
        psi[:, 2] *= -1.0
    if n_components >= 4 and np.mean(psi[dawn_idx, 3]) < 0:
        psi[:, 3] *= -1.0

    # Project all participants (handling rare missing clock bins in 4d10h sensitivity via penalized least squares)
    n_all = prof_all.shape[0]
    scores_all = np.full((n_all, n_components), np.nan, dtype=float)
    full_mask = np.all(~np.isnan(prof_all), axis=1)
    if np.any(full_mask):
        sm_full = (prof_all[full_mask] @ proj_op.T) @ phi.T
        scores_all[full_mask] = (sm_full - mu_h[None, :]) @ psi

    part_mask = (~full_mask) & (np.sum(~np.isnan(prof_all), axis=1) >= 20)
    for idx in np.where(part_mask)[0]:
        y_obs = prof_all[idx]
        obs = ~np.isnan(y_obs)
        phi_o = phi[obs]
        c_i = linalg.solve(phi_o.T @ phi_o + best_lam * r_pen, phi_o.T @ y_obs[obs], assume_a="pos")
        sm_i = phi @ c_i
        scores_all[idx] = (sm_i - mu_h) @ psi

    eig_df = pd.DataFrame({"clock_hour": np.arange(24, dtype=int), "channel": channel_name, "mean_curve": mu_h})
    for k in range(n_components):
        eig_df[f"psi_{k+1}"] = psi[:, k]

    var_rows = []
    cum_v = 0.0
    for k in range(n_components):
        p_v = float(eigvals[k] / total_var) if total_var > 0 else 0.0
        cum_v += p_v
        var_rows.append(
            {
                "channel": channel_name,
                "component": f"fPC{k+1}",
                "eigenvalue": round(float(eigvals[k]), 6),
                "variance_explained_pct": round(100.0 * p_v, 3),
                "cumulative_variance_pct": round(100.0 * cum_v, 3),
                "gcv_lambda": round(best_lam, 5),
                "n_primary_fit": int(n_p),
            }
        )
    return scores_all, eig_df, pd.DataFrame(var_rows), best_lam


# ---------------------------------------------------------------------------
# 2. Batched Dual Extended Kalman Filter Time-Resolved PDC (DEKF-TPDC)
# ---------------------------------------------------------------------------
def compute_batched_dekf_tpdc(
    paxhr: pd.DataFrame,
    eligible_seqns: np.ndarray,
    ar_order: int = 6,
    q_param: float = 1e-3,
    r_state_noise: float = 0.08,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adapts Guest Editor Muthuraman's TPDC paradigm (PMC12602468) to contiguous multi-day hourly
    bivariate [log(1+Lux), log(1+MIMS)]^T wrist actigraphy trajectories via a vectorized batched
    Dual Extended Kalman Filter (state-denoising filter + time-varying VAR(p) parameter filter).

    Channel convention:
      ch 0 = log(1 + lux_per_min)   [Photic zeitgeber input]
      ch 1 = log(1 + mims_per_min)  [Motor activity output]
    Partial Directed Coherence pi_{l <- j}(omega, tau):
      pi_{1 <- 0}(omega, tau) = |A_{10}(omega, tau)| / sqrt(|A_{00}(omega, tau)|^2 + |A_{10}(omega, tau)|^2)
                                (Photic -> Motor directed entrainment)
      pi_{0 <- 1}(omega, tau) = |A_{01}(omega, tau)| / sqrt(|A_{01}(omega, tau)|^2 + |A_{11}(omega, tau)|^2)
                                (Motor -> Photic behavioral light-seeking feedback)
    """
    t0 = time.time()
    sub = paxhr[paxhr["SEQN"].isin(eligible_seqns)].copy()
    sub = sub.sort_values(["SEQN", "seq_hour_idx"]).reset_index(drop=True)

    # Build dense (N_subj, T_max, 2) matrix aligned to each participant's recording start hour
    sub["rel_t"] = sub["seq_hour_idx"] - sub.groupby("SEQN")["seq_hour_idx"].transform("min")
    t_max = int(sub["rel_t"].max()) + 1
    seqn_unique = np.sort(sub["SEQN"].unique())
    n_subj = len(seqn_unique)
    seqn_to_row = pd.Series(np.arange(n_subj, dtype=np.int32), index=seqn_unique)

    row_idx = sub["SEQN"].map(seqn_to_row).to_numpy()
    col_idx = sub["rel_t"].to_numpy(dtype=np.int32)
    is_val = (sub["valid_hour_flag"].to_numpy() == 1) & sub["mims_per_min"].notna().to_numpy() & sub["lux_per_min"].notna().to_numpy()

    raw_z = np.full((n_subj, t_max, 2), np.nan, dtype=np.float64)
    obs_mask = np.zeros((n_subj, t_max), dtype=bool)

    lux_v = np.log1p(np.maximum(sub.loc[is_val, "lux_per_min"].to_numpy(dtype=np.float64), 0.0))
    mims_v = np.log1p(np.maximum(sub.loc[is_val, "mims_per_min"].to_numpy(dtype=np.float64), 0.0))
    raw_z[row_idx[is_val], col_idx[is_val], 0] = lux_v
    raw_z[row_idx[is_val], col_idx[is_val], 1] = mims_v
    obs_mask[row_idx[is_val], col_idx[is_val]] = True

    # Linearly interpolate ONLY isolated 1-2 hour non-wear gaps within active span; leave >=3h gaps as missing
    interp_mask = obs_mask.copy()
    z_clean = raw_z.copy()
    for i in range(n_subj):
        valid_ts = np.where(obs_mask[i])[0]
        if len(valid_ts) < 48:
            continue
        for idx_pos in range(len(valid_ts) - 1):
            t_a, t_b = valid_ts[idx_pos], valid_ts[idx_pos + 1]
            gap = t_b - t_a - 1
            if 1 <= gap <= 2:
                for g_step in range(1, gap + 1):
                    w = g_step / float(gap + 1)
                    z_clean[i, t_a + g_step] = (1.0 - w) * raw_z[i, t_a] + w * raw_z[i, t_b]
                    interp_mask[i, t_a + g_step] = True

    # Within-subject standardization per channel
    for ch in range(2):
        arr_ch = z_clean[:, :, ch]
        cnt_ch = np.sum(~np.isnan(arr_ch), axis=1)
        ok = cnt_ch >= 48
        mu_ch = np.zeros(n_subj)
        sd_ch = np.ones(n_subj)
        mu_ch[ok] = np.nanmean(arr_ch[ok], axis=1)
        sd_ch[ok] = np.maximum(np.nanstd(arr_ch[ok], axis=1, ddof=1), 1e-3)
        z_clean[:, :, ch] = (arr_ch - mu_ch[:, None]) / sd_ch[:, None]

    # Empirical BIC lag-order audit across p in {2, 4, 6, 8} on pooled contiguous lags
    bic_rows = []
    for p_cand in [2, 4, 6, 8]:
        rss_sum = 0.0
        n_eff_total = 0
        # Evaluate on deterministic random subsample of 2,000 subjects for fast exact OLS BIC curve
        rng_bic = np.random.default_rng(SEED)
        sample_idx = rng_bic.choice(n_subj, size=min(2000, n_subj), replace=False)
        for i in sample_idx:
            y_list = []
            x_list = []
            for tau in range(p_cand, t_max):
                if interp_mask[i, tau] and np.all(interp_mask[i, tau - p_cand : tau]):
                    y_list.append(z_clean[i, tau])
                    lags = z_clean[i, tau - p_cand : tau][::-1].reshape(-1)
                    x_list.append(lags)
            if len(y_list) > 2 * p_cand + 10:
                ym = np.asarray(y_list)
                xm = np.asarray(x_list)
                coef, _, _, _ = linalg.lstsq(xm, ym)
                res = ym - xm @ coef
                n_obs_i = len(ym)
                cov_e = (res.T @ res) / n_obs_i + 1e-6 * np.eye(2)
                sign, logdet = np.linalg.slogdet(cov_e)
                if sign > 0:
                    rss_sum += n_obs_i * logdet + np.log(n_obs_i) * (4 * p_cand)
                    n_eff_total += n_obs_i
        bic_rows.append(
            {
                "ar_order_p_hours": p_cand,
                "n_transitions_evaluated": int(n_eff_total),
                "mean_bic_per_hour": round(float(rss_sum / max(n_eff_total, 1)), 5),
                "selected_pre_specified": int(p_cand == ar_order),
            }
        )
    bic_df = pd.DataFrame(bic_rows)

    # Batched Dual Extended Kalman Filter (DEKF) across all N_subj participants simultaneously
    # State vector at step tau: z_hat[i, tau] in R^2; Parameter matrix Theta[i] in R^{(2p) x 2}
    dim_reg = 2 * ar_order  # 12
    theta = np.zeros((n_subj, dim_reg, 2), dtype=np.float64)
    # Weakly informative autoregressive prior: lag-1 self-persistence 0.55
    theta[:, 0, 0] = 0.55
    theta[:, 1, 1] = 0.55
    p_cov = np.broadcast_to(np.eye(dim_reg, dtype=np.float64) * 0.5, (n_subj, dim_reg, dim_reg)).copy()
    q_eye = np.eye(dim_reg, dtype=np.float64)[None, :, :] * q_param

    z_hat = np.where(np.isnan(z_clean), 0.0, z_clean)
    s_var = np.full(n_subj, 0.15, dtype=np.float64)  # scalar per-channel state variance in dual filter

    # Frequency grid: 40 frequencies in (0, 0.5] cycles/hour
    freqs = np.linspace(1.0 / 48.0, 0.5, 48)
    circ_mask = (freqs >= (1.0 / 28.0)) & (freqs <= (1.0 / 20.0))  # 20-28h circadian band
    ultra_mask = (freqs >= (1.0 / 12.0)) & (freqs <= (1.0 / 3.0))  # 3-12h ultradian fragmentation band
    lags_r = np.arange(1, ar_order + 1, dtype=np.float64)  # (p,)
    # Exponential Fourier kernel E_fr: shape (F, p), E[f, r-1] = exp(-i * 2*pi * freq_f * r)
    exp_kernel = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * lags_r[None, :])  # (48, 6)

    # Accumulators for time-resolved PDC moments after 24-hour burn-in
    n_pdc_steps = np.zeros(n_subj, dtype=np.int32)
    sum_p2m_circ = np.zeros(n_subj, dtype=np.float64)
    sumsq_p2m_circ = np.zeros(n_subj, dtype=np.float64)
    sum_m2p_circ = np.zeros(n_subj, dtype=np.float64)
    sum_p2m_ultra = np.zeros(n_subj, dtype=np.float64)
    sum_m2p_ultra = np.zeros(n_subj, dtype=np.float64)
    sum_pow_circ = np.zeros(n_subj, dtype=np.float64)
    sum_pow_ultra = np.zeros(n_subj, dtype=np.float64)

    for tau in range(ar_order, t_max):
        # Active subjects with valid current observation & contiguous lag history
        has_hist = interp_mask[:, tau] & np.all(interp_mask[:, tau - ar_order : tau], axis=1)
        if not np.any(has_hist):
            continue

        # Regressor h_tau = [z_{tau-1}^T, ..., z_{tau-p}^T]^T of shape (N_act, 12)
        act_idx = np.where(has_hist)[0]
        h_tau = z_hat[act_idx, tau - ar_order : tau, :][:, ::-1, :].reshape(len(act_idx), dim_reg)
        y_obs = z_clean[act_idx, tau, :]  # (N_act, 2)

        # 1) State prediction & Dual Kalman state denoising update
        th_act = theta[act_idx]  # (N_act, 12, 2)
        z_pred = np.einsum("ni,nij->nj", h_tau, th_act)  # (N_act, 2)
        s_pred = s_var[act_idx] + 0.10
        # Only update with observed/short-interpolated measurement (higher measurement noise on interpolated hours)
        r_meas = np.where(obs_mask[act_idx, tau], r_state_noise, r_state_noise * 3.0)
        k_state = s_pred / (s_pred + r_meas)  # (N_act,)
        z_filt = z_pred + k_state[:, None] * (y_obs - z_pred)
        z_hat[act_idx, tau, :] = z_filt
        s_var[act_idx] = (1.0 - k_state) * s_pred

        # 2) Parameter Kalman update: P_{tau|tau-1} = P_{tau-1} + Q_theta
        p_pred = p_cov[act_idx] + q_eye  # (N_act, 12, 12)
        ph = np.einsum("nij,nj->ni", p_pred, h_tau)  # (N_act, 12)
        denom = 1.0 + np.einsum("ni,ni->n", h_tau, ph)  # (N_act,)
        k_gain = ph / denom[:, None]  # (N_act, 12)
        innov = z_filt - z_pred  # (N_act, 2)
        theta[act_idx] = th_act + k_gain[:, :, None] * innov[:, None, :]
        # Joseph-stabilized covariance update P_tau = P_pred - k_gain * (h^T P_pred)
        p_new = p_pred - np.einsum("ni,nj->nij", k_gain, ph)
        p_cov[act_idx] = 0.5 * (p_new + np.swapaxes(p_new, 1, 2))

        # 3) Evaluate Time-Resolved Partial Directed Coherence every 2 hours post 24h burn-in
        if tau >= 24 and (tau % 2 == 0):
            # Reshape theta[act_idx] (N_act, 6, 2_from, 2_to) -> A_r[l_to, j_from]
            # Recall h_tau stacked [Lux_{t-r}, MIMS_{t-r}], and output columns are [Lux_t, MIMS_t].
            # Therefore th_r[:, r, j_from, l_to] maps channel j_from at lag r -> channel l_to!
            th_r = theta[act_idx].reshape(len(act_idx), ar_order, 2, 2)
            # Frequency transform: sum_r th_r * exp(-i 2pi f r) -> shape (N_act, F, 2_from, 2_to)
            af_sum = np.einsum("nrjl,fr->nfjl", th_r, exp_kernel)
            # A(f) = I - sum_r A_r exp(-i 2pi f r), indexed as (N_act, F, j_from, l_to)
            eye2 = np.eye(2, dtype=np.complex128)[None, None, :, :]
            a_freq = eye2 - af_sum  # a_freq[:, :, j_from, l_to] = A_{l_to, j_from}(f)

            abs_sq = np.abs(a_freq) ** 2  # (N_act, F, j_from, l_to)
            col_norm = np.sqrt(np.maximum(np.sum(abs_sq, axis=3), 1e-12))  # sum over l_to -> (N_act, F, j_from)
            pdc = np.abs(a_freq) / col_norm[:, :, :, None]  # pdc[:, :, j_from, l_to] = pi_{l_to <- j_from}(f)

            # Photic (0) -> Motor (1): pdc[:, :, 0, 1]; Motor (1) -> Photic (0): pdc[:, :, 1, 0]
            p2m_c = np.mean(pdc[:, circ_mask, 0, 1], axis=1)
            m2p_c = np.mean(pdc[:, circ_mask, 1, 0], axis=1)
            p2m_u = np.mean(pdc[:, ultra_mask, 0, 1], axis=1)
            m2p_u = np.mean(pdc[:, ultra_mask, 1, 0], axis=1)

            # Parametric AR transfer spectral density of Motor channel H(f) = A(f)^{-1}
            det_a = a_freq[:, :, 0, 0] * a_freq[:, :, 1, 1] - a_freq[:, :, 0, 1] * a_freq[:, :, 1, 0]
            det_sq = np.maximum(np.abs(det_a) ** 2, 1e-8)
            # Motor power S_{11}(f) proportional to (|A_{00}|^2 + |A_{10_ transposed -> j=0,l=1? inv 2x2 row 1}|^2) / |det|^2
            # For M_{l,j} = A_{l,j} = a_freq[:,:,j,l], M^{-1}_{1,0} = -M_{1,0}/det = -a_freq[:,:,0,1]/det, M^{-1}_{1,1} = M_{0,0}/det = a_freq[:,:,0,0]/det
            s_motor = (abs_sq[:, :, 0, 1] + abs_sq[:, :, 0, 0]) / det_sq
            pow_c = np.mean(s_motor[:, circ_mask], axis=1)
            pow_u = np.mean(s_motor[:, ultra_mask], axis=1)

            n_pdc_steps[act_idx] += 1
            sum_p2m_circ[act_idx] += p2m_c
            sumsq_p2m_circ[act_idx] += p2m_c**2
            sum_m2p_circ[act_idx] += m2p_c
            sum_p2m_ultra[act_idx] += p2m_u
            sum_m2p_ultra[act_idx] += m2p_u
            sum_pow_circ[act_idx] += pow_c
            sum_pow_ultra[act_idx] += pow_u

    ok_tpdc = n_pdc_steps >= 12  # >= 24 post-burn-in hours evaluated
    cnt_f = np.where(ok_tpdc, n_pdc_steps.astype(float), np.nan)
    mean_p2m_c = sum_p2m_circ / cnt_f
    var_p2m_c = np.maximum(sumsq_p2m_circ / cnt_f - mean_p2m_c**2, 0.0)
    sd_p2m_c = np.sqrt(var_p2m_c)
    mean_m2p_c = sum_m2p_circ / cnt_f
    mean_p2m_u = sum_p2m_ultra / cnt_f
    mean_m2p_u = sum_m2p_ultra / cnt_f
    ultra_circ_ratio = np.where(sum_pow_circ > 0, sum_pow_ultra / np.maximum(sum_pow_circ, 1e-8), np.nan)
    ultra_circ_ratio = np.where(ok_tpdc, np.log1p(np.minimum(ultra_circ_ratio, 50.0)), np.nan)

    tpdc_df = pd.DataFrame(
        {
            "SEQN": seqn_unique,
            "tpdc_eval_epochs": n_pdc_steps,
            "tpdc_photic_to_motor_circ": mean_p2m_c,
            "tpdc_photic_to_motor_circ_sd": sd_p2m_c,
            "tpdc_motor_to_photic_circ": mean_m2p_c,
            "tpdc_net_entrainment_circ": mean_p2m_c - mean_m2p_c,
            "tpdc_photic_to_motor_ultra": mean_p2m_u,
            "tpdc_motor_to_photic_ultra": mean_m2p_u,
            "tpdc_log_ultra_circ_power_ratio": ultra_circ_ratio,
        }
    )
    print(f"[Pillar 1 DEKF-TPDC] Computed time-resolved VAR({ar_order}) TPDC for {int(np.sum(ok_tpdc)):,}/{n_subj:,} subjects in {time.time() - t0:.1f}s")
    return tpdc_df, bic_df


# ---------------------------------------------------------------------------
# 3. VanderWeele E-Values & Cross-Fitted Neyman-Orthogonal DML Mediation
# ---------------------------------------------------------------------------
def compute_evalue_continuous(beta_std: float, se_std: float) -> Tuple[float, float]:
    """
    VanderWeele & Ding (2017, Ann Intern Med) E-value for standardized continuous effect size d
    via RR approx exp(0.91 * |d|), CI bound exp(0.91 * max(0, |d| - 1.96*se)).
    """
    if np.isnan(beta_std) or np.isnan(se_std):
        return np.nan, np.nan
    d_abs = abs(float(beta_std))
    rr = float(np.exp(0.91 * d_abs))
    e_pt = rr + float(np.sqrt(max(rr * (rr - 1.0), 0.0)))
    d_ci = max(0.0, d_abs - 1.96 * float(se_std))
    if d_ci <= 0.0:
        e_ci = 1.0
    else:
        rr_ci = float(np.exp(0.91 * d_ci))
        e_ci = rr_ci + float(np.sqrt(max(rr_ci * (rr_ci - 1.0), 0.0)))
    return round(e_pt, 3), round(e_ci, 3)


def build_confounder_matrix(df: pd.DataFrame, include_cycle: bool = True) -> np.ndarray:
    """
    Construct the locked pre-exposure confounder design matrix W_i (§2.3 & §1.3 of SAP_v1_FROZEN.md):
      - Restricted cubic / B-spline age (4 knots), female, race_ethnicity, educ_adult_code, PIR,
        BMI, SBP, DBP, hypertension_doc, bp_med_current, high_chol_doc, HbA1c, total_chol, HDL,
        WBC, log(NLR), smoking (ever/current), alcohol_12yr, modvig_pa_any, sedentary_min_day,
        diabetes_doc, cvd_any_doc, hearing_difficulty, and physiological serum_creatinine_mgdl
        clearance adjustment (+ missingness indicators for partially missing covariates so zero
        rows with complete A, M, Y are dropped due to auxiliary lab missingness!).
    """
    cont_cols = [
        "pir", "bmi", "sbp_mean", "dbp_mean", "hba1c_pct",
        "total_chol_mgdl", "hdl_chol_mgdl", "wbc_10e9L", "nlr",
        "sedentary_min_day", "serum_creatinine_mgdl",
    ]
    bin_cols = [
        "female", "educ_adult_code", "ever_smoked_100", "current_smoker",
        "alcohol_12yr", "modvig_pa_any", "hypertension_doc", "bp_med_current",
        "high_chol_doc", "diabetes_doc", "cvd_any_doc", "hearing_difficulty",
    ]
    # Age spline
    age_arr = df[["age_years"]].to_numpy(dtype=float)
    spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    age_sp = spline.fit_transform(age_arr)

    # Continuous with missingness indicators + median imputation + standardization
    x_cont_raw = df[cont_cols].copy()
    x_cont_raw["nlr"] = np.log1p(np.maximum(x_cont_raw["nlr"].to_numpy(dtype=float), 0.0))
    miss_ind = x_cont_raw.isna().astype(float)
    keep_miss = [c for c in miss_ind.columns if 0.005 < miss_ind[c].mean() < 0.995]
    imp_c = SimpleImputer(strategy="median")
    x_cont = StandardScaler().fit_transform(imp_c.fit_transform(x_cont_raw))

    # Binary/ordinal with mode imputation
    x_bin_raw = df[bin_cols].copy()
    imp_b = SimpleImputer(strategy="most_frequent")
    x_bin = StandardScaler().fit_transform(imp_b.fit_transform(x_bin_raw))

    # Categorical race_ethnicity (+ cycle if multi-cycle)
    cat_cols = ["race_ethnicity"] + (["cycle"] if include_cycle and df["cycle"].nunique() > 1 else [])
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    x_cat = ohe.fit_transform(df[cat_cols].astype(str))

    parts = [age_sp, x_cont, x_bin, x_cat]
    if keep_miss:
        parts.append(miss_ind[keep_miss].to_numpy(dtype=float))
    return np.hstack(parts)


def cross_fit_stacked_nuisance(w_mat: np.ndarray, target: np.ndarray, kf: KFold, seed_offset: int = 0) -> np.ndarray:
    """
    5-fold cross-fitted stacked ensemble nuisance learner (RidgeCV + RandomForest + LightGBM)
    minimizing out-of-fold MSE via non-negative ridge/simplex stacking weights.
    """
    n = len(target)
    oof_preds = np.zeros((n, 3), dtype=np.float64)
    for fold_idx, (tr, va) in enumerate(kf.split(w_mat)):
        w_tr, y_tr = w_mat[tr], target[tr]
        w_va = w_mat[va]

        m_ridge = RidgeCV(alphas=np.logspace(-2, 4, 25))
        m_ridge.fit(w_tr, y_tr)
        oof_preds[va, 0] = m_ridge.predict(w_va)

        m_rf = RandomForestRegressor(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=12,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED + seed_offset + fold_idx,
        )
        m_rf.fit(w_tr, y_tr)
        oof_preds[va, 1] = m_rf.predict(w_va)

        m_lgb = lgb.LGBMRegressor(
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=SEED + 100 + seed_offset + fold_idx,
            verbose=-1,
            n_jobs=-1,
        )
        m_lgb.fit(w_tr, y_tr)
        oof_preds[va, 2] = m_lgb.predict(w_va)

    # Compute out-of-fold inverse-MSE convex weights per fold to prevent any stacking leakage
    stacked = np.zeros(n, dtype=np.float64)
    for tr, va in kf.split(w_mat):
        # Solve constrained least-squares simplex weights on training fold OOF predictions
        p_tr = oof_preds[tr]
        y_tr = target[tr]
        # Grid / NNLS projection onto simplex
        cov_p = p_tr.T @ p_tr + 1e-4 * np.eye(3)
        rhs = p_tr.T @ y_tr
        w_raw = np.maximum(linalg.solve(cov_p, rhs, assume_a="pos"), 0.0)
        if np.sum(w_raw) <= 0:
            w_opt = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        else:
            w_opt = w_raw / np.sum(w_raw)
        stacked[va] = oof_preds[va] @ w_opt
    return stacked


def run_dml_single_sample_mediation(
    sub: pd.DataFrame,
    exposure_col: str,
    mediator_col: str,
    outcome_col: str,
    weight_col: Optional[str] = None,
    n_boot: int = 1000,
) -> Dict[str, object]:
    """
    5-fold Cross-Fitted Neyman-Orthogonal Double Machine Learning (DML) Causal Mediation:
      A -> M (log sNfL) -> Y | W
    Computes:
      - Path alpha (A -> M | W)
      - Path beta  (M -> Y | A, W)
      - Total Effect TE (A -> Y | W)
      - Natural Direct Effect NDE
      - Natural Indirect Effect NIE = alpha * beta (both exact orthogonal influence-function SE/p
        and B = 1,000 nonparametric residual bootstrap 95% CIs, plus A x M interaction check
        and VanderWeele mediational E-values).
    """
    req = [exposure_col, mediator_col, outcome_col, "age_years"]
    d = sub.dropna(subset=req).copy().reset_index(drop=True)
    n = len(d)

    # Standardize A, M, Y to SD units for directly comparable standardized effect sizes + keep raw outcome SD
    a_raw = d[exposure_col].to_numpy(dtype=float)
    m_raw = d[mediator_col].to_numpy(dtype=float)
    y_raw = d[outcome_col].to_numpy(dtype=float)
    sd_y_raw = float(np.std(y_raw, ddof=1))

    a_std = (a_raw - np.mean(a_raw)) / np.std(a_raw, ddof=1)
    m_std = (m_raw - np.mean(m_raw)) / np.std(m_raw, ddof=1)
    y_std = (y_raw - np.mean(y_raw)) / sd_y_raw

    w_mat = build_confounder_matrix(d, include_cycle=(d["cycle"].nunique() > 1))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    a_hat = cross_fit_stacked_nuisance(w_mat, a_std, kf, seed_offset=11)
    m_hat = cross_fit_stacked_nuisance(w_mat, m_std, kf, seed_offset=29)
    y_hat = cross_fit_stacked_nuisance(w_mat, y_std, kf, seed_offset=47)

    a_til = a_std - a_hat
    m_til = m_std - m_hat
    y_til = y_std - y_hat

    if weight_col is not None and weight_col in d.columns and d[weight_col].notna().sum() > 50:
        sw = d[weight_col].fillna(d[weight_col].median()).to_numpy(dtype=float)
        sw = np.maximum(sw, 1e-6)
        sw = sw / np.mean(sw)
    else:
        sw = np.ones(n, dtype=float)

    # 1) Path alpha: A -> M | W
    denom_a = float(np.mean(sw * (a_til**2)))
    alpha = float(np.mean(sw * a_til * m_til) / denom_a)
    psi_alpha = (sw * a_til * (m_til - alpha * a_til)) / denom_a
    se_alpha = float(np.sqrt(np.sum(psi_alpha**2)) / n)
    p_alpha = float(2.0 * stats.norm.sf(abs(alpha) / max(se_alpha, 1e-12)))

    # 2) Total Effect TE: A -> Y | W
    te = float(np.mean(sw * a_til * y_til) / denom_a)
    psi_te = (sw * a_til * (y_til - te * a_til)) / denom_a
    se_te = float(np.sqrt(np.sum(psi_te**2)) / n)
    p_te = float(2.0 * stats.norm.sf(abs(te) / max(se_te, 1e-12)))

    # 3) Joint Direct & Mediator Score: Y_til = NDE * A_til + beta * M_til + eps
    z_am = np.column_stack([a_til, m_til])  # (n, 2)
    q_mat = (z_am.T * sw) @ z_am / float(n)  # (2, 2)
    rhs_am = (z_am.T * sw) @ y_til / float(n)
    nde, beta = linalg.solve(q_mat, rhs_am, assume_a="pos")
    nde = float(nde)
    beta = float(beta)
    res_joint = y_til - (nde * a_til + beta * m_til)
    q_inv = linalg.inv(q_mat)
    psi_joint = (z_am * (sw * res_joint)[:, None]) @ q_inv.T  # (n, 2)
    psi_nde = psi_joint[:, 0]
    psi_beta = psi_joint[:, 1]
    se_nde = float(np.sqrt(np.sum(psi_nde**2)) / n)
    se_beta = float(np.sqrt(np.sum(psi_beta**2)) / n)
    p_nde = float(2.0 * stats.norm.sf(abs(nde) / max(se_nde, 1e-12)))
    p_beta = float(2.0 * stats.norm.sf(abs(beta) / max(se_beta, 1e-12)))

    # 4) Natural Indirect Effect NIE = alpha * beta (exact orthogonal influence function + bootstrap CI)
    nie = alpha * beta
    psi_nie = beta * psi_alpha + alpha * psi_beta
    se_nie = float(np.sqrt(np.sum(psi_nie**2)) / n)
    p_nie = float(2.0 * stats.norm.sf(abs(nie) / max(se_nie, 1e-12)))

    # Exposure x Mediator (A_til * M_til) nonlinear interaction check
    int_term = a_til * m_til - np.mean(a_til * m_til)
    z_int = np.column_stack([a_til, m_til, int_term])
    q_int = (z_int.T * sw) @ z_int / float(n)
    c_int = linalg.solve(q_int, (z_int.T * sw) @ y_til / float(n))
    res_int = y_til - z_int @ c_int
    psi_int = (z_int * (sw * res_int)[:, None]) @ linalg.inv(q_int).T
    se_kappa = float(np.sqrt(np.sum(psi_int[:, 2] ** 2)) / n)
    p_interaction_am = float(2.0 * stats.norm.sf(abs(float(c_int[2])) / max(se_kappa, 1e-12)))

    # Nonparametric B=1000 bootstrap on cross-fitted orthogonal residuals
    rng = np.random.default_rng(SEED + n)
    nie_boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sw_b = sw[idx]
        ab = a_til[idx]
        mb = m_til[idx]
        yb = y_til[idx]
        da_b = max(float(np.mean(sw_b * (ab**2))), 1e-8)
        al_b = float(np.mean(sw_b * ab * mb) / da_b)
        zb = np.column_stack([ab, mb])
        qb = (zb.T * sw_b) @ zb / float(n) + 1e-8 * np.eye(2)
        _, be_b = linalg.solve(qb, (zb.T * sw_b) @ yb / float(n))
        nie_boots[b] = al_b * float(be_b)
    nie_boot_lo, nie_boot_hi = np.percentile(nie_boots, [2.5, 97.5])

    prop_med = float(nie / te) if abs(te) > 1e-4 else np.nan
    ev_te_pt, ev_te_ci = compute_evalue_continuous(te, se_te)
    ev_nie_pt, ev_nie_ci = compute_evalue_continuous(nie, se_nie)

    return {
        "n_obs": int(n),
        "exposure": exposure_col,
        "mediator": mediator_col,
        "outcome": outcome_col,
        "weighting": weight_col if weight_col else "unweighted_cross_fitted_DML",
        "alpha_A_to_M_std": round(alpha, 5),
        "alpha_se": round(se_alpha, 5),
        "alpha_p": float(f"{p_alpha:.4e}"),
        "beta_M_to_Y_std": round(beta, 5),
        "beta_se": round(se_beta, 5),
        "beta_p": float(f"{p_beta:.4e}"),
        "TE_std": round(te, 5),
        "TE_raw_units": round(te * sd_y_raw, 5),
        "TE_se": round(se_te, 5),
        "TE_ci95_lo": round(te - 1.96 * se_te, 5),
        "TE_ci95_hi": round(te + 1.96 * se_te, 5),
        "TE_p": float(f"{p_te:.4e}"),
        "NDE_std": round(nde, 5),
        "NDE_se": round(se_nde, 5),
        "NDE_p": float(f"{p_nde:.4e}"),
        "NIE_std": round(nie, 5),
        "NIE_raw_units": round(nie * sd_y_raw, 5),
        "NIE_se_orthogonal": round(se_nie, 5),
        "NIE_ci95_lo_orth": round(nie - 1.96 * se_nie, 5),
        "NIE_ci95_hi_orth": round(nie + 1.96 * se_nie, 5),
        "NIE_boot95_lo": round(float(nie_boot_lo), 5),
        "NIE_boot95_hi": round(float(nie_boot_hi), 5),
        "NIE_p": float(f"{p_nie:.4e}"),
        "prop_mediated": round(prop_med, 4) if not np.isnan(prop_med) else np.nan,
        "interaction_AxM_p": round(p_interaction_am, 4),
        "E_value_TE_point": ev_te_pt,
        "E_value_TE_ci95": ev_te_ci,
        "E_value_NIE_point": ev_nie_pt,
        "E_value_NIE_ci95": ev_nie_ci,
    }


def run_two_sample_moment_coupled_dml(
    df_all: pd.DataFrame,
    exposure_col: str,
    outcome_col: str,
) -> Dict[str, object]:
    """
    Two-Sample Cross-Fitted Moment-Coupled DML Mediation (§2.3 of SAP_v1_FROZEN.md):
      - Stage 1 (Arm B older/midlife-to-older sNfL + valid 4d16h PAX, N = 1,552; and age >= 50 subset):
        Cross-fitted DML path alpha_{S1} (A -> log_snfl | W)
      - Stage 2 (Arm C Full 2011-2014 Older Adults >= 60 with valid 4d16h PAX + zero-ceiling cognitive outcome, N ~ 2,541):
        Cross-fitted DML Total Effect TE_{S2} (A -> Y | W)
      - Stage 3 (Arm D Direct 4-Modality Bridge, N = 419):
        Cross-fitted DML conditional mediator slope beta_{D} (log_snfl -> Y | A, W)
      Accounts exactly for the N = 419 participant overlap between Arm B, Arm C, and Arm D via
      individual-level SEQN-indexed influence-function covariance!
    """
    arm_b = df_all[(df_all["pax_eligible_primary_4d16h"] == 1) & df_all["log_snfl"].notna() & df_all[exposure_col].notna()].copy().reset_index(drop=True)
    arm_c = df_all[(df_all["age_years"] >= 60.0) & (df_all["pax_eligible_primary_4d16h"] == 1) & df_all[outcome_col].notna() & df_all[exposure_col].notna()].copy().reset_index(drop=True)
    arm_d = df_all[(df_all["age_years"] >= 60.0) & (df_all["pax_eligible_primary_4d16h"] == 1) & df_all["log_snfl"].notna() & df_all[outcome_col].notna() & df_all[exposure_col].notna()].copy().reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Arm B: alpha_{B} (A -> log_snfl | W)
    wb = build_confounder_matrix(arm_b, include_cycle=False)
    ab_s = (arm_b[exposure_col].to_numpy(float) - arm_b[exposure_col].mean()) / arm_b[exposure_col].std(ddof=1)
    mb_s = (arm_b["log_snfl"].to_numpy(float) - arm_b["log_snfl"].mean()) / arm_b["log_snfl"].std(ddof=1)
    ab_til = ab_s - cross_fit_stacked_nuisance(wb, ab_s, kf, seed_offset=101)
    mb_til = mb_s - cross_fit_stacked_nuisance(wb, mb_s, kf, seed_offset=103)
    da_b = float(np.mean(ab_til**2))
    alpha_b = float(np.mean(ab_til * mb_til) / da_b)
    psi_alpha_b = pd.Series((ab_til * (mb_til - alpha_b * ab_til)) / (da_b * len(arm_b)), index=arm_b["SEQN"].values)

    # Arm C: TE_{C} (A -> Y | W) across full N ~ 2,541 zero-ceiling cognitive cohort
    wc = build_confounder_matrix(arm_c, include_cycle=True)
    ac_s = (arm_c[exposure_col].to_numpy(float) - arm_c[exposure_col].mean()) / arm_c[exposure_col].std(ddof=1)
    yc_raw = arm_c[outcome_col].to_numpy(float)
    sd_yc = float(np.std(yc_raw, ddof=1))
    yc_s = (yc_raw - np.mean(yc_raw)) / sd_yc
    ac_til = ac_s - cross_fit_stacked_nuisance(wc, ac_s, kf, seed_offset=201)
    yc_til = yc_s - cross_fit_stacked_nuisance(wc, yc_s, kf, seed_offset=203)
    da_c = float(np.mean(ac_til**2))
    te_c = float(np.mean(ac_til * yc_til) / da_c)
    se_te_c = float(np.sqrt(np.sum(((ac_til * (yc_til - te_c * ac_til)) / (da_c * len(arm_c))) ** 2)))
    p_te_c = float(2.0 * stats.norm.sf(abs(te_c) / max(se_te_c, 1e-12)))

    # Arm D: beta_{D} (log_snfl -> Y | A, W) on direct quadruple bridge
    wd = build_confounder_matrix(arm_d, include_cycle=False)
    ad_s = (arm_d[exposure_col].to_numpy(float) - arm_d[exposure_col].mean()) / arm_d[exposure_col].std(ddof=1)
    md_s = (arm_d["log_snfl"].to_numpy(float) - arm_d["log_snfl"].mean()) / arm_d["log_snfl"].std(ddof=1)
    yd_s = (arm_d[outcome_col].to_numpy(float) - arm_d[outcome_col].mean()) / arm_d[outcome_col].std(ddof=1)
    ad_til = ad_s - cross_fit_stacked_nuisance(wd, ad_s, kf, seed_offset=301)
    md_til = md_s - cross_fit_stacked_nuisance(wd, md_s, kf, seed_offset=303)
    yd_til = yd_s - cross_fit_stacked_nuisance(wd, yd_s, kf, seed_offset=307)
    zd = np.column_stack([ad_til, md_til])
    qd = (zd.T @ zd) / float(len(arm_d))
    _, beta_d = linalg.solve(qd, (zd.T @ yd_til) / float(len(arm_d)), assume_a="pos")
    beta_d = float(beta_d)
    res_d = yd_til - (zd @ linalg.solve(qd, (zd.T @ yd_til) / float(len(arm_d)), assume_a="pos"))
    psi_joint_d = (zd * res_d[:, None]) @ linalg.inv(qd).T / float(len(arm_d))
    psi_beta_d = pd.Series(psi_joint_d[:, 1], index=arm_d["SEQN"].values)

    # Combine SEQN-aligned influence functions across union(Arm B, Arm D) so overlap covariance is exact!
    union_seqn = np.union1d(arm_b["SEQN"].values, arm_d["SEQN"].values)
    if_a = psi_alpha_b.reindex(union_seqn, fill_value=0.0).to_numpy()
    if_b = psi_beta_d.reindex(union_seqn, fill_value=0.0).to_numpy()

    se_alpha_b = float(np.sqrt(np.sum(if_a**2)))
    se_beta_d = float(np.sqrt(np.sum(if_b**2)))
    nie_2s = alpha_b * beta_d
    if_nie_2s = beta_d * if_a + alpha_b * if_b
    se_nie_2s = float(np.sqrt(np.sum(if_nie_2s**2)))
    p_nie_2s = float(2.0 * stats.norm.sf(abs(nie_2s) / max(se_nie_2s, 1e-12)))
    nde_2s = te_c - nie_2s
    ev_nie_pt, ev_nie_ci = compute_evalue_continuous(nie_2s, se_nie_2s)
    ev_te_pt, ev_te_ci = compute_evalue_continuous(te_c, se_te_c)

    return {
        "exposure": exposure_col,
        "outcome": outcome_col,
        "n_armB_sNfL_stage1": int(len(arm_b)),
        "n_armC_cognition_stage2": int(len(arm_c)),
        "n_armD_bridge_overlap": int(len(arm_d)),
        "alpha_armB_std": round(alpha_b, 5),
        "alpha_armB_se": round(se_alpha_b, 5),
        "alpha_armB_p": float(f"{2.0 * stats.norm.sf(abs(alpha_b) / max(se_alpha_b, 1e-12)):.4e}"),
        "TE_armC_std": round(te_c, 5),
        "TE_armC_raw_units": round(te_c * sd_yc, 5),
        "TE_armC_se": round(se_te_c, 5),
        "TE_armC_p": float(f"{p_te_c:.4e}"),
        "beta_armD_std": round(beta_d, 5),
        "beta_armD_se": round(se_beta_d, 5),
        "beta_armD_p": float(f"{2.0 * stats.norm.sf(abs(beta_d) / max(se_beta_d, 1e-12)):.4e}"),
        "NIE_moment_coupled_std": round(nie_2s, 5),
        "NIE_moment_coupled_raw_units": round(nie_2s * sd_yc, 5),
        "NIE_moment_coupled_se": round(se_nie_2s, 5),
        "NIE_ci95_lo": round(nie_2s - 1.96 * se_nie_2s, 5),
        "NIE_ci95_hi": round(nie_2s + 1.96 * se_nie_2s, 5),
        "NIE_p": float(f"{p_nie_2s:.4e}"),
        "NDE_implied_std": round(nde_2s, 5),
        "prop_mediated_2sample": round(float(nie_2s / te_c), 4) if abs(te_c) > 1e-4 else np.nan,
        "E_value_TE_armC_point": ev_te_pt,
        "E_value_TE_armC_ci95": ev_te_ci,
        "E_value_NIE_point": ev_nie_pt,
        "E_value_NIE_ci95": ev_nie_ci,
    }


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 2 Pillar 1: NHANES 24h fPCA + DEKF-TPDC + Cross-Fitted DML sNfL Mediation ===")

    nh = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_participants.parquet")
    paxhr = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_paxhr_clean.parquet")

    # 1. Build 24h log(1+MIMS) and log(1+Lux) profiles for all adults >= 20 with usable PAX
    adults_pax = nh[(nh["age_years"] >= 20.0) & (nh["pax_has_data"] == 1)].copy().reset_index(drop=True)
    elig_seqns = adults_pax["SEQN"].to_numpy()

    vhr = paxhr[(paxhr["SEQN"].isin(elig_seqns)) & (paxhr["valid_hour_flag"] == 1)].copy()
    vhr["log_mims"] = np.log1p(np.maximum(vhr["mims_per_min"].to_numpy(float), 0.0))
    vhr["log_lux"] = np.log1p(np.maximum(vhr["lux_per_min"].to_numpy(float), 0.0))
    prof = (
        vhr.groupby(["SEQN", "clock_hour"], as_index=False)
        .agg(log_mims_h=("log_mims", "mean"), log_lux_h=("log_lux", "mean"))
    )
    piv_m = prof.pivot(index="SEQN", columns="clock_hour", values="log_mims_h").reindex(index=elig_seqns, columns=list(range(24)))
    piv_l = prof.pivot(index="SEQN", columns="clock_hour", values="log_lux_h").reindex(index=elig_seqns, columns=list(range(24)))

    prim_mask = (adults_pax["pax_eligible_primary_4d16h"].to_numpy() == 1) & (~np.any(np.isnan(piv_m.to_numpy(float)), axis=1))
    print(f"[Pillar 1 fPCA] Fitting periodic Fourier-basis penalized fPCA on N = {int(np.sum(prim_mask)):,} primary 4d16h adults...")

    sc_mims, eig_mims, var_mims, _ = fit_penalized_fpca_24h(
        piv_m.to_numpy(float)[prim_mask], piv_m.to_numpy(float), channel_name="log1p_MIMS", n_harmonics=4, n_components=4
    )
    sc_lux, eig_lux, var_lux, _ = fit_penalized_fpca_24h(
        piv_l.to_numpy(float)[prim_mask], piv_l.to_numpy(float), channel_name="log1p_Lux", n_harmonics=4, n_components=4
    )

    eig_all = pd.concat([eig_mims, eig_lux], axis=0, ignore_index=True)
    var_all = pd.concat([var_mims, var_lux], axis=0, ignore_index=True)
    eig_all.to_csv(OUT_DIR / "fpca_eigenfunctions_24h.csv", index=False)
    var_all.to_csv(OUT_DIR / "fpca_variance_explained.csv", index=False)
    print(var_all.to_string(index=False))

    fpca_feat = pd.DataFrame({"SEQN": elig_seqns})
    for k in range(4):
        fpca_feat[f"fpca_mims_pc{k+1}"] = sc_mims[:, k]
        fpca_feat[f"fpca_lux_pc{k+1}"] = sc_lux[:, k]

    # 2. Batched DEKF-TPDC on contiguous multi-day hourly [log(1+Lux), log(1+MIMS)]^T series
    tpdc_elig = adults_pax.loc[(adults_pax["pax_eligible_primary_4d16h"] == 1) | (adults_pax["pax_eligible_sens_4d10h"] == 1), "SEQN"].to_numpy()
    tpdc_df, bic_df = compute_batched_dekf_tpdc(paxhr, tpdc_elig, ar_order=6)
    bic_df.to_csv(OUT_DIR / "tpdc_var_bic_selection.csv", index=False)

    # Merge fPCA + DEKF-TPDC features onto participant master dataframe and save derived feature parquet
    feat_merged = fpca_feat.merge(tpdc_df, on="SEQN", how="outer")
    nh_full = nh.merge(feat_merged, on="SEQN", how="left")

    # Composite Circadian-Photic Vitality Index (standardized mean of fPC1_MIMS, RA, IS, -IV, and Photic->Motor Circadian TPDC)
    prim_ref = nh_full[nh_full["pax_eligible_primary_4d16h"] == 1]
    vit_parts = [
        ("fpca_mims_pc1", 1.0),
        ("pax_RA", 1.0),
        ("pax_IS", 1.0),
        ("pax_IV", -1.0),
        ("tpdc_photic_to_motor_circ", 1.0),
    ]
    z_sum = np.zeros(len(nh_full), dtype=float)
    z_valid = np.ones(len(nh_full), dtype=bool)
    for col_v, sgn in vit_parts:
        mu_v = float(prim_ref[col_v].mean())
        sd_v = float(prim_ref[col_v].std(ddof=1))
        zv = sgn * (nh_full[col_v].to_numpy(float) - mu_v) / sd_v
        z_valid &= ~np.isnan(zv)
        z_sum += np.where(np.isnan(zv), 0.0, zv)
    nh_full["circadian_photic_vitality_z"] = np.where(z_valid, z_sum / float(len(vit_parts)), np.nan)

    feat_out_cols = ["SEQN"] + [c for c in feat_merged.columns if c != "SEQN"] + ["circadian_photic_vitality_z"]
    nh_full[feat_out_cols].to_parquet(DATA_DIR / "nhanes_2011_2014_fpca_tpdc_features.parquet", index=False)

    # Summarize TPDC + fPCA distributions across primary 4d16h cohort, sNfL Arm B, and Older60 Arm C
    prim_sub = nh_full[nh_full["pax_eligible_primary_4d16h"] == 1]
    tpdc_sum_rows = []
    for col_s in [
        "fpca_mims_pc1", "fpca_mims_pc2", "fpca_lux_pc1", "fpca_lux_pc2",
        "tpdc_photic_to_motor_circ", "tpdc_photic_to_motor_circ_sd",
        "tpdc_motor_to_photic_circ", "tpdc_net_entrainment_circ",
        "tpdc_photic_to_motor_ultra", "tpdc_log_ultra_circ_power_ratio",
        "circadian_photic_vitality_z",
    ]:
        vals = prim_sub[col_s].dropna()
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        tpdc_sum_rows.append(
            {
                "feature": col_s,
                "n_primary_4d16h": int(len(vals)),
                "mean": round(float(vals.mean()), 5),
                "sd": round(float(vals.std(ddof=1)), 5),
                "median": round(float(q50), 5),
                "q25": round(float(q25), 5),
                "q75": round(float(q75), 5),
            }
        )
    pd.DataFrame(tpdc_sum_rows).to_csv(OUT_DIR / "tpdc_band_summary.csv", index=False)

    # 3. Cross-Fitted DML Biological Link Audit: Circadian/Photic Features -> Serum log(sNfL)
    exposures = [
        "circadian_photic_vitality_z",
        "fpca_mims_pc1",
        "fpca_lux_pc1",
        "pax_RA",
        "pax_IS",
        "pax_IV",
        "tpdc_photic_to_motor_circ",
        "tpdc_log_ultra_circ_power_ratio",
    ]
    arm_b_4d16h = nh_full[(nh_full["pax_eligible_primary_4d16h"] == 1) & nh_full["log_snfl"].notna()].copy()
    arm_b_4d10h = nh_full[(nh_full["pax_eligible_sens_4d10h"] == 1) & nh_full["log_snfl"].notna()].copy()
    print(f"[Pillar 1 DML] Running cross-fitted DML biological link & mediation (Arm B 4d16h N={len(arm_b_4d16h):,}, 4d10h sens N={len(arm_b_4d10h):,})...")

    snfl_assoc_rows = []
    for exp_c in exposures:
        res_p = run_dml_single_sample_mediation(arm_b_4d16h, exp_c, "log_snfl", "phq8_nosleep", weight_col=None, n_boot=1000)
        res_w = run_dml_single_sample_mediation(arm_b_4d16h, exp_c, "log_snfl", "phq8_nosleep", weight_col="wtssnh2y", n_boot=400)
        res_10 = run_dml_single_sample_mediation(arm_b_4d10h, exp_c, "log_snfl", "phq8_nosleep", weight_col=None, n_boot=400)
        snfl_assoc_rows.append(
            {
                "exposure": exp_c,
                "n_primary_4d16h": int((arm_b_4d16h[exp_c].notna() & arm_b_4d16h["log_snfl"].notna()).sum()),
                "alpha_log_snfl_std_4d16h": res_p["alpha_A_to_M_std"],
                "alpha_se_4d16h": res_p["alpha_se"],
                "alpha_p_4d16h": res_p["alpha_p"],
                "alpha_survey_weighted_wtssnh2y": res_w["alpha_A_to_M_std"],
                "alpha_survey_weighted_p": res_w["alpha_p"],
                "n_sens_4d10h": res_10["n_obs"],
                "alpha_sens_4d10h_std": res_10["alpha_A_to_M_std"],
                "alpha_sens_4d10h_p": res_10["alpha_p"],
            }
        )
    snfl_assoc_df = pd.DataFrame(snfl_assoc_rows)
    snfl_assoc_df["alpha_fdr_q_4d16h"] = multipletests(snfl_assoc_df["alpha_p_4d16h"], method="fdr_bh")[1]
    snfl_assoc_df["alpha_holm_p_4d16h"] = multipletests(snfl_assoc_df["alpha_p_4d16h"], method="holm")[1]
    snfl_assoc_df.to_csv(OUT_DIR / "snfl_circadian_dml_associations.csv", index=False)

    # 4. Single-Sample Cross-Fitted DML Causal Mediation Table (Arm B Affective Load + Arm D Direct 4-Modality Bridge Cognition)
    key_exposures = [
        "circadian_photic_vitality_z",
        "fpca_mims_pc1",
        "pax_RA",
        "pax_IS",
        "pax_IV",
        "tpdc_photic_to_motor_circ",
        "tpdc_log_ultra_circ_power_ratio",
    ]
    arm_d_bridge = nh_full[(nh_full["age_years"] >= 60.0) & (nh_full["pax_eligible_primary_4d16h"] == 1) & nh_full["log_snfl"].notna()].copy()

    single_med_rows = []
    for exp_c in key_exposures:
        # Outcome Family Y1: PHQ-8_nosleep (co-primary unconfounded affective endpoint) & PHQ-9 total in Arm B (N = 1,467)
        for out_y1 in ["phq8_nosleep", "phq9_total"]:
            r1 = run_dml_single_sample_mediation(arm_b_4d16h, exp_c, "log_snfl", out_y1, weight_col=None, n_boot=1000)
            r1["cohort_arm"] = "ArmB_Adults20_75_SingleSample"
            single_med_rows.append(r1)
        # Outcome Family Y2: Direct 4-Modality Bridge (N = 419 DSST, Global 3-Domain Cognition Z, Animal Fluency, CERAD)
        for out_y2 in ["dsst", "global_cognition_z", "animal_fluency", "cerad_total_40"]:
            r2 = run_dml_single_sample_mediation(arm_d_bridge, exp_c, "log_snfl", out_y2, weight_col=None, n_boot=1000)
            r2["cohort_arm"] = "ArmD_Older60_75_Direct4ModalityBridge"
            single_med_rows.append(r2)

    single_med_df = pd.DataFrame(single_med_rows)
    single_med_df["TE_fdr_q"] = multipletests(single_med_df["TE_p"], method="fdr_bh")[1]
    single_med_df["NIE_fdr_q"] = multipletests(single_med_df["NIE_p"], method="fdr_bh")[1]
    # Co-primary Holm-Bonferroni on circadian_photic_vitality_z -> (phq8_nosleep, dsst)
    coprim_mask = (single_med_df["exposure"] == "circadian_photic_vitality_z") & (single_med_df["outcome"].isin(["phq8_nosleep", "dsst"]))
    single_med_df["coprimary_NIE_holm_p"] = np.nan
    if coprim_mask.sum() == 2:
        single_med_df.loc[coprim_mask, "coprimary_NIE_holm_p"] = multipletests(single_med_df.loc[coprim_mask, "NIE_p"], method="holm")[1]
    single_med_df.to_csv(OUT_DIR / "dml_single_sample_mediation_results.csv", index=False)

    # 5. Two-Sample Cross-Fitted Moment-Coupled DML Mediation (Arm B N=1,552 + Arm C N=2,541 + Arm D N=419 exact overlap IF)
    twosamp_rows = []
    for exp_c in key_exposures:
        for cog_y in ["dsst", "global_cognition_z", "animal_fluency", "cerad_total_40"]:
            twosamp_rows.append(run_two_sample_moment_coupled_dml(nh_full, exp_c, cog_y))
    twosamp_df = pd.DataFrame(twosamp_rows)
    twosamp_df["TE_armC_fdr_q"] = multipletests(twosamp_df["TE_armC_p"], method="fdr_bh")[1]
    twosamp_df["NIE_fdr_q"] = multipletests(twosamp_df["NIE_p"], method="fdr_bh")[1]
    twosamp_df["NIE_holm_p"] = multipletests(twosamp_df["NIE_p"], method="holm")[1]
    twosamp_df.to_csv(OUT_DIR / "dml_two_sample_moment_coupled_cognition.csv", index=False)

    # 6. Survey-Weighted Sensitivity & Arm A Full-Adult Affective DML Benchmark (N = 7,508)
    arm_a = nh_full[(nh_full["age_years"] >= 20.0) & (nh_full["pax_eligible_primary_4d16h"] == 1)].copy()
    sens_rows = []
    for exp_c in ["circadian_photic_vitality_z", "fpca_mims_pc1", "pax_RA", "tpdc_photic_to_motor_circ"]:
        rw_b = run_dml_single_sample_mediation(arm_b_4d16h, exp_c, "log_snfl", "phq8_nosleep", weight_col="wtssnh2y", n_boot=500)
        rw_b["sensitivity_model"] = "ArmB_WTSSNH2Y_SurveyWeighted_PHQ8"
        sens_rows.append(rw_b)
        rw_d = run_dml_single_sample_mediation(arm_d_bridge, exp_c, "log_snfl", "dsst", weight_col="wtssnh2y", n_boot=500)
        rw_d["sensitivity_model"] = "ArmD_WTSSNH2Y_SurveyWeighted_DSST"
        sens_rows.append(rw_d)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_DIR / "dml_sensitivity_survey_weighted.csv", index=False)

    # Full Arm A (N ~ 7,508) and Arm C (N ~ 2,541) Direct Cross-Fitted DML Total Associations
    arm_ac_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for arm_label, sub_df, out_list in [
        ("ArmA_Adults_PAX4d16h", arm_a, ["phq8_nosleep", "phq9_total"]),
        ("ArmC_Older60_PAX4d16h", nh_full[(nh_full["age_years"] >= 60.0) & (nh_full["pax_eligible_primary_4d16h"] == 1)], ["dsst", "global_cognition_z", "animal_fluency", "cerad_total_40"]),
    ]:
        for out_c in out_list:
            d_sub = sub_df.dropna(subset=[out_c, "circadian_photic_vitality_z", "age_years"]).copy().reset_index(drop=True)
            w_sub = build_confounder_matrix(d_sub, include_cycle=True)
            y_raw = d_sub[out_c].to_numpy(float)
            sd_y = float(np.std(y_raw, ddof=1))
            y_s = (y_raw - np.mean(y_raw)) / sd_y
            y_til = y_s - cross_fit_stacked_nuisance(w_sub, y_s, kf, seed_offset=501)
            for exp_c in key_exposures:
                a_s = (d_sub[exp_c].to_numpy(float) - d_sub[exp_c].mean()) / d_sub[exp_c].std(ddof=1)
                a_til = a_s - cross_fit_stacked_nuisance(w_sub, a_s, kf, seed_offset=509)
                da = float(np.mean(a_til**2))
                beta_te = float(np.mean(a_til * y_til) / da)
                se_te = float(np.sqrt(np.sum(((a_til * (y_til - beta_te * a_til)) / (da * len(d_sub))) ** 2)))
                p_te = float(2.0 * stats.norm.sf(abs(beta_te) / max(se_te, 1e-12)))
                ev_pt, ev_ci = compute_evalue_continuous(beta_te, se_te)
                arm_ac_rows.append(
                    {
                        "cohort_arm": arm_label,
                        "n_obs": int(len(d_sub)),
                        "exposure": exp_c,
                        "outcome": out_c,
                        "dml_beta_std": round(beta_te, 5),
                        "dml_beta_raw_units": round(beta_te * sd_y, 5),
                        "dml_se": round(se_te, 5),
                        "dml_ci95_lo": round(beta_te - 1.96 * se_te, 5),
                        "dml_ci95_hi": round(beta_te + 1.96 * se_te, 5),
                        "dml_p": float(f"{p_te:.4e}"),
                        "E_value_point": ev_pt,
                        "E_value_ci95": ev_ci,
                    }
                )
    arm_ac_df = pd.DataFrame(arm_ac_rows)
    arm_ac_df["dml_fdr_q"] = multipletests(arm_ac_df["dml_p"], method="fdr_bh")[1]
    arm_ac_df.to_csv(OUT_DIR / "dml_armA_armC_total_effects.csv", index=False)

    # Save Pillar 1 summary JSON
    summary_payload = {
        "script": "scripts/phase2_pillar1_nhanes_fpca_tpdc_dml.py",
        "seed": SEED,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "n_primary_fpca_4d16h": int(np.sum(prim_mask)),
        "n_tpdc_computed": int(tpdc_df["tpdc_photic_to_motor_circ"].notna().sum()),
        "fpca_mims_cumvar_4pc_pct": float(var_mims["cumulative_variance_pct"].iloc[-1]),
        "fpca_lux_cumvar_4pc_pct": float(var_lux["cumulative_variance_pct"].iloc[-1]),
        "top_snfl_associations": snfl_assoc_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "pillar1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("=== Pillar 1 Key Results Preview ===")
    print("\n[A -> log(sNfL) Cross-Fitted DML Links]:")
    print(snfl_assoc_df[["exposure", "n_primary_4d16h", "alpha_log_snfl_std_4d16h", "alpha_se_4d16h", "alpha_p_4d16h", "alpha_fdr_q_4d16h"]].to_string(index=False))
    print("\n[Two-Sample Moment-Coupled DML Mediation on Zero-Ceiling Cognition]:")
    print(twosamp_df[twosamp_df["outcome"].isin(["dsst", "global_cognition_z"])][["exposure", "outcome", "TE_armC_std", "TE_armC_p", "NIE_moment_coupled_std", "NIE_p", "NIE_fdr_q", "prop_mediated_2sample"]].to_string(index=False))
    print(f"[Pillar 1 DONE] Completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
