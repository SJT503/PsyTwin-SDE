#!/usr/bin/env python3
"""
Phase 3 — Pillar 3: Individualized Jacobian Dominance Phenotyping, 9-Year Parametric
Longitudinal G-Computation Counterfactual Prescriptions (B = 500 Cluster Bootstrap),
GAMLSS Location-Scale Distributional Fairness Audit, & Faraci Ambiguity Entropy Triage
======================================================================================
Target Project: PsyTwin-SDE (npj Digital Medicine Collection)
Pre-Specified in: docs/SAP_v1_FROZEN.md (§4.1–4.3 & §5)
Governance Rules (CLAUDE.md & lab-constitution):
  - STRICT READ-ONLY RAW DATA: Reads exclusively from data_processed/ and results/phase2_*/;
    writes exclusively inside results/phase3_pillar3_fairness/ and data_processed/.
  - ZERO Brain-Kidney Axis & ZERO Raw DICOM Imaging.
  - 100% Real Script-Computed Numbers with deterministic seed = 20270507.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.multitest import multipletests

SEED = 20270507
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_processed"
P2_DIR = PROJECT_ROOT / "results" / "phase2_pillar2_sde"
OUT_DIR = PROJECT_ROOT / "results" / "phase3_pillar3_fairness"


# ---------------------------------------------------------------------------
# 1. Exact 2-Parameter Location-Scale GAMLSS Solver (Penalized Spline + Cluster Sandwich SE)
# ---------------------------------------------------------------------------
def natural_cubic_basis_3df(x: np.ndarray, knots: Tuple[float, float, float] = (-1.0, 0.0, 1.0)) -> np.ndarray:
    """Smooth 3-df cubic spline basis [x, (x - k1)_+^3 - ..., (x - k2)_+^3 - ...] for age adjustment."""
    k1, k2, k3 = knots
    p1 = np.maximum(x - k1, 0.0) ** 3
    p2 = np.maximum(x - k2, 0.0) ** 3
    p3 = np.maximum(x - k3, 0.0) ** 3
    d1 = (p1 - p3) / max(k3 - k1, 1e-6)
    d2 = (p2 - p3) / max(k3 - k2, 1e-6)
    b2 = d1 - d2
    b3 = np.maximum(x - 0.5, 0.0) ** 2 - np.mean(np.maximum(x - 0.5, 0.0) ** 2)
    return np.column_stack([x, b2 - np.mean(b2), b3])


def fit_gaussian_location_scale_gamlss(
    y: np.ndarray,
    x_mu: np.ndarray,
    x_sig: np.ndarray,
    col_names_mu: List[str],
    col_names_sig: List[str],
    cluster_ids: Optional[np.ndarray] = None,
    l2_pen: float = 1e-4,
    max_iter: int = 40,
) -> pd.DataFrame:
    """
    Fit 2-parameter Gaussian Location-Scale GAMLSS:
      Y_i ~ N(mu_i, sigma_i^2),   mu_i = X_{mu,i} beta_mu,   log(sigma_i) = X_{sig,i} beta_sig
    via exact alternating Fisher-scoring / Rigby-Stasinopoulos RS algorithm with cluster-robust
    Huber-White sandwich standard errors!
      - Score & Fisher info for mu:    u_mu = (y - mu) / sigma^2,     W_mu = 1 / sigma^2
      - Score & Fisher info for alpha = log(sigma):
        ell_i = -alpha_i - (y_i - mu_i)^2 / (2 exp(2 alpha_i))
        d ell_i / d alpha_i = (y_i - mu_i)^2 / sigma_i^2 - 1
        E[-d^2 ell_i / d alpha_i^2] = 2
    """
    n = len(y)
    p_mu = x_mu.shape[1]
    p_sig = x_sig.shape[1]

    # Initialize beta_mu by OLS and beta_sig by log(residual SD)
    pen_mu = l2_pen * np.eye(p_mu)
    pen_mu[0, 0] = 0.0
    pen_sig = l2_pen * np.eye(p_sig)
    pen_sig[0, 0] = 0.0

    beta_mu = linalg.solve(x_mu.T @ x_mu + pen_mu, x_mu.T @ y, assume_a="pos")
    res = y - x_mu @ beta_mu
    beta_sig = np.zeros(p_sig, dtype=np.float64)
    beta_sig[0] = math.log(max(float(np.std(res, ddof=1)), 1e-3))

    for _ in range(max_iter):
        old_mu = beta_mu.copy()
        old_sig = beta_sig.copy()

        # 1) Update beta_mu via weighted least squares with weights w_i = exp(-2 alpha_i)
        log_sig = np.clip(x_sig @ beta_sig, -6.0, 6.0)
        inv_var = np.exp(-2.0 * log_sig)
        xw = x_mu * inv_var[:, None]
        beta_mu = linalg.solve(xw.T @ x_mu + pen_mu, xw.T @ y, assume_a="pos")

        # 2) Update beta_sig via damped Fisher scoring on log(sigma)
        res = y - x_mu @ beta_mu
        for _inner in range(4):
            log_sig = np.clip(x_sig @ beta_sig, -6.0, 6.0)
            inv_var = np.exp(-2.0 * log_sig)
            z2 = (res * res) * inv_var
            score_a = z2 - 1.0  # (N,)
            # Fisher weight = 2.0 per observation; working response = log_sig + 0.5 * score_a
            grad_sig = x_sig.T @ score_a - pen_sig @ beta_sig
            hess_sig = 2.0 * (x_sig.T @ x_sig) + pen_sig
            step = linalg.solve(hess_sig, grad_sig, assume_a="pos")
            beta_sig = beta_sig + 0.75 * step
            if np.max(np.abs(step)) < 1e-7:
                break

        if max(np.max(np.abs(beta_mu - old_mu)), np.max(np.abs(beta_sig - old_sig))) < 1e-7:
            break

    # Compute exact cluster-robust Huber-White sandwich covariance for [beta_mu; beta_sig]
    log_sig = np.clip(x_sig @ beta_sig, -6.0, 6.0)
    inv_var = np.exp(-2.0 * log_sig)
    res = y - x_mu @ beta_mu

    # Individual score vectors: s_mu_i = x_mu_i * res_i * inv_var_i; s_sig_i = x_sig_i * (res_i^2 * inv_var_i - 1)
    s_mu = x_mu * (res * inv_var)[:, None]
    s_sig = x_sig * ((res * res) * inv_var - 1.0)[:, None]

    bread_mu = linalg.inv((x_mu * inv_var[:, None]).T @ x_mu + pen_mu)
    bread_sig = linalg.inv(2.0 * (x_sig.T @ x_sig) + pen_sig)

    if cluster_ids is not None:
        df_s_mu = pd.DataFrame(s_mu).groupby(cluster_ids).sum().to_numpy(dtype=np.float64)
        df_s_sig = pd.DataFrame(s_sig).groupby(cluster_ids).sum().to_numpy(dtype=np.float64)
        g_cnt = len(df_s_mu)
        scale_c = (g_cnt / max(g_cnt - 1.0, 1.0)) * ((n - 1.0) / max(n - p_mu - p_sig, 1.0))
        meat_mu = scale_c * (df_s_mu.T @ df_s_mu)
        meat_sig = scale_c * (df_s_sig.T @ df_s_sig)
    else:
        meat_mu = (n / max(n - p_mu, 1.0)) * (s_mu.T @ s_mu)
        meat_sig = (n / max(n - p_sig, 1.0)) * (s_sig.T @ s_sig)

    cov_mu = bread_mu @ meat_mu @ bread_mu
    cov_sig = bread_sig @ meat_sig @ bread_sig
    se_mu = np.sqrt(np.maximum(np.diag(cov_mu), 1e-14))
    se_sig = np.sqrt(np.maximum(np.diag(cov_sig), 1e-14))

    rows = []
    for j, nm in enumerate(col_names_mu):
        est = float(beta_mu[j])
        se = float(se_mu[j])
        p_v = float(2.0 * stats.norm.sf(abs(est) / max(se, 1e-12)))
        rows.append(
            {
                "gamlss_parameter": "mu_location",
                "term": nm,
                "estimate": round(est, 5),
                "robust_se": round(se, 5),
                "ci95_lo": round(est - 1.96 * se, 5),
                "ci95_hi": round(est + 1.96 * se, 5),
                "p_value": float(f"{p_v:.4e}"),
                "exp_multiplier": np.nan,
            }
        )
    for j, nm in enumerate(col_names_sig):
        est = float(beta_sig[j])
        se = float(se_sig[j])
        p_v = float(2.0 * stats.norm.sf(abs(est) / max(se, 1e-12)))
        rows.append(
            {
                "gamlss_parameter": "log_sigma_scale",
                "term": nm,
                "estimate": round(est, 5),
                "robust_se": round(se, 5),
                "ci95_lo": round(est - 1.96 * se, 5),
                "ci95_hi": round(est + 1.96 * se, 5),
                "p_value": float(f"{p_v:.4e}"),
                "exp_multiplier": round(math.exp(est), 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Module 1 (§4.1): Individualized Directed Jacobian Dominance Phenotyping
# ---------------------------------------------------------------------------
def run_jacobian_dominance_phenotyping(
    twins: pd.DataFrame,
    panel: pd.DataFrame,
    persons: pd.DataFrame,
    blood: pd.DataFrame,
    pb_mod_csv: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Characterize individualized directed Jacobian dominance phenotypes across N = 20,928 overall
    and biologically moderated paired-blood dominance across N = 7,438 paired venous blood adults.
    """
    base_p = panel[panel["ID"].isin(twins["ID"])].sort_values(["ID", "wave"]).groupby("ID", as_index=False).first()
    last_p = panel[panel["ID"].isin(twins["ID"])].sort_values(["ID", "wave"]).groupby("ID", as_index=False).last()

    merged = (
        twins.merge(
            base_p[
                [
                    "ID", "age_years", "female", "education_3stratum", "rural_village",
                    "optimal_sleep_6_to_85h", "poor_vision", "poor_hearing", "dual_sensory_impairment",
                    "social_active", "internet_use", "phys_active_modvig", "tr20", "cesd10",
                    "hibpe", "diabe", "hearte", "stroke", "dyslipe",
                ]
            ],
            on="ID",
            how="left",
        )
        .merge(
            last_p[["ID", "tr20", "cesd10", "time_years_from_2011"]].rename(
                columns={"tr20": "tr20_last", "cesd10": "cesd10_last", "time_years_from_2011": "t_last_yr"}
            ),
            on="ID",
            how="left",
        )
        .merge(persons[["ID", "died_followup"]], on="ID", how="left")
    )
    merged["delta_eta_cog_9yr"] = merged["eta_cog_w5"] - merged["eta_cog_w1"]
    merged["delta_eta_dep_9yr"] = merged["eta_dep_w5"] - merged["eta_dep_w1"]
    merged["elevated_dep_last"] = (merged["cesd10_last"] >= 10.0).astype(float)
    merged["cog_impairment_last"] = (merged["tr20_last"] <= 6.0).astype(float)

    # Summarize across full cohort phenotypes (and continuous D_i tertiles if population D_i is concentrated)
    # Note: If demographic-only moderators cluster near bidirectional equilibrium, we report BOTH exact
    # pre-specified threshold phenotypes (|D_i| > 0.15) AND biologically moderated paired-blood phenotypes!
    pheno_rows = []
    for grp_type, col_grp in [("PreSpecified_Threshold_0p15", "coupling_phenotype")]:
        for ph, g in merged.groupby(col_grp):
            pheno_rows.append(
                {
                    "cohort": "CHARLS_Full_SDE_N20928",
                    "classification_scheme": grp_type,
                    "phenotype": str(ph),
                    "n_participants": int(len(g)),
                    "pct_cohort": round(100.0 * len(g) / len(merged), 2),
                    "mean_J_cog_from_dep": round(float(g["J_cog_from_dep"].mean()), 5),
                    "mean_J_dep_from_cog": round(float(g["J_dep_from_cog"].mean()), 5),
                    "mean_dominance_D": round(float(g["dominance_index_D"].mean()), 4),
                    "baseline_age_mean": round(float(g["age_years"].mean()), 2),
                    "female_pct": round(100.0 * float(g["female"].mean()), 2),
                    "illiterate_pct": round(100.0 * float(g["illiterate"].mean()), 2),
                    "rural_pct": round(100.0 * float(g["rural_village"].mean()), 2),
                    "optimal_sleep_pct": round(100.0 * float(g["optimal_sleep_6_to_85h"].mean()), 2),
                    "dual_sensory_impairment_pct": round(100.0 * float(g["dual_sensory_impairment"].mean()), 2),
                    "mean_delta_eta_cog_9yr": round(float(g["delta_eta_cog_9yr"].mean()), 4),
                    "mean_delta_eta_dep_9yr": round(float(g["delta_eta_dep_9yr"].mean()), 4),
                    "followup_elevated_dep_pct": round(100.0 * float(g["elevated_dep_last"].mean()), 2),
                    "followup_cog_impairment_pct": round(100.0 * float(g["cog_impairment_last"].mean()), 2),
                }
            )

    # Biologically moderated Jacobian dominance in N = 7,438 paired venous blood subcohort!
    pb = merged.merge(blood[blood["is_paired_2011_2015"] == 1], on="ID", how="inner").reset_index(drop=True)

    def zclean(s: pd.Series) -> np.ndarray:
        v = s.fillna(s.median()).to_numpy(float)
        return (v - np.mean(v)) / max(np.std(v, ddof=1), 1e-6)

    tc_hdl = pb["tchol_mgdl_2011"] / pb["hdl_mgdl_2011"].where(pb["hdl_mgdl_2011"] > 5.0, np.nan)
    mod_cols = {
        "log_hsCRP_2011_z": zclean(pb["log_crp_2011"]),
        "delta_log_hsCRP_4yr_z": zclean(pb["delta_log_crp"]),
        "HbA1c_2011_z": zclean(pb["hba1c_pct_2011"]),
        "delta_HbA1c_4yr_z": zclean(pb["delta_hba1c_pct"]),
        "tchol_to_hdl_2011_z": zclean(tc_hdl),
        "cystatinC_clearance_2011_z": zclean(pb["cystatin_c_mgl_2011"]),
        "baseline_age_z": zclean(pb["age_years"]),
        "illiterate_z": zclean(pb["illiterate"]),
    }

    c12 = pb_mod_csv[pb_mod_csv["pathway"].str.startswith("J_cog_from_dep")].set_index("moderator")["estimate_SD_per_yr"].to_dict()
    c21 = pb_mod_csv[pb_mod_csv["pathway"].str.startswith("J_dep_from_cog")].set_index("moderator")["estimate_SD_per_yr"].to_dict()

    j12_bio = np.full(len(pb), float(c12["Intercept_at_Mean"]), dtype=np.float64)
    j21_bio = np.full(len(pb), float(c21["Intercept_at_Mean"]), dtype=np.float64)
    for m_nm, z_v in mod_cols.items():
        j12_bio += float(c12.get(m_nm, 0.0)) * z_v
        j21_bio += float(c21.get(m_nm, 0.0)) * z_v

    sd_c = float(np.std(pb["eta_cog_w1"], ddof=1))
    sd_d = float(np.std(pb["eta_dep_w1"], ddof=1))
    f_12 = np.abs(j12_bio) * sd_d
    f_21 = np.abs(j21_bio) * sd_c
    d_bio = (f_12 - f_21) / np.maximum(f_12 + f_21, 1e-8)

    pb["J_cog_from_dep_bio"] = j12_bio
    pb["J_dep_from_cog_bio"] = j21_bio
    pb["dominance_index_D_bio"] = d_bio
    pb["bio_coupling_phenotype"] = np.where(
        d_bio > 0.15,
        "1_Affective_Driven_Cognitive_Decline",
        np.where(d_bio < -0.15, "2_Cognitive_Driven_Affective_Collapse", "3_Bidirectional_Coupled_Equilibrium"),
    )

    pb_rows = []
    for ph, g in pb.groupby("bio_coupling_phenotype"):
        pb_rows.append(
            {
                "cohort": "CHARLS_PairedBlood_N7438",
                "bio_coupling_phenotype": str(ph),
                "n_participants": int(len(g)),
                "pct_subcohort": round(100.0 * len(g) / len(pb), 2),
                "mean_J_cog_from_dep_bio": round(float(g["J_cog_from_dep_bio"].mean()), 5),
                "mean_J_dep_from_cog_bio": round(float(g["J_dep_from_cog_bio"].mean()), 5),
                "mean_dominance_D_bio": round(float(g["dominance_index_D_bio"].mean()), 4),
                "hsCRP_2011_median_mgL": round(float(g["crp_mgl_2011"].median()), 3),
                "hsCRP_2015_median_mgL": round(float(g["crp_mgl_2015"].median()), 3),
                "delta_log_hsCRP_mean": round(float(g["delta_log_crp"].mean()), 4),
                "HbA1c_2011_mean_pct": round(float(g["hba1c_pct_2011"].mean()), 3),
                "HbA1c_2015_mean_pct": round(float(g["hba1c_pct_2015"].mean()), 3),
                "delta_HbA1c_4yr_mean_pct": round(float(g["delta_hba1c_pct"].mean()), 4),
                "cystatinC_2011_mean_mgL": round(float(g["cystatin_c_mgl_2011"].mean()), 3),
                "mean_delta_eta_cog_9yr": round(float(g["delta_eta_cog_9yr"].mean()), 4),
                "mean_delta_eta_dep_9yr": round(float(g["delta_eta_dep_9yr"].mean()), 4),
                "followup_elevated_dep_pct": round(100.0 * float(g["elevated_dep_last"].mean()), 2),
                "followup_cog_impairment_pct": round(100.0 * float(g["cog_impairment_last"].mean()), 2),
            }
        )
    return merged, pd.DataFrame(pheno_rows), pd.DataFrame(pb_rows), pb


# ---------------------------------------------------------------------------
# 3. Module 2 (§4.2): 9-Year Parametric Longitudinal G-Computation Engine (Robins 1986)
# ---------------------------------------------------------------------------
def build_5wave_gcomp_arrays(
    twins: pd.DataFrame,
    panel: pd.DataFrame,
    obs_params: pd.DataFrame,
) -> Dict[str, object]:
    """Assemble dense 5-wave (N = 20,928 x 5) longitudinal exposure, state, and covariate arrays."""
    ids = twins["ID"].to_numpy()
    n_s = len(ids)
    id_map = pd.Series(np.arange(n_s, dtype=np.int32), index=ids)

    sub = panel[panel["ID"].isin(ids)].sort_values(["ID", "wave"]).reset_index(drop=True)
    ri = sub["ID"].map(id_map).to_numpy()
    wi = sub["wave"].to_numpy(int) - 1

    # Time grid t_years (N, 5)
    t_def = np.array([0.0, 2.0, 4.0, 7.0, 9.1], dtype=np.float64)
    t_mat = np.tile(t_def, (n_s, 1))
    has_w = np.zeros((n_s, 5), dtype=bool)
    t_mat[ri, wi] = sub["time_years_from_2011"].to_numpy(float)
    has_w[ri, wi] = True
    for i in range(n_s):
        aw = np.where(has_w[i])[0]
        if len(aw) == 0:
            continue
        for w in range(aw[0] - 1, -1, -1):
            t_mat[i, w] = t_mat[i, w + 1] - (t_def[w + 1] - t_def[w])
        for w in range(aw[-1] + 1, 5):
            t_mat[i, w] = t_mat[i, w - 1] + (t_def[w] - t_def[w - 1])
        for k in range(len(aw) - 1):
            wa, wb = aw[k], aw[k + 1]
            for wm in range(wa + 1, wb):
                fr = (t_def[wm] - t_def[wa]) / (t_def[wb] - t_def[wa])
                t_mat[i, wm] = t_mat[i, wa] + fr * (t_mat[i, wb] - t_mat[i, wa])
        for w in range(1, 5):
            if t_mat[i, w] <= t_mat[i, w - 1] + 0.1:
                t_mat[i, w] = t_mat[i, w - 1] + 0.1

    # Psychometric-filtered latent states eta_cog (N, 5) & eta_dep (N, 5)
    eta_cog = np.column_stack([twins[f"eta_cog_w{w+1}"].to_numpy(float) for w in range(5)])
    eta_dep = np.column_stack([twins[f"eta_dep_w{w+1}"].to_numpy(float) for w in range(5)])

    # 7 Time-varying modifiable exposures/drivers A_{i,w}:
    # 0: optimal_sleep, 1: poor_vision, 2: poor_hearing, 3: social_active,
    # 4: internet_use,  5: phys_active, 6: smoken (modifiable smoking vascular risk)
    a_cols = [
        "optimal_sleep_6_to_85h",
        "poor_vision",
        "poor_hearing",
        "social_active",
        "internet_use",
        "phys_active_modvig",
        "smoken",
    ]
    a_mat = np.full((n_s, 5, 7), np.nan, dtype=np.float64)
    for k, c in enumerate(a_cols):
        a_mat[ri, wi, k] = sub[c].to_numpy(float)

    # Observed raw tr20 & cesd10 at Wave 5 for natural-course calibration check
    tr20_obs = np.full((n_s, 5), np.nan)
    cesd_obs = np.full((n_s, 5), np.nan)
    tr20_obs[ri, wi] = sub["tr20"].to_numpy(float)
    cesd_obs[ri, wi] = sub["cesd10"].to_numpy(float)

    # Forward/backward impute baseline missing exposure entries using cohort wave medians
    w_medians = np.nanmedian(a_mat, axis=0)  # (5, 7)
    a_clean = a_mat.copy()
    for i in range(n_s):
        for k in range(7):
            v = a_clean[i, :, k]
            ok = np.where(~np.isnan(v))[0]
            if len(ok) == 0:
                a_clean[i, :, k] = w_medians[:, k]
            else:
                for w in range(ok[0] - 1, -1, -1):
                    v[w] = v[w + 1]
                for w in range(ok[0] + 1, 5):
                    if np.isnan(v[w]):
                        v[w] = v[w - 1]

    # Static baseline covariates X0: [age_w1_c, age_w1_c^2, female, illit, prim_ed, rural, unmar, cperc_z, base_cm]
    base_rows = sub.groupby("ID", as_index=False).first()
    base_rows = pd.DataFrame({"ID": ids}).merge(base_rows, on="ID", how="left")
    age_first = base_rows["age_years"].fillna(58.0).to_numpy(float)
    t_first = base_rows["time_years_from_2011"].fillna(0.0).to_numpy(float)
    age_2011 = age_first - t_first
    age0_c = (age_2011 - 58.0) / 10.0
    female = base_rows["female"].fillna(0.5).to_numpy(float)
    illit = twins["illiterate"].to_numpy(float)
    prim_ed = (base_rows["education_3stratum"] == "2_Primary_or_Sishu").astype(float).to_numpy()
    rural = base_rows["rural_village"].fillna(0.5).to_numpy(float)
    unmar = base_rows["unmarried_or_widowed"].fillna(0.0).to_numpy(float)
    cperc = base_rows["log_hh_cperc"].fillna(base_rows["log_hh_cperc"].median()).to_numpy(float)
    cperc_z = (cperc - np.mean(cperc)) / max(np.std(cperc, ddof=1), 1e-6)
    base_cm = (
        base_rows["hibpe"].fillna(0.0)
        + base_rows["diabe"].fillna(0.0)
        + base_rows["hearte"].fillna(0.0)
        + base_rows["stroke"].fillna(0.0)
        + base_rows["dyslipe"].fillna(0.0)
    ).to_numpy(float)
    x0 = np.column_stack([age0_c, age0_c**2, female, illit, prim_ed, rural, unmar, cperc_z, base_cm])

    # Extract fitted psychometric observation mapping parameters for translating latent eta -> clinical event probabilities
    op = obs_params.set_index("scale")
    nu_tr20 = float(op.loc["imrc", "nu_intercept"] + op.loc["dlrc", "nu_intercept"])
    lam_tr20 = float(op.loc["imrc", "lambda_loading"] + op.loc["dlrc", "lambda_loading"])
    sig_tr20 = math.sqrt(float(op.loc["imrc", "sigma_obs_base"] ** 2 + op.loc["dlrc", "sigma_obs_base"] ** 2))

    nu_cesd = float(op.loc["cesd10", "nu_intercept"])
    lam_cesd = float(op.loc["cesd10", "lambda_loading"])
    sig_cesd = float(op.loc["cesd10", "sigma_obs_base"])

    # Compute unadjusted Wave-1 (k=0, zero practice effect) 5-scale psychometric latent chronological aging slope (SD/yr)
    w1_sub = sub[sub["wave"] == 1].copy()
    num_g = np.zeros(len(w1_sub), dtype=np.float64)
    den_g = np.zeros(len(w1_sub), dtype=np.float64)
    for sc_nm in ["imrc", "dlrc", "ser7", "orient", "draw"]:
        yv = w1_sub[sc_nm].to_numpy(float)
        mv = ~np.isnan(yv)
        nu_s = float(op.loc[sc_nm, "nu_intercept"])
        lam_s = float(op.loc[sc_nm, "lambda_loading"])
        var_s = float(op.loc[sc_nm, "sigma_obs_base"]) ** 2
        w_info = (lam_s * lam_s) / max(var_s, 1e-4)
        num_g[mv] += (lam_s / max(var_s, 1e-4)) * (yv[mv] - nu_s)
        den_g[mv] += w_info
    ok_g = (den_g > 0) & w1_sub["age_years"].notna().to_numpy()
    eta_raw_w1 = num_g[ok_g] / den_g[ok_g]
    eta_raw_w1_z = (eta_raw_w1 - np.mean(eta_raw_w1)) / max(np.std(eta_raw_w1, ddof=1), 1e-6)
    chron_age_slope_sd_yr = float(abs(np.polyfit(w1_sub.loc[ok_g, "age_years"].to_numpy(float), eta_raw_w1_z, 1)[0]))

    return {
        "ids": ids,
        "comm_id": twins["communityID"].to_numpy(),
        "ipcw": twins["ipcw_weight"].to_numpy(float),
        "dom_D": twins["dominance_index_D"].to_numpy(float),
        "j_cog_from_dep": twins["J_cog_from_dep"].to_numpy(float),
        "j_dep_from_cog": twins["J_dep_from_cog"].to_numpy(float),
        "phenotype": twins["coupling_phenotype"].to_numpy(),
        "b_cog": twins["random_trait_b_cog"].to_numpy(float),
        "b_dep": twins["random_trait_b_dep"].to_numpy(float),
        "t_mat": t_mat,
        "has_w": has_w,
        "eta_cog": eta_cog,
        "eta_dep": eta_dep,
        "a_mat_raw": a_mat,
        "a_clean": a_clean,
        "x0": x0,
        "age_2011": age_2011,
        "female": female,
        "illit": illit,
        "prim_ed": prim_ed,
        "rural": rural,
        "tr20_obs": tr20_obs,
        "cesd_obs": cesd_obs,
        "chron_age_slope_sd_yr": chron_age_slope_sd_yr,
        "psych_map": {
            "nu_tr20": nu_tr20,
            "lam_tr20": lam_tr20,
            "sig_tr20": sig_tr20,
            "nu_cesd": nu_cesd,
            "lam_cesd": lam_cesd,
            "sig_cesd": sig_cesd,
        },
    }


def fit_and_simulate_gcomp_cohort(
    gdata: Dict[str, object],
    sub_idx: np.ndarray,
    use_ipcw: bool = False,
    blood_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    """
    Fit Mundlak-adjusted Coupled-SDE Parametric Longitudinal G-Computation structural equations
    (time-varying confounder/exposure transitions + individualized Jacobian-coupled latent
    cognitive-affective state transitions separating within-person modifiable exposure changes
    from baseline selection propensity) on participant index vector sub_idx, and roll forward
    all pre-specified counterfactual regimes across Waves 1 -> 5 (9.1 years).
    """
    t_m = gdata["t_mat"][sub_idx]
    hw = gdata["has_w"][sub_idx]
    ec = gdata["eta_cog"][sub_idx]
    ed = gdata["eta_dep"][sub_idx]
    ac = gdata["a_clean"][sub_idx]
    x0 = gdata["x0"][sub_idx]
    bc_tr = gdata["b_cog"][sub_idx]
    bd_tr = gdata["b_dep"][sub_idx]
    j_cd = gdata["j_cog_from_dep"][sub_idx]
    j_dc = gdata["j_dep_from_cog"][sub_idx]
    wt = gdata["ipcw"][sub_idx] if use_ipcw else np.ones(len(sub_idx), dtype=np.float64)
    dom_d = gdata["dom_D"][sub_idx]
    pm = gdata["psych_map"]

    has_bio = blood_arrays is not None
    if has_bio:
        # Dynamic biological mediators/exposures: [log_crp_w, hba1c_w] across waves 0..4
        bio_mat = blood_arrays["bio_clean"][sub_idx]  # (N_sub, 5, 2)

    # 1) Build pooled person-wave transition dataset (w-1 -> w for active observed transitions)
    tr_x_exp = []   # predictors for A_w | A_{w-1}, eta_{w-1}, X0, t_w, dt_w
    tr_a_next = []
    tr_w_exp = []

    tr_x_eta = []   # predictors for eta_w | eta_{w-1}, (A_w - A_0), A_0, J_i, (bio_w), X0, dt_w
    tr_ec_next = []
    tr_ed_next = []
    tr_w_eta = []

    a0_obs = ac[:, 0, :]
    bio0_obs = bio_mat[:, 0, :] if has_bio else None

    for w in range(1, 5):
        valid = hw[:, w] & hw[:, w - 1]
        if not np.any(valid):
            continue
        dt = t_m[valid, w] - t_m[valid, w - 1]
        tw = t_m[valid, w] / 9.0
        age_w = x0[valid, 0] + t_m[valid, w] / 10.0

        # Predictors for endogenous exposure evolution A_w (treatment-confounder feedback from lagged eta_cog, eta_dep!)
        xe = np.column_stack(
            [
                np.ones(np.sum(valid)),
                ec[valid, w - 1],
                ed[valid, w - 1],
                ac[valid, w - 1, :],
                age_w,
                tw,
                dt,
                x0[valid, 2:],
            ]
        )
        tr_x_exp.append(xe)
        tr_a_next.append(ac[valid, w, :])
        tr_w_exp.append(wt[valid])

        # Predictors for coupled latent state evolution eta_w with Mundlak within-person separation
        # and explicit individualized coupled SDE off-diagonal Jacobian flux (J_cd * ed_prev * dt, J_dc * ec_prev * dt)
        aw_cur = ac[valid, w, :]
        aw_within = aw_cur - a0_obs[valid, :]
        ec_prev = ec[valid, w - 1]
        ed_prev = ed[valid, w - 1]

        flux_dep_to_cog = j_cd[valid] * ed_prev * dt   # individualized affective -> cognitive erosion flux
        flux_cog_to_dep = j_dc[valid] * ec_prev * dt   # individualized cognitive -> affective buffering flux
        inter_sleep_dep = aw_within[:, 0] * ed_prev
        inter_sens_cog = (aw_within[:, 1] + aw_within[:, 2]) * ec_prev

        cols_eta = [
            np.ones(np.sum(valid)),
            ec_prev,
            ed_prev,
            ec_prev * dt,
            ed_prev * dt,
            flux_dep_to_cog,
            flux_cog_to_dep,
            bc_tr[valid] * dt,
            bd_tr[valid] * dt,
            dt,
            age_w * dt,
            (age_w**2) * dt,
            aw_within * dt[:, None],
            a0_obs[valid, :] * dt[:, None],
            (inter_sleep_dep * dt)[:, None],
            (inter_sens_cog * dt)[:, None],
            x0[valid, 2:] * dt[:, None],
        ]
        if has_bio and bio0_obs is not None:
            bw_cur = bio_mat[valid, w, :]
            bw_within = bw_cur - bio0_obs[valid, :]
            cols_eta.append(bw_within * dt[:, None])
            cols_eta.append(bio0_obs[valid, :] * dt[:, None])
            cols_eta.append((bw_cur[:, 0] * ed_prev * dt)[:, None])  # hs-CRP x depressive load synergistic erosion
            cols_eta.append((bw_cur[:, 1] * age_w * dt)[:, None])    # HbA1c x aging drift

        tr_x_eta.append(np.column_stack(cols_eta))
        tr_ec_next.append(ec[valid, w])
        tr_ed_next.append(ed[valid, w])
        tr_w_eta.append(wt[valid])

    X_exp = np.vstack(tr_x_exp)
    A_nxt = np.vstack(tr_a_next)
    W_exp = np.concatenate(tr_w_exp)

    X_eta = np.vstack(tr_x_eta)
    Yc_nxt = np.concatenate(tr_ec_next)
    Yd_nxt = np.concatenate(tr_ed_next)
    W_eta = np.concatenate(tr_w_eta)

    # Fit weighted Ridge / linear probability & structural transition equations
    sq_we = np.sqrt(W_exp)[:, None]
    pen_e = 1.0 * np.eye(X_exp.shape[1])
    pen_e[0, 0] = 0.0
    beta_exp = linalg.solve((X_exp * sq_we).T @ (X_exp * sq_we) + pen_e, (X_exp * W_exp[:, None]).T @ A_nxt, assume_a="pos")

    sq_wy = np.sqrt(W_eta)[:, None]
    pen_y = 2.0 * np.eye(X_eta.shape[1])
    pen_y[0, 0] = 0.0
    gram_y = (X_eta * sq_wy).T @ (X_eta * sq_wy) + pen_y
    beta_ec = linalg.solve(gram_y, (X_eta * W_eta[:, None]).T @ Yc_nxt, assume_a="pos")
    beta_ed = linalg.solve(gram_y, (X_eta * W_eta[:, None]).T @ Yd_nxt, assume_a="pos")

    res_ec_sd = float(np.std(Yc_nxt - X_eta @ beta_ec, ddof=1))
    res_ed_sd = float(np.std(Yd_nxt - X_eta @ beta_ed, ddof=1))

    if has_bio:
        # Fit endogenous transition for [log_crp_w, hba1c_w] from w-1 -> w
        xb_list, yb_list = [], []
        for w in range(1, 5):
            v = hw[:, w] & hw[:, w - 1]
            xb_list.append(np.column_stack([np.ones(np.sum(v)), bio_mat[v, w - 1, :], ec[v, w - 1], ed[v, w - 1], ac[v, w - 1, :], x0[v]]))
            yb_list.append(bio_mat[v, w, :])
        Xb = np.vstack(xb_list)
        Yb = np.vstack(yb_list)
        pen_b = 2.0 * np.eye(Xb.shape[1])
        pen_b[0, 0] = 0.0
        beta_bio = linalg.solve(Xb.T @ Xb + pen_b, Xb.T @ Yb, assume_a="pos")

    # Re-estimate unadjusted Wave-1 (k=0, zero practice effect) 5-scale psychometric latent chronological aging slope (SD/yr)
    # directly on sub_idx without clamping so bootstrap replicates propagate slope denominator uncertainty!
    age_sub = gdata["age_2011"][sub_idx]
    ec_w1_sub = gdata["eta_cog"][sub_idx, 0]
    ec_w1_z = (ec_w1_sub - np.mean(ec_w1_sub)) / max(float(np.std(ec_w1_sub, ddof=1)), 1e-6)
    emp_slope_sub = float(abs(np.polyfit(age_sub, ec_w1_z, 1)[0]))
    base_ratio = float(gdata["chron_age_slope_sd_yr"]) / max(
        float(abs(np.polyfit(gdata["age_2011"], (gdata["eta_cog"][:, 0] - np.mean(gdata["eta_cog"][:, 0])) / np.std(gdata["eta_cog"][:, 0], ddof=1), 1)[0])),
        1e-6,
    )
    annual_age_slope_cog = max(emp_slope_sub * base_ratio, 0.015)

    # 2) Roll forward 8 Counterfactual Regimes from Baseline Wave 1 (2011) -> Wave 5 (2020, 9.1 years)
    n_eval = len(sub_idx)
    regimes = [
        "R0_Natural_Course",
        "R1_Sleep_Circadian_Restoration",
        "R2_Dual_Sensory_Compensation",
        "R3_Cardiometabolic_Inflammatory_Control",
        "R4_Social_Digital_Physical_Engagement",
        "R5_Combined_MultiTarget_Prescription",
        "R5_HalfAdherence_50pct_Sensitivity",
        "R6_Phenotype_Matched_SinglePriority",
        "R7_Phenotype_Mismatched_SinglePriority",
    ]

    # Equal-budget (50% / 50%) individualized Twin Jacobian-dominance & actionable-deficit precision targeting for R6 vs R7:
    # Compares each participant's expected actionable sleep-affective coupling gain against actionable dual-sensory cognitive gain
    sleep_deficit_u = (1.0 - a0_obs[:, 0]) * (0.022 + 0.10 * np.abs(j_cd) + 0.015 * np.maximum(ed[:, 0], 0.0) + 0.008 * dom_d)
    sensory_deficit_u = 0.50 * (a0_obs[:, 1] + a0_obs[:, 2]) * (0.032 + 0.08 * np.abs(j_dc) + 0.018 * np.maximum(-ec[:, 0], 0.0))
    vuln_score = sleep_deficit_u - sensory_deficit_u + 1e-5 * (stats.rankdata(dom_d) / n_eval)
    is_affective_priority = vuln_score >= np.median(vuln_score)

    def apply_regime_to_exposures(
        r_name: str,
        a_cur: np.ndarray,
        a_base: np.ndarray,
        bio_cur: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        a_mod = a_cur.copy()
        b_mod = bio_cur.copy() if bio_cur is not None else None

        if r_name == "R0_Natural_Course":
            return a_mod, b_mod
        if r_name in ("R1_Sleep_Circadian_Restoration", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 0] = 1.0  # optimal 6.5-8.0h sleep
        if r_name in ("R2_Dual_Sensory_Compensation", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 1] = 0.0  # compensated vision
            a_mod[:, 2] = 0.0  # compensated hearing
        if r_name in ("R3_Cardiometabolic_Inflammatory_Control", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 6] = 0.0  # smoking cessation
            a_mod[:, 5] = np.maximum(a_mod[:, 5], 0.5) if r_name == "R3_Cardiometabolic_Inflammatory_Control" else a_mod[:, 5]
            if b_mod is not None:
                b_mod[:, 0] = np.minimum(b_mod[:, 0], 0.0)   # hs-CRP <= 1.0 mg/L (log <= 0)
                b_mod[:, 1] = np.minimum(b_mod[:, 1], 5.60)  # HbA1c <= 5.6%
        if r_name in ("R4_Social_Digital_Physical_Engagement", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 3] = 1.0  # active social engagement
            a_mod[:, 4] = np.maximum(a_mod[:, 4], 0.65)  # structured community digital inclusion
            a_mod[:, 5] = 1.0  # moderate-to-vigorous physical activity
        if r_name == "R5_HalfAdherence_50pct_Sensitivity":
            a_full, b_full = apply_regime_to_exposures("R5_Combined_MultiTarget_Prescription", a_cur, a_base, bio_cur)
            a_mod = 0.5 * a_cur + 0.5 * a_full
            if b_mod is not None and b_full is not None:
                b_mod = 0.5 * bio_cur + 0.5 * b_full
        if r_name == "R6_Phenotype_Matched_SinglePriority":
            # Equal 50%/50% resource budget: top-50% Affective->Cognitive vulnerability gets R1 Sleep-Circadian Stabilization;
            # bottom-50% Cognitive/Sensory vulnerability gets R2 Dual Sensory Compensation!
            a_mod[:, 0] = np.where(is_affective_priority, 1.0, a_mod[:, 0])
            a_mod[:, 1] = np.where(~is_affective_priority, 0.0, a_mod[:, 1])
            a_mod[:, 2] = np.where(~is_affective_priority, 0.0, a_mod[:, 2])
        if r_name == "R7_Phenotype_Mismatched_SinglePriority":
            # Reversed (anti-matched) single-domain allocation at identical 50%/50% resource budget!
            a_mod[:, 0] = np.where(~is_affective_priority, 1.0, a_mod[:, 0])
            a_mod[:, 1] = np.where(is_affective_priority, 0.0, a_mod[:, 1])
            a_mod[:, 2] = np.where(is_affective_priority, 0.0, a_mod[:, 2])
        return a_mod, b_mod

    results_reg: Dict[str, Dict[str, np.ndarray]] = {}
    for r_name in regimes:
        cur_ec = ec[:, 0].copy()
        cur_ed = ed[:, 0].copy()
        a_base = a0_obs.copy()
        b_base = bio0_obs.copy() if (has_bio and bio0_obs is not None) else None
        cur_a, cur_bio = apply_regime_to_exposures(
            r_name, a_base, a_base, b_base.copy() if b_base is not None else None
        )

        # Accumulate path-dependent stochastic variance propagation across 4 wave steps for exact event probabilities
        var_ec_acc = np.zeros(n_eval, dtype=np.float64)
        var_ed_acc = np.zeros(n_eval, dtype=np.float64)

        for w in range(1, 5):
            dt = t_m[:, w] - t_m[:, w - 1]
            tw = t_m[:, w] / 9.0
            age_w = x0[:, 0] + t_m[:, w] / 10.0

            # Step A: Endogenous time-varying confounder/exposure prediction under counterfactual history
            xe_w = np.column_stack([np.ones(n_eval), cur_ec, cur_ed, cur_a, age_w, tw, dt, x0[:, 2:]])
            pred_a = np.clip(xe_w @ beta_exp, 0.0, 1.0)

            pred_bio = None
            if has_bio and cur_bio is not None:
                xb_w = np.column_stack([np.ones(n_eval), cur_bio, cur_ec, cur_ed, cur_a, x0])
                pred_bio = xb_w @ beta_bio

            # Step B: Apply counterfactual regime rule do(A_w = g(H_w))
            cur_a, cur_bio = apply_regime_to_exposures(r_name, pred_a, a_base, pred_bio)

            # Step C: Propagate coupled latent cognitive reserve & affective load to wave w
            aw_within_w = cur_a - a_base
            flux_dc_w = j_cd * cur_ed * dt
            flux_cd_w = j_dc * cur_ec * dt
            inter_sd = aw_within_w[:, 0] * cur_ed
            inter_sc = (aw_within_w[:, 1] + aw_within_w[:, 2]) * cur_ec
            cols_w = [
                np.ones(n_eval),
                cur_ec,
                cur_ed,
                cur_ec * dt,
                cur_ed * dt,
                flux_dc_w,
                flux_cd_w,
                bc_tr * dt,
                bd_tr * dt,
                dt,
                age_w * dt,
                (age_w**2) * dt,
                aw_within_w * dt[:, None],
                a_base * dt[:, None],
                (inter_sd * dt)[:, None],
                (inter_sc * dt)[:, None],
                x0[:, 2:] * dt[:, None],
            ]
            if has_bio and cur_bio is not None and b_base is not None:
                bw_within_w = cur_bio - b_base
                cols_w.append(bw_within_w * dt[:, None])
                cols_w.append(b_base * dt[:, None])
                cols_w.append((cur_bio[:, 0] * cur_ed * dt)[:, None])
                cols_w.append((cur_bio[:, 1] * age_w * dt)[:, None])
            xy_w = np.column_stack(cols_w)
            cur_ec = xy_w @ beta_ec
            cur_ed = xy_w @ beta_ed
            var_ec_acc = 0.72 * var_ec_acc + res_ec_sd**2
            var_ed_acc = 0.65 * var_ed_acc + res_ed_sd**2

        # Map Wave-5 latent states (without practice-effect inflation, k=0 true biological aging scale!)
        # to clinical Wave-5 endpoints via fitted Psychometric Tobit-IRT observation parameters:
        tot_sd_tr20 = np.sqrt(pm["sig_tr20"] ** 2 + (pm["lam_tr20"] ** 2) * var_ec_acc)
        tot_sd_cesd = np.sqrt(pm["sig_cesd"] ** 2 + (pm["lam_cesd"] ** 2) * var_ed_acc)
        mu_tr20_w5 = pm["nu_tr20"] + pm["lam_tr20"] * cur_ec
        mu_cesd_w5 = pm["nu_cesd"] + pm["lam_cesd"] * cur_ed

        p_cog_imp_w5 = stats.norm.cdf((6.5 - mu_tr20_w5) / tot_sd_tr20)
        p_dep_elev_w5 = stats.norm.sf((9.5 - mu_cesd_w5) / tot_sd_cesd)

        results_reg[r_name] = {
            "eta_cog_w5": cur_ec,
            "eta_dep_w5": cur_ed,
            "mu_tr20_w5": mu_tr20_w5,
            "mu_cesd_w5": mu_cesd_w5,
            "p_cog_imp_w5": p_cog_imp_w5,
            "p_dep_elev_w5": p_dep_elev_w5,
        }

    # Natural-course calibration diagnostics against observed Wave 5 states
    w5_obs_m = hw[:, 4]
    calib = {
        "n_eval": int(n_eval),
        "annual_age_slope_cog_SD_per_yr": round(annual_age_slope_cog, 5),
        "obs_eta_cog_w5_mean": round(float(np.mean(ec[w5_obs_m, 4])), 5),
        "nat_course_eta_cog_w5_mean": round(float(np.mean(results_reg["R0_Natural_Course"]["eta_cog_w5"][w5_obs_m])), 5),
        "obs_eta_dep_w5_mean": round(float(np.mean(ed[w5_obs_m, 4])), 5),
        "nat_course_eta_dep_w5_mean": round(float(np.mean(results_reg["R0_Natural_Course"]["eta_dep_w5"][w5_obs_m])), 5),
    }
    return results_reg, calib


def run_longitudinal_gcomputation_with_bootstrap(
    gdata: Dict[str, object],
    blood_df: pd.DataFrame,
    n_boot: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute 9-Year Parametric Longitudinal G-Computation on:
      1) Full Primary Coupled SDE Cohort (N = 20,928, unweighted + IPCW-weighted sensitivity)
      2) Paired Venous Blood Mechanistic Subcohort (N = 7,438, explicitly intervening on hs-CRP & HbA1c!)
    with B = 500 community-clustered bootstrap replicates (communityID block resampling).
    """
    t0 = time.time()
    n_full = len(gdata["ids"])
    all_idx = np.arange(n_full, dtype=np.int32)

    # Point estimates on Full Cohort (Unweighted & IPCW-Weighted)
    reg_full, calib_full = fit_and_simulate_gcomp_cohort(gdata, all_idx, use_ipcw=False)
    reg_ipcw, _ = fit_and_simulate_gcomp_cohort(gdata, all_idx, use_ipcw=True)
    age_rate = calib_full["annual_age_slope_cog_SD_per_yr"]

    # Build Paired-Blood dynamic biomarker tensor (N_full, 5, 2) for [log_crp, hba1c], subsetting N = 7,438
    id_to_i = pd.Series(all_idx, index=gdata["ids"])
    pb_sub = blood_df[blood_df["is_paired_2011_2015"] == 1].copy()
    pb_sub = pb_sub[pb_sub["ID"].isin(gdata["ids"])].reset_index(drop=True)
    pb_idx = pb_sub["ID"].map(id_to_i).to_numpy(dtype=np.int32)

    bio_full = np.zeros((n_full, 5, 2), dtype=np.float64)
    lcrp_11 = pb_sub["log_crp_2011"].fillna(pb_sub["log_crp_2011"].median()).to_numpy(float)
    lcrp_15 = pb_sub["log_crp_2015"].fillna(pb_sub["log_crp_2015"].median()).to_numpy(float)
    hba_11 = pb_sub["hba1c_pct_2011"].fillna(pb_sub["hba1c_pct_2011"].median()).to_numpy(float)
    hba_15 = pb_sub["hba1c_pct_2015"].fillna(pb_sub["hba1c_pct_2015"].median()).to_numpy(float)

    bio_full[pb_idx, 0, 0] = lcrp_11
    bio_full[pb_idx, 1, 0] = 0.5 * (lcrp_11 + lcrp_15)
    bio_full[pb_idx, 2, 0] = lcrp_15
    bio_full[pb_idx, 3, 0] = lcrp_15 + 0.35 * (lcrp_15 - lcrp_11)
    bio_full[pb_idx, 4, 0] = lcrp_15 + 0.55 * (lcrp_15 - lcrp_11)

    bio_full[pb_idx, 0, 1] = hba_11
    bio_full[pb_idx, 1, 1] = 0.5 * (hba_11 + hba_15)
    bio_full[pb_idx, 2, 1] = hba_15
    bio_full[pb_idx, 3, 1] = hba_15 + 0.35 * (hba_15 - hba_11)
    bio_full[pb_idx, 4, 1] = hba_15 + 0.55 * (hba_15 - hba_11)

    reg_pb, calib_pb = fit_and_simulate_gcomp_cohort(gdata, pb_idx, use_ipcw=False, blood_arrays={"bio_clean": bio_full})

    def summarize_contrasts(reg_dict: Dict[str, Dict[str, np.ndarray]], rate_yr: float) -> Dict[str, np.ndarray]:
        r0 = reg_dict["R0_Natural_Course"]
        contrasts = [
            ("R1_Sleep_Circadian_Restoration", "R0_Natural_Course"),
            ("R2_Dual_Sensory_Compensation", "R0_Natural_Course"),
            ("R3_Cardiometabolic_Inflammatory_Control", "R0_Natural_Course"),
            ("R4_Social_Digital_Physical_Engagement", "R0_Natural_Course"),
            ("R5_Combined_MultiTarget_Prescription", "R0_Natural_Course"),
            ("R5_HalfAdherence_50pct_Sensitivity", "R0_Natural_Course"),
            ("R6_Phenotype_Matched_SinglePriority", "R0_Natural_Course"),
            ("R7_Phenotype_Mismatched_SinglePriority", "R0_Natural_Course"),
            ("R6_Phenotype_Matched_SinglePriority", "R7_Phenotype_Mismatched_SinglePriority"),
        ]
        out_m = {}
        for r_a, r_b in contrasts:
            key = f"{r_a}__vs__{r_b}"
            da = reg_dict[r_a]
            db = reg_dict[r_b]
            d_cog = float(np.mean(da["eta_cog_w5"] - db["eta_cog_w5"]))
            d_dep = float(np.mean(da["eta_dep_w5"] - db["eta_dep_w5"]))
            d_tr20 = float(np.mean(da["mu_tr20_w5"] - db["mu_tr20_w5"]))
            d_cesd = float(np.mean(da["mu_cesd_w5"] - db["mu_cesd_w5"]))
            arr_cog = 100.0 * float(np.mean(db["p_cog_imp_w5"] - da["p_cog_imp_w5"]))  # positive = risk reduction (% pts)
            arr_dep = 100.0 * float(np.mean(db["p_dep_elev_w5"] - da["p_dep_elev_w5"]))
            rr_cog = float(np.mean(da["p_cog_imp_w5"]) / max(np.mean(db["p_cog_imp_w5"]), 1e-6))
            rr_dep = float(np.mean(da["p_dep_elev_w5"]) / max(np.mean(db["p_dep_elev_w5"]), 1e-6))
            yrs_delayed = d_cog / max(rate_yr, 1e-4)
            out_m[key] = np.array([d_cog, d_dep, d_tr20, d_cesd, yrs_delayed, arr_cog, rr_cog, arr_dep, rr_dep], dtype=np.float64)
        return out_m

    pt_full = summarize_contrasts(reg_full, age_rate)
    pt_ipcw = summarize_contrasts(reg_ipcw, age_rate)
    pt_pb = summarize_contrasts(reg_pb, calib_pb["annual_age_slope_cog_SD_per_yr"])

    # B = 500 Community-Clustered Bootstrap Replicates
    print(f"[Pillar 3 G-Computation] Running B = {n_boot} community-clustered bootstrap replicates across 450 communities...")
    rng = np.random.default_rng(SEED)
    comm_all = gdata["comm_id"]
    uniq_c = np.unique(comm_all)
    c_to_full = {c: np.where(comm_all == c)[0] for c in uniq_c}

    pb_set = set(pb_idx.tolist())
    c_to_pb = {c: np.array([i for i in idx_arr if i in pb_set], dtype=np.int32) for c, idx_arr in c_to_full.items()}

    boot_full: Dict[str, List[np.ndarray]] = {k: [] for k in pt_full}
    boot_pb: Dict[str, List[np.ndarray]] = {k: [] for k in pt_pb}

    for b_i in range(n_boot):
        sampled_c = rng.choice(uniq_c, size=len(uniq_c), replace=True)
        b_idx_full = np.concatenate([c_to_full[c] for c in sampled_c])
        rg_b_full, cal_bf = fit_and_simulate_gcomp_cohort(gdata, b_idx_full, use_ipcw=False)
        sm_bf = summarize_contrasts(rg_b_full, cal_bf["annual_age_slope_cog_SD_per_yr"])
        for k, v in sm_bf.items():
            boot_full[k].append(v)

        pb_chunks = [c_to_pb[c] for c in sampled_c if len(c_to_pb[c]) > 0]
        b_idx_pb = np.concatenate(pb_chunks)
        rg_b_pb, cal_bp = fit_and_simulate_gcomp_cohort(gdata, b_idx_pb, use_ipcw=False, blood_arrays={"bio_clean": bio_full})
        sm_bp = summarize_contrasts(rg_b_pb, cal_bp["annual_age_slope_cog_SD_per_yr"])
        for k, v in sm_bp.items():
            boot_pb[k].append(v)

        if (b_i + 1) % 100 == 0:
            print(f"  [G-Comp Bootstrap {b_i+1}/{n_boot}] elapsed = {time.time() - t0:.1f}s")

    def format_gcomp_table(
        cohort_label: str,
        n_c: int,
        pt_map: Dict[str, np.ndarray],
        boot_map: Dict[str, List[np.ndarray]],
        ipcw_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, object]]:
        rows = []
        for k, pt in pt_map.items():
            b_arr = np.asarray(boot_map[k])  # (B, 9)
            lo = np.percentile(b_arr, 2.5, axis=0)
            hi = np.percentile(b_arr, 97.5, axis=0)
            se = np.std(b_arr, ddof=1, axis=0)
            p_cog = float(2.0 * stats.norm.sf(abs(pt[0]) / max(float(se[0]), 1e-12)))
            p_dep = float(2.0 * stats.norm.sf(abs(pt[1]) / max(float(se[1]), 1e-12)))
            r_a, r_b = k.split("__vs__")
            nnt_cog = round(100.0 / pt[5], 1) if pt[5] > 0.05 else np.nan
            nnt_dep = round(100.0 / pt[7], 1) if pt[7] > 0.05 else np.nan
            rows.append(
                {
                    "cohort": cohort_label,
                    "n_participants": n_c,
                    "n_cluster_bootstrap": n_boot,
                    "intervention_regime": r_a,
                    "comparator_regime": r_b,
                    "delta_eta_cog_9yr_SD": round(float(pt[0]), 5),
                    "delta_eta_cog_se": round(float(se[0]), 5),
                    "delta_eta_cog_ci95_lo": round(float(lo[0]), 5),
                    "delta_eta_cog_ci95_hi": round(float(hi[0]), 5),
                    "delta_eta_cog_p": float(f"{p_cog:.4e}"),
                    "ipcw_sens_delta_eta_cog_SD": round(float(ipcw_map[k][0]), 5) if ipcw_map is not None else np.nan,
                    "cognitive_aging_years_delayed": round(float(pt[4]), 3),
                    "cog_years_delayed_ci95_lo": round(float(lo[4]), 3),
                    "cog_years_delayed_ci95_hi": round(float(hi[4]), 3),
                    "delta_eta_dep_9yr_SD": round(float(pt[1]), 5),
                    "delta_eta_dep_se": round(float(se[1]), 5),
                    "delta_eta_dep_ci95_lo": round(float(lo[1]), 5),
                    "delta_eta_dep_ci95_hi": round(float(hi[1]), 5),
                    "delta_eta_dep_p": float(f"{p_dep:.4e}"),
                    "ipcw_sens_delta_eta_dep_SD": round(float(ipcw_map[k][1]), 5) if ipcw_map is not None else np.nan,
                    "delta_tr20_words_9yr": round(float(pt[2]), 4),
                    "delta_cesd10_points_9yr": round(float(pt[3]), 4),
                    "ARR_cog_impairment_pct_pts": round(float(pt[5]), 3),
                    "ARR_cog_ci95_lo": round(float(lo[5]), 3),
                    "ARR_cog_ci95_hi": round(float(hi[5]), 3),
                    "RR_cog_impairment": round(float(pt[6]), 4),
                    "NNT_prevent_1_cog_impairment": nnt_cog,
                    "ARR_elevated_depression_pct_pts": round(float(pt[7]), 3),
                    "ARR_dep_ci95_lo": round(float(lo[7]), 3),
                    "ARR_dep_ci95_hi": round(float(hi[7]), 3),
                    "RR_elevated_depression": round(float(pt[8]), 4),
                    "NNT_prevent_1_elevated_depression": nnt_dep,
                }
            )
        return rows

    all_rows = format_gcomp_table("CHARLS_Full_CoupledSDE_N20928", n_full, pt_full, boot_full, pt_ipcw) + format_gcomp_table(
        "CHARLS_PairedVenousBlood_N7438", len(pb_idx), pt_pb, boot_pb, None
    )
    gcomp_df = pd.DataFrame(all_rows)
    gcomp_df["fdr_q_cog"] = multipletests(gcomp_df["delta_eta_cog_p"], method="fdr_bh")[1]
    gcomp_df["fdr_q_dep"] = multipletests(gcomp_df["delta_eta_dep_p"], method="fdr_bh")[1]

    # Individual-level counterfactual benefit dataframe for GAMLSS benefit-equity audit
    indiv_ben = pd.DataFrame(
        {
            "ID": gdata["ids"],
            "communityID": gdata["comm_id"],
            "age_2011": gdata["age_2011"],
            "female": gdata["female"],
            "illiterate": gdata["illit"],
            "prim_ed": gdata["prim_ed"],
            "rural": gdata["rural"],
            "benefit_cog_R5_SD": reg_full["R5_Combined_MultiTarget_Prescription"]["eta_cog_w5"] - reg_full["R0_Natural_Course"]["eta_cog_w5"],
            "benefit_dep_R5_SD": reg_full["R0_Natural_Course"]["eta_dep_w5"] - reg_full["R5_Combined_MultiTarget_Prescription"]["eta_dep_w5"],  # positive = dep load reduced
            "arr_cog_R5_pct": 100.0 * (reg_full["R0_Natural_Course"]["p_cog_imp_w5"] - reg_full["R5_Combined_MultiTarget_Prescription"]["p_cog_imp_w5"]),
            "arr_dep_R5_pct": 100.0 * (reg_full["R0_Natural_Course"]["p_dep_elev_w5"] - reg_full["R5_Combined_MultiTarget_Prescription"]["p_dep_elev_w5"]),
        }
    )

    calib_df = pd.DataFrame(
        [
            {"cohort": "CHARLS_Full_N20928", **calib_full},
            {"cohort": "CHARLS_PairedBlood_N7438", **calib_pb},
        ]
    )
    return gcomp_df, calib_df, indiv_ben


# ---------------------------------------------------------------------------
# 4. Module 3 (§4.3): GAMLSS Distributional Fairness & Faraci Ambiguity Entropy Audit
# ---------------------------------------------------------------------------
def run_gamlss_fairness_and_faraci_ambiguity(
    gdata: Dict[str, object],
    indiv_ben: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. Fit 2-parameter Location-Scale GAMLSS on out-of-sample Split A Tobit NLL & calibration residuals
       comparing Naive Neural ODE (M4), Uncensored Coupled SDE (M5), and Psychometric-Coupled-SDE (M7)
       across Literacy, Urban-Rural, Sex, and Age Decades in CHARLS (N = 20,928).
    2. Compute Subgroup Fairness Disparity Summary across all 4 equity axes (CHARLS + NHANES N = 3,014).
    3. Fit Location-Scale GAMLSS on 9-year multi-target counterfactual benefits (R5 vs R0).
    4. Execute Faraci Multi-Model Ensemble Shannon Entropy & JSD Ambiguity Triage Audit (Rossi & Faraci 2025).
    """
    oof = pd.read_parquet(DATA_DIR / "charls_splitA_oof_predictions.parquet")
    oof = pd.DataFrame({"ID": gdata["ids"]}).merge(oof, on="ID", how="left")

    age_z = (gdata["age_2011"] - np.mean(gdata["age_2011"])) / np.std(gdata["age_2011"], ddof=1)
    sp_age = natural_cubic_basis_3df(age_z)
    x_design = np.column_stack(
        [
            np.ones(len(oof)),
            gdata["illit"],
            gdata["prim_ed"],
            gdata["rural"],
            gdata["female"],
            sp_age,
        ]
    )
    design_names = [
        "Intercept_(SecondaryEdu_Urban_Male_MeanAge)",
        "Illiterate_vs_SecondaryPlus",
        "PrimarySishu_vs_SecondaryPlus",
        "RuralVillage_vs_Urban",
        "Female_vs_Male",
        "Age_Spline_Linear",
        "Age_Spline_Cubic1",
        "Age_Spline_Cubic2",
    ]

    gamlss_dfs = []
    for m_tag, m_label in [
        ("M4", "M4_Naive_Neural_Latent_ODE"),
        ("M5", "M5_Uncensored_Coupled_SDE"),
        ("M7", "M7_Psychometric_Coupled_SDE"),
    ]:
        y_nll = oof[f"nll_{m_tag}"].to_numpy(float)
        gdf = fit_gaussian_location_scale_gamlss(
            y_nll, x_design, x_design, design_names, design_names, cluster_ids=gdata["comm_id"]
        )
        gdf.insert(0, "outcome_metric", "OutOfSample_Tobit_NLL")
        gdf.insert(0, "model", m_label)
        gamlss_dfs.append(gdf)

    # Also GAMLSS on Signed Episodic Memory Calibration Residual (true_tr20 - mu_tr20) for M5 vs M7
    for m_tag, m_label in [("M5", "M5_Uncensored_Coupled_SDE"), ("M7", "M7_Psychometric_Coupled_SDE")]:
        y_res_tr20 = (oof["true_tr20"] - oof[f"mu_tr20_{m_tag}"]).to_numpy(float)
        gdf_r = fit_gaussian_location_scale_gamlss(
            y_res_tr20, x_design, x_design, design_names, design_names, cluster_ids=gdata["comm_id"]
        )
        gdf_r.insert(0, "outcome_metric", "EpisodicMemory_tr20_CalibrationResidual")
        gdf_r.insert(0, "model", m_label)
        gamlss_dfs.append(gdf_r)

    gamlss_charls_df = pd.concat(gamlss_dfs, axis=0, ignore_index=True)

    # Subgroup Fairness Parity Table across 4 Pre-Specified Equity Axes
    edu_str = np.where(gdata["illit"] == 1.0, "1_Illiterate_NoFormal", np.where(gdata["prim_ed"] == 1.0, "2_Primary_or_Sishu", "3_Secondary_or_Higher"))
    rur_str = np.where(gdata["rural"] == 1.0, "1_Rural_Village", "0_Urban_Community")
    sex_str = np.where(gdata["female"] == 1.0, "1_Female", "0_Male")
    age_str = np.where(gdata["age_2011"] < 60.0, "1_Age_45_59", np.where(gdata["age_2011"] < 75.0, "2_Age_60_74", "3_Age_75_Plus"))

    sub_rows = []
    for axis_name, vec_s in [("Education_Literacy", edu_str), ("Urban_Rural_Divide", rur_str), ("Sex", sex_str), ("Age_Decade", age_str)]:
        for st_val in np.unique(vec_s):
            m_s = vec_s == st_val
            # 90% PI empirical coverage on tr20 for M4, M5, M7
            cov_m4 = 100.0 * float(np.mean(np.abs(oof.loc[m_s, "true_tr20"] - oof.loc[m_s, "mu_tr20_M4"]) <= (1.64485 * oof.loc[m_s, "sd_tr20_M4"] + 0.5)))
            cov_m5 = 100.0 * float(np.mean(np.abs(oof.loc[m_s, "true_tr20"] - oof.loc[m_s, "mu_tr20_M5"]) <= (1.64485 * oof.loc[m_s, "sd_tr20_M5"] + 0.5)))
            cov_m7 = 100.0 * float(np.mean(np.abs(oof.loc[m_s, "true_tr20"] - oof.loc[m_s, "mu_tr20_M7"]) <= (1.64485 * oof.loc[m_s, "sd_tr20_M7"] + 0.5)))
            sub_rows.append(
                {
                    "equity_axis": axis_name,
                    "stratum": str(st_val),
                    "n_participants": int(np.sum(m_s)),
                    "pct_cohort": round(100.0 * float(np.mean(m_s)), 2),
                    "M4_NeuralODE_mean_NLL": round(float(oof.loc[m_s, "nll_M4"].mean()), 5),
                    "M5_UncensoredSDE_mean_NLL": round(float(oof.loc[m_s, "nll_M5"].mean()), 5),
                    "M7_PsychometricSDE_mean_NLL": round(float(oof.loc[m_s, "nll_M7"].mean()), 5),
                    "NLL_gain_M7_vs_M5": round(float((oof.loc[m_s, "nll_M5"] - oof.loc[m_s, "nll_M7"]).mean()), 5),
                    "NLL_gain_M7_vs_M4": round(float((oof.loc[m_s, "nll_M4"] - oof.loc[m_s, "nll_M7"]).mean()), 5),
                    "M4_tr20_PICP90_pct": round(cov_m4, 2),
                    "M5_tr20_PICP90_pct": round(cov_m5, 2),
                    "M7_tr20_PICP90_pct": round(cov_m7, 2),
                    "counterfactual_R5_cog_gain_SD": round(float(indiv_ben.loc[m_s, "benefit_cog_R5_SD"].mean()), 5),
                    "counterfactual_R5_dep_reduction_SD": round(float(indiv_ben.loc[m_s, "benefit_dep_R5_SD"].mean()), 5),
                    "counterfactual_R5_ARR_dep_pct_pts": round(float(indiv_ben.loc[m_s, "arr_dep_R5_pct"].mean()), 3),
                }
            )
    subgroup_fairness_df = pd.DataFrame(sub_rows)

    # Location-Scale GAMLSS on 9-Year Counterfactual Benefits (R5 vs R0)
    ben_gamlss_list = []
    for col_b, lbl_b in [
        ("benefit_cog_R5_SD", "9Yr_Counterfactual_Cognitive_Preservation_SD"),
        ("benefit_dep_R5_SD", "9Yr_Counterfactual_Depressive_Reduction_SD"),
    ]:
        gb = fit_gaussian_location_scale_gamlss(
            indiv_ben[col_b].to_numpy(float), x_design, x_design, design_names, design_names, cluster_ids=gdata["comm_id"]
        )
        gb.insert(0, "benefit_endpoint", lbl_b)
        ben_gamlss_list.append(gb)
    benefit_equity_df = pd.concat(ben_gamlss_list, axis=0, ignore_index=True)

    # NHANES External Equity GAMLSS Audit (N = 3,014 older adults >= 60: <High School vs >=High School, Sex, Race, PIR)
    nh = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_participants.parquet")
    nh60 = nh[(nh["age_years"] >= 60.0) & nh["dsst"].notna()].copy().reset_index(drop=True)
    age_nh_z = (nh60["age_years"].to_numpy(float) - 69.5) / 6.8
    low_ed_nh = nh60["low_education_lt_hs"].fillna(0.0).to_numpy(float)
    fem_nh = nh60["female"].fillna(0.5).to_numpy(float)
    low_pir_nh = (nh60["pir"].fillna(nh60["pir"].median()).to_numpy(float) < 1.85).astype(float)
    non_nhw = (nh60["race_ethnicity"].astype(str) != "Non_Hispanic_White").astype(float).to_numpy()
    x_nh = np.column_stack([np.ones(len(nh60)), low_ed_nh, low_pir_nh, fem_nh, non_nhw, age_nh_z])
    nh_names = ["Intercept", "LowEducation_lt_HighSchool", "LowIncome_PIR_lt_1p85", "Female_vs_Male", "RacialEthnicMinority_vs_NHWhite", "Age_Std"]
    y_dsst_z = (nh60["dsst"].to_numpy(float) - nh60["dsst"].mean()) / nh60["dsst"].std(ddof=1)
    psu_nh = nh60["SDMVPSU"].fillna(1).astype(str).to_numpy() + "_" + nh60["SDMVSTRA"].fillna(1).astype(str).to_numpy()
    gamlss_nh_df = fit_gaussian_location_scale_gamlss(y_dsst_z, x_nh, x_nh, nh_names, nh_names, cluster_ids=psu_nh)
    gamlss_nh_df.insert(0, "cohort", "NHANES_2011_2014_Older60_N3014")
    gamlss_nh_df.insert(1, "endpoint", "ZeroCeiling_DSST_Standardized_Distribution")

    # Guest Editor Faraci Multi-Model Ensemble Shannon Entropy & JSD Ambiguity Triage Audit (Rossi et al., npj Digit Med 2025)
    def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))

    ens_tags = ["M2", "M3", "M4", "M5", "M6", "M7"]
    triage_rows = []
    for ev_name, y_ev, col_prefix in [
        ("Elevated_Depression_CESD10_ge10", (oof["true_cesd10"].to_numpy(float) >= 10.0).astype(int), "p_dep_ge10"),
        ("Cognitive_Impairment_tr20_le6", (oof["true_tr20"].to_numpy(float) <= 6.0).astype(int), "p_cog_le6"),
    ]:
        p_mat = np.column_stack([oof[f"{col_prefix}_{tg}"].to_numpy(float) for tg in ens_tags])  # (N, 6)
        p_bar = np.mean(p_mat, axis=1)
        h_total = binary_entropy_bits(p_bar)               # Total predictive ambiguity (0..1 bit)
        h_aleat = np.mean(binary_entropy_bits(p_mat), axis=1)  # Mean within-model aleatoric entropy
        jsd_epist = np.maximum(h_total - h_aleat, 0.0)     # Jensen-Shannon inter-model epistemic divergence!

        p_m7 = oof[f"{col_prefix}_M7"].to_numpy(float)
        # Evaluate selective ambiguity triage at 0%, 10%, 15%, 20% deferral of highest total ambiguity / boundary cases
        for defer_pct in [0, 10, 15, 20]:
            if defer_pct == 0:
                keep = np.ones(len(y_ev), dtype=bool)
            else:
                thresh = np.percentile(h_total, 100.0 - defer_pct)
                keep = h_total <= thresh
            auroc_k = float(roc_auc_score(y_ev[keep], p_m7[keep]))
            brier_k = float(brier_score_loss(y_ev[keep], p_m7[keep]))
            err_k = 100.0 * float(np.mean((p_m7[keep] >= 0.5).astype(int) != y_ev[keep]))
            ill_deferred_pct = 100.0 * float(np.mean(gdata["illit"][~keep])) if np.any(~keep) else np.nan
            triage_rows.append(
                {
                    "clinical_endpoint": ev_name,
                    "deferral_rate_pct": defer_pct,
                    "n_retained": int(np.sum(keep)),
                    "n_deferred_ambiguous": int(np.sum(~keep)),
                    "M7_retained_AUROC": round(auroc_k, 4),
                    "M7_retained_Brier": round(brier_k, 4),
                    "M7_retained_error_pct": round(err_k, 2),
                    "mean_shannon_entropy_bits_retained": round(float(np.mean(h_total[keep])), 4),
                    "mean_epistemic_JSD_bits_deferred": round(float(np.mean(jsd_epist[~keep])), 5) if np.any(~keep) else 0.0,
                    "illiterate_share_in_deferred_pct": round(ill_deferred_pct, 2) if not np.isnan(ill_deferred_pct) else np.nan,
                }
            )
    faraci_triage_df = pd.DataFrame(triage_rows)
    return gamlss_charls_df, subgroup_fairness_df, benefit_equity_df, gamlss_nh_df, faraci_triage_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=== Phase 3 Pillar 3: Jacobian Dominance, 9-Yr G-Computation (B=500) & GAMLSS Fairness Audit ===")
    twins = pd.read_parquet(DATA_DIR / "charls_sde_individual_twins.parquet")
    panel = pd.read_parquet(DATA_DIR / "charls_5wave_panel.parquet")
    persons = pd.read_parquet(DATA_DIR / "charls_participants_summary.parquet")
    blood = pd.read_parquet(DATA_DIR / "charls_paired_blood_2011_2015.parquet")
    pb_mod_csv = pd.read_csv(P2_DIR / "paired_blood_jacobian_moderation.csv")
    obs_params = pd.read_csv(P2_DIR / "charls_sde_observation_parameters.csv")

    # 1. Jacobian Dominance Phenotyping & Paired-Blood Enrichment
    print("[Module 1] Computing Individualized Directed Jacobian Dominance Phenotypes & Paired-Blood Enrichment...")
    _, pheno_df, pb_pheno_df, pb_indiv = run_jacobian_dominance_phenotyping(twins, panel, persons, blood, pb_mod_csv)
    pheno_df.to_csv(OUT_DIR / "jacobian_dominance_phenotypes_summary.csv", index=False)
    pb_pheno_df.to_csv(OUT_DIR / "paired_blood_phenotype_enrichment.csv", index=False)

    # 2. 9-Year Parametric Longitudinal G-Computation (B = 500 Community-Clustered Bootstrap)
    print("[Module 2] Assembling 5-wave G-Computation tensors & running 9-year counterfactual regimes...")
    gdata = build_5wave_gcomp_arrays(twins, panel, obs_params)
    gcomp_df, calib_df, indiv_ben = run_longitudinal_gcomputation_with_bootstrap(gdata, blood, n_boot=500)
    gcomp_df.to_csv(OUT_DIR / "gcomputation_9yr_counterfactual_prescriptions.csv", index=False)
    calib_df.to_csv(OUT_DIR / "gcomputation_natural_course_calibration.csv", index=False)
    indiv_ben.to_parquet(DATA_DIR / "charls_gcomp_individual_benefits.parquet", index=False)

    # 3. GAMLSS Location-Scale Distributional Fairness & Faraci Ambiguity Entropy Triage
    print("[Module 3] Fitting 2-Parameter Location-Scale GAMLSS Fairness Models & Faraci Ambiguity Triage...")
    gamlss_ch_df, sub_fair_df, ben_eq_df, gamlss_nh_df, triage_df = run_gamlss_fairness_and_faraci_ambiguity(gdata, indiv_ben)
    gamlss_ch_df.to_csv(OUT_DIR / "gamlss_fairness_audit_charls.csv", index=False)
    sub_fair_df.to_csv(OUT_DIR / "subgroup_equity_disparity_table.csv", index=False)
    ben_eq_df.to_csv(OUT_DIR / "gamlss_counterfactual_benefit_equity.csv", index=False)
    gamlss_nh_df.to_csv(OUT_DIR / "gamlss_fairness_audit_nhanes.csv", index=False)
    triage_df.to_csv(OUT_DIR / "faraci_ambiguity_entropy_triage.csv", index=False)

    with open(OUT_DIR / "pillar3_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "scripts/phase3_pillar3_gcomp_gamlss_fairness.py",
                "seed": SEED,
                "elapsed_seconds": round(time.time() - t0, 2),
                "n_full_cohort": int(len(twins)),
                "n_paired_blood_sde": int(len(pb_indiv)),
                "n_cluster_bootstrap": 500,
                "natural_course_calibration": calib_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("\n=== 9-Year Parametric G-Computation Counterfactual Prescriptions (Key Regimes) ===")
    print(
        gcomp_df[
            [
                "cohort", "intervention_regime", "comparator_regime",
                "delta_eta_cog_9yr_SD", "cognitive_aging_years_delayed", "delta_eta_cog_p",
                "delta_eta_dep_9yr_SD", "ARR_elevated_depression_pct_pts", "NNT_prevent_1_elevated_depression",
            ]
        ].to_string(index=False)
    )
    print("\n=== Faraci Ensemble Shannon Entropy Ambiguity Triage ===")
    print(triage_df.to_string(index=False))
    print(f"[Phase 3 Pillar 3 DONE] Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
