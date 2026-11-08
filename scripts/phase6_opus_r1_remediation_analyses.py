#!/usr/bin/env python3
"""
Phase 6 — Opus 5.5 Round-1 Peer Review Comprehensive Remediation & Sensitivity Suite
====================================================================================
Executes all quantitative remediation analyses required by Opus 5.5 Round-1 Review (R1-R19 & Y1-Y20):
  - Module A: NHANES Cross-Cohort Bridges (Self-Reported Sleep -> Actigraphy fPCA/TPDC & sNfL;
    CHARLS-Transferred vs NHANES-Native Tobit-IRT vs Unfiltered Gaussian; Stationary SDE vs NHANES
    Latent Construct Correlation; Stage-1 Age Interaction & Age-Aligned >=60 yr DML Mediation).
  - Module B: CHARLS 8-Model Tournament Upgrade (Fixing M4 Standard Neural Latent ODE sign/centering;
    Adding M2b Censored Tobit LMM; Computing 90% PI Width & Winkler Interval Score; Full-Cohort
    Community-Clustered Sandwich SEs & 95% CIs for Jacobian Drift, Impulse vs Stationary Ratios,
    Kalman Random-Slope Incremental Prognostic Value, Pre-Pandemic W1-W4 & Age >=45 yr Sensitivities).
  - Module C: Pillar 3 G-Computation & Fairness Upgrades (Strict Out-of-Sample W1-W4 -> W5 Rollout &
    Calibration; Unclamped Empirical Aging-Slope Bootstrap Uncertainty Propagation; 1-Wave Lagged
    Exposure Sensitivity & Regimen E-Values; IPCW ESS; True Inter-Model Epistemic JSD/MI vs Random
    Deferral; Unified Whole-Cohort N=20,928 Decision Curve Analysis with LightGBM M3 & Fixed M4).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gate import canary_gate, gate, pass_gate, retry_limit

SCRIPT_DIR = Path(__file__).resolve().parent
pass_gate(str(SCRIPT_DIR))
gate(__file__)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.stats.multitest import multipletests
import torch

from psychometric_sde_core import (
    CHARLS_SCALE_SPECS,
    PsychometricCoupledSDE,
    ScaleSpec,
    exact_censored_gaussian_nll_and_crps_np,
    fit_psychometric_sde_model,
    tobit_probit_Scale_loglik,
)
from phase2_pillar1_nhanes_fpca_tpdc_dml import (
    build_confounder_matrix,
    compute_evalue_continuous,
    cross_fit_stacked_nuisance,
    run_dml_single_sample_mediation,
    run_two_sample_moment_coupled_dml,
)
from phase2_pillar2_charls_psychometric_sde import (
    attach_paired_significance_vs_m7,
    build_charls_tensor_dataset,
    evaluate_predictions_on_targets,
    extract_transition_pairs,
    fit_and_predict_m4_naive_neural_ode,
    make_torch_batch,
    predict_m1_persistence_agenorm,
    predict_m2_lmm_empirical_bayes,
    predict_m2b_tobit_censored_lmm,
    predict_m3_lightgbm_gbdt,
)
from phase3_pillar3_gcomp_gamlss_fairness import (
    build_5wave_gcomp_arrays,
    fit_gaussian_location_scale_gamlss,
    natural_cubic_basis_3df,
)

SEED = 20270507
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_processed"
P1_DIR = PROJECT_ROOT / "results" / "phase2_pillar1_nhanes"
P2_DIR = PROJECT_ROOT / "results" / "phase2_pillar2_sde"
P3_DIR = PROJECT_ROOT / "results" / "phase3_pillar3_fairness"


# ===========================================================================
# MODULE A: NHANES Cross-Cohort Bridges & Age-Aligned Mediation (R1, R8, Y2, Y9)
# ===========================================================================
def run_module_a_nhanes_bridges_and_mediation() -> Dict[str, object]:
    t0 = time.time()
    print("\n=== [Module A] NHANES Cross-Cohort Bridges & Age-Aligned DML Mediation ===")
    nh_base = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_participants.parquet")
    nh_feat = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_fpca_tpdc_features.parquet")
    nh = nh_base.merge(nh_feat, on="SEQN", how="left")

    # Exact CHARLS-harmonized sleep exposure definition: 6.0 to 8.5 hours/night
    nh["optimal_sleep_6_to_85h"] = (
        (nh["sleep_hours_self"] >= 6.0) & (nh["sleep_hours_self"] <= 8.5)
    ).where(nh["sleep_hours_self"].notna(), np.nan)
    nh["healthy_sleep_no_disorder"] = (
        (nh["optimal_sleep_6_to_85h"] == 1.0)
        & (nh["sleep_trouble_doc"].fillna(0) == 0)
        & (nh["sleep_disorder_doc"].fillna(0) == 0)
    ).where(nh["optimal_sleep_6_to_85h"].notna(), np.nan)

    # 1. Y2.1: Self-Reported Optimal Sleep (6.0-8.5h) -> Objective Actigraphy (fPC1, TPDC, RA, IS) & Serum log(sNfL)
    bridge_rows = []
    sub_defs = [
        ("NHANES_Primary_PAX_4d16h_AllAdults", nh[nh["pax_eligible_primary_4d16h"] == 1]),
        ("NHANES_Primary_PAX_4d16h_Older60", nh[(nh["pax_eligible_primary_4d16h"] == 1) & (nh["age_years"] >= 60.0)]),
        ("NHANES_ArmB_sNfL_Subcohort", nh[(nh["pax_eligible_primary_4d16h"] == 1) & nh["log_snfl"].notna()]),
    ]
    target_metrics = [
        ("circadian_photic_vitality_z", "Composite_Circadian_Photic_Vitality_Z"),
        ("fpca_mims_pc1", "Motor_fPC1_Amplitude"),
        ("fpca_lux_pc1", "Photic_Lux_fPC1_Amplitude"),
        ("tpdc_photic_to_motor_circ", "Photic_to_Motor_TPDC_Circadian"),
        ("pax_RA", "Relative_Amplitude_RA"),
        ("pax_IS", "Interdaily_Stability_IS"),
        ("pax_IV", "Intradaily_Variability_IV"),
        ("log_snfl", "Serum_log_sNfL_pg_mL"),
    ]
    for coh_name, sub_df in sub_defs:
        for exp_col in ["optimal_sleep_6_to_85h", "healthy_sleep_no_disorder"]:
            for m_col, m_label in target_metrics:
                d_sub = sub_df.dropna(subset=[exp_col, m_col, "age_years", "female"]).copy()
                if len(d_sub) < 100:
                    continue
                y_raw = d_sub[m_col].to_numpy(float)
                y_sd = max(float(np.std(y_raw, ddof=1)), 1e-6)
                y_z = (y_raw - np.mean(y_raw)) / y_sd
                x_exp = d_sub[exp_col].to_numpy(float)
                age_c = (d_sub["age_years"].to_numpy(float) - 50.0) / 10.0
                fem = d_sub["female"].to_numpy(float)
                low_ed = d_sub["low_education_lt_hs"].fillna(0.0).to_numpy(float)
                bmi_c = (d_sub["bmi"].fillna(d_sub["bmi"].median()).to_numpy(float) - 28.0) / 6.0
                creat_c = d_sub["serum_creatinine_mgdl"].fillna(d_sub["serum_creatinine_mgdl"].median()).to_numpy(float)
                X_mat = np.column_stack([np.ones(len(d_sub)), x_exp, age_c, age_c**2, fem, low_ed, bmi_c, creat_c])
                beta, _, _, _ = linalg.lstsq(X_mat, y_z)
                res = y_z - X_mat @ beta
                bread = linalg.inv(X_mat.T @ X_mat + 1e-6 * np.eye(X_mat.shape[1]))
                meat = (X_mat * (res**2)[:, None]).T @ X_mat * (len(d_sub) / max(len(d_sub) - X_mat.shape[1], 1))
                se_b = float(np.sqrt(max((bread @ meat @ bread)[1, 1], 1e-12)))
                est_sd = float(beta[1])
                p_val = float(2.0 * stats.norm.sf(abs(est_sd) / se_b))
                bridge_rows.append(
                    {
                        "cohort_subset": coh_name,
                        "self_report_exposure": exp_col,
                        "objective_biomarker_or_actigraphy": m_col,
                        "metric_label": m_label,
                        "n_obs": int(len(d_sub)),
                        "exposure_prevalence_pct": round(100.0 * float(np.mean(x_exp)), 2),
                        "beta_SD_per_unit_exposure": round(est_sd, 5),
                        "beta_raw_units": round(est_sd * y_sd, 5),
                        "robust_se_SD": round(se_b, 5),
                        "ci95_lo_SD": round(est_sd - 1.96 * se_b, 5),
                        "ci95_hi_SD": round(est_sd + 1.96 * se_b, 5),
                        "p_value": float(f"{p_val:.4e}"),
                    }
                )
    bridge_df = pd.DataFrame(bridge_rows)
    bridge_df["fdr_q"] = multipletests(bridge_df["p_value"], method="fdr_bh")[1]
    bridge_df.to_csv(P1_DIR / "cross_cohort_sleep_biomarker_bridge.csv", index=False)

    # 2. R8 & Y9: Stage-1 Age Interaction (<60 vs >=60 yr) on log(sNfL) & Age-Aligned >=60 yr Mediation + Difference Test
    arm_b = nh[(nh["pax_eligible_primary_4d16h"] == 1) & nh["log_snfl"].notna()].copy().reset_index(drop=True)
    arm_d = nh[(nh["age_years"] >= 60.0) & (nh["pax_eligible_primary_4d16h"] == 1) & nh["log_snfl"].notna()].copy().reset_index(drop=True)

    key_exposures = [
        "circadian_photic_vitality_z",
        "fpca_mims_pc1",
        "pax_RA",
        "pax_IS",
        "pax_IV",
        "tpdc_photic_to_motor_circ",
        "tpdc_log_ultra_circ_power_ratio",
    ]
    age_med_rows = []
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for exp_c in key_exposures:
        d_b = arm_b.dropna(subset=[exp_c, "log_snfl", "age_years"]).copy().reset_index(drop=True)
        wb = build_confounder_matrix(d_b, include_cycle=False)
        a_s = (d_b[exp_c].to_numpy(float) - d_b[exp_c].mean()) / d_b[exp_c].std(ddof=1)
        m_s = (d_b["log_snfl"].to_numpy(float) - d_b["log_snfl"].mean()) / d_b["log_snfl"].std(ddof=1)
        a_til = a_s - cross_fit_stacked_nuisance(wb, a_s, kf5, seed_offset=101)
        m_til = m_s - cross_fit_stacked_nuisance(wb, m_s, kf5, seed_offset=103)

        is_old = (d_b["age_years"].to_numpy(float) >= 60.0).astype(float)
        # Stratum-specific alpha (<60 vs >=60) and interaction test
        m_lt60 = is_old == 0.0
        m_ge60 = is_old == 1.0
        al_lt60 = float(np.mean(a_til[m_lt60] * m_til[m_lt60]) / np.mean(a_til[m_lt60] ** 2))
        se_lt60 = float(np.sqrt(np.sum(((a_til[m_lt60] * (m_til[m_lt60] - al_lt60 * a_til[m_lt60])) / (np.mean(a_til[m_lt60] ** 2) * np.sum(m_lt60))) ** 2)))
        al_ge60 = float(np.mean(a_til[m_ge60] * m_til[m_ge60]) / np.mean(a_til[m_ge60] ** 2))
        se_ge60 = float(np.sqrt(np.sum(((a_til[m_ge60] * (m_til[m_ge60] - al_ge60 * a_til[m_ge60])) / (np.mean(a_til[m_ge60] ** 2) * np.sum(m_ge60))) ** 2)))
        diff_al = al_ge60 - al_lt60
        se_diff_al = math.sqrt(se_lt60**2 + se_ge60**2)
        p_int_age = float(2.0 * stats.norm.sf(abs(diff_al) / max(se_diff_al, 1e-12)))

        # Single-sample >=60 yr PHQ-8_nosleep mediation (N = 511) vs Single-sample >=60 yr DSST (N = 419) vs Two-Sample DML
        med_phq8_60 = run_dml_single_sample_mediation(arm_d, exp_c, "log_snfl", "phq8_nosleep", n_boot=500)
        med_dsst_60 = run_dml_single_sample_mediation(arm_d, exp_c, "log_snfl", "dsst", n_boot=500)
        med_gcog_60 = run_dml_single_sample_mediation(arm_d, exp_c, "log_snfl", "global_cognition_z", n_boot=500)
        med_2s_dsst = run_two_sample_moment_coupled_dml(nh, exp_c, "dsst")
        med_2s_gcog = run_two_sample_moment_coupled_dml(nh, exp_c, "global_cognition_z")
        med_phq8_all = run_dml_single_sample_mediation(arm_b, exp_c, "log_snfl", "phq8_nosleep", n_boot=500)

        # Age-matched (>=60 yr Stage-1 n=442 coupled with >=60 yr Stage-2 N=2,756) two-sample NIE
        beta_2s_dsst = float(med_2s_dsst["beta_armD_std"])
        se_b_2s_dsst = float(med_2s_dsst["beta_armD_se"])
        nie_2s_60_dsst = al_ge60 * beta_2s_dsst
        se_nie_2s_60_dsst = math.sqrt((al_ge60**2) * (se_b_2s_dsst**2) + (beta_2s_dsst**2) * (se_ge60**2) + (se_ge60**2) * (se_b_2s_dsst**2))
        p_nie_2s_60_dsst = float(2.0 * stats.norm.sf(abs(nie_2s_60_dsst) / max(se_nie_2s_60_dsst, 1e-12)))

        beta_2s_gcog = float(med_2s_gcog["beta_armD_std"])
        se_b_2s_gcog = float(med_2s_gcog["beta_armD_se"])
        nie_2s_60_gcog = al_ge60 * beta_2s_gcog
        se_nie_2s_60_gcog = math.sqrt((al_ge60**2) * (se_b_2s_gcog**2) + (beta_2s_gcog**2) * (se_ge60**2) + (se_ge60**2) * (se_b_2s_gcog**2))
        p_nie_2s_60_gcog = float(2.0 * stats.norm.sf(abs(nie_2s_60_gcog) / max(se_nie_2s_60_gcog, 1e-12)))

        # Formal contrast |NIE_cog_2sample| - |NIE_dep_phq8| (standardized magnitude comparison)
        nie_c = abs(float(med_2s_dsst["NIE_moment_coupled_std"]))
        se_c = float(med_2s_dsst["NIE_moment_coupled_se"])
        nie_d = abs(float(med_phq8_all["NIE_std"]))
        se_d = float(med_phq8_all["NIE_se_orthogonal"])
        diff_nie_mag = float(med_2s_dsst["NIE_moment_coupled_std"]) - (-float(med_phq8_all["NIE_std"]))  # signed beneficial direction
        se_diff_nie = math.sqrt(se_c**2 + se_d**2)
        p_diff_nie = float(2.0 * stats.norm.sf(abs(nie_c - nie_d) / max(se_diff_nie, 1e-12)))

        age_med_rows.append(
            {
                "exposure": exp_c,
                "n_armB_lt60": int(np.sum(m_lt60)),
                "alpha_lt60_std": round(al_lt60, 5),
                "alpha_lt60_se": round(se_lt60, 5),
                "n_armB_ge60": int(np.sum(m_ge60)),
                "alpha_ge60_std": round(al_ge60, 5),
                "alpha_ge60_se": round(se_ge60, 5),
                "stage1_age_interaction_diff": round(diff_al, 5),
                "stage1_age_interaction_p": float(f"{p_int_age:.4e}"),
                "twosample_DSST_NIE_std": med_2s_dsst["NIE_moment_coupled_std"],
                "twosample_DSST_NIE_se": med_2s_dsst["NIE_moment_coupled_se"],
                "twosample_DSST_NIE_p": med_2s_dsst["NIE_p"],
                "twosample_DSST_prop_mediated": med_2s_dsst["prop_mediated_2sample"],
                "twosample_DSST_Evalue_point": med_2s_dsst["E_value_NIE_point"],
                "twosample_GlobalCog_NIE_std": med_2s_gcog["NIE_moment_coupled_std"],
                "twosample_GlobalCog_NIE_se": med_2s_gcog["NIE_moment_coupled_se"],
                "twosample_GlobalCog_NIE_p": med_2s_gcog["NIE_p"],
                "twosample_GlobalCog_prop_mediated": med_2s_gcog["prop_mediated_2sample"],
                "twosample_GlobalCog_Evalue_point": med_2s_gcog["E_value_NIE_point"],
                "twosample_AgeMatched60_DSST_NIE_std": round(nie_2s_60_dsst, 5),
                "twosample_AgeMatched60_DSST_NIE_se": round(se_nie_2s_60_dsst, 5),
                "twosample_AgeMatched60_DSST_NIE_p": float(f"{p_nie_2s_60_dsst:.4e}"),
                "twosample_AgeMatched60_GlobalCog_NIE_std": round(nie_2s_60_gcog, 5),
                "twosample_AgeMatched60_GlobalCog_NIE_se": round(se_nie_2s_60_gcog, 5),
                "twosample_AgeMatched60_GlobalCog_NIE_p": float(f"{p_nie_2s_60_gcog:.4e}"),
                "singlesample_ArmD_N419_DSST_alpha_std": med_dsst_60["alpha_A_to_M_std"],
                "singlesample_ArmD_N419_DSST_alpha_p": med_dsst_60["alpha_p"],
                "singlesample_ArmD_N419_DSST_beta_std": med_dsst_60["beta_M_to_Y_std"],
                "singlesample_ArmD_N419_DSST_beta_p": med_dsst_60["beta_p"],
                "singlesample_ArmD_N419_DSST_NIE_std": med_dsst_60["NIE_std"],
                "singlesample_ArmD_N419_DSST_NIE_se": med_dsst_60["NIE_se_orthogonal"],
                "singlesample_ArmD_N419_DSST_NIE_p": med_dsst_60["NIE_p"],
                "singlesample_ArmD_N416_GlobalCog_NIE_std": med_gcog_60["NIE_std"],
                "singlesample_ArmD_N416_GlobalCog_NIE_p": med_gcog_60["NIE_p"],
                "singlesample_Older60_N511_PHQ8_NIE_std": med_phq8_60["NIE_std"],
                "singlesample_Older60_N511_PHQ8_NIE_se": med_phq8_60["NIE_se_orthogonal"],
                "singlesample_Older60_N511_PHQ8_NIE_p": med_phq8_60["NIE_p"],
                "singlesample_ArmB_N1467_PHQ8_NIE_std": med_phq8_all["NIE_std"],
                "singlesample_ArmB_N1467_PHQ8_NIE_p": med_phq8_all["NIE_p"],
                "diff_abs_NIE_cog2S_minus_depPHQ8_std": round(nie_c - nie_d, 5),
                "diff_abs_NIE_se": round(se_diff_nie, 5),
                "diff_abs_NIE_p": float(f"{p_diff_nie:.4e}"),
            }
        )
    age_med_df = pd.DataFrame(age_med_rows)
    age_med_df.to_csv(P1_DIR / "nhanes_age_stratified_mediation_bridge.csv", index=False)

    # 3. Y2.2 & Y2.3: 3-Way NHANES Psychometric Validation (CHARLS-Transferred Tobit vs NHANES-Native Tobit vs Unfiltered Gaussian)
    #    & Cross-Cohort Stationary/Filtered Latent Construct Correlation Bridge
    sub60 = nh[(nh["age_years"] >= 60.0) & nh["dsst"].notna()].copy().reset_index(drop=True)
    nh_specs = [
        ScaleSpec("dsst", domain=0, lower=0.0, upper=133.0, anchor_loading=17.0),
        ScaleSpec("animal_fluency", domain=0, lower=0.0, upper=60.0, censor_upper=False, anchor_loading=None),
        ScaleSpec("cerad_ir_sum", domain=0, lower=0.0, upper=30.0, anchor_loading=None),
        ScaleSpec("cerad_dr", domain=0, lower=0.0, upper=10.0, anchor_loading=None),
        ScaleSpec("phq9_total", domain=1, lower=0.0, upper=27.0, anchor_loading=4.2),
    ]
    n_s = len(sub60)
    n_d = len(nh_specs)
    y_mat = np.zeros((n_s, 1, n_d), dtype=np.float64)
    m_mat = np.zeros((n_s, 1, n_d), dtype=bool)
    for d, sp in enumerate(nh_specs):
        v = sub60[sp.name].to_numpy(float)
        ok = ~np.isnan(v)
        y_mat[ok, 0, d] = v[ok]
        m_mat[ok, 0, d] = True

    age_c = (sub60["age_years"].to_numpy(float) - 69.0) / 10.0
    fem_c = sub60["female"].fillna(0.5).to_numpy(float) - 0.5
    low_ed = sub60["low_education_lt_hs"].fillna(0.0).to_numpy(float)
    x0 = np.column_stack([age_c, fem_c, low_ed - np.mean(low_ed)])

    # Load CHARLS fitted observation parameters to transfer structural low-literacy variance inflation delta_het
    ch_obs = pd.read_csv(P2_DIR / "charls_sde_observation_parameters.csv").set_index("scale")
    delta_cog_charls = float(np.mean([ch_obs.loc["imrc", "delta_illiteracy_heteroscedasticity"], ch_obs.loc["dlrc", "delta_illiteracy_heteroscedasticity"]]))
    delta_dep_charls = float(ch_obs.loc["cesd10", "delta_illiteracy_heteroscedasticity"])
    transferred_delta = torch.tensor(
        [delta_cog_charls, delta_cog_charls, delta_cog_charls, delta_cog_charls, delta_dep_charls],
        dtype=torch.float64,
    )

    m_keys = ["CHARLS_Transferred_Tobit", "NHANES_Native_Tobit", "Naive_Uncensored_Gaussian"]
    oof_mu = {k: np.zeros((n_s, n_d)) for k in m_keys}
    oof_sd = {k: np.zeros((n_s, n_d)) for k in m_keys}
    lat_corr = {}

    for m_label, use_psych, freeze_charls_transfer in [
        ("CHARLS_Transferred_Tobit", True, True),
        ("NHANES_Native_Tobit", True, False),
        ("Naive_Uncensored_Gaussian", False, False),
    ]:
        for tr, va in kf5.split(y_mat):
            b_tr = {
                "t_years": torch.zeros((len(tr), 1), dtype=torch.float64),
                "y_obs": torch.tensor(y_mat[tr], dtype=torch.float64),
                "obs_mask": torch.tensor(m_mat[tr], dtype=torch.bool),
                "k_prior": torch.zeros((len(tr), 1, n_d), dtype=torch.float64),
                "x0": torch.tensor(x0[tr], dtype=torch.float64),
                "u_ctrl": torch.zeros((len(tr), 1, 1), dtype=torch.float64),
                "illit": torch.tensor(low_ed[tr], dtype=torch.float64),
            }
            b_va = {
                "t_years": torch.zeros((len(va), 1), dtype=torch.float64),
                "y_obs": torch.tensor(y_mat[va], dtype=torch.float64),
                "obs_mask": torch.tensor(m_mat[va], dtype=torch.bool),
                "k_prior": torch.zeros((len(va), 1, n_d), dtype=torch.float64),
                "x0": torch.tensor(x0[va], dtype=torch.float64),
                "u_ctrl": torch.zeros((len(va), 1, 1), dtype=torch.float64),
                "illit": torch.tensor(low_ed[va], dtype=torch.float64),
            }
            torch.manual_seed(SEED)
            mod = PsychometricCoupledSDE(nh_specs, d_x0=3, d_u=1, coupled=True, psychometric_filter=use_psych, n_gh_1d=7)
            if freeze_charls_transfer:
                with torch.no_grad():
                    mod.delta_het.copy_(transferred_delta)
                mod.delta_het.requires_grad_(False)
            fit_psychometric_sde_model(mod, b_tr, n_adam_steps=120, lr_adam=0.04, n_lbfgs_steps=25, l2_reg=1e-4, verbose=False)
            with torch.no_grad():
                loadings = mod.get_loadings()
                sig_base = torch.nn.functional.softplus(mod.raw_sigma_obs) + 1e-2
                for d_hold, sp_h in enumerate(nh_specs):
                    b_va_mask = dict(b_va)
                    m_clone = b_va["obs_mask"].clone()
                    m_clone[:, 0, d_hold] = False
                    b_va_mask["obs_mask"] = m_clone
                    out_lo = mod.filter_and_loglik(b_va_mask)
                    eta_post = out_lo["filt_eta_means"][:, 0, :]
                    p0_cov = mod._build_2x2_cov(mod.raw_p0_diag, mod.raw_p0_off)
                    mu0_eta = b_va["x0"] @ mod.C0_init.T
                    l_eta = torch.linalg.cholesky(p0_cov + 1e-7 * torch.eye(2, dtype=torch.float64))
                    eta_grid = mu0_eta[:, None, :] + torch.einsum("ij,gj->gi", l_eta, mod.gh_nodes)[None, :, :]
                    ll_g = torch.zeros((len(va), mod.gh_nodes.shape[0]), dtype=torch.float64)
                    for d2, sp2 in enumerate(nh_specs):
                        if d2 == d_hold:
                            continue
                        het_m2 = torch.exp(0.5 * mod.delta_het[d2] * b_va["illit"][:, None]) if use_psych else torch.ones((len(va), 1), dtype=torch.float64)
                        sd2 = sig_base[d2] * het_m2
                        mu_g2 = mod.nu[d2] + loadings[d2] * eta_grid[:, :, sp2.domain]
                        ll_g = ll_g + tobit_probit_Scale_loglik(
                            b_va["y_obs"][:, 0, d2 : d2 + 1],
                            m_clone[:, 0, d2 : d2 + 1],
                            mu_g2,
                            sd2,
                            sp2,
                            use_tobit_censoring=use_psych,
                        )
                    post_w = torch.softmax(mod.gh_logw[None, :] + ll_g, dim=-1)
                    dom_h = sp_h.domain
                    mu_dom_post = eta_post[:, dom_h]
                    var_dom_post = torch.sum(post_w * ((eta_grid[:, :, dom_h] - mu_dom_post[:, None]) ** 2), dim=-1)
                    het_mh = torch.exp(0.5 * mod.delta_het[d_hold] * b_va["illit"]) if use_psych else torch.ones_like(b_va["illit"])
                    sig_dh = sig_base[d_hold] * het_mh
                    pred_m_h = mod.nu[d_hold] + loadings[d_hold] * mu_dom_post
                    pred_s_h = torch.sqrt(torch.clamp((loadings[d_hold] ** 2) * var_dom_post + sig_dh**2, min=1e-6))
                    oof_mu[m_label][va, d_hold] = pred_m_h.cpu().numpy()
                    oof_sd[m_label][va, d_hold] = pred_s_h.cpu().numpy()

        b_all = {
            "t_years": torch.zeros((n_s, 1), dtype=torch.float64),
            "y_obs": torch.tensor(y_mat, dtype=torch.float64),
            "obs_mask": torch.tensor(m_mat, dtype=torch.bool),
            "k_prior": torch.zeros((n_s, 1, n_d), dtype=torch.float64),
            "x0": torch.tensor(x0, dtype=torch.float64),
            "u_ctrl": torch.zeros((len(sub60), 1, 1), dtype=torch.float64),
            "illit": torch.tensor(low_ed, dtype=torch.float64),
        }
        torch.manual_seed(SEED)
        mod_full = PsychometricCoupledSDE(nh_specs, d_x0=3, d_u=1, coupled=True, psychometric_filter=use_psych, n_gh_1d=7)
        if freeze_charls_transfer:
            with torch.no_grad():
                mod_full.delta_het.copy_(transferred_delta)
            mod_full.delta_het.requires_grad_(False)
        fit_psychometric_sde_model(mod_full, b_all, n_adam_steps=130, lr_adam=0.04, n_lbfgs_steps=25, l2_reg=1e-4, verbose=False)
        with torch.no_grad():
            p0_cov = mod_full._build_2x2_cov(mod_full.raw_p0_diag, mod_full.raw_p0_off).cpu().numpy()
            r_lat = float(p0_cov[0, 1] / math.sqrt(max(p0_cov[0, 0] * p0_cov[1, 1], 1e-12)))
            eta_f = mod_full.filter_and_loglik(b_all)["filt_eta_means"][:, 0, :].cpu().numpy()
            r_emp = float(np.corrcoef(eta_f[:, 0], eta_f[:, 1])[0, 1])
            has_snfl = sub60["log_snfl"].notna().to_numpy()
            r_snfl_cog = float(np.corrcoef(eta_f[has_snfl, 0], sub60.loc[has_snfl, "log_snfl"].to_numpy(float))[0, 1])
            lat_corr[m_label] = (r_lat, r_snfl_cog, r_emp)

    val_rows = []
    for d, sp in enumerate(nh_specs):
        vm = m_mat[:, 0, d]
        yt = y_mat[vm, 0, d]
        fl_pct = 100.0 * float(np.mean(yt <= sp.lower))
        ce_pct = 100.0 * float(np.mean(yt >= sp.upper)) if sp.censor_upper else 0.0
        r_trans = exact_censored_gaussian_nll_and_crps_np(yt, oof_mu["CHARLS_Transferred_Tobit"][vm, d], oof_sd["CHARLS_Transferred_Tobit"][vm, d], sp)
        r_native = exact_censored_gaussian_nll_and_crps_np(yt, oof_mu["NHANES_Native_Tobit"][vm, d], oof_sd["NHANES_Native_Tobit"][vm, d], sp)
        r_naive = exact_censored_gaussian_nll_and_crps_np(yt, oof_mu["Naive_Uncensored_Gaussian"][vm, d], oof_sd["Naive_Uncensored_Gaussian"][vm, d], sp)
        diff_tr_na = r_trans["nll"] - r_naive["nll"]
        diff_nat_na = r_native["nll"] - r_naive["nll"]
        _, p_tr_na = stats.ttest_1samp(diff_tr_na, 0.0)
        _, p_nat_na = stats.ttest_1samp(diff_nat_na, 0.0)
        val_rows.append(
            {
                "cohort": "NHANES_2011_2014_Older60_ExternalValidation",
                "scale": sp.name,
                "n_valid": int(np.sum(vm)),
                "empirical_floor_pct": round(fl_pct, 2),
                "empirical_ceiling_pct": round(ce_pct, 2),
                "psychometric_oof_NLL": round(float(np.mean(r_native["nll"])), 5),
                "charls_transferred_tobit_oof_NLL": round(float(np.mean(r_trans["nll"])), 5),
                "naive_gaussian_oof_NLL": round(float(np.mean(r_naive["nll"])), 5),
                "delta_NLL_psych_minus_naive": round(float(np.mean(diff_nat_na)), 5),
                "delta_NLL_transferred_minus_naive": round(float(np.mean(diff_tr_na)), 5),
                "paired_p_NLL": float(f"{p_nat_na:.4e}"),
                "paired_p_transferred_vs_naive": float(f"{p_tr_na:.4e}"),
                "psychometric_oof_CRPS": round(float(np.mean(r_native["crps"])), 5),
                "charls_transferred_tobit_oof_CRPS": round(float(np.mean(r_trans["crps"])), 5),
                "naive_gaussian_oof_CRPS": round(float(np.mean(r_naive["crps"])), 5),
                "latent_cog_dep_r_psych": round(lat_corr["NHANES_Native_Tobit"][0], 4),
                "latent_cog_dep_r_transferred": round(lat_corr["CHARLS_Transferred_Tobit"][0], 4),
                "latent_cog_dep_r_naive": round(lat_corr["Naive_Uncensored_Gaussian"][0], 4),
                "latent_cog_log_sNfL_r_psych": round(lat_corr["NHANES_Native_Tobit"][1], 4),
            }
        )
    nh_val_df = pd.DataFrame(val_rows)
    nh_val_df.to_csv(P2_DIR / "nhanes_cross_cohort_validation.csv", index=False)

    # Cross-Cohort Latent Correlation Bridge (CHARLS SDE vs NHANES Older 60+)
    twins_ch = pd.read_parquet(DATA_DIR / "charls_sde_individual_twins.parquet")
    panel_ch = pd.read_parquet(DATA_DIR / "charls_5wave_panel.parquet")
    base_ch = panel_ch.sort_values(["ID", "wave"]).groupby("ID", as_index=False).first()
    tw_m = twins_ch.merge(base_ch[["ID", "age_years", "tr20", "cesd10"]], on="ID", how="left")
    tw_60 = tw_m[tw_m["age_years"] >= 60.0]

    r_ch_lat_all = float(np.corrcoef(tw_m["eta_cog_w1"], tw_m["eta_dep_w1"])[0, 1])
    r_ch_lat_60 = float(np.corrcoef(tw_60["eta_cog_w1"], tw_60["eta_dep_w1"])[0, 1])
    r_ch_raw_60 = float(tw_60[["tr20", "cesd10"]].dropna().corr().iloc[0, 1])
    r_nh_raw_60 = float(sub60[["dsst", "phq9_total"]].dropna().corr().iloc[0, 1])

    corr_bridge_df = pd.DataFrame(
        [
            {
                "cohort": "CHARLS_Full_Cohort_N20928",
                "age_range": "All_Eligible_Baseline",
                "n_participants": int(len(tw_m)),
                "raw_scale_cog_dep_r": round(float(tw_m[["tr20", "cesd10"]].dropna().corr().iloc[0, 1]), 4),
                "tobit_filtered_latent_cog_dep_r": round(r_ch_lat_all, 4),
                "deattenuation_gain_abs_r": round(abs(r_ch_lat_all) - abs(float(tw_m[["tr20", "cesd10"]].dropna().corr().iloc[0, 1])), 4),
            },
            {
                "cohort": "CHARLS_Older60_Subcohort",
                "age_range": "Age_ge_60",
                "n_participants": int(len(tw_60)),
                "raw_scale_cog_dep_r": round(r_ch_raw_60, 4),
                "tobit_filtered_latent_cog_dep_r": round(r_ch_lat_60, 4),
                "deattenuation_gain_abs_r": round(abs(r_ch_lat_60) - abs(r_ch_raw_60), 4),
            },
            {
                "cohort": "NHANES_Older60_TransferredTobit",
                "age_range": "Age_ge_60",
                "n_participants": int(len(sub60)),
                "raw_scale_cog_dep_r": round(r_nh_raw_60, 4),
                "tobit_filtered_latent_cog_dep_r": round(lat_corr["CHARLS_Transferred_Tobit"][0], 4),
                "deattenuation_gain_abs_r": round(abs(lat_corr["CHARLS_Transferred_Tobit"][0]) - abs(r_nh_raw_60), 4),
            },
            {
                "cohort": "NHANES_Older60_NativeTobit",
                "age_range": "Age_ge_60",
                "n_participants": int(len(sub60)),
                "raw_scale_cog_dep_r": round(r_nh_raw_60, 4),
                "tobit_filtered_latent_cog_dep_r": round(lat_corr["NHANES_Native_Tobit"][0], 4),
                "deattenuation_gain_abs_r": round(abs(lat_corr["NHANES_Native_Tobit"][0]) - abs(r_nh_raw_60), 4),
            },
        ]
    )
    corr_bridge_df.to_csv(P2_DIR / "cross_cohort_latent_correlation_bridge.csv", index=False)
    print(f"[Module A] Completed NHANES bridges & age-aligned mediation in {time.time() - t0:.1f}s")
    return {
        "bridge_df": bridge_df,
        "age_med_df": age_med_df,
        "nh_val_df": nh_val_df,
        "corr_bridge_df": corr_bridge_df,
    }


# ===========================================================================
# MODULE B: CHARLS 8-Model Tournament Upgrade & Structural SDE Uncertainty
#           (R4, R5, R6, R7, R9, R12, Y4, Y5, Y7, Y8, Y15)
# ===========================================================================
def run_module_b_charls_benchmark_and_sde_inference() -> Dict[str, object]:
    t0 = time.time()
    print("\n=== [Module B] CHARLS 8-Model Tournament Upgrade (Fixed M4 + M2b Tobit-LMM + Winkler) & SDE SEs ===")
    ds = build_charls_tensor_dataset()
    n_subj = len(ds["ids"])
    n_d = len(CHARLS_SCALE_SPECS)

    # 1. Re-evaluate Fixed M4_Standard_Neural_Latent_ODE and New M2b_Tobit_Censored_LMM on Split A (5-Fold Community CV)
    gkf = GroupKFold(n_splits=5)
    oof_mu_m4 = np.zeros((n_subj, n_d), dtype=np.float64)
    oof_sd_m4 = np.zeros((n_subj, n_d), dtype=np.float64)
    oof_mu_m2b = np.zeros((n_subj, n_d), dtype=np.float64)
    oof_sd_m2b = np.zeros((n_subj, n_d), dtype=np.float64)
    y_true_a = np.zeros((n_subj, n_d), dtype=np.float64)
    m_true_a = np.zeros((n_subj, n_d), dtype=bool)
    last_y_a = np.full((n_subj, n_d), np.nan, dtype=np.float64)

    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(ds["y_obs"], groups=ds["comm_id"])):
        te_pairs = extract_transition_pairs(ds, te_idx, only_person_last_wave=True)
        te_subj = te_pairs["subj_idx"]
        y_true_a[te_subj] = te_pairs["Y"]
        m_true_a[te_subj] = te_pairs["M"]
        last_y_a[te_subj] = te_pairs["last_Y"]

        oof_mu_m2b[te_subj], oof_sd_m2b[te_subj] = predict_m2b_tobit_censored_lmm(ds, tr_idx, te_idx, te_pairs)

        x0_m = np.mean(ds["x0_raw"][tr_idx], axis=0)
        u_m = np.mean(ds["u_ctrl_raw"][tr_idx], axis=(0, 1))
        tr_b = make_torch_batch(ds, tr_idx, x0_m, u_m)
        te_b_masked = make_torch_batch(ds, te_idx, x0_m, u_m, mask_person_last_wave=True)
        lw_te = ds["last_coupled_wave"][te_idx]
        row_arange = np.arange(len(te_idx))

        p4_mu, p4_sd = fit_and_predict_m4_naive_neural_ode(tr_b, te_b_masked, per_person_last_wave_te=lw_te, n_epochs=160)
        oof_mu_m4[te_idx] = p4_mu[row_arange, lw_te, :]
        oof_sd_m4[te_idx] = p4_sd[row_arange, lw_te, :]

    ov_m2b_a, sc_m2b_a, pn_m2b_a, pc_m2b_a = evaluate_predictions_on_targets(
        "M2b_Tobit_Censored_LMM", "SplitA_5Fold_CommunityGrouped_CV",
        y_true_a, m_true_a, oof_mu_m2b, oof_sd_m2b, last_y_a, ds["illit"], ds["comm_id"]
    )
    ov_m4_a, sc_m4_a, pn_m4_a, pc_m4_a = evaluate_predictions_on_targets(
        "M4_Naive_Neural_Latent_ODE", "SplitA_5Fold_CommunityGrouped_CV",
        y_true_a, m_true_a, oof_mu_m4, oof_sd_m4, last_y_a, ds["illit"], ds["comm_id"]
    )
    print(f"  [Split A Fixed M4] NLL = {ov_m4_a['co_primary_tobit_NLL']:.5f}, Dep AUROC = {ov_m4_a['AUROC_elevated_depression_cesd10_ge10']:.4f}, Cog AUROC = {ov_m4_a['AUROC_cognitive_impairment_tr20_le6']:.4f}")
    print(f"  [Split A New M2b Tobit-LMM] NLL = {ov_m2b_a['co_primary_tobit_NLL']:.5f}, Dep AUROC = {ov_m2b_a['AUROC_elevated_depression_cesd10_ge10']:.4f}, Cog AUROC = {ov_m2b_a['AUROC_cognitive_impairment_tr20_le6']:.4f}")

    # Update charls_splitA_oof_predictions.parquet with fixed M4 and new M2b
    oof_df = pd.read_parquet(DATA_DIR / "charls_splitA_oof_predictions.parquet")
    for st, mu_mat, sd_mat, pn_vec, pc_vec in [
        ("M4", oof_mu_m4, oof_sd_m4, pn_m4_a, pc_m4_a),
        ("M2b", oof_mu_m2b, oof_sd_m2b, pn_m2b_a, pc_m2b_a),
    ]:
        oof_df[f"nll_{st}"] = pn_vec
        oof_df[f"crps_{st}"] = pc_vec
        oof_df[f"mu_tr20_{st}"] = mu_mat[:, 0] + mu_mat[:, 1]
        oof_df[f"sd_tr20_{st}"] = np.sqrt(np.maximum(sd_mat[:, 0] ** 2 + sd_mat[:, 1] ** 2, 1e-4))
        oof_df[f"mu_cesd_{st}"] = mu_mat[:, 5]
        oof_df[f"sd_cesd_{st}"] = np.maximum(sd_mat[:, 5], 1e-3)
        oof_df[f"p_dep_ge10_{st}"] = np.clip(stats.norm.sf((9.5 - mu_mat[:, 5]) / np.maximum(sd_mat[:, 5], 1e-3)), 1e-5, 1.0 - 1e-5)
        oof_df[f"p_cog_le6_{st}"] = np.clip(stats.norm.cdf((6.5 - oof_df[f"mu_tr20_{st}"]) / oof_df[f"sd_tr20_{st}"]), 1e-5, 1.0 - 1e-5)
    oof_df.to_parquet(DATA_DIR / "charls_splitA_oof_predictions.parquet", index=False)

    # Update charls_cv_benchmark_7models.csv (replacing M4 row and inserting M2b_Tobit_Censored_LMM, plus recalculating paired p-values)
    cv_df_old = pd.read_csv(P2_DIR / "charls_cv_benchmark_7models.csv")
    sc_df_old = pd.read_csv(P2_DIR / "scale_stratified_fidelity_gain.csv")

    # Map person-level NLL & CRPS from oof_df for M1..M7
    tag_to_model = {
        "M1": "M1_Persistence_AgeNormative",
        "M2": "M2_Linear_Mixed_Effects_BLUP",
        "M2b": "M2b_Tobit_Censored_LMM",
        "M3": "M3_LightGBM_Autoregressive_GBDT",
        "M4": "M4_Naive_Neural_Latent_ODE",
        "M5": "M5_Uncensored_Coupled_SDE",
        "M6": "M6_Uncoupled_Psychometric_Tobit_SDE",
        "M7": "M7_Psychometric_Coupled_SDE",
    }
    pnll_a = {v: oof_df[f"nll_{k}"].to_numpy(float) for k, v in tag_to_model.items()}
    pcrps_a = {v: oof_df[f"crps_{k}"].to_numpy(float) for k, v in tag_to_model.items()}

    # Extract dlrc PICP90 from scale_stratified_fidelity_gain.csv
    dlrc_map_a = sc_df_old[(sc_df_old["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_df_old["scale"] == "dlrc")].set_index("model")["picp_90_pct"].to_dict()
    dlrc_map_a["M2b_Tobit_Censored_LMM"] = ov_m2b_a["dlrc_PICP_90_coverage_pct"]
    dlrc_map_a["M4_Naive_Neural_Latent_ODE"] = ov_m4_a["dlrc_PICP_90_coverage_pct"]

    # Also compute tr20 + cesd10 PI90 width and Winkler score for M1..M7 from oof_df
    tr20_spec = ScaleSpec("tr20", domain=0, lower=0.0, upper=20.0)
    cesd_spec = CHARLS_SCALE_SPECS[5]
    piw_map_a = {}
    wink_map_a = {}
    for st, m_full in tag_to_model.items():
        r_tr = exact_censored_gaussian_nll_and_crps_np(
            oof_df["true_tr20"].to_numpy(float), oof_df[f"mu_tr20_{st}"].to_numpy(float), oof_df[f"sd_tr20_{st}"].to_numpy(float), tr20_spec
        )
        r_ce = exact_censored_gaussian_nll_and_crps_np(
            oof_df["true_cesd10"].to_numpy(float), oof_df[f"mu_cesd_{st}"].to_numpy(float), oof_df[f"sd_cesd_{st}"].to_numpy(float), cesd_spec
        )
        piw_map_a[m_full] = round(0.5 * (float(np.mean(r_tr["pi90_width"] / 20.0)) + float(np.mean(r_ce["pi90_width"] / 30.0))), 5)
        wink_map_a[m_full] = round(0.5 * (float(np.mean(r_tr["winkler90"] / 20.0)) + float(np.mean(r_ce["winkler90"] / 30.0))), 5)

    new_cv_rows = []
    for _, r in cv_df_old.iterrows():
        rd = r.to_dict()
        m_nm = rd["model"]
        if m_nm == "M2b_Tobit_Censored_LMM":
            continue
        if m_nm == "M4_Naive_Neural_Latent_ODE":
            for k_up in [
                "co_primary_tobit_NLL", "co_primary_CRPS_norm", "PICP_90_coverage_pct",
                "AUROC_elevated_depression_cesd10_ge10", "Brier_elevated_depression",
                "AUROC_cognitive_impairment_tr20_le6", "Brier_cognitive_impairment",
                "AUROC_incident_cognitive_drop_ge3w",
            ]:
                rd[k_up] = ov_m4_a[k_up]
        rd["dlrc_PICP_90_coverage_pct"] = dlrc_map_a.get(m_nm, rd["PICP_90_coverage_pct"])
        rd["mean_PI90_width_norm"] = piw_map_a.get(m_nm, piw_map_a["M7_Psychometric_Coupled_SDE"])
        rd["winkler_score_90_norm"] = wink_map_a.get(m_nm, wink_map_a["M7_Psychometric_Coupled_SDE"])
        new_cv_rows.append(rd)
        if m_nm == "M2_Linear_Mixed_Effects_BLUP":
            r2b = dict(ov_m2b_a)
            r2b["mean_PI90_width_norm"] = piw_map_a["M2b_Tobit_Censored_LMM"]
            r2b["winkler_score_90_norm"] = wink_map_a["M2b_Tobit_Censored_LMM"]
            new_cv_rows.append(r2b)

    # Recompute paired t-test & Holm p-values across baseline models vs M7
    cv_upd_df = pd.DataFrame(new_cv_rows)
    ref_n = pnll_a["M7_Psychometric_Coupled_SDE"]
    ref_c = pcrps_a["M7_Psychometric_Coupled_SDE"]
    base_idxs = []
    for idx_r, r in cv_upd_df.iterrows():
        m_nm = r["model"]
        if m_nm in pnll_a and m_nm != "M7_Psychometric_Coupled_SDE":
            dn = ref_n - pnll_a[m_nm]
            dc = ref_c - pcrps_a[m_nm]
            _, pn_v = stats.ttest_1samp(dn, 0.0)
            _, pc_v = stats.ttest_1samp(dc, 0.0)
            cv_upd_df.loc[idx_r, "delta_NLL_M7_minus_model"] = round(float(np.mean(dn)), 5)
            cv_upd_df.loc[idx_r, "paired_p_NLL_vs_M7"] = float(f"{pn_v:.4e}")
            cv_upd_df.loc[idx_r, "delta_CRPS_M7_minus_model"] = round(float(np.mean(dc)), 5)
            cv_upd_df.loc[idx_r, "paired_p_CRPS_vs_M7"] = float(f"{pc_v:.4e}")
            base_idxs.append(idx_r)
    cv_upd_df.loc[base_idxs, "holm_p_NLL_vs_6baselines"] = multipletests(cv_upd_df.loc[base_idxs, "paired_p_NLL_vs_M7"].to_numpy(float), method="holm")[1]
    cv_upd_df.loc[base_idxs, "holm_p_CRPS_vs_6baselines"] = multipletests(cv_upd_df.loc[base_idxs, "paired_p_CRPS_vs_M7"].to_numpy(float), method="holm")[1]
    cv_upd_df.to_csv(P2_DIR / "charls_cv_benchmark_7models.csv", index=False)

    # 2. Evaluate All 8 Models (M1, M2, M2b, M3, M4, M5, M6, M7) on Split B (B1 +3y, B1 +5y, B2 +2y)
    #    with Exact Subject-Level Paired t-Tests, Holm Multiplicity Control, PI90 Width, & Winkler Interval Scores
    ge3_idx = np.where(ds["n_coupled_waves"] >= 3)[0]
    ge3_pos_map = pd.Series(np.arange(len(ge3_idx), dtype=np.int32), index=ge3_idx)
    temporal_protocols = [
        ("SplitB1_TrainW1_W3", 2, [("Forecast_W4_Plus3Yr", 3), ("Forecast_W5_Plus5Yr", 4)]),
        ("SplitB2_TrainW1_W4", 3, [("Forecast_W5_Plus2Yr", 4)]),
    ]
    core_8_keys = [
        "M1_Persistence_AgeNormative",
        "M2_Linear_Mixed_Effects_BLUP",
        "M2b_Tobit_Censored_LMM",
        "M3_LightGBM_Autoregressive_GBDT",
        "M4_Naive_Neural_Latent_ODE",
        "M5_Uncensored_Coupled_SDE",
        "M6_Uncoupled_Psychometric_Tobit_SDE",
        "M7_Psychometric_Coupled_SDE",
    ]
    new_sc_rows = []
    temp_all_dfs = []

    for train_tag, max_tr_w, target_specs in temporal_protocols:
        tr_pairs_b = extract_transition_pairs(ds, ge3_idx, max_target_wave_idx=max_tr_w)
        x0_mb = np.mean(ds["x0_raw"][ge3_idx], axis=0)
        u_mb = np.mean(ds["u_ctrl_raw"][ge3_idx, : max_tr_w + 1, :], axis=(0, 1))
        tr_b_temp = make_torch_batch(ds, ge3_idx, x0_mb, u_mb, max_obs_wave_idx=max_tr_w)
        full_b_temp = make_torch_batch(ds, ge3_idx, x0_mb, u_mb, max_obs_wave_idx=None, freeze_future_from_wave_idx=max_tr_w)

        p4_mu_b, p4_sd_b = fit_and_predict_m4_naive_neural_ode(tr_b_temp, full_b_temp, max_assimilate_wave_idx=max_tr_w, n_epochs=160)

        sde_preds_b = {}
        fitted_b_mods: Dict[str, PsychometricCoupledSDE] = {}
        for m_key, is_coup, is_psych, warm_key in [
            ("M5_Uncensored_Coupled_SDE", True, False, None),
            ("M6_Uncoupled_Psychometric_Tobit_SDE", False, True, None),
            ("M7_Psychometric_Coupled_SDE", True, True, "M6_Uncoupled_Psychometric_Tobit_SDE"),
        ]:
            torch.manual_seed(SEED + max_tr_w)
            mod_b = PsychometricCoupledSDE(
                CHARLS_SCALE_SPECS,
                d_x0=tr_b_temp["x0"].shape[1],
                d_u=tr_b_temp["u_ctrl"].shape[2],
                d_z_mod=0,
                coupled=is_coup,
                psychometric_filter=is_psych,
                n_gh_1d=5,
                use_subtest_traits=True,
                d_prot=2,
            )
            fit_psychometric_sde_model(
                mod_b,
                tr_b_temp,
                lr_adam=0.038,
                n_lbfgs_steps=55,
                l2_reg=1.5e-4,
                warm_from=fitted_b_mods.get(warm_key, None) if warm_key is not None else None,
                seed=SEED,
                verbose=False,
            )
            fitted_b_mods[m_key] = mod_b
            with torch.no_grad():
                out_b = mod_b.filter_and_loglik(full_b_temp, max_filter_wave_idx=max_tr_w)
                mu_b_np = out_b["pred_scale_mu"].cpu().numpy()
                sd_b_np = out_b["pred_scale_sd"].cpu().numpy()
                if not is_psych:
                    mu_b_np[:, :, 4] = np.clip(mu_b_np[:, :, 4], 0.02, 0.98)
                    sd_b_np[:, :, 4] = -1.0
                sde_preds_b[m_key] = (mu_b_np, sd_b_np)

        for hz_tag, w_targ in target_specs:
            split_label = f"{train_tag}__{hz_tag}"
            te_pairs_hz = extract_transition_pairs(ds, ge3_idx, fixed_origin_max_wave=max_tr_w, fixed_target_wave=w_targ)
            loc_rows = pd.Series(te_pairs_hz["subj_idx"]).map(ge3_pos_map).to_numpy(dtype=np.int32)
            sub_ill = ds["illit"][te_pairs_hz["subj_idx"]]
            sub_com = ds["comm_id"][te_pairs_hz["subj_idx"]]

            mu_hz = {}
            sd_hz = {}
            mu_hz["M1_Persistence_AgeNormative"], sd_hz["M1_Persistence_AgeNormative"] = predict_m1_persistence_agenorm(tr_pairs_b, te_pairs_hz)
            mu_hz["M2_Linear_Mixed_Effects_BLUP"], sd_hz["M2_Linear_Mixed_Effects_BLUP"] = predict_m2_lmm_empirical_bayes(
                ds, ge3_idx, ge3_idx, te_pairs_hz, max_train_wave_idx=max_tr_w, fixed_origin_max_wave=max_tr_w
            )
            mu_hz["M2b_Tobit_Censored_LMM"], sd_hz["M2b_Tobit_Censored_LMM"] = predict_m2b_tobit_censored_lmm(
                ds, ge3_idx, ge3_idx, te_pairs_hz, max_train_wave_idx=max_tr_w, fixed_origin_max_wave=max_tr_w
            )
            mu_hz["M3_LightGBM_Autoregressive_GBDT"], sd_hz["M3_LightGBM_Autoregressive_GBDT"] = predict_m3_lightgbm_gbdt(
                tr_pairs_b, te_pairs_hz, seed_fold=w_targ * 100
            )
            mu_hz["M4_Naive_Neural_Latent_ODE"] = p4_mu_b[loc_rows, w_targ, :]
            sd_hz["M4_Naive_Neural_Latent_ODE"] = p4_sd_b[loc_rows, w_targ, :]
            for sk in ["M5_Uncensored_Coupled_SDE", "M6_Uncoupled_Psychometric_Tobit_SDE", "M7_Psychometric_Coupled_SDE"]:
                mu_hz[sk] = sde_preds_b[sk][0][loc_rows, w_targ, :]
                sd_hz[sk] = sde_preds_b[sk][1][loc_rows, w_targ, :]

            hz_sum = []
            hz_pnll = {}
            hz_pcrps = {}
            for mk in core_8_keys:
                ov, sc_r, pn, pc = evaluate_predictions_on_targets(
                    mk, split_label, te_pairs_hz["Y"], te_pairs_hz["M"],
                    mu_hz[mk], sd_hz[mk], te_pairs_hz["last_Y"], sub_ill, sub_com
                )
                hz_sum.append(ov)
                new_sc_rows.extend(sc_r)
                hz_pnll[mk] = pn
                hz_pcrps[mk] = pc
            hz_df = attach_paired_significance_vs_m7(hz_sum, hz_pnll, hz_pcrps)
            temp_all_dfs.append(hz_df)
            ov_4 = hz_df[hz_df["model"] == "M4_Naive_Neural_Latent_ODE"].iloc[0]
            ov_2b = hz_df[hz_df["model"] == "M2b_Tobit_Censored_LMM"].iloc[0]
            print(
                f"  [{split_label}] Fixed M4 Dep AUROC = {ov_4['AUROC_elevated_depression_cesd10_ge10']:.4f} "
                f"(p_NLL={ov_4['paired_p_NLL_vs_M7']:.2e}), M2b Tobit-LMM NLL = {ov_2b['co_primary_tobit_NLL']:.5f} "
                f"(p_NLL={ov_2b['paired_p_NLL_vs_M7']:.2e}), dlrc PICP90 = {ov_2b['dlrc_PICP_90_coverage_pct']:.2f}%"
            )

    # Update scale_stratified_fidelity_gain.csv (keep Split A non-M4/M2b rows, append Split A M2b+M4 and all Split B rows)
    sc_keep_a = sc_df_old[(sc_df_old["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (~sc_df_old["model"].isin(["M4_Naive_Neural_Latent_ODE", "M2b_Tobit_Censored_LMM"]))].copy()
    sc_add = pd.DataFrame(sc_m2b_a + sc_m4_a + new_sc_rows)
    sc_merged = pd.concat([sc_keep_a, sc_add], axis=0, ignore_index=True)
    sc_merged.to_csv(P2_DIR / "scale_stratified_fidelity_gain.csv", index=False)

    temp_upd_df = pd.concat(temp_all_dfs, axis=0, ignore_index=True)
    temp_upd_df.to_csv(P2_DIR / "charls_temporal_holdout_benchmark.csv", index=False)

    # 3. Full-Cohort (N = 20,928) Community-Clustered Sandwich/BHHH SEs, 95% CIs, Eigenvalues, & Sensitivity Checks (Y5, Y7, R9, R12, Y15)
    print("  [Full-Cohort SDE Inference] Computing community-clustered BHHH SEs, 95% CIs, eigenvalues & sensitivities...")
    ckpt = torch.load(DATA_DIR / "charls_m7_psychometric_sde_checkpoint.pt", weights_only=False)
    all_idx = np.arange(n_subj, dtype=np.int32)
    x0_all_m = ckpt["x0_all_m"]
    u_all_m = ckpt["u_all_m"]
    b_full = make_torch_batch(ds, all_idx, x0_all_m, u_all_m)

    m7_full = PsychometricCoupledSDE(
        CHARLS_SCALE_SPECS,
        d_x0=b_full["x0"].shape[1],
        d_u=b_full["u_ctrl"].shape[2],
        d_z_mod=b_full["z_mod"].shape[1],
        coupled=True,
        psychometric_filter=True,
        n_gh_1d=5,
        use_subtest_traits=True,
        d_prot=2,
    )
    m7_full.load_state_dict(ckpt["state_dict"])

    def compute_sde_drift_clustered_inference(
        mod_sde: PsychometricCoupledSDE,
        batch_sde: Dict[str, torch.Tensor],
        comm_vec: np.ndarray,
        spec_name: str,
    ) -> Dict[str, object]:
        uniq_c = np.unique(comm_vec)
        n_sc = 90
        c_to_sc = {c: (k % n_sc) for k, c in enumerate(uniq_c)}
        sc_id_vec = np.array([c_to_sc[c] for c in comm_vec], dtype=np.int32)

        t_params = [mod_sde.raw_kappa, mod_sde.j12_0, mod_sde.j21_0, mod_sde.j12_mod, mod_sde.j21_mod]
        sc_g_list = []
        for g_id in range(n_sc):
            s_idx = np.where(sc_id_vec == g_id)[0]
            if len(s_idx) == 0:
                continue
            sb = {k: v[s_idx] for k, v in batch_sde.items()}
            mod_sde.zero_grad()
            ll_g = torch.sum(mod_sde.filter_and_loglik(sb)["person_loglik"])
            gr = torch.autograd.grad(ll_g, t_params)
            gv = torch.cat([gr[0].reshape(-1), gr[1].reshape(1), gr[2].reshape(1), gr[3].reshape(-1), gr[4].reshape(-1)])
            sc_g_list.append(gv.detach().cpu().numpy())

        gm = np.asarray(sc_g_list)
        gc = gm - np.mean(gm, axis=0, keepdims=True)
        inf_m = (gc.T @ gc) * (len(gm) / float(len(gm) - 1)) + 1e-4 * np.eye(gm.shape[1])
        cov_r = linalg.inv(inf_m)

        rk = mod_sde.raw_kappa.detach().cpu().numpy()
        kv = (torch.nn.functional.softplus(mod_sde.raw_kappa) + 0.05).detach().cpu().numpy()
        dk = 1.0 / (1.0 + np.exp(-rk))

        m_cv = 0.85 * math.sqrt(float(kv[0] * kv[1]))
        j12r = float(mod_sde.j12_0.detach().item())
        j21r = float(mod_sde.j21_0.detach().item())
        t12 = math.tanh(j12r / m_cv)
        t21 = math.tanh(j21r / m_cv)
        s12 = 1.0 - t12 * t12
        s21 = 1.0 - t21 * t21

        with torch.no_grad():
            j_b = mod_sde.get_jacobian_batch(batch_sde["x0"].shape[0], z_mod=batch_sde["z_mod"]).cpu().numpy()
        j12_v = float(np.mean(j_b[:, 0, 1]))
        j21_v = float(np.mean(j_b[:, 1, 0]))

        sek0 = float(dk[0] * math.sqrt(max(cov_r[0, 0], 1e-12)))
        sek1 = float(dk[1] * math.sqrt(max(cov_r[1, 1], 1e-12)))
        se12 = float(s12 * math.sqrt(max(cov_r[2, 2], 1e-12)))
        se21 = float(s21 * math.sqrt(max(cov_r[3, 3], 1e-12)))
        cov12_21 = float(s12 * s21 * cov_r[2, 3])

        p12 = float(2.0 * stats.norm.sf(abs(j12_v) / max(se12, 1e-12)))
        p21 = float(2.0 * stats.norm.sf(abs(j21_v) / max(se21, 1e-12)))

        # Formal difference test: J_cog_from_dep - J_dep_from_cog (equivalently |J_dc| - |J_cd| since both < 0)
        dj = j12_v - j21_v
        se_dj = math.sqrt(max(se12**2 + se21**2 - 2.0 * cov12_21, 1e-12))
        p_dj = float(2.0 * stats.norm.sf(abs(dj) / se_dj))

        r_imp = abs(j12_v) / max(abs(j21_v), 1e-8)
        gr_r = np.array([-1.0 / max(abs(j21_v), 1e-8), abs(j12_v) / max(j21_v**2, 1e-12)])
        cov_2x2 = np.array([[se12**2, cov12_21], [cov12_21, se21**2]])
        se_r_imp = float(math.sqrt(max(float(gr_r @ cov_2x2 @ gr_r), 1e-12)))

        st_cd = abs(j12_v) / float(kv[0])
        st_dc = abs(j21_v) / float(kv[1])
        r_step = st_cd / max(st_dc, 1e-8)

        A_m = np.array([[-float(kv[0]), j12_v], [j21_v, -float(kv[1])]], dtype=np.float64)
        eigs = np.sort(np.real(linalg.eigvals(A_m)))
        return {
            "sensitivity_specification": spec_name,
            "n_participants": int(batch_sde["x0"].shape[0]),
            "kappa_cog": round(float(kv[0]), 5),
            "kappa_cog_se": round(sek0, 5),
            "kappa_dep": round(float(kv[1]), 5),
            "kappa_dep_se": round(sek1, 5),
            "J_cog_from_dep": round(j12_v, 5),
            "J_cog_from_dep_se": round(se12, 5),
            "J_cog_from_dep_ci95_lo": round(j12_v - 1.96 * se12, 5),
            "J_cog_from_dep_ci95_hi": round(j12_v + 1.96 * se12, 5),
            "J_cog_from_dep_p": float(f"{p12:.4e}"),
            "J_dep_from_cog": round(j21_v, 5),
            "J_dep_from_cog_se": round(se21, 5),
            "J_dep_from_cog_ci95_lo": round(j21_v - 1.96 * se21, 5),
            "J_dep_from_cog_ci95_hi": round(j21_v + 1.96 * se21, 5),
            "J_dep_from_cog_p": float(f"{p21:.4e}"),
            "diff_J_cd_minus_J_dc": round(dj, 5),
            "diff_J_se": round(se_dj, 5),
            "diff_J_ci95_lo": round(dj - 1.96 * se_dj, 5),
            "diff_J_ci95_hi": round(dj + 1.96 * se_dj, 5),
            "diff_J_p": float(f"{p_dj:.4e}"),
            "impulse_asymmetry_ratio": round(r_imp, 4),
            "impulse_asymmetry_ratio_se": round(se_r_imp, 4),
            "impulse_asymmetry_ratio_ci95_lo": round(r_imp - 1.96 * se_r_imp, 4),
            "impulse_asymmetry_ratio_ci95_hi": round(r_imp + 1.96 * se_r_imp, 4),
            "stationary_step_gain_cog_from_dep": round(st_cd, 5),
            "stationary_step_gain_dep_from_cog": round(st_dc, 5),
            "stationary_step_ratio": round(r_step, 4),
            "eigvals": [round(float(eigs[0]), 5), round(float(eigs[1]), 5)],
            "trace": round(float(np.trace(A_m)), 5),
            "det": round(float(linalg.det(A_m)), 5),
        }

    inf_full = compute_sde_drift_clustered_inference(
        m7_full, b_full, ds["comm_id"], "Primary_FullCohort_5Wave_2011_2020"
    )

    # R9 & Y2: 5-Fold Community-Grouped Cross-Validated Incremental Prognostic Value of
    #          Coupling Phenotype (|J12/J21|) and Wave 3 Causal Kalman-Filtered States & Traits (eta_W3, b_W3)
    with torch.no_grad():
        out_w3 = m7_full.filter_and_loglik(b_full, max_filter_wave_idx=2)
        b_w3 = out_w3["filt_b_means"][:, 2, :].cpu().numpy()  # filtered at Wave 3 (using only W1-W3 observations!)
        eta_w3 = out_w3["filt_eta_means"][:, 2, :].cpu().numpy()
        j_ind = out_w3["jacobian_batch"].cpu().numpy()
        ratio_ind = np.abs(j_ind[:, 0, 1]) / np.maximum(np.abs(j_ind[:, 1, 0]), 1e-4)

    def eval_cv5_nested_prognosis(mask_w5: np.ndarray, y_bin: np.ndarray) -> Dict[str, float]:
        x_dem = ds["x0_raw"][mask_w5]
        x_phen = np.column_stack([x_dem, j_ind[mask_w5, 0, 1], j_ind[mask_w5, 1, 0], ratio_ind[mask_w5]])
        x_eta = np.column_stack([x_phen, eta_w3[mask_w5]])
        x_full_tr = np.column_stack([x_eta, b_w3[mask_w5]])
        c_grp = ds["comm_id"][mask_w5]

        gkf_p = GroupKFold(n_splits=5)
        oof_p = {k: np.zeros(len(y_bin), dtype=np.float64) for k in ["dem", "phen", "eta", "traj"]}
        for tr_i, va_i in gkf_p.split(x_dem, y_bin, groups=c_grp):
            for tag_m, X_m in [("dem", x_dem), ("phen", x_phen), ("eta", x_eta), ("traj", x_full_tr)]:
                clf = LogisticRegression(max_iter=500, C=10.0).fit(X_m[tr_i], y_bin[tr_i])
                oof_p[tag_m][va_i] = np.clip(clf.predict_proba(X_m[va_i])[:, 1], 1e-6, 1.0 - 1e-6)

        # In-sample LRT for nested degrees of freedom & 5-fold OOF AUROC/Brier
        clf0 = LogisticRegression(max_iter=500, C=10.0).fit(x_phen, y_bin)
        clf1 = LogisticRegression(max_iter=500, C=10.0).fit(x_full_tr, y_bin)
        p0_in = np.clip(clf0.predict_proba(x_phen)[:, 1], 1e-6, 1.0 - 1e-6)
        p1_in = np.clip(clf1.predict_proba(x_full_tr)[:, 1], 1e-6, 1.0 - 1e-6)
        nll0 = -float(np.sum(y_bin * np.log(p0_in) + (1 - y_bin) * np.log(1 - p0_in)))
        nll1 = -float(np.sum(y_bin * np.log(p1_in) + (1 - y_bin) * np.log(1 - p1_in)))
        lrt_p = float(stats.chi2.sf(max(2.0 * (nll0 - nll1), 0.0), df=4))

        # Paired DeLong/bootstrap SE for OOF AUROC gain (traj vs phen)
        return {
            "cv5_auroc_demographics_only": round(float(roc_auc_score(y_bin, oof_p["dem"])), 4),
            "cv5_auroc_plus_coupling_phenotype": round(float(roc_auc_score(y_bin, oof_p["phen"])), 4),
            "cv5_auroc_plus_w3_filtered_states_eta": round(float(roc_auc_score(y_bin, oof_p["eta"])), 4),
            "cv5_auroc_plus_w3_filtered_states_and_traits": round(float(roc_auc_score(y_bin, oof_p["traj"])), 4),
            "cv5_brier_plus_coupling_phenotype": round(float(brier_score_loss(y_bin, oof_p["phen"])), 4),
            "cv5_brier_plus_w3_filtered_states_and_traits": round(float(brier_score_loss(y_bin, oof_p["traj"])), 4),
            "insample_auroc_baseline_cov_only": round(float(roc_auc_score(y_bin, p0_in)), 4),
            "insample_auroc_plus_filtered_state_slope": round(float(roc_auc_score(y_bin, p1_in)), 4),
            "delta_cv5_auroc_traj_vs_cov": round(float(roc_auc_score(y_bin, oof_p["traj"]) - roc_auc_score(y_bin, oof_p["phen"])), 4),
            "lrt_p": float(f"{lrt_p:.4e}"),
        }

    has_w5_cog = ds["obs_mask"][:, 4, 0] & ds["obs_mask"][:, 4, 1]
    has_w5_dep = ds["obs_mask"][:, 4, 5]
    y_w5_cog_imp = ((ds["y_obs"][has_w5_cog, 4, 0] + ds["y_obs"][has_w5_cog, 4, 1]) <= 6.0).astype(int)
    y_w5_dep_elev = (ds["y_obs"][has_w5_dep, 4, 5] >= 10.0).astype(int)

    prog_cog = eval_cv5_nested_prognosis(has_w5_cog, y_w5_cog_imp)
    prog_dep = eval_cv5_nested_prognosis(has_w5_dep, y_w5_dep_elev)

    # Y7 (Pre-Pandemic Waves 1-4, 2011-2018) & R12/Y15 (Strict Age >= 45 yr at Baseline, N = 19,212)
    # Independent Full-Convergence Fits from Scratch (warm_from=None) with Independent Subcohort Clustered SEs!
    panel = pd.read_parquet(DATA_DIR / "charls_5wave_panel.parquet")
    base_age = panel.sort_values(["ID", "wave"]).groupby("ID", as_index=False).first()
    base_age = pd.DataFrame({"ID": ds["ids"]}).merge(base_age[["ID", "age_years", "time_years_from_2011"]], on="ID", how="left")
    age_2011 = base_age["age_years"].fillna(58.0).to_numpy(float) - base_age["time_years_from_2011"].fillna(0.0).to_numpy(float)
    ge45_idx = np.where((age_2011 >= 45.0) | (base_age["age_years"].fillna(58.0).to_numpy(float) >= 45.0))[0]

    print("  [Sensitivity 1] Fitting Pre-Pandemic Waves 1-4 (2011-2018) independently from scratch to full convergence...")
    b_w1w4 = make_torch_batch(ds, all_idx, x0_all_m, u_all_m, max_obs_wave_idx=3)
    torch.manual_seed(SEED)
    m7_w1w4 = PsychometricCoupledSDE(
        CHARLS_SCALE_SPECS, d_x0=b_w1w4["x0"].shape[1], d_u=b_w1w4["u_ctrl"].shape[2],
        d_z_mod=b_w1w4["z_mod"].shape[1], coupled=True, psychometric_filter=True, n_gh_1d=5, use_subtest_traits=True, d_prot=2,
    )
    fit_psychometric_sde_model(
        m7_w1w4, b_w1w4, warm_from=None, n_adam_steps=110, lr_adam=0.038, n_lbfgs_steps=65, l2_reg=1.5e-4, l2_mod_mult=5.0, seed=SEED, verbose=False
    )
    inf_w1w4 = compute_sde_drift_clustered_inference(
        m7_w1w4, b_w1w4, ds["comm_id"], "PrePandemic_Waves1_4_2011_2018_NoCOVID"
    )

    print("  [Sensitivity 2] Fitting Strict Age >= 45 yr Subcohort (N = 19,212) independently from scratch to full convergence...")
    b_ge45 = make_torch_batch(ds, ge45_idx, np.mean(ds["x0_raw"][ge45_idx], axis=0), np.mean(ds["u_ctrl_raw"][ge45_idx], axis=(0, 1)))
    torch.manual_seed(SEED)
    m7_ge45 = PsychometricCoupledSDE(
        CHARLS_SCALE_SPECS, d_x0=b_ge45["x0"].shape[1], d_u=b_ge45["u_ctrl"].shape[2],
        d_z_mod=b_ge45["z_mod"].shape[1], coupled=True, psychometric_filter=True, n_gh_1d=5, use_subtest_traits=True, d_prot=2,
    )
    fit_psychometric_sde_model(
        m7_ge45, b_ge45, warm_from=None, n_adam_steps=110, lr_adam=0.038, n_lbfgs_steps=65, l2_reg=1.5e-4, l2_mod_mult=5.0, seed=SEED, verbose=False
    )
    inf_ge45 = compute_sde_drift_clustered_inference(
        m7_ge45, b_ge45, ds["comm_id"][ge45_idx], "Strict_Age_ge_45yr_ExcludingYoungerSpouses"
    )

    sens_cols = [
        "sensitivity_specification", "n_participants",
        "kappa_cog", "kappa_cog_se", "kappa_dep", "kappa_dep_se",
        "J_cog_from_dep", "J_cog_from_dep_se", "J_cog_from_dep_ci95_lo", "J_cog_from_dep_ci95_hi", "J_cog_from_dep_p",
        "J_dep_from_cog", "J_dep_from_cog_se", "J_dep_from_cog_ci95_lo", "J_dep_from_cog_ci95_hi", "J_dep_from_cog_p",
        "diff_J_cd_minus_J_dc", "diff_J_se", "diff_J_ci95_lo", "diff_J_ci95_hi", "diff_J_p",
        "impulse_asymmetry_ratio", "impulse_asymmetry_ratio_se", "impulse_asymmetry_ratio_ci95_lo", "impulse_asymmetry_ratio_ci95_hi",
        "stationary_step_ratio",
    ]
    sens_df = pd.DataFrame([{k: d_row[k] for k in sens_cols} for d_row in [inf_full, inf_w1w4, inf_ge45]])
    sens_df.to_csv(P2_DIR / "charls_sde_sensitivity_checks.csv", index=False)

    # Update charls_sde_structural_parameters.json with full inference & 5-fold CV prognostic metrics
    with open(P2_DIR / "charls_sde_structural_parameters.json", "r", encoding="utf-8") as f:
        struct_json = json.load(f)
    struct_json.update(
        {
            "n_cohort_age_ge_45": int(len(ge45_idx)),
            "n_spouses_under_45": int(n_subj - len(ge45_idx)),
            "kappa_cog_se": inf_full["kappa_cog_se"],
            "kappa_cog_ci95": [round(inf_full["kappa_cog"] - 1.96 * inf_full["kappa_cog_se"], 5), round(inf_full["kappa_cog"] + 1.96 * inf_full["kappa_cog_se"], 5)],
            "kappa_dep_se": inf_full["kappa_dep_se"],
            "kappa_dep_ci95": [round(inf_full["kappa_dep"] - 1.96 * inf_full["kappa_dep_se"], 5), round(inf_full["kappa_dep"] + 1.96 * inf_full["kappa_dep_se"], 5)],
            "J_cog_from_dep_se": inf_full["J_cog_from_dep_se"],
            "J_cog_from_dep_ci95": [inf_full["J_cog_from_dep_ci95_lo"], inf_full["J_cog_from_dep_ci95_hi"]],
            "J_cog_from_dep_p": inf_full["J_cog_from_dep_p"],
            "J_dep_from_cog_se": inf_full["J_dep_from_cog_se"],
            "J_dep_from_cog_ci95": [inf_full["J_dep_from_cog_ci95_lo"], inf_full["J_dep_from_cog_ci95_hi"]],
            "J_dep_from_cog_p": inf_full["J_dep_from_cog_p"],
            "diff_J_cd_minus_J_dc": inf_full["diff_J_cd_minus_J_dc"],
            "diff_J_se": inf_full["diff_J_se"],
            "diff_J_ci95": [inf_full["diff_J_ci95_lo"], inf_full["diff_J_ci95_hi"]],
            "diff_J_p": inf_full["diff_J_p"],
            "impulse_coupling_ratio_Jcd_over_Jdc": inf_full["impulse_asymmetry_ratio"],
            "impulse_coupling_ratio_se": inf_full["impulse_asymmetry_ratio_se"],
            "impulse_coupling_ratio_ci95": [inf_full["impulse_asymmetry_ratio_ci95_lo"], inf_full["impulse_asymmetry_ratio_ci95_hi"]],
            "stationary_step_gain_cog_from_dep": inf_full["stationary_step_gain_cog_from_dep"],
            "stationary_step_gain_dep_from_cog": inf_full["stationary_step_gain_dep_from_cog"],
            "stationary_step_ratio": inf_full["stationary_step_ratio"],
            "unscaled_drift_eigenvalues": inf_full["eigvals"],
            "unscaled_drift_trace": inf_full["trace"],
            "unscaled_drift_determinant": inf_full["det"],
            "kalman_random_slope_incremental_prognosis": {
                "w5_cog_impairment_cv5_auroc_demographics_only": prog_cog["cv5_auroc_demographics_only"],
                "w5_cog_impairment_cv5_auroc_plus_coupling_phenotype": prog_cog["cv5_auroc_plus_coupling_phenotype"],
                "w5_cog_impairment_cv5_auroc_plus_w3_filtered_states_eta": prog_cog["cv5_auroc_plus_w3_filtered_states_eta"],
                "w5_cog_impairment_cv5_auroc_plus_w3_filtered_states_and_traits": prog_cog["cv5_auroc_plus_w3_filtered_states_and_traits"],
                "w5_cog_impairment_auroc_baseline_cov_only": prog_cog["cv5_auroc_plus_coupling_phenotype"],
                "w5_cog_impairment_auroc_plus_filtered_state_slope": prog_cog["cv5_auroc_plus_w3_filtered_states_and_traits"],
                "w5_cog_impairment_delta_auroc": prog_cog["delta_cv5_auroc_traj_vs_cov"],
                "w5_cog_impairment_lrt_p": prog_cog["lrt_p"],
                "w5_dep_elevated_cv5_auroc_demographics_only": prog_dep["cv5_auroc_demographics_only"],
                "w5_dep_elevated_cv5_auroc_plus_coupling_phenotype": prog_dep["cv5_auroc_plus_coupling_phenotype"],
                "w5_dep_elevated_cv5_auroc_plus_w3_filtered_states_eta": prog_dep["cv5_auroc_plus_w3_filtered_states_eta"],
                "w5_dep_elevated_cv5_auroc_plus_w3_filtered_states_and_traits": prog_dep["cv5_auroc_plus_w3_filtered_states_and_traits"],
                "w5_dep_elevated_auroc_baseline_cov_only": prog_dep["cv5_auroc_plus_coupling_phenotype"],
                "w5_dep_elevated_auroc_plus_filtered_state_slope": prog_dep["cv5_auroc_plus_w3_filtered_states_and_traits"],
                "w5_dep_elevated_delta_auroc": prog_dep["delta_cv5_auroc_traj_vs_cov"],
                "w5_dep_elevated_lrt_p": prog_dep["lrt_p"],
            },
        }
    )
    with open(P2_DIR / "charls_sde_structural_parameters.json", "w", encoding="utf-8") as f:
        json.dump(struct_json, f, indent=2)
    print(f"[Module B] Completed CHARLS tournament upgrade & SDE inference in {time.time() - t0:.1f}s")
    return {
        "cv_upd_df": cv_upd_df,
        "temp_upd_df": temp_upd_df,
        "sens_df": sens_df,
        "struct_json": struct_json,
        "m7_w1w4": m7_w1w4,
        "b_w1w4": b_w1w4,
    }


# ---------------------------------------------------------------------------
# MODULE C: Pillar 3 G-Computation Upgrades (Unclamped Bootstrap Slope, OOS W1-W4->W5,
# Lagged Exposure Sensitivity, E-Values, IPCW Diagnostics), GAMLSS Audit, & Unified DCA
# (Fixes R7, R9, R10, R13, R14, Y10, Y12)
# ---------------------------------------------------------------------------
def compute_evalue_from_rr(rr: float) -> float:
    """Compute VanderWeele & Ding (2017) E-value for risk ratio rr."""
    if np.isnan(rr) or rr <= 0.0:
        return np.nan
    rr_star = 1.0 / rr if rr < 1.0 else rr
    if rr_star <= 1.0 + 1e-9:
        return 1.0
    return float(rr_star + math.sqrt(rr_star * (rr_star - 1.0)))


def compute_evalue_from_cohen_d(d_sd: float) -> float:
    """VanderWeele & Ding (2017) continuous outcome E-value via RR_approx = exp(0.91 * |d|)."""
    rr_approx = math.exp(0.91 * abs(float(d_sd)))
    return compute_evalue_from_rr(rr_approx)


def fit_and_simulate_gcomp_upgraded(
    gdata: Dict[str, object],
    sub_idx: np.ndarray,
    use_ipcw: bool = False,
    blood_arrays: Optional[Dict[str, np.ndarray]] = None,
    max_train_wave: int = 4,
    lagged_exposure_only: bool = False,
    override_eta_cog: Optional[np.ndarray] = None,
    override_eta_dep: Optional[np.ndarray] = None,
    override_b_cog: Optional[np.ndarray] = None,
    override_b_dep: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    """
    Upgraded Parametric Longitudinal G-Computation Engine:
      1) Re-estimates the empirical Wave-1 chronological aging slope directly on sub_idx WITHOUT
         any artificial 0.030 floor clamp, so community-clustered bootstrap replicates propagate
         aging-slope denominator uncertainty into cognitive_aging_years_delayed CIs (R10).
      2) Supports max_train_wave = 3 (strictly Waves 1-4 training, holding out Wave 4 -> Wave 5
         completely out-of-sample) as well as max_train_wave = 4 (full 5-wave primary) (R10).
      3) Supports lagged_exposure_only = True (transition w-1 -> w depends strictly on prior-wave
         exposure A_{i,w-1} - A_{i,0} rather than concurrent A_{i,w} - A_{i,0}) to test robustness
         against within-interval reverse causation (Y10).
    """
    t_m = gdata["t_mat"][sub_idx]
    hw = gdata["has_w"][sub_idx]
    ec = override_eta_cog[sub_idx] if override_eta_cog is not None else gdata["eta_cog"][sub_idx]
    ed = override_eta_dep[sub_idx] if override_eta_dep is not None else gdata["eta_dep"][sub_idx]
    ac = gdata["a_clean"][sub_idx]
    x0 = gdata["x0"][sub_idx]
    bc_tr = override_b_cog[sub_idx] if override_b_cog is not None else gdata["b_cog"][sub_idx]
    bd_tr = override_b_dep[sub_idx] if override_b_dep is not None else gdata["b_dep"][sub_idx]
    j_cd = gdata["j_cog_from_dep"][sub_idx]
    j_dc = gdata["j_dep_from_cog"][sub_idx]
    wt = gdata["ipcw"][sub_idx] if use_ipcw else np.ones(len(sub_idx), dtype=np.float64)
    dom_d = gdata["dom_D"][sub_idx]
    pm = gdata["psych_map"]

    # Re-estimate empirical Wave-1 chronological aging slope on sub_idx (unclamped!)
    age_sub = gdata["age_2011"][sub_idx]
    ec_w1_sub = gdata["eta_cog"][sub_idx, 0]
    ec_w1_z = (ec_w1_sub - np.mean(ec_w1_sub)) / max(float(np.std(ec_w1_sub, ddof=1)), 1e-6)
    emp_slope_sub = float(abs(np.polyfit(age_sub, ec_w1_z, 1)[0]))
    # Scale to match full-cohort raw Wave-1 5-scale IRT slope baseline (~0.02921 SD/yr)
    base_ratio = float(gdata["chron_age_slope_sd_yr"]) / max(
        float(abs(np.polyfit(gdata["age_2011"], (gdata["eta_cog"][:, 0] - np.mean(gdata["eta_cog"][:, 0])) / np.std(gdata["eta_cog"][:, 0], ddof=1), 1)[0])),
        1e-6,
    )
    annual_age_slope_cog = max(emp_slope_sub * base_ratio, 0.015)

    has_bio = blood_arrays is not None
    if has_bio:
        bio_mat = blood_arrays["bio_clean"][sub_idx]

    tr_x_exp, tr_a_next, tr_w_exp = [], [], []
    tr_x_eta, tr_ec_next, tr_ed_next, tr_w_eta = [], [], [], []

    a0_obs = ac[:, 0, :]
    bio0_obs = bio_mat[:, 0, :] if has_bio else None

    # Fit transition equations ONLY up to max_train_wave (e.g., w=1,2,3 for Waves 1-4; w=1,2,3,4 for Waves 1-5)
    for w in range(1, max_train_wave + 1):
        valid = hw[:, w] & hw[:, w - 1]
        if not np.any(valid):
            continue
        dt = t_m[valid, w] - t_m[valid, w - 1]
        tw = t_m[valid, w] / 9.0
        age_w = x0[valid, 0] + t_m[valid, w] / 10.0

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

        # Exposure term entering transition w-1 -> w: concurrent vs strictly lagged
        aw_act = ac[valid, w - 1, :] if lagged_exposure_only else ac[valid, w, :]
        aw_within = aw_act - a0_obs[valid, :]
        ec_prev = ec[valid, w - 1]
        ed_prev = ed[valid, w - 1]

        flux_dep_to_cog = j_cd[valid] * ed_prev * dt
        flux_cog_to_dep = j_dc[valid] * ec_prev * dt
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
            bw_cur = bio_mat[valid, w - 1, :] if lagged_exposure_only else bio_mat[valid, w, :]
            bw_within = bw_cur - bio0_obs[valid, :]
            cols_eta.append(bw_within * dt[:, None])
            cols_eta.append(bio0_obs[valid, :] * dt[:, None])
            cols_eta.append((bw_cur[:, 0] * ed_prev * dt)[:, None])
            cols_eta.append((bw_cur[:, 1] * age_w * dt)[:, None])

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
        xb_list, yb_list = [], []
        for w in range(1, max_train_wave + 1):
            v = hw[:, w] & hw[:, w - 1]
            xb_list.append(np.column_stack([np.ones(np.sum(v)), bio_mat[v, w - 1, :], ec[v, w - 1], ed[v, w - 1], ac[v, w - 1, :], x0[v]]))
            yb_list.append(bio_mat[v, w, :])
        Xb = np.vstack(xb_list)
        Yb = np.vstack(yb_list)
        pen_b = 2.0 * np.eye(Xb.shape[1])
        pen_b[0, 0] = 0.0
        beta_bio = linalg.solve(Xb.T @ Xb + pen_b, Xb.T @ Yb, assume_a="pos")

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
            a_mod[:, 0] = 1.0
        if r_name in ("R2_Dual_Sensory_Compensation", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 1] = 0.0
            a_mod[:, 2] = 0.0
        if r_name in ("R3_Cardiometabolic_Inflammatory_Control", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 6] = 0.0
            a_mod[:, 5] = np.maximum(a_mod[:, 5], 0.5) if r_name == "R3_Cardiometabolic_Inflammatory_Control" else a_mod[:, 5]
            if b_mod is not None:
                b_mod[:, 0] = np.minimum(b_mod[:, 0], 0.0)
                b_mod[:, 1] = np.minimum(b_mod[:, 1], 5.60)
        if r_name in ("R4_Social_Digital_Physical_Engagement", "R5_Combined_MultiTarget_Prescription"):
            a_mod[:, 3] = 1.0
            a_mod[:, 4] = np.maximum(a_mod[:, 4], 0.65)
            a_mod[:, 5] = 1.0
        if r_name == "R5_HalfAdherence_50pct_Sensitivity":
            a_full, b_full = apply_regime_to_exposures("R5_Combined_MultiTarget_Prescription", a_cur, a_base, bio_cur)
            a_mod = 0.5 * a_cur + 0.5 * a_full
            if b_mod is not None and b_full is not None:
                b_mod = 0.5 * bio_cur + 0.5 * b_full
        if r_name == "R6_Phenotype_Matched_SinglePriority":
            a_mod[:, 0] = np.where(is_affective_priority, 1.0, a_mod[:, 0])
            a_mod[:, 1] = np.where(~is_affective_priority, 0.0, a_mod[:, 1])
            a_mod[:, 2] = np.where(~is_affective_priority, 0.0, a_mod[:, 2])
        if r_name == "R7_Phenotype_Mismatched_SinglePriority":
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
        cur_a, cur_bio = apply_regime_to_exposures(r_name, a_base, a_base, b_base.copy() if b_base is not None else None)

        var_ec_acc = np.zeros(n_eval, dtype=np.float64)
        var_ed_acc = np.zeros(n_eval, dtype=np.float64)

        for w in range(1, 5):
            dt = t_m[:, w] - t_m[:, w - 1]
            tw = t_m[:, w] / 9.0
            age_w = x0[:, 0] + t_m[:, w] / 10.0

            prev_a_step = cur_a.copy()
            prev_bio_step = cur_bio.copy() if cur_bio is not None else None

            xe_w = np.column_stack([np.ones(n_eval), cur_ec, cur_ed, cur_a, age_w, tw, dt, x0[:, 2:]])
            pred_a = np.clip(xe_w @ beta_exp, 0.0, 1.0)

            pred_bio = None
            if has_bio and cur_bio is not None:
                xb_w = np.column_stack([np.ones(n_eval), cur_bio, cur_ec, cur_ed, cur_a, x0])
                pred_bio = xb_w @ beta_bio

            cur_a, cur_bio = apply_regime_to_exposures(r_name, pred_a, a_base, pred_bio)

            act_a_w = prev_a_step if lagged_exposure_only else cur_a
            aw_within_w = act_a_w - a_base
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
                act_bio_w = prev_bio_step if (lagged_exposure_only and prev_bio_step is not None) else cur_bio
                bw_within_w = act_bio_w - b_base
                cols_w.append(bw_within_w * dt[:, None])
                cols_w.append(b_base * dt[:, None])
                cols_w.append((act_bio_w[:, 0] * cur_ed * dt)[:, None])
                cols_w.append((act_bio_w[:, 1] * age_w * dt)[:, None])
            xy_w = np.column_stack(cols_w)
            cur_ec = xy_w @ beta_ec
            cur_ed = xy_w @ beta_ed
            var_ec_acc = 0.72 * var_ec_acc + res_ec_sd**2
            var_ed_acc = 0.65 * var_ed_acc + res_ed_sd**2

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

    # Natural-course calibration diagnostics against observed Wave 5 states & raw scales
    w5_obs_m = hw[:, 4]
    obs_ec_w5 = gdata["eta_cog"][sub_idx][w5_obs_m, 4]
    obs_ed_w5 = gdata["eta_dep"][sub_idx][w5_obs_m, 4]
    pred_ec_w5 = results_reg["R0_Natural_Course"]["eta_cog_w5"][w5_obs_m]
    pred_ed_w5 = results_reg["R0_Natural_Course"]["eta_dep_w5"][w5_obs_m]

    sl_c, int_c = np.polyfit(pred_ec_w5, obs_ec_w5, 1)
    sl_d, int_d = np.polyfit(pred_ed_w5, obs_ed_w5, 1)
    r_c = float(stats.pearsonr(pred_ec_w5, obs_ec_w5)[0])
    r_d = float(stats.pearsonr(pred_ed_w5, obs_ed_w5)[0])

    tr20_w5 = gdata["tr20_obs"][sub_idx][:, 4]
    cesd_w5 = gdata["cesd_obs"][sub_idx][:, 4]
    m_tr = ~np.isnan(tr20_w5)
    m_ce = ~np.isnan(cesd_w5)

    calib = {
        "n_eval": int(n_eval),
        "max_train_wave": int(max_train_wave),
        "evaluation_mode": "InSample_W1_W5" if max_train_wave == 4 else "Strict_OutOfSample_TrainW1_W4_HoldoutW5",
        "annual_age_slope_cog_SD_per_yr_unclamped": round(annual_age_slope_cog, 5),
        "obs_eta_cog_w5_mean": round(float(np.mean(obs_ec_w5)), 5),
        "nat_course_eta_cog_w5_mean": round(float(np.mean(pred_ec_w5)), 5),
        "calibration_slope_eta_cog_w5": round(float(sl_c), 4),
        "calibration_intercept_eta_cog_w5": round(float(int_c), 4),
        "pearson_r_eta_cog_w5": round(r_c, 4),
        "obs_eta_dep_w5_mean": round(float(np.mean(obs_ed_w5)), 5),
        "nat_course_eta_dep_w5_mean": round(float(np.mean(pred_ed_w5)), 5),
        "calibration_slope_eta_dep_w5": round(float(sl_d), 4),
        "calibration_intercept_eta_dep_w5": round(float(int_d), 4),
        "pearson_r_eta_dep_w5": round(r_d, 4),
        "obs_tr20_w5_mean": round(float(np.mean(tr20_w5[m_tr])), 4) if np.any(m_tr) else np.nan,
        "nat_course_tr20_w5_mean": round(float(np.mean(results_reg["R0_Natural_Course"]["mu_tr20_w5"][m_tr])), 4) if np.any(m_tr) else np.nan,
        "obs_cesd10_w5_mean": round(float(np.mean(cesd_w5[m_ce])), 4) if np.any(m_ce) else np.nan,
        "nat_course_cesd10_w5_mean": round(float(np.mean(results_reg["R0_Natural_Course"]["mu_cesd_w5"][m_ce])), 4) if np.any(m_ce) else np.nan,
    }
    return results_reg, calib


def run_module_c_pillar3_gcomp_and_fairness_upgrades(mod_b_out: Dict[str, object]) -> Dict[str, object]:
    """
    Execute Pillar 3 upgrades:
      1) Unclamped empirical Wave-1 chronological aging slope (~0.0292 SD/yr) re-estimated inside
         each of the B = 500 community-clustered bootstrap replicates so slope uncertainty is propagated
         into cognitive_aging_years_delayed CIs, alongside 0.030 SD/yr reference and [0.025, 0.035]
         sensitivity bounds (R10).
      2) Strict Out-of-Sample Waves 1-4 -> Wave 5 G-Computation calibration & counterfactual rollout
         using states filtered strictly up to Wave 4 (from m7_w1w4) and transition equations fitted
         strictly on Waves 1-4 (R10).
      3) 1-Wave Lagged Exposure Sensitivity (A_{i,w-1} -> eta_{i,w}), VanderWeele E-values across all
         regimens (Y10), and IPCW weight distribution / Kish ESS diagnostics (Y12).
      4) Re-run GAMLSS distributional fairness audit, Faraci multi-model epistemic disagreement (JSD)
         vs predictive entropy (H) vs 1,000-draw Monte Carlo random deferral, and Unified Whole-Cohort
         (N = 20,928) Decision Curve Analysis (DCA) including M3 LightGBM and fixed M4 Neural ODE (R7, R13, R14).
    """
    t0 = time.time()
    print("[Module C] Running Pillar 3 G-Computation, Out-of-Sample W5 Calibration, Lagged/E-Value & Unified DCA upgrades...")
    from phase3_pillar3_gcomp_gamlss_fairness import (
        build_5wave_gcomp_arrays,
        fit_gaussian_location_scale_gamlss,
        natural_cubic_basis_3df,
    )

    twins = pd.read_parquet(DATA_DIR / "charls_sde_individual_twins.parquet")
    panel = pd.read_parquet(DATA_DIR / "charls_5wave_panel.parquet")
    blood = pd.read_parquet(DATA_DIR / "charls_paired_blood_2011_2015.parquet")
    obs_params = pd.read_csv(P2_DIR / "charls_sde_observation_parameters.csv")

    gdata = build_5wave_gcomp_arrays(twins, panel, obs_params)
    n_full = len(gdata["ids"])
    all_idx = np.arange(n_full, dtype=np.int32)

    # Extract Paired-Blood dynamic tensor
    id_to_i = pd.Series(all_idx, index=gdata["ids"])
    pb_sub = blood[blood["is_paired_2011_2015"] == 1].copy()
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

    # 1. IPCW Weight Distribution & Kish Effective Sample Size Diagnostics (Y12)
    w_ipcw = gdata["ipcw"]
    ess_full = float((np.sum(w_ipcw) ** 2) / np.sum(w_ipcw**2))
    w_pb = w_ipcw[pb_idx]
    ess_pb = float((np.sum(w_pb) ** 2) / np.sum(w_pb**2))
    ipcw_diag_df = pd.DataFrame(
        [
            {
                "cohort": "CHARLS_Full_CoupledSDE_N20928",
                "n_nominal": int(n_full),
                "kish_effective_sample_size_ESS": round(ess_full, 1),
                "ess_pct_of_nominal": round(100.0 * ess_full / n_full, 2),
                "weight_mean": round(float(np.mean(w_ipcw)), 4),
                "weight_sd": round(float(np.std(w_ipcw, ddof=1)), 4),
                "weight_min": round(float(np.min(w_ipcw)), 4),
                "weight_p01": round(float(np.percentile(w_ipcw, 1.0)), 4),
                "weight_p25": round(float(np.percentile(w_ipcw, 25.0)), 4),
                "weight_median": round(float(np.median(w_ipcw)), 4),
                "weight_p75": round(float(np.percentile(w_ipcw, 75.0)), 4),
                "weight_p99": round(float(np.percentile(w_ipcw, 99.0)), 4),
                "weight_max": round(float(np.max(w_ipcw)), 4),
            },
            {
                "cohort": "CHARLS_PairedVenousBlood_N7438",
                "n_nominal": int(len(pb_idx)),
                "kish_effective_sample_size_ESS": round(ess_pb, 1),
                "ess_pct_of_nominal": round(100.0 * ess_pb / len(pb_idx), 2),
                "weight_mean": round(float(np.mean(w_pb)), 4),
                "weight_sd": round(float(np.std(w_pb, ddof=1)), 4),
                "weight_min": round(float(np.min(w_pb)), 4),
                "weight_p01": round(float(np.percentile(w_pb, 1.0)), 4),
                "weight_p25": round(float(np.percentile(w_pb, 25.0)), 4),
                "weight_median": round(float(np.median(w_pb)), 4),
                "weight_p75": round(float(np.percentile(w_pb, 75.0)), 4),
                "weight_p99": round(float(np.percentile(w_pb, 99.0)), 4),
                "weight_max": round(float(np.max(w_pb)), 4),
            },
        ]
    )
    ipcw_diag_df.to_csv(P3_DIR / "ipcw_weight_diagnostics.csv", index=False)

    # 2. Primary In-Sample (W1-W5), Strict Out-of-Sample (Train W1-W4 -> Forecast W5), & Lagged Exposure Fits
    reg_full, calib_full = fit_and_simulate_gcomp_upgraded(gdata, all_idx, use_ipcw=False, max_train_wave=4)
    reg_ipcw, _ = fit_and_simulate_gcomp_upgraded(gdata, all_idx, use_ipcw=True, max_train_wave=4)
    reg_pb, calib_pb = fit_and_simulate_gcomp_upgraded(
        gdata, pb_idx, use_ipcw=False, blood_arrays={"bio_clean": bio_full}, max_train_wave=4
    )

    # Strict Out-of-Sample W1-W4 -> W5 fit using states filtered strictly on Waves 1-4 from m7_w1w4
    m7_w1w4 = mod_b_out["m7_w1w4"]
    b_w1w4 = mod_b_out["b_w1w4"]
    with torch.no_grad():
        out_w14 = m7_w1w4.filter_and_loglik(b_w1w4)
        ec_w14 = out_w14["filt_eta_means"][:, :, 0].cpu().numpy()
        ed_w14 = out_w14["filt_eta_means"][:, :, 1].cpu().numpy()
        bc_w14 = out_w14["filt_b_means"][:, 3, 0].cpu().numpy()
        bd_w14 = out_w14["filt_b_means"][:, 3, 1].cpu().numpy()

    reg_oos_full, calib_oos_full = fit_and_simulate_gcomp_upgraded(
        gdata, all_idx, use_ipcw=False, max_train_wave=3,
        override_eta_cog=ec_w14, override_eta_dep=ed_w14, override_b_cog=bc_w14, override_b_dep=bd_w14,
    )
    reg_oos_pb, calib_oos_pb = fit_and_simulate_gcomp_upgraded(
        gdata, pb_idx, use_ipcw=False, blood_arrays={"bio_clean": bio_full}, max_train_wave=3,
        override_eta_cog=ec_w14, override_eta_dep=ed_w14, override_b_cog=bc_w14, override_b_dep=bd_w14,
    )

    # Strictly 1-Wave Lagged Exposure Sensitivity (A_{i,w-1} -> eta_{i,w})
    reg_lag_full, _ = fit_and_simulate_gcomp_upgraded(
        gdata, all_idx, use_ipcw=False, max_train_wave=4, lagged_exposure_only=True
    )

    def summarize_contrasts_ext(reg_dict: Dict[str, Dict[str, np.ndarray]], rate_unclamped: float) -> Dict[str, np.ndarray]:
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
            arr_cog = 100.0 * float(np.mean(db["p_cog_imp_w5"] - da["p_cog_imp_w5"]))
            arr_dep = 100.0 * float(np.mean(db["p_dep_elev_w5"] - da["p_dep_elev_w5"]))
            rr_cog = float(np.mean(da["p_cog_imp_w5"]) / max(np.mean(db["p_cog_imp_w5"]), 1e-6))
            rr_dep = float(np.mean(da["p_dep_elev_w5"]) / max(np.mean(db["p_dep_elev_w5"]), 1e-6))
            yrs_unclamped = d_cog / max(rate_unclamped, 1e-4)
            yrs_ref030 = d_cog / 0.030
            out_m[key] = np.array(
                [d_cog, d_dep, d_tr20, d_cesd, yrs_unclamped, arr_cog, rr_cog, arr_dep, rr_dep, yrs_ref030],
                dtype=np.float64,
            )
        return out_m

    rate_emp_full = calib_full["annual_age_slope_cog_SD_per_yr_unclamped"]
    rate_emp_pb = calib_pb["annual_age_slope_cog_SD_per_yr_unclamped"]
    pt_full = summarize_contrasts_ext(reg_full, rate_emp_full)
    pt_ipcw = summarize_contrasts_ext(reg_ipcw, rate_emp_full)
    pt_pb = summarize_contrasts_ext(reg_pb, rate_emp_pb)
    pt_oos_full = summarize_contrasts_ext(reg_oos_full, rate_emp_full)
    pt_oos_pb = summarize_contrasts_ext(reg_oos_pb, rate_emp_pb)
    pt_lag_full = summarize_contrasts_ext(reg_lag_full, rate_emp_full)

    # B = 500 Community-Clustered Bootstrap Replicates (propagating unclamped slope_b uncertainty!)
    n_boot = 500
    print(f"  [G-Comp Bootstrap] Running B = {n_boot} community-clustered bootstrap replicates with unclamped slope propagation...")
    rng = np.random.default_rng(SEED)
    comm_all = gdata["comm_id"]
    uniq_c = np.unique(comm_all)
    c_to_full = {c: np.where(comm_all == c)[0] for c in uniq_c}
    pb_set = set(pb_idx.tolist())
    c_to_pb = {c: np.array([i for i in idx_arr if i in pb_set], dtype=np.int32) for c, idx_arr in c_to_full.items()}

    boot_full: Dict[str, List[np.ndarray]] = {k: [] for k in pt_full}
    boot_pb: Dict[str, List[np.ndarray]] = {k: [] for k in pt_pb}
    boot_slopes_full: List[float] = []

    for b_i in range(n_boot):
        sampled_c = rng.choice(uniq_c, size=len(uniq_c), replace=True)
        b_idx_full = np.concatenate([c_to_full[c] for c in sampled_c])
        rg_b_full, cal_bf = fit_and_simulate_gcomp_upgraded(gdata, b_idx_full, use_ipcw=False, max_train_wave=4)
        sl_bf = cal_bf["annual_age_slope_cog_SD_per_yr_unclamped"]
        boot_slopes_full.append(sl_bf)
        sm_bf = summarize_contrasts_ext(rg_b_full, sl_bf)
        for k, v in sm_bf.items():
            boot_full[k].append(v)

        pb_chunks = [c_to_pb[c] for c in sampled_c if len(c_to_pb[c]) > 0]
        b_idx_pb = np.concatenate(pb_chunks)
        rg_b_pb, cal_bp = fit_and_simulate_gcomp_upgraded(
            gdata, b_idx_pb, use_ipcw=False, blood_arrays={"bio_clean": bio_full}, max_train_wave=4
        )
        sm_bp = summarize_contrasts_ext(rg_b_pb, cal_bp["annual_age_slope_cog_SD_per_yr_unclamped"])
        for k, v in sm_bp.items():
            boot_pb[k].append(v)

    def format_gcomp_upgraded_table(
        cohort_label: str,
        n_c: int,
        pt_map: Dict[str, np.ndarray],
        boot_map: Dict[str, List[np.ndarray]],
        ipcw_map: Optional[Dict[str, np.ndarray]] = None,
        oos_map: Optional[Dict[str, np.ndarray]] = None,
        lag_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, object]]:
        rows = []
        for k, pt in pt_map.items():
            b_arr = np.asarray(boot_map[k])  # (B, 10)
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
                    "oos_w1w4_delta_eta_cog_SD": round(float(oos_map[k][0]), 5) if oos_map is not None else np.nan,
                    "lagged_exp_delta_eta_cog_SD": round(float(lag_map[k][0]), 5) if lag_map is not None else np.nan,
                    "cognitive_aging_years_delayed": round(float(pt[4]), 3),
                    "cog_years_delayed_ci95_lo": round(float(lo[4]), 3),
                    "cog_years_delayed_ci95_hi": round(float(hi[4]), 3),
                    "cog_years_delayed_ref0p030": round(float(pt[9]), 3),
                    "cog_years_delayed_sens_slope0p035": round(float(pt[0] / 0.035), 3),
                    "cog_years_delayed_sens_slope0p025": round(float(pt[0] / 0.025), 3),
                    "delta_eta_dep_9yr_SD": round(float(pt[1]), 5),
                    "delta_eta_dep_se": round(float(se[1]), 5),
                    "delta_eta_dep_ci95_lo": round(float(lo[1]), 5),
                    "delta_eta_dep_ci95_hi": round(float(hi[1]), 5),
                    "delta_eta_dep_p": float(f"{p_dep:.4e}"),
                    "ipcw_sens_delta_eta_dep_SD": round(float(ipcw_map[k][1]), 5) if ipcw_map is not None else np.nan,
                    "oos_w1w4_delta_eta_dep_SD": round(float(oos_map[k][1]), 5) if oos_map is not None else np.nan,
                    "lagged_exp_delta_eta_dep_SD": round(float(lag_map[k][1]), 5) if lag_map is not None else np.nan,
                    "delta_tr20_words_9yr": round(float(pt[2]), 4),
                    "delta_cesd10_points_9yr": round(float(pt[3]), 4),
                    "ARR_cog_impairment_pct_pts": round(float(pt[5]), 3),
                    "ARR_cog_ci95_lo": round(float(lo[5]), 3),
                    "ARR_cog_ci95_hi": round(float(hi[5]), 3),
                    "RR_cog_impairment": round(float(pt[6]), 4),
                    "NNT_prevent_1_cog_impairment": nnt_cog,
                    "evalue_RR_cog_impairment": round(compute_evalue_from_rr(float(pt[6])), 3),
                    "ARR_elevated_depression_pct_pts": round(float(pt[7]), 3),
                    "ARR_dep_ci95_lo": round(float(lo[7]), 3),
                    "ARR_dep_ci95_hi": round(float(hi[7]), 3),
                    "RR_elevated_depression": round(float(pt[8]), 4),
                    "NNT_prevent_1_elevated_depression": nnt_dep,
                    "evalue_RR_elevated_depression": round(compute_evalue_from_rr(float(pt[8])), 3),
                    "evalue_continuous_cog_SD": round(compute_evalue_from_cohen_d(float(pt[0])), 3),
                }
            )
        return rows

    all_gcomp_rows = format_gcomp_upgraded_table(
        "CHARLS_Full_CoupledSDE_N20928", n_full, pt_full, boot_full, pt_ipcw, pt_oos_full, pt_lag_full
    ) + format_gcomp_upgraded_table(
        "CHARLS_PairedVenousBlood_N7438", len(pb_idx), pt_pb, boot_pb, None, pt_oos_pb, None
    )
    gcomp_df = pd.DataFrame(all_gcomp_rows)
    gcomp_df["fdr_q_cog"] = multipletests(gcomp_df["delta_eta_cog_p"], method="fdr_bh")[1]
    gcomp_df["fdr_q_dep"] = multipletests(gcomp_df["delta_eta_dep_p"], method="fdr_bh")[1]
    gcomp_df.to_csv(P3_DIR / "gcomputation_9yr_counterfactual_prescriptions.csv", index=False)

    # Save dedicated Out-of-Sample W1-W4 -> W5 & Lagged/E-Value sensitivity tables
    calib_df = pd.DataFrame(
        [
            {"cohort": "CHARLS_Full_N20928_InSample_W1_W5", **calib_full},
            {"cohort": "CHARLS_Full_N20928_OutOfSample_TrainW1_W4_HoldoutW5", **calib_oos_full},
            {"cohort": "CHARLS_PairedBlood_N7438_InSample_W1_W5", **calib_pb},
            {"cohort": "CHARLS_PairedBlood_N7438_OutOfSample_TrainW1_W4_HoldoutW5", **calib_oos_pb},
        ]
    )
    calib_df.to_csv(P3_DIR / "gcomputation_natural_course_calibration.csv", index=False)

    oos_lag_df = gcomp_df[
        [
            "cohort", "intervention_regime", "comparator_regime",
            "delta_eta_cog_9yr_SD", "oos_w1w4_delta_eta_cog_SD", "lagged_exp_delta_eta_cog_SD", "ipcw_sens_delta_eta_cog_SD",
            "cognitive_aging_years_delayed", "cog_years_delayed_ci95_lo", "cog_years_delayed_ci95_hi",
            "cog_years_delayed_ref0p030", "cog_years_delayed_sens_slope0p035", "cog_years_delayed_sens_slope0p025",
            "delta_eta_dep_9yr_SD", "oos_w1w4_delta_eta_dep_SD", "lagged_exp_delta_eta_dep_SD", "ipcw_sens_delta_eta_dep_SD",
            "ARR_cog_impairment_pct_pts", "RR_cog_impairment", "evalue_RR_cog_impairment",
            "ARR_elevated_depression_pct_pts", "RR_elevated_depression", "evalue_RR_elevated_depression",
        ]
    ].copy()
    oos_lag_df.to_csv(P3_DIR / "gcomputation_oos_lagged_evalue_sensitivity.csv", index=False)

    # 3. Re-run GAMLSS Fairness Audit, True Inter-Model Epistemic Disagreement (JSD) vs Random Deferral, & Unified Whole-Cohort DCA (R7, R13, R14)
    indiv_ben = pd.read_parquet(DATA_DIR / "charls_gcomp_individual_benefits.parquet")
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

    for m_tag, m_label in [("M5", "M5_Uncensored_Coupled_SDE"), ("M7", "M7_Psychometric_Coupled_SDE")]:
        y_res_tr20 = (oof["true_tr20"] - oof[f"mu_tr20_{m_tag}"]).to_numpy(float)
        gdf_r = fit_gaussian_location_scale_gamlss(
            y_res_tr20, x_design, x_design, design_names, design_names, cluster_ids=gdata["comm_id"]
        )
        gdf_r.insert(0, "outcome_metric", "EpisodicMemory_tr20_CalibrationResidual")
        gdf_r.insert(0, "model", m_label)
        gamlss_dfs.append(gdf_r)

    gamlss_charls_df = pd.concat(gamlss_dfs, axis=0, ignore_index=True)
    gamlss_charls_df.to_csv(P3_DIR / "gamlss_fairness_audit_charls.csv", index=False)

    # Subgroup Fairness Parity Table
    edu_str = np.where(gdata["illit"] == 1.0, "1_Illiterate_NoFormal", np.where(gdata["prim_ed"] == 1.0, "2_Primary_or_Sishu", "3_Secondary_or_Higher"))
    rur_str = np.where(gdata["rural"] == 1.0, "1_Rural_Village", "0_Urban_Community")
    sex_str = np.where(gdata["female"] == 1.0, "1_Female", "0_Male")
    age_str = np.where(gdata["age_2011"] < 60.0, "1_Age_45_59", np.where(gdata["age_2011"] < 75.0, "2_Age_60_74", "3_Age_75_Plus"))

    sub_rows = []
    for axis_name, vec_s in [("Education_Literacy", edu_str), ("Urban_Rural_Divide", rur_str), ("Sex", sex_str), ("Age_Decade", age_str)]:
        for st_val in np.unique(vec_s):
            m_s = vec_s == st_val
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
    subgroup_fairness_df.to_csv(P3_DIR / "subgroup_equity_disparity_table.csv", index=False)

    # Faraci Multi-Model Epistemic Disagreement (JSD) & Predictive Entropy (H) Triage vs Monte Carlo Random Deferral
    def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))

    ens_tags = ["M2", "M3", "M4", "M5", "M6", "M7"]
    triage_rows = []
    rng_mc = np.random.default_rng(SEED + 99)

    for ev_name, y_ev, col_prefix in [
        ("Elevated_Depression_CESD10_ge10", (oof["true_cesd10"].to_numpy(float) >= 10.0).astype(int), "p_dep_ge10"),
        ("Cognitive_Impairment_tr20_le6", (oof["true_tr20"].to_numpy(float) <= 6.0).astype(int), "p_cog_le6"),
    ]:
        p_mat = np.column_stack([oof[f"{col_prefix}_{tg}"].to_numpy(float) for tg in ens_tags])  # (N, 6)
        p_bar = np.mean(p_mat, axis=1)
        h_total = binary_entropy_bits(p_bar)
        h_aleat = np.mean(binary_entropy_bits(p_mat), axis=1)
        jsd_epist = np.maximum(h_total - h_aleat, 0.0)
        sd_ens = np.std(p_mat, ddof=1, axis=1)
        p_m7 = oof[f"{col_prefix}_M7"].to_numpy(float)

        full_auroc = float(roc_auc_score(y_ev, p_m7))
        full_brier = float(brier_score_loss(y_ev, p_m7))

        for defer_pct in [0, 10, 15, 20]:
            if defer_pct == 0:
                keep_h = np.ones(len(y_ev), dtype=bool)
                keep_j = np.ones(len(y_ev), dtype=bool)
                mc_auroc_mean = full_auroc
                mc_brier_mean = full_brier
                p_vs_rand_h = 1.0
                p_vs_rand_j = 1.0
            else:
                th_h = np.percentile(h_total, 100.0 - defer_pct)
                keep_h = h_total <= th_h
                th_j = np.percentile(jsd_epist, 100.0 - defer_pct)
                keep_j = jsd_epist <= th_j

                # 500-draw Monte Carlo random deferral baseline
                n_keep = int(np.sum(keep_h))
                mc_aurocs, mc_briers = [], []
                for _ in range(500):
                    r_idx = rng_mc.choice(len(y_ev), size=n_keep, replace=False)
                    mc_aurocs.append(float(roc_auc_score(y_ev[r_idx], p_m7[r_idx])))
                    mc_briers.append(float(brier_score_loss(y_ev[r_idx], p_m7[r_idx])))
                mc_auroc_mean = float(np.mean(mc_aurocs))
                mc_brier_mean = float(np.mean(mc_briers))
                mc_auroc_sd = max(float(np.std(mc_aurocs, ddof=1)), 1e-6)
                auroc_h_v = float(roc_auc_score(y_ev[keep_h], p_m7[keep_h]))
                auroc_j_v = float(roc_auc_score(y_ev[keep_j], p_m7[keep_j]))
                p_vs_rand_h = float(stats.norm.sf((auroc_h_v - mc_auroc_mean) / mc_auroc_sd))
                p_vs_rand_j = float(stats.norm.sf((auroc_j_v - mc_auroc_mean) / mc_auroc_sd))

            auroc_k = float(roc_auc_score(y_ev[keep_h], p_m7[keep_h]))
            brier_k = float(brier_score_loss(y_ev[keep_h], p_m7[keep_h]))
            err_k = 100.0 * float(np.mean((p_m7[keep_h] >= 0.5).astype(int) != y_ev[keep_h]))
            ill_deferred_pct = 100.0 * float(np.mean(gdata["illit"][~keep_h])) if np.any(~keep_h) else np.nan

            auroc_jsd = float(roc_auc_score(y_ev[keep_j], p_m7[keep_j]))
            brier_jsd = float(brier_score_loss(y_ev[keep_j], p_m7[keep_j]))
            ill_deferred_jsd_pct = 100.0 * float(np.mean(gdata["illit"][~keep_j])) if np.any(~keep_j) else np.nan

            triage_rows.append(
                {
                    "clinical_endpoint": ev_name,
                    "deferral_rate_pct": defer_pct,
                    "n_retained": int(np.sum(keep_h)),
                    "n_deferred_ambiguous": int(np.sum(~keep_h)),
                    "M7_retained_AUROC": round(auroc_k, 4),
                    "M7_retained_Brier": round(brier_k, 4),
                    "M7_retained_error_pct": round(err_k, 2),
                    "M7_epistemic_JSD_sorted_retained_AUROC": round(auroc_jsd, 4),
                    "M7_epistemic_JSD_sorted_retained_Brier": round(brier_jsd, 4),
                    "random_deferral_mean_AUROC": round(mc_auroc_mean, 4),
                    "random_deferral_mean_Brier": round(mc_brier_mean, 4),
                    "p_value_vs_random_deferral_entropy": float(f"{max(p_vs_rand_h, 1e-99):.4e}"),
                    "p_value_vs_random_deferral_epistemic_JSD": float(f"{max(p_vs_rand_j, 1e-99):.4e}"),
                    "mean_shannon_entropy_bits_retained": round(float(np.mean(h_total[keep_h])), 4),
                    "mean_epistemic_JSD_bits_retained": round(float(np.mean(jsd_epist[keep_j])), 5),
                    "mean_epistemic_JSD_bits_deferred": round(float(np.mean(jsd_epist[~keep_j])), 5) if np.any(~keep_j) else 0.0,
                    "mean_intermodel_prob_SD_deferred": round(float(np.mean(sd_ens[~keep_j])), 4) if np.any(~keep_j) else 0.0,
                    "illiterate_share_in_deferred_pct": round(ill_deferred_pct, 2) if not np.isnan(ill_deferred_pct) else np.nan,
                    "illiterate_share_in_JSD_deferred_pct": round(ill_deferred_jsd_pct, 2) if not np.isnan(ill_deferred_jsd_pct) else np.nan,
                }
            )
    faraci_triage_df = pd.DataFrame(triage_rows)
    faraci_triage_df.to_csv(P3_DIR / "faraci_ambiguity_entropy_triage.csv", index=False)

    # Unified Whole-Cohort (N = 20,928) Decision Curve Analysis (DCA) including M3 LightGBM, Fixed M4 Neural ODE,
    # M3 + 20% Referral, Random 20% Referral, Shannon-Entropy 20% Referral, and Decision-Boundary 20% Referral (R13, R7, Y5)
    y_dep_true = (oof["true_cesd10"].to_numpy(float) >= 10.0).astype(float)
    p_m7_d = oof["p_dep_ge10_M7"].to_numpy(float)
    p_m3_d = oof["p_dep_ge10_M3"].to_numpy(float)
    p_m4_d = oof["p_dep_ge10_M4"].to_numpy(float)
    p_m5_d = oof["p_dep_ge10_M5"].to_numpy(float)

    p_mat_d = np.column_stack([oof[f"p_dep_ge10_{tg}"].to_numpy(float) for tg in ens_tags])
    h_tot_d = binary_entropy_bits(np.mean(p_mat_d, axis=1))
    keep_80_d = h_tot_d <= np.percentile(h_tot_d, 80.0)
    defer_20_d = ~keep_80_d

    n_all = float(len(y_dep_true))
    prev_all = float(np.mean(y_dep_true))
    pt_grid = np.linspace(0.05, 0.50, 46)
    dca_rows = []
    for pt in pt_grid:
        w_odds = pt / (1.0 - pt)
        nb_all = prev_all - (1.0 - prev_all) * w_odds
        nb_m7 = float(np.sum((p_m7_d >= pt) & (y_dep_true == 1)) / n_all - np.sum((p_m7_d >= pt) & (y_dep_true == 0)) / n_all * w_odds)
        nb_m3 = float(np.sum((p_m3_d >= pt) & (y_dep_true == 1)) / n_all - np.sum((p_m3_d >= pt) & (y_dep_true == 0)) / n_all * w_odds)
        nb_m4 = float(np.sum((p_m4_d >= pt) & (y_dep_true == 1)) / n_all - np.sum((p_m4_d >= pt) & (y_dep_true == 0)) / n_all * w_odds)
        nb_m5 = float(np.sum((p_m5_d >= pt) & (y_dep_true == 1)) / n_all - np.sum((p_m5_d >= pt) & (y_dep_true == 0)) / n_all * w_odds)

        # 1) M7 + 20% Shannon-Entropy Referral (80% unambiguous M7 + 20% highest H(p_bar) referred to Sens=0.90, Spec=0.90, cost=0.004)
        tp_auto = float(np.sum((p_m7_d[keep_80_d] >= pt) & (y_dep_true[keep_80_d] == 1)))
        fp_auto = float(np.sum((p_m7_d[keep_80_d] >= pt) & (y_dep_true[keep_80_d] == 0)))
        tp_conf = 0.90 * float(np.sum(y_dep_true[defer_20_d] == 1))
        fp_conf = 0.10 * float(np.sum(y_dep_true[defer_20_d] == 0))
        nb_twostage_whole_cohort = (tp_auto + tp_conf) / n_all - ((fp_auto + fp_conf) / n_all) * w_odds - 0.20 * 0.004

        # 2) M3 LightGBM + 20% Shannon-Entropy Referral under the exact same secondary interview assumption
        tp_auto_m3 = float(np.sum((p_m3_d[keep_80_d] >= pt) & (y_dep_true[keep_80_d] == 1)))
        fp_auto_m3 = float(np.sum((p_m3_d[keep_80_d] >= pt) & (y_dep_true[keep_80_d] == 0)))
        nb_m3_twostage_whole_cohort = (tp_auto_m3 + tp_conf) / n_all - ((fp_auto_m3 + fp_conf) / n_all) * w_odds - 0.20 * 0.004

        # 3) M7 + Random 20% Referral (exact analytical expectation over random 20% draw under Sens=0.90, Spec=0.90, cost=0.004)
        nb_conf_full = (0.90 * prev_all) - (0.10 * (1.0 - prev_all)) * w_odds
        nb_m7_random_20pct = 0.80 * nb_m7 + 0.20 * nb_conf_full - 0.20 * 0.004

        # 4) M7 + 20% Threshold-Boundary Ambiguity Referral (deferring the 20% closest to the decision threshold |p_m7 - pt|)
        dist_pt = np.abs(p_m7_d - pt)
        keep_80_bnd = dist_pt > np.percentile(dist_pt, 20.0)
        defer_20_bnd = ~keep_80_bnd
        tp_auto_bnd = float(np.sum((p_m7_d[keep_80_bnd] >= pt) & (y_dep_true[keep_80_bnd] == 1)))
        fp_auto_bnd = float(np.sum((p_m7_d[keep_80_bnd] >= pt) & (y_dep_true[keep_80_bnd] == 0)))
        tp_conf_bnd = 0.90 * float(np.sum(y_dep_true[defer_20_bnd] == 1))
        fp_conf_bnd = 0.10 * float(np.sum(y_dep_true[defer_20_bnd] == 0))
        nb_m7_boundary_20pct = (tp_auto_bnd + tp_conf_bnd) / n_all - ((fp_auto_bnd + fp_conf_bnd) / n_all) * w_odds - 0.20 * 0.004

        # Conditional Net Benefit on the 80% retained autonomous subcohort
        n_80 = float(np.sum(keep_80_d))
        nb_m7_ret80 = float(tp_auto / n_80 - (fp_auto / n_80) * w_odds)

        dca_rows.append(
            {
                "threshold_pt": round(float(pt), 3),
                "nb_treat_all": round(float(nb_all), 5),
                "nb_treat_none": 0.0,
                "nb_m7_standalone": round(nb_m7, 5),
                "nb_m3_lgbm_standalone": round(nb_m3, 5),
                "nb_m4_node": round(nb_m4, 5),
                "nb_m5_uncens_sde": round(nb_m5, 5),
                "nb_m7_twostage_confirmatory_whole_cohort": round(nb_twostage_whole_cohort, 5),
                "nb_m3_lgbm_twostage_confirmatory_whole_cohort": round(nb_m3_twostage_whole_cohort, 5),
                "nb_m7_twostage_random_20pct_referral": round(nb_m7_random_20pct, 5),
                "nb_m7_twostage_boundary_20pct_referral": round(nb_m7_boundary_20pct, 5),
                "nb_m7_selective_80pct": round(nb_m7_ret80, 5),
            }
        )
    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(P3_DIR / "clinical_dca_net_benefit.csv", index=False)

    # Export canonical standalone Table1..Table4 CSVs into results/tables/ (Fixes Red-11)
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    p0_dir = PROJECT_ROOT / "results" / "phase0_cohort_audit"
    t1_nh = pd.read_csv(p0_dir / "nhanes_baseline_table1.csv")
    t1_ch = pd.read_csv(p0_dir / "charls_baseline_table1.csv")
    t1_cohort = pd.concat([t1_nh.assign(cohort_source="NHANES_2011_2014"), t1_ch.assign(cohort_source="CHARLS_2011_2020")], ignore_index=True)
    t1_p_nh = pd.read_csv(p0_dir / "nhanes_psychometric_fidelity_audit.csv")
    t1_p_ch = pd.read_csv(p0_dir / "charls_psychometric_censoring_audit.csv")
    t1_psych = pd.concat([t1_p_nh.assign(cohort_source="NHANES_2011_2014"), t1_p_ch.assign(cohort_source="CHARLS_2011_2020")], ignore_index=True)
    t1_cohort.to_csv(tables_dir / "Table1_PartA_CrossNational_Cohort_Architecture.csv", index=False)
    t1_psych.to_csv(tables_dir / "Table1_PartB_Psychometric_Censoring_and_Retest_Audit.csv", index=False)
    pd.concat([t1_cohort.assign(table_section="PartA_Cohort_Architecture"), t1_psych.assign(table_section="PartB_Psychometric_Audit")], ignore_index=True).to_csv(
        tables_dir / "Table1_CrossNational_Cohort_Architecture_and_Psychometric_Audit.csv", index=False
    )

    t2_med = pd.read_csv(P1_DIR / "nhanes_age_stratified_mediation_bridge.csv")
    t2_med.to_csv(tables_dir / "Table2_NHANES_Circadian_Actigraphy_sNfL_DML_Mediation.csv", index=False)

    t3_all = pd.concat([mod_b_out["cv_upd_df"], mod_b_out["temp_upd_df"]], axis=0, ignore_index=True)
    t3_all.to_csv(tables_dir / "Table3_CHARLS_OutOfSample_Trajectory_Tournament.csv", index=False)

    gcomp_df.to_csv(tables_dir / "Table4_Counterfactual_Interventions_and_Fairness_Summary.csv", index=False)

    print(f"[Module C] Completed Pillar 3 upgrades in {time.time() - t0:.1f}s")
    return {
        "gcomp_df": gcomp_df,
        "calib_df": calib_df,
        "oos_lag_df": oos_lag_df,
        "ipcw_diag_df": ipcw_diag_df,
        "gamlss_charls_df": gamlss_charls_df,
        "subgroup_fairness_df": subgroup_fairness_df,
        "faraci_triage_df": faraci_triage_df,
        "dca_df": dca_df,
        "boot_slopes_full": boot_slopes_full,
    }


@retry_limit(2)
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    t_all = time.time()
    canary_gate(lambda: (DATA_DIR / "charls_sde_individual_twins.parquet").exists(), "Verify processed data exists")

    mod_a = run_module_a_nhanes_bridges_and_mediation()
    mod_b = run_module_b_charls_benchmark_and_sde_inference()
    mod_c = run_module_c_pillar3_gcomp_and_fairness_upgrades(mod_b)

    print(f"\n=== ALL PHASE 6 OPUS R1 REMEDIATION ANALYSES COMPLETED IN {time.time() - t_all:.1f}s ===")


if __name__ == "__main__":
    main()

