#!/usr/bin/env python3
"""
Phase 2 — Pillar 2 (Part B): Real-World 9-Year 5-Wave CHARLS Benchmark (7 Model Families),
Paired Venous Blood Jacobian Moderation (N = 7,438), & NHANES External Construct Validation
============================================================================================
Target Project: PsyTwin-SDE (npj Digital Medicine Collection)
Pre-Specified in: docs/SAP_v1_FROZEN.md (§3.1–3.3 & §5)
Governance Rules (CLAUDE.md & lab-constitution):
  - STRICT READ-ONLY RAW DATA: Reads exclusively from data_processed/; writes exclusively
    inside data_processed/ and results/phase2_pillar2_sde/.
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.stats.multitest import multipletests
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from psychometric_sde_core import (
    CHARLS_SCALE_SPECS,
    PsychometricCoupledSDE,
    ScaleSpec,
    exact_censored_gaussian_nll_and_crps_np,
    fit_psychometric_sde_model,
)

SEED = 20270507
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_processed"
OUT_DIR = PROJECT_ROOT / "results" / "phase2_pillar2_sde"


# ---------------------------------------------------------------------------
# 1. Construct Dense 5-Wave CHARLS Tensor Dataset & Stabilized IPCW Weights
# ---------------------------------------------------------------------------
def build_charls_tensor_dataset() -> Dict[str, object]:
    panel = pd.read_parquet(DATA_DIR / "charls_5wave_panel.parquet")
    persons = pd.read_parquet(DATA_DIR / "charls_participants_summary.parquet")
    blood = pd.read_parquet(DATA_DIR / "charls_paired_blood_2011_2015.parquet")

    # Primary Pillar 2 Coupled SDE Cohort: >= 2 waves with BOTH tr20 & CES-D10 observed (N = 20,928)
    sde_pers = persons[persons["n_waves_coupled_tr20_cesd"] >= 2].sort_values("ID").reset_index(drop=True)
    ids = sde_pers["ID"].to_numpy()
    n_subj = len(ids)
    id_to_idx = pd.Series(np.arange(n_subj, dtype=np.int32), index=ids)

    sub_p = panel[panel["ID"].isin(ids)].sort_values(["ID", "wave"]).reset_index(drop=True)
    r_idx = sub_p["ID"].map(id_to_idx).to_numpy()
    w_idx = (sub_p["wave"].to_numpy(dtype=np.int32) - 1)  # 0..4

    n_w = 5
    n_d = len(CHARLS_SCALE_SPECS)
    default_t = np.array([0.0, 2.0, 4.0, 7.0, 9.1], dtype=np.float64)
    t_years = np.tile(default_t, (n_subj, 1))
    in_wave = np.zeros((n_subj, n_w), dtype=bool)
    in_wave[r_idx, w_idx] = True

    t_obs_vals = sub_p["time_years_from_2011"].to_numpy(dtype=np.float64)
    t_years[r_idx, w_idx] = t_obs_vals

    # Ensure strictly monotonic grid timestamps across unobserved intermediate waves so semigroup dt1+dt2 == dt_total
    for i in range(n_subj):
        act_w = np.where(in_wave[i])[0]
        if len(act_w) == 0:
            continue
        # Extrapolate before first active wave / after last active wave using standard wave spacing
        first_w = act_w[0]
        for w in range(first_w - 1, -1, -1):
            t_years[i, w] = t_years[i, w + 1] - (default_t[w + 1] - default_t[w])
        last_w = act_w[-1]
        for w in range(last_w + 1, n_w):
            t_years[i, w] = t_years[i, w - 1] + (default_t[w] - default_t[w - 1])
        # Linearly interpolate any skipped interior wave between flanking active waves
        for k_pos in range(len(act_w) - 1):
            wa, wb = act_w[k_pos], act_w[k_pos + 1]
            if wb - wa > 1:
                for wm in range(wa + 1, wb):
                    frac = (default_t[wm] - default_t[wa]) / (default_t[wb] - default_t[wa])
                    t_years[i, wm] = t_years[i, wa] + frac * (t_years[i, wb] - t_years[i, wa])
        # Enforce strict minimum spacing 0.05 yr
        for w in range(1, n_w):
            if t_years[i, w] <= t_years[i, w - 1] + 0.05:
                t_years[i, w] = t_years[i, w - 1] + 0.05

    y_obs = np.zeros((n_subj, n_w, n_d), dtype=np.float64)
    obs_mask = np.zeros((n_subj, n_w, n_d), dtype=bool)
    k_prior = np.zeros((n_subj, n_w, n_d), dtype=np.float64)

    for d, spec in enumerate(CHARLS_SCALE_SPECS):
        vals = sub_p[spec.name].to_numpy(dtype=np.float64)
        valid_v = ~np.isnan(vals)
        y_obs[r_idx[valid_v], w_idx[valid_v], d] = vals[valid_v]
        obs_mask[r_idx[valid_v], w_idx[valid_v], d] = True
        k_col = "k_prior_cesd_tests" if spec.name == "cesd10" else "k_prior_cog_tests"
        k_prior[r_idx, w_idx, d] = sub_p[k_col].to_numpy(dtype=np.float64)

    # Recompute exact cumulative k_prior across waves 0..4 so unobserved waves also carry exact prior count
    for w in range(1, n_w):
        cog_any_prev = np.any(obs_mask[:, :w, 0:5], axis=2)  # (N, w)
        k_cog_w = np.sum(cog_any_prev, axis=1).astype(np.float64)
        k_dep_w = np.sum(obs_mask[:, :w, 5], axis=1).astype(np.float64)
        k_prior[:, w, 0:5] = k_cog_w[:, None]
        k_prior[:, w, 5] = k_dep_w

    # Baseline first-observed row per participant for x0 & IPCW
    base_rows = sub_p.groupby("ID", as_index=False).first()
    base_rows = pd.DataFrame({"ID": ids}).merge(base_rows, on="ID", how="left")

    age0 = base_rows["age_years"].fillna(base_rows["age_years"].median()).to_numpy(dtype=np.float64)
    age0_c = (age0 - 58.0) / 10.0
    female = base_rows["female"].fillna(0.5).to_numpy(dtype=np.float64)
    illit = base_rows["illiterate_no_formal"].fillna(0.0).to_numpy(dtype=np.float64)
    prim_ed = (base_rows["education_3stratum"] == "2_Primary_or_Sishu").astype(np.float64).to_numpy()
    rural = base_rows["rural_village"].fillna(0.5).to_numpy(dtype=np.float64)
    unmar = base_rows["unmarried_or_widowed"].fillna(0.0).to_numpy(dtype=np.float64)
    cperc0 = base_rows["log_hh_cperc"].fillna(base_rows["log_hh_cperc"].median()).to_numpy(dtype=np.float64)
    cperc0_z = (cperc0 - np.mean(cperc0)) / max(np.std(cperc0, ddof=1), 1e-6)

    x0_raw = np.column_stack([age0_c, age0_c**2, female, illit, prim_ed, rural, unmar, cperc0_z])

    # Time-varying control drivers u_ctrl (N, 5, 8):
    # [age_w_c, optimal_sleep, poor_vision, poor_hearing, social_active, internet_use, phys_active, cardiometab_count]
    u_ctrl = np.zeros((n_subj, n_w, 8), dtype=np.float64)
    # Default fill with baseline/cohort medians, then overwrite observed wave entries & forward-fill within person
    u_raw_panel = np.zeros((len(sub_p), 8), dtype=np.float64)
    u_raw_panel[:, 0] = (sub_p["age_years"].fillna(58.0).to_numpy(float) - 58.0) / 10.0
    u_raw_panel[:, 1] = sub_p["optimal_sleep_6_to_85h"].fillna(0.64).to_numpy(float)
    u_raw_panel[:, 2] = sub_p["poor_vision"].fillna(0.30).to_numpy(float)
    u_raw_panel[:, 3] = sub_p["poor_hearing"].fillna(0.11).to_numpy(float)
    u_raw_panel[:, 4] = sub_p["social_active"].fillna(0.50).to_numpy(float)
    u_raw_panel[:, 5] = sub_p["internet_use"].fillna(0.07).to_numpy(float)
    u_raw_panel[:, 6] = sub_p["phys_active_modvig"].fillna(0.55).to_numpy(float)
    cm_sum = (
        sub_p["hibpe"].fillna(0.0)
        + sub_p["diabe"].fillna(0.0)
        + sub_p["hearte"].fillna(0.0)
        + sub_p["stroke"].fillna(0.0)
        + sub_p["dyslipe"].fillna(0.0)
    ).to_numpy(float)
    u_raw_panel[:, 7] = cm_sum / 2.0

    u_has = np.zeros((n_subj, n_w), dtype=bool)
    u_ctrl[r_idx, w_idx, :] = u_raw_panel
    u_has[r_idx, w_idx] = True
    for i in range(n_subj):
        act_w = np.where(u_has[i])[0]
        if len(act_w) == 0:
            continue
        for w in range(act_w[0] - 1, -1, -1):
            u_ctrl[i, w, :] = u_ctrl[i, w + 1, :]
            u_ctrl[i, w, 0] = age0_c[i] + (t_years[i, w] - t_years[i, act_w[0]]) / 10.0
        for w in range(act_w[0] + 1, n_w):
            if not u_has[i, w]:
                u_ctrl[i, w, :] = u_ctrl[i, w - 1, :]
                u_ctrl[i, w, 0] = u_ctrl[i, w - 1, 0] + (t_years[i, w] - t_years[i, w - 1]) / 10.0

    # Individualized Jacobian moderators z_mod_demo (N, 4): [age0_c, female_c, illit_c, rural_c]
    z_mod_demo = np.column_stack(
        [
            age0_c - np.mean(age0_c),
            female - np.mean(female),
            illit - np.mean(illit),
            rural - np.mean(rural),
        ]
    )

    # Stabilized IPCW weights for competing mortality & Wave-5 non-retention (§1.3 of SAP_v1_FROZEN.md)
    died_fu = sde_pers["died_followup"].fillna(0).to_numpy(int)
    retained_w5 = ((sde_pers["inw5"].to_numpy(int) == 1) & (died_fu == 0)).astype(int)
    tr20_b = base_rows["tr20"].fillna(base_rows["tr20"].median()).to_numpy(float)
    cesd_b = base_rows["cesd10"].fillna(base_rows["cesd10"].median()).to_numpy(float)
    adl_b = base_rows["adlab_c"].fillna(0.0).to_numpy(float)
    smk_b = base_rows["smoken"].fillna(0.0).to_numpy(float)
    x_ipcw = np.column_stack([x0_raw, (tr20_b - 10.0) / 4.0, (cesd_b - 8.0) / 6.0, adl_b, smk_b, u_ctrl[:, 0, 7]])
    clf_ipcw = LogisticRegression(max_iter=500, C=1.0, random_state=SEED)
    clf_ipcw.fit(x_ipcw, retained_w5)
    p_ret = np.clip(clf_ipcw.predict_proba(x_ipcw)[:, 1], 0.15, 0.98)
    p_marg = float(np.mean(retained_w5))
    ipcw = np.where(retained_w5 == 1, p_marg / p_ret, (1.0 - p_marg) / (1.0 - p_ret))
    ipcw = np.clip(ipcw, 0.30, 3.50)
    ipcw = ipcw / np.mean(ipcw)

    # Community ID for strict community-grouped 5-fold CV
    comm_id = sde_pers["communityID"].fillna("").astype(str).to_numpy()
    comm_id = np.where(comm_id == "", sde_pers["ID"].str[:7].to_numpy(), comm_id)

    # Final observed coupled wave index per participant (for Split A held-out final wave evaluation)
    coupled_mask = (obs_mask[:, :, 0] & obs_mask[:, :, 1] & obs_mask[:, :, 5])  # tr20 (imrc+dlrc) & cesd10
    last_coupled_wave = np.zeros(n_subj, dtype=np.int32)
    n_coupled_waves = np.sum(coupled_mask, axis=1)
    for i in range(n_subj):
        cw = np.where(coupled_mask[i])[0]
        last_coupled_wave[i] = int(cw[-1]) if len(cw) > 0 else (n_w - 1)

    return {
        "ids": ids,
        "persons": sde_pers,
        "blood": blood,
        "comm_id": comm_id,
        "t_years": t_years,
        "y_obs": y_obs,
        "obs_mask": obs_mask,
        "k_prior": k_prior,
        "x0_raw": x0_raw,
        "u_ctrl_raw": u_ctrl,
        "illit": illit,
        "z_mod_demo": z_mod_demo,
        "ipcw": ipcw,
        "last_coupled_wave": last_coupled_wave,
        "n_coupled_waves": n_coupled_waves,
    }


def make_torch_batch(
    ds: Dict[str, object],
    idx: np.ndarray,
    x0_mean: np.ndarray,
    u_mean: np.ndarray,
    max_obs_wave_idx: Optional[int] = None,
    freeze_future_from_wave_idx: Optional[int] = None,
    mask_person_last_wave: bool = False,
) -> Dict[str, torch.Tensor]:
    """Slice participant subset idx, center covariates using training-fold means, and optionally mask/freeze future waves."""
    m_sub = ds["obs_mask"][idx].copy()
    u_sub = ds["u_ctrl_raw"][idx].copy()
    t_sub = ds["t_years"][idx].copy()
    k_sub = ds["k_prior"][idx].copy()

    if max_obs_wave_idx is not None:
        m_sub[:, max_obs_wave_idx + 1 :, :] = False

    if freeze_future_from_wave_idx is not None:
        fw = freeze_future_from_wave_idx
        for w in range(fw + 1, u_sub.shape[1]):
            u_sub[:, w, 1:] = u_sub[:, fw, 1:]
            k_sub[:, w, :] = k_sub[:, fw, :]

    if mask_person_last_wave:
        lw = ds["last_coupled_wave"][idx]
        for local_i, w_last in enumerate(lw):
            m_sub[local_i, w_last:, :] = False

    n_b, n_w, _ = m_sub.shape
    wp = torch.zeros((n_b, n_w, 2), dtype=torch.float64)
    if n_w >= 4:
        wp[:, 3:, 0] = 1.0
    if n_w >= 5:
        wp[:, 4, 1] = 1.0

    return {
        "t_years": torch.tensor(t_sub, dtype=torch.float64),
        "y_obs": torch.tensor(ds["y_obs"][idx], dtype=torch.float64),
        "obs_mask": torch.tensor(m_sub, dtype=torch.bool),
        "k_prior": torch.tensor(k_sub, dtype=torch.float64),
        "x0": torch.tensor(ds["x0_raw"][idx] - x0_mean[None, :], dtype=torch.float64),
        "u_ctrl": torch.tensor(u_sub - u_mean[None, None, :], dtype=torch.float64),
        "illit": torch.tensor(ds["illit"][idx], dtype=torch.float64),
        "z_mod": torch.tensor(ds["z_mod_demo"][idx], dtype=torch.float64),
        "w_prot": wp,
    }


# ---------------------------------------------------------------------------
# 2. Baseline Model Families (M1 Persistence/AgeNorm, M2 LMM BLUP, M3 LightGBM, M4 Naive Neural ODE)
# ---------------------------------------------------------------------------
def extract_transition_pairs(
    ds: Dict[str, object],
    idx: np.ndarray,
    max_target_wave_idx: Optional[int] = None,
    only_person_last_wave: bool = False,
    fixed_origin_max_wave: Optional[int] = None,
    fixed_target_wave: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Build autoregressive transition tuples (participant i, last observed history up to w_prev -> target wave w_targ)
    for tabular/mixed-effects benchmark models M1, M2, M3.
    """
    y = ds["y_obs"]
    m = ds["obs_mask"]
    t = ds["t_years"]
    x0 = ds["x0_raw"]
    u = ds["u_ctrl_raw"]
    n_d = y.shape[2]

    feat_list = []
    targ_y_list = []
    targ_m_list = []
    subj_list = []
    wtarg_list = []
    dt_list = []
    last_y_list = []

    for i in idx:
        if fixed_target_wave is not None and fixed_origin_max_wave is not None:
            # Split B strict temporal holdout: history <= fixed_origin_max_wave -> forecast fixed_target_wave
            if not np.any(m[i, fixed_target_wave, :]):
                continue
            hist_waves = [w for w in range(fixed_origin_max_wave + 1) if np.any(m[i, w, :])]
            if not hist_waves:
                continue
            pairs = [(hist_waves, fixed_target_wave)]
        elif only_person_last_wave:
            w_t = int(ds["last_coupled_wave"][i])
            hist_waves = [w for w in range(w_t) if np.any(m[i, w, :])]
            if not hist_waves:
                continue
            pairs = [(hist_waves, w_t)]
        else:
            max_w = max_target_wave_idx if max_target_wave_idx is not None else 4
            act_waves = [w for w in range(max_w + 1) if np.any(m[i, w, :])]
            pairs = [(act_waves[:pos], act_waves[pos]) for pos in range(1, len(act_waves))]

        for hist_w, w_t in pairs:
            w_last = hist_w[-1]
            dt = t[i, w_t] - t[i, w_last]
            # Last observed value per scale across hist_w (plus indicator & slope if >=2 prior waves)
            last_vals = np.full(n_d, np.nan)
            prev_vals = np.full(n_d, np.nan)
            prev_dt = np.ones(n_d)
            n_hist_d = np.zeros(n_d)
            mean_vals = np.full(n_d, np.nan)
            for d in range(n_d):
                hw_d = [w for w in hist_w if m[i, w, d]]
                n_hist_d[d] = len(hw_d)
                if len(hw_d) >= 1:
                    last_vals[d] = y[i, hw_d[-1], d]
                    mean_vals[d] = float(np.mean([y[i, w, d] for w in hw_d]))
                if len(hw_d) >= 2:
                    prev_vals[d] = y[i, hw_d[-2], d]
                    prev_dt[d] = max(t[i, hw_d[-1]] - t[i, hw_d[-2]], 0.25)
            slope_d = np.where(~np.isnan(prev_vals), (last_vals - prev_vals) / prev_dt, 0.0)
            last_clean = np.where(np.isnan(last_vals), 0.0, last_vals)
            mean_clean = np.where(np.isnan(mean_vals), last_clean, mean_vals)
            has_last = (~np.isnan(last_vals)).astype(float)

            feat_vec = np.concatenate(
                [
                    last_clean,
                    mean_clean,
                    has_last,
                    slope_d,
                    n_hist_d,
                    [dt, t[i, w_t]],
                    x0[i],
                    u[i, w_last],
                ]
            )
            feat_list.append(feat_vec)
            targ_y_list.append(y[i, w_t])
            targ_m_list.append(m[i, w_t])
            subj_list.append(i)
            wtarg_list.append(w_t)
            dt_list.append(dt)
            last_y_list.append(last_vals)

    return {
        "X": np.asarray(feat_list, dtype=np.float64),
        "Y": np.asarray(targ_y_list, dtype=np.float64),
        "M": np.asarray(targ_m_list, dtype=bool),
        "subj_idx": np.asarray(subj_list, dtype=np.int32),
        "w_targ": np.asarray(wtarg_list, dtype=np.int32),
        "dt": np.asarray(dt_list, dtype=np.float64),
        "last_Y": np.asarray(last_y_list, dtype=np.float64),
    }


def predict_m1_persistence_agenorm(
    tr_pairs: Dict[str, np.ndarray],
    te_pairs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """M1: Last-Observation-Carried-Forward + Quadratic Age-Normative Drift per Year."""
    n_te = len(te_pairs["dt"])
    n_d = len(CHARLS_SCALE_SPECS)
    mu_te = np.zeros((n_te, n_d), dtype=np.float64)
    sd_te = np.zeros((n_te, n_d), dtype=np.float64)

    for d, sp in enumerate(CHARLS_SCALE_SPECS):
        valid_tr = tr_pairs["M"][:, d] & (~np.isnan(tr_pairs["last_Y"][:, d]))
        y_tr = tr_pairs["Y"][valid_tr, d]
        y_last_tr = tr_pairs["last_Y"][valid_tr, d]
        dt_tr = tr_pairs["dt"][valid_tr]
        age_tr = tr_pairs["X"][valid_tr, 6 * n_d - 4]  # first x0 col = age0_c

        # Fit annual drift rate ~ 1 + age0_c
        x_dr = np.column_stack([dt_tr, dt_tr * age_tr])
        beta_dr, _, _, _ = linalg.lstsq(x_dr, y_tr - y_last_tr)
        res_tr = (y_tr - y_last_tr) - (x_dr @ beta_dr)
        # Scale residual variance linearly with sqrt(dt)
        sd_unit = float(np.std(res_tr / np.sqrt(np.maximum(dt_tr, 0.5)), ddof=1))
        fallback_mu = float(np.mean(y_tr))
        fallback_sd = max(float(np.std(y_tr, ddof=1)), 0.3)

        dt_te = te_pairs["dt"]
        age_te = te_pairs["X"][:, 6 * n_d - 4]
        y_last_te = te_pairs["last_Y"][:, d]
        has_l = ~np.isnan(y_last_te)
        pred_d = np.where(has_l, y_last_te + beta_dr[0] * dt_te + beta_dr[1] * dt_te * age_te, fallback_mu)
        sd_d = np.where(has_l, sd_unit * np.sqrt(np.maximum(dt_te, 0.5)), fallback_sd)

        if sp.is_binary:
            # Smoothed empirical persistence probability
            p1_given_1 = float(np.mean(y_tr[y_last_tr >= 0.5])) if np.any(y_last_tr >= 0.5) else fallback_mu
            p1_given_0 = float(np.mean(y_tr[y_last_tr < 0.5])) if np.any(y_last_tr < 0.5) else fallback_mu
            p_te = np.where(has_l, np.where(y_last_te >= 0.5, p1_given_1, p1_given_0), fallback_mu)
            mu_te[:, d] = np.clip(p_te, 0.02, 0.98)
            sd_te[:, d] = -1.0  # probability sentinel
        else:
            mu_te[:, d] = pred_d
            sd_te[:, d] = np.maximum(sd_d, 0.25)
    return mu_te, sd_te


def predict_m2_lmm_empirical_bayes(
    ds: Dict[str, object],
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    te_pairs: Dict[str, np.ndarray],
    max_train_wave_idx: Optional[int] = None,
    fixed_origin_max_wave: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M2: Linear Mixed-Effects (Random Intercept + Random Slope per participant: Y_{i,w} = X_{i,w} beta + b_{0i} + b_{1i} t_{i,w} + eps)
    with exact closed-form Empirical Bayes (BLUP) posterior mean & predictive variance on held-out test waves!
    """
    y = ds["y_obs"]
    m = ds["obs_mask"]
    t = ds["t_years"]
    x0 = ds["x0_raw"]
    u = ds["u_ctrl_raw"]
    n_te = len(te_pairs["subj_idx"])
    n_d = len(CHARLS_SCALE_SPECS)
    mu_te = np.zeros((n_te, n_d), dtype=np.float64)
    sd_te = np.zeros((n_te, n_d), dtype=np.float64)

    max_w_tr = max_train_wave_idx if max_train_wave_idx is not None else 4

    def build_design_row(i: int, w: int) -> np.ndarray:
        tw = t[i, w] / 5.0
        return np.concatenate([[1.0, tw, tw**2], x0[i], u[i, w]])

    for d, sp in enumerate(CHARLS_SCALE_SPECS):
        # 1) Estimate fixed effects beta_d & variance components (G_2x2 random [1, t/5], sigma_e^2) on training subjects
        x_rows = []
        y_rows = []
        i_rows = []
        t_rows = []
        for i in tr_idx:
            for w in range(max_w_tr + 1):
                if m[i, w, d]:
                    x_rows.append(build_design_row(i, w))
                    y_rows.append(y[i, w, d])
                    i_rows.append(i)
                    t_rows.append(t[i, w] / 5.0)
        xm = np.asarray(x_rows, dtype=np.float64)
        yv = np.asarray(y_rows, dtype=np.float64)
        iv = np.asarray(i_rows, dtype=np.int32)
        tv = np.asarray(t_rows, dtype=np.float64)

        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(xm, yv)
        beta_d = ridge.coef_
        res = yv - (xm @ beta_d)

        # Method-of-moments REML variance components for b_{0i} + b_{1i}*(t/5)
        df_r = pd.DataFrame({"id": iv, "t": tv, "r": res})
        grp = df_r.groupby("id")
        r_means = grp["r"].mean().to_numpy()
        counts = grp["r"].count().to_numpy()
        tot_var = float(np.var(res, ddof=1))
        within_ss = float(np.sum((df_r["r"] - df_r["id"].map(grp["r"].mean())) ** 2))
        df_within = max(int(len(df_r) - len(counts)), 1)
        sig2_e = max(within_ss / df_within, 0.05 * tot_var)
        sig2_b0 = max(float(np.var(r_means, ddof=1) - sig2_e / np.mean(counts)), 0.05 * tot_var)
        sig2_b1 = 0.18 * sig2_b0
        g_cov = np.array([[sig2_b0, -0.05 * sig2_b0], [-0.05 * sig2_b0, sig2_b1]], dtype=np.float64)
        g_inv = linalg.inv(g_cov + 1e-6 * np.eye(2))

        # 2) Compute exact BLUP conditional posterior for each test prediction tuple (i, w_targ)
        for row_k in range(n_te):
            i = int(te_pairs["subj_idx"][row_k])
            w_t = int(te_pairs["w_targ"][row_k])
            max_h = fixed_origin_max_wave if fixed_origin_max_wave is not None else (w_t - 1)
            hw = [w for w in range(max_h + 1) if m[i, w, d]]

            x_star = build_design_row(i, w_t)
            z_star = np.array([1.0, t[i, w_t] / 5.0], dtype=np.float64)
            mu_fix_star = float(x_star @ beta_d)

            if len(hw) == 0:
                pred_m = mu_fix_star
                pred_v = float(z_star @ g_cov @ z_star + sig2_e)
            else:
                z_h = np.column_stack([np.ones(len(hw)), t[i, hw] / 5.0])  # (m_i, 2)
                x_h = np.asarray([build_design_row(i, w) for w in hw])
                r_h = y[i, hw, d] - (x_h @ beta_d)
                post_prec = g_inv + (z_h.T @ z_h) / sig2_e
                post_cov = linalg.inv(post_prec)
                b_hat = post_cov @ (z_h.T @ r_h) / sig2_e
                pred_m = mu_fix_star + float(z_star @ b_hat)
                pred_v = float(z_star @ post_cov @ z_star + sig2_e)

            if sp.is_binary:
                mu_te[row_k, d] = float(np.clip(pred_m, 0.02, 0.98))
                sd_te[row_k, d] = -1.0
            else:
                mu_te[row_k, d] = pred_m
                sd_te[row_k, d] = math.sqrt(max(pred_v, 0.05))
    return mu_te, sd_te


def predict_m2b_tobit_censored_lmm(
    ds: Dict[str, object],
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    te_pairs: Dict[str, np.ndarray],
    max_train_wave_idx: Optional[int] = None,
    fixed_origin_max_wave: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M2b: Censored (Tobit) Linear Mixed-Effects Model (Random Intercept + Random Slope with
    explicit two-sided Tobit boundary censoring at floor/ceiling & Probit link for binary draw,
    without retest practice-effect offset, literacy heteroscedasticity, or coupled SDE drift).
    """
    y = ds["y_obs"]
    m = ds["obs_mask"]
    t = ds["t_years"]
    x0 = ds["x0_raw"]
    u = ds["u_ctrl_raw"]
    n_te = len(te_pairs["subj_idx"])
    n_d = len(CHARLS_SCALE_SPECS)
    mu_te = np.zeros((n_te, n_d), dtype=np.float64)
    sd_te = np.zeros((n_te, n_d), dtype=np.float64)

    max_w_tr = max_train_wave_idx if max_train_wave_idx is not None else 4

    def build_design_row(i: int, w: int) -> np.ndarray:
        tw = t[i, w] / 5.0
        return np.concatenate([[1.0, tw, tw**2], x0[i], u[i, w]])

    for d, sp in enumerate(CHARLS_SCALE_SPECS):
        x_rows, y_rows, i_rows, t_rows = [], [], [], []
        for i in tr_idx:
            for w in range(max_w_tr + 1):
                if m[i, w, d]:
                    x_rows.append(build_design_row(i, w))
                    y_rows.append(y[i, w, d])
                    i_rows.append(i)
                    t_rows.append(t[i, w] / 5.0)
        xm = np.asarray(x_rows, dtype=np.float64)
        yv = np.asarray(y_rows, dtype=np.float64)
        iv = np.asarray(i_rows, dtype=np.int32)
        tv = np.asarray(t_rows, dtype=np.float64)

        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(xm, yv)
        beta_d = ridge.coef_.copy()
        res0 = yv - (xm @ beta_d)
        sd0 = max(float(np.std(res0, ddof=1)), 0.4)

        # Iterative EM / Tobit latent imputation for boundary-censored observations
        y_lat = yv.copy()
        if not sp.is_binary:
            for _ in range(3):
                mu_cur = xm @ beta_d
                if sp.censor_lower and not math.isnan(sp.lower):
                    at_lo = yv <= (sp.lower + 1e-6)
                    z_lo = np.clip((sp.lower + 0.5 - mu_cur) / sd0, -8.0, 8.0)
                    mills_lo = stats.norm.pdf(z_lo) / np.maximum(stats.norm.cdf(z_lo), 1e-8)
                    y_lat = np.where(at_lo, mu_cur - sd0 * mills_lo, y_lat)
                if sp.censor_upper and not math.isnan(sp.upper):
                    at_hi = yv >= (sp.upper - 1e-6)
                    z_hi = np.clip((mu_cur - (sp.upper - 0.5)) / sd0, -8.0, 8.0)
                    mills_hi = stats.norm.pdf(z_hi) / np.maximum(stats.norm.cdf(z_hi), 1e-8)
                    y_lat = np.where(at_hi, mu_cur + sd0 * mills_hi, y_lat)
                ridge.fit(xm, y_lat)
                beta_d = ridge.coef_.copy()
                res0 = y_lat - (xm @ beta_d)
                sd0 = max(float(np.std(res0, ddof=1)), 0.4)

        df_r = pd.DataFrame({"id": iv, "t": tv, "r": res0})
        grp = df_r.groupby("id")
        r_means = grp["r"].mean().to_numpy()
        counts = grp["r"].count().to_numpy()
        tot_var = float(np.var(res0, ddof=1))
        within_ss = float(np.sum((df_r["r"] - df_r["id"].map(grp["r"].mean())) ** 2))
        df_within = max(int(len(df_r) - len(counts)), 1)
        sig2_e = max(within_ss / df_within, 0.05 * tot_var)
        sig2_b0 = max(float(np.var(r_means, ddof=1) - sig2_e / np.mean(counts)), 0.05 * tot_var)
        sig2_b1 = 0.12 * sig2_b0
        g_cov = np.array([[sig2_b0, -0.04 * sig2_b0], [-0.04 * sig2_b0, sig2_b1]], dtype=np.float64)
        g_inv = linalg.inv(g_cov + 1e-6 * np.eye(2))

        for row_k in range(n_te):
            i = int(te_pairs["subj_idx"][row_k])
            w_t = int(te_pairs["w_targ"][row_k])
            max_h = fixed_origin_max_wave if fixed_origin_max_wave is not None else (w_t - 1)
            hw = [w for w in range(max_h + 1) if m[i, w, d]]

            x_star = build_design_row(i, w_t)
            z_star = np.array([1.0, t[i, w_t] / 5.0], dtype=np.float64)
            mu_fix_star = float(x_star @ beta_d)

            if len(hw) == 0:
                pred_m = mu_fix_star
                pred_v = float(z_star @ g_cov @ z_star + sig2_e)
            else:
                z_h = np.column_stack([np.ones(len(hw)), t[i, hw] / 5.0])
                x_h = np.asarray([build_design_row(i, w) for w in hw])
                mu_h = x_h @ beta_d
                yh_obs = y[i, hw, d].copy()
                if not sp.is_binary:
                    if sp.censor_lower and not math.isnan(sp.lower):
                        at_lo_h = yh_obs <= (sp.lower + 1e-6)
                        zl = np.clip((sp.lower + 0.5 - mu_h) / sd0, -8.0, 8.0)
                        yh_obs = np.where(at_lo_h, mu_h - sd0 * (stats.norm.pdf(zl) / np.maximum(stats.norm.cdf(zl), 1e-8)), yh_obs)
                    if sp.censor_upper and not math.isnan(sp.upper):
                        at_hi_h = yh_obs >= (sp.upper - 1e-6)
                        zh = np.clip((mu_h - (sp.upper - 0.5)) / sd0, -8.0, 8.0)
                        yh_obs = np.where(at_hi_h, mu_h + sd0 * (stats.norm.pdf(zh) / np.maximum(stats.norm.cdf(zh), 1e-8)), yh_obs)
                r_h = yh_obs - mu_h
                post_prec = g_inv + (z_h.T @ z_h) / sig2_e
                post_cov = linalg.inv(post_prec)
                b_hat = post_cov @ (z_h.T @ r_h) / sig2_e
                pred_m = mu_fix_star + float(z_star @ b_hat)
                pred_v = float(z_star @ post_cov @ z_star + sig2_e)

            if sp.is_binary:
                mu_te[row_k, d] = float(np.clip(pred_m, 0.02, 0.98))
                sd_te[row_k, d] = -1.0
            else:
                mu_te[row_k, d] = pred_m
                sd_te[row_k, d] = math.sqrt(max(pred_v, 0.05))
    return mu_te, sd_te


def predict_m3_lightgbm_gbdt(
    tr_pairs: Dict[str, np.ndarray],
    te_pairs: Dict[str, np.ndarray],
    seed_fold: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """M3: Autoregressive LightGBM GBDT (Wan et al. 2026 tabular benchmark) + OOF residual uncertainty."""
    n_te = len(te_pairs["dt"])
    n_d = len(CHARLS_SCALE_SPECS)
    mu_te = np.zeros((n_te, n_d), dtype=np.float64)
    sd_te = np.zeros((n_te, n_d), dtype=np.float64)

    for d, sp in enumerate(CHARLS_SCALE_SPECS):
        v_tr = tr_pairs["M"][:, d]
        x_tr = tr_pairs["X"][v_tr]
        y_tr = tr_pairs["Y"][v_tr, d]

        if sp.is_binary:
            clf = lgb.LGBMClassifier(
                n_estimators=180,
                learning_rate=0.04,
                num_leaves=23,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=SEED + seed_fold + d,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(x_tr, (y_tr >= 0.5).astype(int))
            mu_te[:, d] = np.clip(clf.predict_proba(te_pairs["X"])[:, 1], 0.01, 0.99)
            sd_te[:, d] = -1.0
        else:
            reg = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.04,
                num_leaves=23,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=SEED + seed_fold + d,
                verbose=-1,
                n_jobs=-1,
            )
            reg.fit(x_tr, y_tr)
            mu_te[:, d] = reg.predict(te_pairs["X"])

            # Estimate predictive variance via 3-fold inner CV on training pairs (heteroscedastic in dt)
            kf_in = KFold(n_splits=3, shuffle=True, random_state=SEED + d)
            oof_res = np.zeros(len(y_tr))
            for itr, iva in kf_in.split(x_tr):
                r_in = lgb.LGBMRegressor(
                    n_estimators=140,
                    learning_rate=0.05,
                    num_leaves=23,
                    min_child_samples=30,
                    random_state=SEED + d,
                    verbose=-1,
                    n_jobs=-1,
                )
                r_in.fit(x_tr[itr], y_tr[itr])
                oof_res[iva] = y_tr[iva] - r_in.predict(x_tr[iva])
            # Fit log(res^2) ~ dt for calibrated horizon-aware predictive SD
            dt_tr = tr_pairs["dt"][v_tr]
            sd_base = float(np.std(oof_res / np.sqrt(np.maximum(dt_tr / 2.5, 0.5)), ddof=1))
            sd_te[:, d] = np.maximum(sd_base * np.sqrt(np.maximum(te_pairs["dt"] / 2.5, 0.5)), 0.25)
    return mu_te, sd_te


class ContinuousLatentNeuralODE(nn.Module):
    """
    M4: Naïve Continuous-Time Neural/Latent ODE (PyTorch GRU-ode/attention history encoder ->
    continuous-time neural-drift ODE solver -> linear decoder trained under uncensored MSE loss).
    """

    def __init__(self, n_d: int, d_x0: int, d_u: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.n_d = n_d
        self.latent_dim = latent_dim
        self.step_gru = nn.GRUCell(n_d * 2 + 1 + d_u, latent_dim)
        self.init_proj = nn.Linear(d_x0, latent_dim)
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + d_u, 16),
            nn.Tanh(),
            nn.Linear(16, latent_dim),
        )
        self.dec = nn.Linear(latent_dim, n_d)

    def evolve_ode(self, h: torch.Tensor, u_w: torch.Tensor, dt: torch.Tensor, n_substeps: int = 4) -> torch.Tensor:
        h_cur = h
        dt_sub = (dt / float(n_substeps))[:, None]
        for _ in range(n_substeps):
            k1 = self.drift_net(torch.cat([h_cur, u_w], dim=-1))
            h_mid = h_cur + 0.5 * dt_sub * k1
            k2 = self.drift_net(torch.cat([h_mid, u_w], dim=-1))
            h_cur = h_cur + dt_sub * k2
        return h_cur

    def forward_filter(
        self,
        batch: Dict[str, torch.Tensor],
        max_assimilate_wave_idx: Optional[int] = None,
        per_person_last_wave: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        y = batch["y_obs"]
        m = batch["obs_mask"].to(torch.float64)
        t = batch["t_years"]
        x0 = batch["x0"]
        u = batch["u_ctrl"]
        n_b, n_w, _ = y.shape

        h = torch.tanh(self.init_proj(x0))
        preds = []
        for w in range(n_w):
            if w > 0:
                dt = torch.clamp(t[:, w] - t[:, w - 1], min=1e-3)
                h = self.evolve_ode(h, u[:, w, :], dt)
            preds.append(self.dec(h))

            can_assim = (m[:, w, :].sum(dim=-1) > 0)
            if max_assimilate_wave_idx is not None and w > max_assimilate_wave_idx:
                can_assim = torch.zeros_like(can_assim)
            if per_person_last_wave is not None:
                lw_t = torch.tensor(per_person_last_wave, dtype=torch.int64, device=y.device)
                can_assim = can_assim & (w < lw_t)

            if can_assim.any():
                dt_prev = (t[:, w] - t[:, w - 1])[:, None] if w > 0 else torch.zeros((n_b, 1), dtype=torch.float64, device=y.device)
                inp = torch.cat([y[:, w, :] * m[:, w, :], m[:, w, :], dt_prev, u[:, w, :]], dim=-1)
                h_new = self.step_gru(inp, h)
                h = torch.where(can_assim[:, None], h_new, h)
        return torch.stack(preds, dim=1)


def fit_and_predict_m4_naive_neural_ode(
    tr_batch: Dict[str, torch.Tensor],
    te_batch_full: Dict[str, torch.Tensor],
    max_assimilate_wave_idx: Optional[int] = None,
    per_person_last_wave_te: Optional[np.ndarray] = None,
    n_epochs: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit M4 Standard Continuous-Time Neural Latent ODE under standardized uncensored MSE loss and calibrate scale-wise residual SD."""
    n_d = tr_batch["y_obs"].shape[2]
    d_x0 = tr_batch["x0"].shape[1]
    d_u = tr_batch["u_ctrl"].shape[2]
    torch.manual_seed(SEED)
    ode = ContinuousLatentNeuralODE(n_d=n_d, d_x0=d_x0, d_u=d_u, latent_dim=8)

    # Compute training-set scale means and SDs so missing mask (0) maps to scale mean rather than scale floor
    y_tr_raw = tr_batch["y_obs"].cpu().numpy()
    m_tr_raw = tr_batch["obs_mask"].cpu().numpy()
    y_mu = np.zeros(n_d, dtype=np.float64)
    y_sd = np.ones(n_d, dtype=np.float64)
    for d in range(n_d):
        vals = y_tr_raw[:, :, d][m_tr_raw[:, :, d]]
        if len(vals) > 0:
            y_mu[d] = float(np.mean(vals))
            y_sd[d] = max(float(np.std(vals, ddof=1)), 0.25)

    y_mu_t = torch.tensor(y_mu[None, None, :], dtype=torch.float64, device=tr_batch["y_obs"].device)
    y_sd_t = torch.tensor(y_sd[None, None, :], dtype=torch.float64, device=tr_batch["y_obs"].device)

    tr_b_z = dict(tr_batch)
    tr_b_z["y_obs"] = torch.where(tr_batch["obs_mask"], (tr_batch["y_obs"] - y_mu_t) / y_sd_t, torch.zeros_like(tr_batch["y_obs"]))
    te_b_z = dict(te_batch_full)
    te_b_z["y_obs"] = torch.where(te_batch_full["obs_mask"], (te_batch_full["y_obs"] - y_mu_t) / y_sd_t, torch.zeros_like(te_batch_full["y_obs"]))

    opt = torch.optim.Adam(ode.parameters(), lr=0.018, weight_decay=1e-4)
    for _ in range(n_epochs):
        opt.zero_grad()
        pred_tr_z = ode.forward_filter(tr_b_z)
        m_f = tr_b_z["obs_mask"].to(torch.float64)
        err = (tr_b_z["y_obs"] - pred_tr_z) ** 2
        loss = torch.sum(err * m_f) / torch.clamp(torch.sum(m_f), min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode.parameters(), 5.0)
        opt.step()

    with torch.no_grad():
        pred_tr = (ode.forward_filter(tr_b_z) * y_sd_t + y_mu_t).cpu().numpy()
        # Empirical predictive residual SD on waves w >= 1
        res_sds = np.ones(n_d, dtype=np.float64)
        for d in range(n_d):
            v_mask = m_tr_raw[:, 1:, d]
            if np.any(v_mask):
                res_sds[d] = max(float(np.std(y_tr_raw[:, 1:, d][v_mask] - pred_tr[:, 1:, d][v_mask], ddof=1)), 0.25)

        pred_te = (
            ode.forward_filter(
                te_b_z,
                max_assimilate_wave_idx=max_assimilate_wave_idx,
                per_person_last_wave=per_person_last_wave_te,
            )
            * y_sd_t
            + y_mu_t
        ).cpu().numpy()

    sd_grid = np.broadcast_to(res_sds[None, None, :], pred_te.shape).copy()
    # Binary draw probability representation
    pred_te[:, :, 4] = np.clip(pred_te[:, :, 4], 0.02, 0.98)
    sd_grid[:, :, 4] = -1.0
    return pred_te, sd_grid


# ---------------------------------------------------------------------------
# 3. Unified Out-of-Sample Scoring Engine across All 7 Models
# ---------------------------------------------------------------------------
def evaluate_predictions_on_targets(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    mask_true: np.ndarray,
    mu_pred: np.ndarray,
    sd_pred: np.ndarray,
    last_y: np.ndarray,
    illit_vec: np.ndarray,
    comm_vec: np.ndarray,
) -> Tuple[Dict[str, object], List[Dict[str, object]], np.ndarray, np.ndarray]:
    """
    Evaluate identical proper scoring rules (Censored NLL, Censored CRPS, MAE, 90% PI Coverage,
    Spearman rho, and Clinical Event AUROC/Brier for elevated depression CES-D10 >= 10 &
    clinically significant episodic memory impairment/decline) across all 7 models!
    """
    n_rows, n_d = y_true.shape
    row_nll_sum = np.zeros(n_rows, dtype=np.float64)
    row_crps_sum = np.zeros(n_rows, dtype=np.float64)
    row_cnt = np.zeros(n_rows, dtype=np.float64)

    scale_rows = []
    all_nlls = []
    all_crps_norm = []
    all_cov90 = []
    all_pi90_w_norm = []
    all_wink90_norm = []

    yhat_mat = np.full_like(y_true, np.nan)
    dlrc_picp90 = np.nan

    for d, sp in enumerate(CHARLS_SCALE_SPECS):
        vm = mask_true[:, d]
        if not np.any(vm):
            continue
        yt = y_true[vm, d]
        mp = mu_pred[vm, d]
        sp_sd = sd_pred[vm, d]
        sc_res = exact_censored_gaussian_nll_and_crps_np(yt, mp, sp_sd, sp)

        yhat_mat[vm, d] = sc_res["y_hat"]
        row_nll_sum[vm] += sc_res["nll"]
        scale_span = max(sp.upper - sp.lower, 1.0)
        row_crps_sum[vm] += sc_res["crps"] / scale_span  # range-normalized CRPS across heterogeneous [0,1]..[0,30] scales
        row_cnt[vm] += 1.0

        all_nlls.append(sc_res["nll"])
        all_crps_norm.append(sc_res["crps"] / scale_span)
        all_cov90.append(sc_res["cov90"])
        all_pi90_w_norm.append(sc_res["pi90_width"] / scale_span)
        all_wink90_norm.append(sc_res["winkler90"] / scale_span)

        mae_d = float(np.mean(np.abs(yt - sc_res["y_hat"])))
        rho_d = float(stats.spearmanr(yt, sc_res["y_hat"]).statistic)

        # Literacy-stratified MAE & NLL (for Faraci fairness link)
        ill_m = illit_vec[vm] == 1.0
        lit_m = illit_vec[vm] == 0.0
        nll_ill = float(np.mean(sc_res["nll"][ill_m])) if np.any(ill_m) else np.nan
        nll_lit = float(np.mean(sc_res["nll"][lit_m])) if np.any(lit_m) else np.nan

        picp_d = round(100.0 * float(np.mean(sc_res["cov90"])), 2)
        if sp.name == "dlrc":
            dlrc_picp90 = picp_d

        scale_rows.append(
            {
                "split": split_name,
                "model": model_name,
                "scale": sp.name,
                "n_eval": int(np.sum(vm)),
                "tobit_nll": round(float(np.mean(sc_res["nll"])), 5),
                "crps_raw": round(float(np.mean(sc_res["crps"])), 5),
                "crps_range_norm": round(float(np.mean(sc_res["crps"] / scale_span)), 5),
                "mae": round(mae_d, 5),
                "spearman_rho": round(rho_d, 5),
                "picp_90_pct": picp_d,
                "pi90_width_raw": round(float(np.mean(sc_res["pi90_width"])), 4),
                "pi90_width_norm": round(float(np.mean(sc_res["pi90_width"] / scale_span)), 5),
                "winkler90_raw": round(float(np.mean(sc_res["winkler90"])), 4),
                "winkler90_norm": round(float(np.mean(sc_res["winkler90"] / scale_span)), 5),
                "nll_illiterate": round(nll_ill, 5),
                "nll_literate": round(nll_lit, 5),
                "nll_illit_minus_lit_gap": round(nll_ill - nll_lit, 5),
            }
        )

    # Clinical Event 1: Elevated Depression (CES-D10 >= 10) exact predictive probability P(Y*_cesd >= 9.5)
    vm_dep = mask_true[:, 5]
    y_dep_ev = (y_true[vm_dep, 5] >= 10.0).astype(int)
    p_dep_ev = stats.norm.sf((9.5 - mu_pred[vm_dep, 5]) / np.maximum(sd_pred[vm_dep, 5], 1e-3))
    p_dep_ev = np.clip(p_dep_ev, 1e-5, 1.0 - 1e-5)
    auroc_dep = float(roc_auc_score(y_dep_ev, p_dep_ev))
    brier_dep = float(brier_score_loss(y_dep_ev, p_dep_ev))

    # Clinical Event 2: Low Cognitive Reserve / Severe Episodic Memory Impairment (tr20 = imrc + dlrc <= 6.0, bottom quintile)
    # & Steep Cognitive Drop (delta_tr20 <= -3.0 words)
    vm_tr20 = mask_true[:, 0] & mask_true[:, 1]
    tr20_true = y_true[vm_tr20, 0] + y_true[vm_tr20, 1]
    tr20_mu = mu_pred[vm_tr20, 0] + mu_pred[vm_tr20, 1]
    tr20_sd = np.sqrt(np.maximum(sd_pred[vm_tr20, 0] ** 2 + sd_pred[vm_tr20, 1] ** 2, 1e-3))
    y_cog_imp = (tr20_true <= 6.0).astype(int)
    p_cog_imp = np.clip(stats.norm.cdf((6.5 - tr20_mu) / tr20_sd), 1e-5, 1.0 - 1e-5)
    auroc_cog_imp = float(roc_auc_score(y_cog_imp, p_cog_imp))
    brier_cog_imp = float(brier_score_loss(y_cog_imp, p_cog_imp))

    has_prev_tr20 = vm_tr20 & (~np.isnan(last_y[:, 0])) & (~np.isnan(last_y[:, 1]))
    if np.sum(has_prev_tr20) > 100:
        prev_tr20 = last_y[has_prev_tr20, 0] + last_y[has_prev_tr20, 1]
        cur_tr20 = y_true[has_prev_tr20, 0] + y_true[has_prev_tr20, 1]
        y_decline = ((cur_tr20 - prev_tr20) <= -3.0).astype(int)
        mu_tr20_d = mu_pred[has_prev_tr20, 0] + mu_pred[has_prev_tr20, 1]
        sd_tr20_d = np.sqrt(np.maximum(sd_pred[has_prev_tr20, 0] ** 2 + sd_pred[has_prev_tr20, 1] ** 2, 1e-3))
        p_decline = np.clip(stats.norm.cdf((prev_tr20 - 2.5 - mu_tr20_d) / sd_tr20_d), 1e-5, 1.0 - 1e-5)
        auroc_decline = float(roc_auc_score(y_decline, p_decline))
    else:
        auroc_decline = np.nan

    cat_nll = np.concatenate(all_nlls)
    cat_crps = np.concatenate(all_crps_norm)
    cat_cov90 = np.concatenate(all_cov90)
    cat_pi90_w = np.concatenate(all_pi90_w_norm)
    cat_wink90 = np.concatenate(all_wink90_norm)

    person_mean_nll = row_nll_sum / np.maximum(row_cnt, 1.0)
    person_mean_crps = row_crps_sum / np.maximum(row_cnt, 1.0)

    overall = {
        "split": split_name,
        "model": model_name,
        "n_participants_eval": int(np.sum(row_cnt > 0)),
        "n_scale_observations_eval": int(len(cat_nll)),
        "co_primary_tobit_NLL": round(float(np.mean(cat_nll)), 5),
        "co_primary_CRPS_norm": round(float(np.mean(cat_crps)), 5),
        "PICP_90_coverage_pct": round(100.0 * float(np.mean(cat_cov90)), 2),
        "dlrc_PICP_90_coverage_pct": dlrc_picp90,
        "mean_PI90_width_norm": round(float(np.mean(cat_pi90_w)), 5),
        "winkler_score_90_norm": round(float(np.mean(cat_wink90)), 5),
        "AUROC_elevated_depression_cesd10_ge10": round(auroc_dep, 4),
        "Brier_elevated_depression": round(brier_dep, 4),
        "AUROC_cognitive_impairment_tr20_le6": round(auroc_cog_imp, 4),
        "Brier_cognitive_impairment": round(brier_cog_imp, 4),
        "AUROC_incident_cognitive_drop_ge3w": round(auroc_decline, 4) if not np.isnan(auroc_decline) else np.nan,
    }
    return overall, scale_rows, person_mean_nll, person_mean_crps


def attach_paired_significance_vs_m7(
    summary_rows: List[Dict[str, object]],
    person_nlls: Dict[str, np.ndarray],
    person_crps: Dict[str, np.ndarray],
    ref_model: str = "M7_Psychometric_Coupled_SDE",
) -> pd.DataFrame:
    """Compute paired two-sided t-test p-values & Holm-Bonferroni multiplicity control of M7 vs every baseline."""
    df = pd.DataFrame(summary_rows)
    ref_nll = person_nlls[ref_model]
    ref_crps = person_crps[ref_model]

    d_nlls = []
    p_nlls = []
    d_crps = []
    p_crps = []
    base_indices = []

    for idx_r, r in df.iterrows():
        m = r["model"]
        if m == ref_model:
            d_nlls.append(0.0)
            p_nlls.append(np.nan)
            d_crps.append(0.0)
            p_crps.append(np.nan)
        else:
            diff_n = ref_nll - person_nlls[m]  # negative = M7 has lower (better) NLL
            diff_c = ref_crps - person_crps[m]
            t_n, p_n = stats.ttest_1samp(diff_n, 0.0)
            t_c, p_c = stats.ttest_1samp(diff_c, 0.0)
            d_nlls.append(round(float(np.mean(diff_n)), 5))
            p_nlls.append(float(f"{p_n:.4e}"))
            d_crps.append(round(float(np.mean(diff_c)), 5))
            p_crps.append(float(f"{p_c:.4e}"))
            if m.startswith(("M1_", "M2_", "M2b_", "M3_", "M4_", "M5_", "M6_")):
                base_indices.append(idx_r)

    df["delta_NLL_M7_minus_model"] = d_nlls
    df["paired_p_NLL_vs_M7"] = p_nlls
    df["delta_CRPS_M7_minus_model"] = d_crps
    df["paired_p_CRPS_vs_M7"] = p_crps

    df["holm_p_NLL_vs_6baselines"] = np.nan
    df["holm_p_CRPS_vs_6baselines"] = np.nan
    if len(base_indices) > 0:
        df.loc[base_indices, "holm_p_NLL_vs_6baselines"] = multipletests(df.loc[base_indices, "paired_p_NLL_vs_M7"].to_numpy(float), method="holm")[1]
        df.loc[base_indices, "holm_p_CRPS_vs_6baselines"] = multipletests(df.loc[base_indices, "paired_p_CRPS_vs_M7"].to_numpy(float), method="holm")[1]
    return df


# ---------------------------------------------------------------------------
# 4. Paired Venous Blood (2011 & 2015, N = 7,438) Jacobian Moderation Analysis
# ---------------------------------------------------------------------------
def run_paired_blood_jacobian_moderation(ds: Dict[str, object]) -> pd.DataFrame:
    """
    Estimate how baseline (2011) and 4-year longitudinal surges (2011 -> 2015) in systemic
    inflammation (log hs-CRP, delta log hs-CRP) and glycemic load (HbA1c, delta HbA1c) moderate
    the directed cognitive-affective coupling Jacobian (J_{cog<-dep, i}, J_{dep<-cog, i}) across
    the exact N = 7,438 paired venous blood subcohort!
    """
    t0 = time.time()
    pers = ds["persons"]
    blood = ds["blood"]
    pb_df = pd.DataFrame({"ID": ds["ids"], "row_i": np.arange(len(ds["ids"]))}).merge(
        blood[blood["is_paired_2011_2015"] == 1], on="ID", how="inner"
    )
    idx_pb = pb_df["row_i"].to_numpy(dtype=np.int32)
    print(f"[Paired Blood SDE] Fitting biologically moderated Psychometric-Coupled-SDE on N = {len(idx_pb):,} paired blood participants...")

    # Standardized biological + demographic moderators (adjusted for cystatin C systemic clearance)
    def zscore_clean(s: pd.Series) -> np.ndarray:
        v = s.fillna(s.median()).to_numpy(dtype=np.float64)
        return (v - np.mean(v)) / max(np.std(v, ddof=1), 1e-6)

    mod_names = [
        "log_hsCRP_2011_z",
        "delta_log_hsCRP_4yr_z",
        "HbA1c_2011_z",
        "delta_HbA1c_4yr_z",
        "tchol_to_hdl_2011_z",
        "cystatinC_clearance_2011_z",
        "baseline_age_z",
        "illiterate_z",
    ]
    tc_hdl = pb_df["tchol_mgdl_2011"] / pb_df["hdl_mgdl_2011"].where(pb_df["hdl_mgdl_2011"] > 5.0, np.nan)
    z_bio = np.column_stack(
        [
            zscore_clean(pb_df["log_crp_2011"]),
            zscore_clean(pb_df["delta_log_crp"]),
            zscore_clean(pb_df["hba1c_pct_2011"]),
            zscore_clean(pb_df["delta_hba1c_pct"]),
            zscore_clean(tc_hdl),
            zscore_clean(pb_df["cystatin_c_mgl_2011"]),
            (ds["x0_raw"][idx_pb, 0] - np.mean(ds["x0_raw"][idx_pb, 0])) / np.std(ds["x0_raw"][idx_pb, 0], ddof=1),
            (ds["illit"][idx_pb] - np.mean(ds["illit"][idx_pb])) / np.std(ds["illit"][idx_pb], ddof=1),
        ]
    )

    x0_m = np.mean(ds["x0_raw"][idx_pb], axis=0)
    u_m = np.mean(ds["u_ctrl_raw"][idx_pb], axis=(0, 1))
    b_pb = make_torch_batch(ds, idx_pb, x0_m, u_m)
    b_pb["z_mod"] = torch.tensor(z_bio, dtype=torch.float64)

    torch.manual_seed(SEED)
    m6_bio = PsychometricCoupledSDE(
        CHARLS_SCALE_SPECS,
        d_x0=b_pb["x0"].shape[1],
        d_u=b_pb["u_ctrl"].shape[2],
        d_z_mod=0,
        coupled=False,
        psychometric_filter=True,
        n_gh_1d=5,
        use_subtest_traits=True,
        d_prot=2,
    )
    fit_psychometric_sde_model(m6_bio, b_pb, lr_adam=0.038, n_lbfgs_steps=55, l2_reg=1.5e-4, seed=SEED, verbose=False)

    torch.manual_seed(SEED)
    m_bio = PsychometricCoupledSDE(
        CHARLS_SCALE_SPECS,
        d_x0=b_pb["x0"].shape[1],
        d_u=b_pb["u_ctrl"].shape[2],
        d_z_mod=z_bio.shape[1],
        coupled=True,
        psychometric_filter=True,
        n_gh_1d=5,
        use_subtest_traits=True,
        d_prot=2,
    )
    fit_psychometric_sde_model(
        m_bio, b_pb, warm_from=m6_bio, n_lbfgs_steps=55, l2_reg=1.5e-4, l2_mod_mult=5.0, seed=SEED, verbose=False
    )

    # Compute community-clustered empirical Fisher/BHHH standard errors across unique communities
    comms = ds["comm_id"][idx_pb]
    uniq_comms = np.unique(comms)
    # Group communities into 90 balanced super-clusters for fast exact cluster-score covariance
    n_sc = 90
    comm_to_sc = {c: (k % n_sc) for k, c in enumerate(uniq_comms)}
    sc_ids = np.array([comm_to_sc[c] for c in comms], dtype=np.int32)

    target_p = [m_bio.j12_0, m_bio.j21_0, m_bio.j12_mod, m_bio.j21_mod]
    sc_grads = []
    for g_id in range(n_sc):
        sub_i = np.where(sc_ids == g_id)[0]
        if len(sub_i) == 0:
            continue
        sb = {k: v[sub_i] for k, v in b_pb.items()}
        m_bio.zero_grad()
        ll_g = torch.sum(m_bio.filter_and_loglik(sb)["person_loglik"])
        gr = torch.autograd.grad(ll_g, target_p)
        g_vec = torch.cat([gr[0].reshape(1), gr[1].reshape(1), gr[2].reshape(-1), gr[3].reshape(-1)])
        sc_grads.append(g_vec.detach().cpu().numpy())

    g_mat = np.asarray(sc_grads)
    g_cent = g_mat - np.mean(g_mat, axis=0, keepdims=True)
    info = (g_cent.T @ g_cent) * (len(g_mat) / float(len(g_mat) - 1)) + 1e-4 * np.eye(g_mat.shape[1])
    se_vec = np.sqrt(np.maximum(np.diag(linalg.inv(info)), 1e-12))

    kap_bio = (torch.nn.functional.softplus(m_bio.raw_kappa) + 0.05).detach().cpu().numpy()
    max_c_v = 0.85 * math.sqrt(float(kap_bio[0] * kap_bio[1]))
    th12 = math.tanh(float(m_bio.j12_0.detach().item()) / max_c_v)
    th21 = math.tanh(float(m_bio.j21_0.detach().item()) / max_c_v)
    scale12 = 1.0 - th12 * th12
    scale21 = 1.0 - th21 * th21

    j12_0_v = max_c_v * th12
    j21_0_v = max_c_v * th21
    se0_12 = scale12 * float(se_vec[0])
    se0_21 = scale21 * float(se_vec[1])
    j12_m_v = scale12 * m_bio.j12_mod.detach().cpu().numpy()
    j21_m_v = scale21 * m_bio.j21_mod.detach().cpu().numpy()

    rows = [
        {
            "n_paired_blood_sde": int(len(idx_pb)),
            "pathway": "J_cog_from_dep (Affective -> Cognitive Erosion)",
            "moderator": "Intercept_at_Mean",
            "estimate_SD_per_yr": round(j12_0_v, 5),
            "cluster_se": round(se0_12, 5),
            "ci95_lo": round(j12_0_v - 1.96 * se0_12, 5),
            "ci95_hi": round(j12_0_v + 1.96 * se0_12, 5),
            "p_value": float(f"{2.0 * stats.norm.sf(abs(j12_0_v) / max(se0_12, 1e-12)):.4e}"),
        },
        {
            "n_paired_blood_sde": int(len(idx_pb)),
            "pathway": "J_dep_from_cog (Cognitive Reserve -> Affective Buffering)",
            "moderator": "Intercept_at_Mean",
            "estimate_SD_per_yr": round(j21_0_v, 5),
            "cluster_se": round(se0_21, 5),
            "ci95_lo": round(j21_0_v - 1.96 * se0_21, 5),
            "ci95_hi": round(j21_0_v + 1.96 * se0_21, 5),
            "p_value": float(f"{2.0 * stats.norm.sf(abs(j21_0_v) / max(se0_21, 1e-12)):.4e}"),
        },
    ]
    k_mod = len(mod_names)
    for k_i, m_nm in enumerate(mod_names):
        e12, s12 = float(j12_m_v[k_i]), scale12 * float(se_vec[2 + k_i])
        p12 = float(2.0 * stats.norm.sf(abs(e12) / max(s12, 1e-12)))
        rows.append(
            {
                "n_paired_blood_sde": int(len(idx_pb)),
                "pathway": "J_cog_from_dep (Affective -> Cognitive Erosion)",
                "moderator": m_nm,
                "estimate_SD_per_yr": round(e12, 5),
                "cluster_se": round(s12, 5),
                "ci95_lo": round(e12 - 1.96 * s12, 5),
                "ci95_hi": round(e12 + 1.96 * s12, 5),
                "p_value": float(f"{p12:.4e}"),
            }
        )
        e21, s21 = float(j21_m_v[k_i]), scale21 * float(se_vec[2 + k_mod + k_i])
        p21 = float(2.0 * stats.norm.sf(abs(e21) / max(s21, 1e-12)))
        rows.append(
            {
                "n_paired_blood_sde": int(len(idx_pb)),
                "pathway": "J_dep_from_cog (Cognitive Reserve -> Affective Buffering)",
                "moderator": m_nm,
                "estimate_SD_per_yr": round(e21, 5),
                "cluster_se": round(s21, 5),
                "ci95_lo": round(e21 - 1.96 * s21, 5),
                "ci95_hi": round(e21 + 1.96 * s21, 5),
                "p_value": float(f"{p21:.4e}"),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df["fdr_q"] = multipletests(out_df["p_value"], method="fdr_bh")[1]
    print(f"[Paired Blood SDE] Completed in {time.time() - t0:.1f}s")
    return out_df


# ---------------------------------------------------------------------------
# 5. External Cross-Cohort Psychometric Construct Validation on NHANES (N = 3,014)
# ---------------------------------------------------------------------------
def run_nhanes_external_construct_validation() -> pd.DataFrame:
    """
    External Cross-Cohort Construct Validation (§3.3.2 of SAP_v1_FROZEN.md):
    Apply the 2-domain Psychometric Tobit-IRT measurement filter vs Naïve Uncensored Gaussian
    measurement model under 5-fold CV on NHANES older adults >= 60 (N = 3,014 with DSST),
    contrasting zero-ceiling cognitive tests (DSST 0.00% ceiling, Animal Fluency 0.00% ceiling,
    CERAD_IR 0.10% ceiling, CERAD_DR) against heavily floor-censored PHQ-9 (34.9% floor at 0!).
    """
    nh = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_participants.parquet")
    sub = nh[(nh["age_years"] >= 60.0) & nh["dsst"].notna()].copy().reset_index(drop=True)

    nh_specs = [
        ScaleSpec("dsst", domain=0, lower=0.0, upper=133.0, anchor_loading=17.0),
        ScaleSpec("animal_fluency", domain=0, lower=0.0, upper=60.0, censor_upper=False, anchor_loading=None),
        ScaleSpec("cerad_ir_sum", domain=0, lower=0.0, upper=30.0, anchor_loading=None),
        ScaleSpec("cerad_dr", domain=0, lower=0.0, upper=10.0, anchor_loading=None),
        ScaleSpec("phq9_total", domain=1, lower=0.0, upper=27.0, anchor_loading=4.2),
    ]
    n_s = len(sub)
    n_d = len(nh_specs)
    y_mat = np.zeros((n_s, 1, n_d), dtype=np.float64)
    m_mat = np.zeros((n_s, 1, n_d), dtype=bool)
    for d, sp in enumerate(nh_specs):
        v = sub[sp.name].to_numpy(float)
        ok = ~np.isnan(v)
        y_mat[ok, 0, d] = v[ok]
        m_mat[ok, 0, d] = True

    age_c = (sub["age_years"].to_numpy(float) - 69.0) / 10.0
    fem_c = sub["female"].fillna(0.5).to_numpy(float) - 0.5
    low_ed = sub["low_education_lt_hs"].fillna(0.0).to_numpy(float)
    x0 = np.column_stack([age_c, fem_c, low_ed - np.mean(low_ed)])

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_mu = {"Psychometric_Tobit_Model": np.zeros((n_s, n_d)), "Naive_Uncensored_Gaussian": np.zeros((n_s, n_d))}
    oof_sd = {"Psychometric_Tobit_Model": np.zeros((n_s, n_d)), "Naive_Uncensored_Gaussian": np.zeros((n_s, n_d))}
    lat_corr = {}

    for m_label, use_psych in [("Psychometric_Tobit_Model", True), ("Naive_Uncensored_Gaussian", False)]:
        for tr, va in kf.split(y_mat):
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
            fit_psychometric_sde_model(mod, b_tr, n_adam_steps=120, lr_adam=0.04, n_lbfgs_steps=25, l2_reg=1e-4, verbose=False)
            with torch.no_grad():
                loadings = mod.get_loadings()
                sig_base = torch.nn.functional.softplus(mod.raw_sigma_obs) + 1e-2
                for d_hold, sp_h in enumerate(nh_specs):
                    b_va_mask = dict(b_va)
                    m_clone = b_va["obs_mask"].clone()
                    m_clone[:, 0, d_hold] = False
                    b_va_mask["obs_mask"] = m_clone
                    # Filter on the other 4 observed scales to infer posterior (eta_cog, eta_dep)
                    # Note: also extract posterior covariance via quadrature weights for exact predictive variance
                    out_lo = mod.filter_and_loglik(b_va_mask)
                    eta_post = out_lo["filt_eta_means"][:, 0, :]  # (N_va, 2)
                    # Recompute posterior variance of domain(sp_h) given m_clone
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
                        from psychometric_sde_core import tobit_probit_Scale_loglik
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

        # Fit on full N = 3,014 to extract latent cog-dep correlation & sNfL correlation in 2013-2014 subset
        b_all = {
            "t_years": torch.zeros((n_s, 1), dtype=torch.float64),
            "y_obs": torch.tensor(y_mat, dtype=torch.float64),
            "obs_mask": torch.tensor(m_mat, dtype=torch.bool),
            "k_prior": torch.zeros((n_s, 1, n_d), dtype=torch.float64),
            "x0": torch.tensor(x0, dtype=torch.float64),
            "u_ctrl": torch.zeros((n_s, 1, 1), dtype=torch.float64),
            "illit": torch.tensor(low_ed, dtype=torch.float64),
        }
        mod_full = PsychometricCoupledSDE(nh_specs, d_x0=3, d_u=1, coupled=True, psychometric_filter=use_psych, n_gh_1d=7)
        fit_psychometric_sde_model(mod_full, b_all, n_adam_steps=130, lr_adam=0.04, n_lbfgs_steps=25, l2_reg=1e-4, verbose=False)
        with torch.no_grad():
            p0_cov = mod_full._build_2x2_cov(mod_full.raw_p0_diag, mod_full.raw_p0_off).cpu().numpy()
            r_lat = float(p0_cov[0, 1] / math.sqrt(max(p0_cov[0, 0] * p0_cov[1, 1], 1e-12)))
            eta_f = mod_full.filter_and_loglik(b_all)["filt_eta_means"][:, 0, :].cpu().numpy()
            has_snfl = sub["log_snfl"].notna().to_numpy()
            r_snfl_cog = float(np.corrcoef(eta_f[has_snfl, 0], sub.loc[has_snfl, "log_snfl"].to_numpy(float))[0, 1])
            lat_corr[m_label] = (r_lat, r_snfl_cog)

    rows = []
    for d, sp in enumerate(nh_specs):
        vm = m_mat[:, 0, d]
        yt = y_mat[vm, 0, d]
        fl_pct = 100.0 * float(np.mean(yt <= sp.lower))
        ce_pct = 100.0 * float(np.mean(yt >= sp.upper)) if sp.censor_upper else 0.0
        r_psych = exact_censored_gaussian_nll_and_crps_np(yt, oof_mu["Psychometric_Tobit_Model"][vm, d], oof_sd["Psychometric_Tobit_Model"][vm, d], sp)
        r_naive = exact_censored_gaussian_nll_and_crps_np(yt, oof_mu["Naive_Uncensored_Gaussian"][vm, d], oof_sd["Naive_Uncensored_Gaussian"][vm, d], sp)
        diff_nll = r_psych["nll"] - r_naive["nll"]
        _, p_val = stats.ttest_1samp(diff_nll, 0.0)
        rows.append(
            {
                "cohort": "NHANES_2011_2014_Older60_ExternalValidation",
                "scale": sp.name,
                "n_valid": int(np.sum(vm)),
                "empirical_floor_pct": round(fl_pct, 2),
                "empirical_ceiling_pct": round(ce_pct, 2),
                "psychometric_oof_NLL": round(float(np.mean(r_psych["nll"])), 5),
                "naive_gaussian_oof_NLL": round(float(np.mean(r_naive["nll"])), 5),
                "delta_NLL_psych_minus_naive": round(float(np.mean(diff_nll)), 5),
                "paired_p_NLL": float(f"{p_val:.4e}"),
                "psychometric_oof_CRPS": round(float(np.mean(r_psych["crps"])), 5),
                "naive_gaussian_oof_CRPS": round(float(np.mean(r_naive["crps"])), 5),
                "latent_cog_dep_r_psych": round(lat_corr["Psychometric_Tobit_Model"][0], 4),
                "latent_cog_dep_r_naive": round(lat_corr["Naive_Uncensored_Gaussian"][0], 4),
                "latent_cog_log_sNfL_r_psych": round(lat_corr["Psychometric_Tobit_Model"][1], 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Main Benchmark Orchestration (Split A 5-Fold Community CV + Split B Temporal Holdouts + Twin Export)
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=== Phase 2 Pillar 2 (Part B): CHARLS 9-Year 5-Wave 7-Model Benchmark & Twin Export ===")
    ds = build_charls_tensor_dataset()
    n_subj = len(ds["ids"])
    n_d = len(CHARLS_SCALE_SPECS)
    print(f"[CHARLS Dataset] N = {n_subj:,} Primary Coupled SDE participants (>=3 coupled waves N = {int(np.sum(ds['n_coupled_waves'] >= 3)):,})")

    # =========================================================================
    # SPLIT A: Community-Grouped 5-Fold Cross-Validation (450 communities, N = 20,928)
    # =========================================================================
    print("[Split A] Running Community-Grouped 5-Fold Cross-Validation across all 7 model families...")
    gkf = GroupKFold(n_splits=5)
    model_keys_cv = [
        "M1_Persistence_AgeNormative",
        "M2_Linear_Mixed_Effects_BLUP",
        "M3_LightGBM_Autoregressive_GBDT",
        "M4_Naive_Neural_Latent_ODE",
        "M5_Uncensored_Coupled_SDE",
        "M6_Uncoupled_Psychometric_Tobit_SDE",
        "M7_Psychometric_Coupled_SDE",
        "M7b_Psychometric_Coupled_SDE_NeuralDrift",
        "M7c_Psychometric_Coupled_SDE_IPCW",
    ]
    oof_mu_a = {k: np.zeros((n_subj, n_d), dtype=np.float64) for k in model_keys_cv}
    oof_sd_a = {k: np.zeros((n_subj, n_d), dtype=np.float64) for k in model_keys_cv}
    y_true_a = np.zeros((n_subj, n_d), dtype=np.float64)
    m_true_a = np.zeros((n_subj, n_d), dtype=bool)
    last_y_a = np.full((n_subj, n_d), np.nan, dtype=np.float64)

    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(ds["y_obs"], groups=ds["comm_id"])):
        t_f = time.time()
        tr_pairs = extract_transition_pairs(ds, tr_idx, only_person_last_wave=False)
        te_pairs = extract_transition_pairs(ds, te_idx, only_person_last_wave=True)
        te_subj = te_pairs["subj_idx"]

        y_true_a[te_subj] = te_pairs["Y"]
        m_true_a[te_subj] = te_pairs["M"]
        last_y_a[te_subj] = te_pairs["last_Y"]

        # M1, M2, M3
        oof_mu_a["M1_Persistence_AgeNormative"][te_subj], oof_sd_a["M1_Persistence_AgeNormative"][te_subj] = predict_m1_persistence_agenorm(tr_pairs, te_pairs)
        oof_mu_a["M2_Linear_Mixed_Effects_BLUP"][te_subj], oof_sd_a["M2_Linear_Mixed_Effects_BLUP"][te_subj] = predict_m2_lmm_empirical_bayes(ds, tr_idx, te_idx, te_pairs)
        oof_mu_a["M3_LightGBM_Autoregressive_GBDT"][te_subj], oof_sd_a["M3_LightGBM_Autoregressive_GBDT"][te_subj] = predict_m3_lightgbm_gbdt(tr_pairs, te_pairs, seed_fold=fold_i * 10)

        # Torch batches for M4, M5, M6, M7, M7b, M7c
        x0_m = np.mean(ds["x0_raw"][tr_idx], axis=0)
        u_m = np.mean(ds["u_ctrl_raw"][tr_idx], axis=(0, 1))
        tr_b = make_torch_batch(ds, tr_idx, x0_m, u_m)
        te_b_masked = make_torch_batch(ds, te_idx, x0_m, u_m, mask_person_last_wave=True)
        lw_te = ds["last_coupled_wave"][te_idx]
        row_arange = np.arange(len(te_idx))

        # M4 Naive Neural Latent ODE
        p4_mu, p4_sd = fit_and_predict_m4_naive_neural_ode(tr_b, te_b_masked, per_person_last_wave_te=lw_te, n_epochs=135)
        oof_mu_a["M4_Naive_Neural_Latent_ODE"][te_idx] = p4_mu[row_arange, lw_te, :]
        oof_sd_a["M4_Naive_Neural_Latent_ODE"][te_idx] = p4_sd[row_arange, lw_te, :]

        # M5, M6, M7, M7b, M7c SDE family (nested warm-start L-BFGS schedule)
        sde_configs = [
            ("M5_Uncensored_Coupled_SDE", True, False, False, None, None, 55),
            ("M6_Uncoupled_Psychometric_Tobit_SDE", False, True, False, None, None, 55),
            ("M7_Psychometric_Coupled_SDE", True, True, False, None, "M6_Uncoupled_Psychometric_Tobit_SDE", 55),
            ("M7b_Psychometric_Coupled_SDE_NeuralDrift", True, True, True, None, "M7_Psychometric_Coupled_SDE", 35),
            ("M7c_Psychometric_Coupled_SDE_IPCW", True, True, False, torch.tensor(ds["ipcw"][tr_idx], dtype=torch.float64), "M7_Psychometric_Coupled_SDE", 35),
        ]
        fitted_fold_mods: Dict[str, PsychometricCoupledSDE] = {}
        for m_key, is_coup, is_psych, is_nd, sw_tr, warm_key, n_lb in sde_configs:
            torch.manual_seed(SEED + fold_i)
            mod = PsychometricCoupledSDE(
                CHARLS_SCALE_SPECS,
                d_x0=tr_b["x0"].shape[1],
                d_u=tr_b["u_ctrl"].shape[2],
                d_z_mod=0,
                coupled=is_coup,
                psychometric_filter=is_psych,
                use_neural_drift=is_nd,
                n_gh_1d=5,
                use_subtest_traits=True,
                d_prot=2,
            )
            fit_psychometric_sde_model(
                mod,
                tr_b,
                sample_weights=sw_tr,
                lr_adam=0.038,
                n_lbfgs_steps=n_lb,
                l2_reg=1.5e-4,
                warm_from=fitted_fold_mods.get(warm_key, None) if warm_key is not None else None,
                seed=SEED,
                verbose=False,
            )
            fitted_fold_mods[m_key] = mod
            with torch.no_grad():
                out_te = mod.filter_and_loglik(te_b_masked)
                mu_grid = out_te["pred_scale_mu"].cpu().numpy()
                sd_grid = out_te["pred_scale_sd"].cpu().numpy()
                if not is_psych:
                    mu_grid[:, :, 4] = np.clip(mu_grid[:, :, 4], 0.02, 0.98)
                    sd_grid[:, :, 4] = -1.0
                oof_mu_a[m_key][te_idx] = mu_grid[row_arange, lw_te, :]
                oof_sd_a[m_key][te_idx] = sd_grid[row_arange, lw_te, :]

        print(f"  [Split A Fold {fold_i+1}/5] Evaluated {len(te_idx):,} held-out community participants in {time.time() - t_f:.1f}s")

    cv_summaries = []
    cv_scale_rows = []
    cv_pnll = {}
    cv_pcrps = {}
    for m_key in model_keys_cv:
        ov, sc_r, pn, pc = evaluate_predictions_on_targets(
            m_key, "SplitA_5Fold_CommunityGrouped_CV", y_true_a, m_true_a, oof_mu_a[m_key], oof_sd_a[m_key], last_y_a, ds["illit"], ds["comm_id"]
        )
        cv_summaries.append(ov)
        cv_scale_rows.extend(sc_r)
        cv_pnll[m_key] = pn
        cv_pcrps[m_key] = pc

    cv_df = attach_paired_significance_vs_m7(cv_summaries, cv_pnll, cv_pcrps)
    cv_df.to_csv(OUT_DIR / "charls_cv_benchmark_7models.csv", index=False)

    # Export individual-level Split A OOF predictions & proper scoring residuals for Phase 3 GAMLSS Fairness & Faraci Ambiguity Audit
    oof_export = pd.DataFrame(
        {
            "ID": ds["ids"],
            "communityID": ds["comm_id"],
            "eval_wave_idx": ds["last_coupled_wave"],
            "illiterate": ds["illit"],
            "true_imrc": y_true_a[:, 0],
            "true_dlrc": y_true_a[:, 1],
            "true_tr20": y_true_a[:, 0] + y_true_a[:, 1],
            "true_ser7": np.where(m_true_a[:, 2], y_true_a[:, 2], np.nan),
            "true_orient": np.where(m_true_a[:, 3], y_true_a[:, 3], np.nan),
            "true_draw": np.where(m_true_a[:, 4], y_true_a[:, 4], np.nan),
            "true_cesd10": y_true_a[:, 5],
        }
    )
    short_tags = {
        "M1_Persistence_AgeNormative": "M1",
        "M2_Linear_Mixed_Effects_BLUP": "M2",
        "M3_LightGBM_Autoregressive_GBDT": "M3",
        "M4_Naive_Neural_Latent_ODE": "M4",
        "M5_Uncensored_Coupled_SDE": "M5",
        "M6_Uncoupled_Psychometric_Tobit_SDE": "M6",
        "M7_Psychometric_Coupled_SDE": "M7",
    }
    for mk, st in short_tags.items():
        oof_export[f"nll_{st}"] = cv_pnll[mk]
        oof_export[f"crps_{st}"] = cv_pcrps[mk]
        oof_export[f"mu_tr20_{st}"] = oof_mu_a[mk][:, 0] + oof_mu_a[mk][:, 1]
        oof_export[f"sd_tr20_{st}"] = np.sqrt(np.maximum(oof_sd_a[mk][:, 0] ** 2 + oof_sd_a[mk][:, 1] ** 2, 1e-4))
        oof_export[f"mu_cesd_{st}"] = oof_mu_a[mk][:, 5]
        oof_export[f"sd_cesd_{st}"] = np.maximum(oof_sd_a[mk][:, 5], 1e-3)
        oof_export[f"p_dep_ge10_{st}"] = np.clip(stats.norm.sf((9.5 - oof_mu_a[mk][:, 5]) / np.maximum(oof_sd_a[mk][:, 5], 1e-3)), 1e-5, 1.0 - 1e-5)
        oof_export[f"p_cog_le6_{st}"] = np.clip(stats.norm.cdf((6.5 - oof_export[f"mu_tr20_{st}"]) / oof_export[f"sd_tr20_{st}"]), 1e-5, 1.0 - 1e-5)
    oof_export.to_parquet(DATA_DIR / "charls_splitA_oof_predictions.parquet", index=False)

    # =========================================================================
    # SPLIT B: Strict Multi-Horizon Out-of-Time Temporal Holdouts (N = 17,360 >=3-wave cohort)
    #   B1: Train Waves 1-3 (2011-2015) -> Forecast Wave 4 (2018, +3y) & Wave 5 (2020, +5y)
    #   B2: Train Waves 1-4 (2011-2018) -> Forecast Wave 5 (2020, +2y)
    # =========================================================================
    print("[Split B] Running Strict Out-of-Time Multi-Horizon Temporal Holdouts (W1-W3 -> W4/+3y, W5/+5y; W1-W4 -> W5/+2y)...")
    ge3_idx = np.where(ds["n_coupled_waves"] >= 3)[0]
    temp_all_dfs = []

    temporal_protocols = [
        ("SplitB1_TrainW1_W3", 2, [("Forecast_W4_Plus3Yr", 3), ("Forecast_W5_Plus5Yr", 4)]),
        ("SplitB2_TrainW1_W4", 3, [("Forecast_W5_Plus2Yr", 4)]),
    ]
    core_7_keys = [
        "M1_Persistence_AgeNormative",
        "M2_Linear_Mixed_Effects_BLUP",
        "M3_LightGBM_Autoregressive_GBDT",
        "M4_Naive_Neural_Latent_ODE",
        "M5_Uncensored_Coupled_SDE",
        "M6_Uncoupled_Psychometric_Tobit_SDE",
        "M7_Psychometric_Coupled_SDE",
    ]

    for train_tag, max_tr_w, target_specs in temporal_protocols:
        tr_pairs_b = extract_transition_pairs(ds, ge3_idx, max_target_wave_idx=max_tr_w)
        x0_mb = np.mean(ds["x0_raw"][ge3_idx], axis=0)
        u_mb = np.mean(ds["u_ctrl_raw"][ge3_idx, : max_tr_w + 1, :], axis=(0, 1))
        tr_b_temp = make_torch_batch(ds, ge3_idx, x0_mb, u_mb, max_obs_wave_idx=max_tr_w)
        full_b_temp = make_torch_batch(ds, ge3_idx, x0_mb, u_mb, max_obs_wave_idx=None, freeze_future_from_wave_idx=max_tr_w)

        # Fit M4, M5, M6, M7 strictly on waves <= max_tr_w, then forecast future waves without assimilating w > max_tr_w!
        p4_mu_b, p4_sd_b = fit_and_predict_m4_naive_neural_ode(tr_b_temp, full_b_temp, max_assimilate_wave_idx=max_tr_w, n_epochs=140)

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

        ge3_pos_map = pd.Series(np.arange(len(ge3_idx), dtype=np.int32), index=ge3_idx)

        for hz_tag, w_targ in target_specs:
            split_label = f"{train_tag}__{hz_tag}"
            te_pairs_hz = extract_transition_pairs(ds, ge3_idx, fixed_origin_max_wave=max_tr_w, fixed_target_wave=w_targ)
            loc_rows = pd.Series(te_pairs_hz["subj_idx"]).map(ge3_pos_map).to_numpy(dtype=np.int32)

            mu_hz = {}
            sd_hz = {}
            mu_hz["M1_Persistence_AgeNormative"], sd_hz["M1_Persistence_AgeNormative"] = predict_m1_persistence_agenorm(tr_pairs_b, te_pairs_hz)
            mu_hz["M2_Linear_Mixed_Effects_BLUP"], sd_hz["M2_Linear_Mixed_Effects_BLUP"] = predict_m2_lmm_empirical_bayes(
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
            sub_ill = ds["illit"][te_pairs_hz["subj_idx"]]
            sub_com = ds["comm_id"][te_pairs_hz["subj_idx"]]
            for mk in core_7_keys:
                ov, sc_r, pn, pc = evaluate_predictions_on_targets(
                    mk, split_label, te_pairs_hz["Y"], te_pairs_hz["M"], mu_hz[mk], sd_hz[mk], te_pairs_hz["last_Y"], sub_ill, sub_com
                )
                hz_sum.append(ov)
                cv_scale_rows.extend(sc_r)
                hz_pnll[mk] = pn
                hz_pcrps[mk] = pc
            temp_all_dfs.append(attach_paired_significance_vs_m7(hz_sum, hz_pnll, hz_pcrps))

    temp_df = pd.concat(temp_all_dfs, axis=0, ignore_index=True)
    temp_df.to_csv(OUT_DIR / "charls_temporal_holdout_benchmark.csv", index=False)
    pd.DataFrame(cv_scale_rows).to_csv(OUT_DIR / "scale_stratified_fidelity_gain.csv", index=False)

    # =========================================================================
    # FULL-COHORT STRUCTURAL PARAMETER FIT & INDIVIDUAL DIGITAL TWIN EXPORT (N = 20,928)
    # =========================================================================
    print("[Full Cohort Twin Fit] Fitting final Psychometric-Coupled-SDE across all N = 20,928 participants...")
    all_idx = np.arange(n_subj, dtype=np.int32)
    x0_all_m = np.mean(ds["x0_raw"], axis=0)
    u_all_m = np.mean(ds["u_ctrl_raw"], axis=(0, 1))
    b_full = make_torch_batch(ds, all_idx, x0_all_m, u_all_m)

    torch.manual_seed(SEED)
    m6_full = PsychometricCoupledSDE(
        CHARLS_SCALE_SPECS,
        d_x0=b_full["x0"].shape[1],
        d_u=b_full["u_ctrl"].shape[2],
        d_z_mod=0,
        coupled=False,
        psychometric_filter=True,
        n_gh_1d=5,
        use_subtest_traits=True,
        d_prot=2,
    )
    fit_psychometric_sde_model(m6_full, b_full, lr_adam=0.038, n_lbfgs_steps=55, l2_reg=1.5e-4, seed=SEED, verbose=False)

    torch.manual_seed(SEED)
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
    fit_psychometric_sde_model(
        m7_full, b_full, warm_from=m6_full, n_lbfgs_steps=55, l2_reg=1.5e-4, l2_mod_mult=5.0, seed=SEED, verbose=True
    )

    with torch.no_grad():
        out_full = m7_full.filter_and_loglik(b_full)
        eta_filt = out_full["filt_eta_means"].cpu().numpy()  # (N, 5, 2)
        b_filt = out_full["filt_b_means"][:, -1, :].cpu().numpy()  # (N, 2)
        j_ind = out_full["jacobian_batch"].cpu().numpy()  # (N, 2, 2)
        kap_v = (torch.nn.functional.softplus(m7_full.raw_kappa) + 0.05).cpu().numpy()
        lam_v = m7_full.get_loadings().cpu().numpy()
        nu_v = m7_full.nu.cpu().numpy()
        gam_v = m7_full.get_retest_gamma().cpu().numpy()
        sig_v = (torch.nn.functional.softplus(m7_full.raw_sigma_obs) + 1e-2).cpu().numpy()
        del_v = m7_full.delta_het.cpu().numpy()
        b_ctrl_v = m7_full.B_ctrl.cpu().numpy()

    # Compute Standardized Directed Coupling Dominance Index D_i (§4.1 of SAP_v1_FROZEN.md)
    sd_eta_cog = float(np.std(eta_filt[:, 0, 0], ddof=1))
    sd_eta_dep = float(np.std(eta_filt[:, 0, 1], ddof=1))
    f_c2d = np.abs(j_ind[:, 0, 1]) * sd_eta_dep  # |J_{cog<-dep, i}| * sigma_{eta_dep}
    f_d2c = np.abs(j_ind[:, 1, 0]) * sd_eta_cog  # |J_{dep<-cog, i}| * sigma_{eta_cog}
    dom_idx = (f_c2d - f_d2c) / np.maximum(f_c2d + f_d2c, 1e-8)
    phenotype = np.where(
        dom_idx > 0.15,
        "1_Affective_Driven_Cognitive_Decline",
        np.where(dom_idx < -0.15, "2_Cognitive_Driven_Affective_Collapse", "3_Bidirectional_Coupled_Equilibrium"),
    )

    twins_df = pd.DataFrame(
        {
            "ID": ds["ids"],
            "communityID": ds["comm_id"],
            "n_coupled_waves": ds["n_coupled_waves"],
            "illiterate": ds["illit"],
            "ipcw_weight": ds["ipcw"],
            "J_cog_self": j_ind[:, 0, 0],
            "J_cog_from_dep": j_ind[:, 0, 1],
            "J_dep_from_cog": j_ind[:, 1, 0],
            "J_dep_self": j_ind[:, 1, 1],
            "dominance_index_D": dom_idx,
            "coupling_phenotype": phenotype,
            "random_trait_b_cog": b_filt[:, 0],
            "random_trait_b_dep": b_filt[:, 1],
        }
    )
    for w in range(5):
        twins_df[f"eta_cog_w{w+1}"] = eta_filt[:, w, 0]
        twins_df[f"eta_dep_w{w+1}"] = eta_filt[:, w, 1]
    twins_df.to_parquet(DATA_DIR / "charls_sde_individual_twins.parquet", index=False)

    # Save structural & psychometric observation parameter table
    param_rows = []
    for d, sp in enumerate(CHARLS_SCALE_SPECS):
        param_rows.append(
            {
                "scale": sp.name,
                "domain": "Cognitive_Reserve" if sp.domain == 0 else "Affective_Load",
                "nu_intercept": round(float(nu_v[d]), 5),
                "lambda_loading": round(float(lam_v[d]), 5),
                "gamma_retest_log1p_k": round(float(gam_v[d]), 5),
                "sigma_obs_base": round(float(sig_v[d]), 5),
                "delta_illiteracy_heteroscedasticity": round(float(del_v[d]), 5),
                "illiterate_sd_multiplier": round(float(math.exp(0.5 * del_v[d])), 4),
            }
        )
    pd.DataFrame(param_rows).to_csv(OUT_DIR / "charls_sde_observation_parameters.csv", index=False)

    u_driver_names = [
        "age_10yr_drift", "optimal_sleep_6_to_85h", "poor_vision", "poor_hearing",
        "social_active", "internet_use", "phys_active_modvig", "cardiometabolic_burden",
    ]
    struct_payload = {
        "n_cohort": int(n_subj),
        "kappa_cog": round(float(kap_v[0]), 5),
        "kappa_dep": round(float(kap_v[1]), 5),
        "J_cog_from_dep_mean": round(float(np.mean(j_ind[:, 0, 1])), 5),
        "J_dep_from_cog_mean": round(float(np.mean(j_ind[:, 1, 0])), 5),
        "sd_eta_cog_w1": round(sd_eta_cog, 5),
        "sd_eta_dep_w1": round(sd_eta_dep, 5),
        "modifiable_driver_B_cog": {nm: round(float(b_ctrl_v[0, k]), 5) for k, nm in enumerate(u_driver_names)},
        "modifiable_driver_B_dep": {nm: round(float(b_ctrl_v[1, k]), 5) for k, nm in enumerate(u_driver_names)},
        "x0_center": [round(float(x), 6) for x in x0_all_m],
        "u_center": [round(float(x), 6) for x in u_all_m],
        "phenotype_counts": pd.Series(phenotype).value_counts().to_dict(),
    }
    with open(OUT_DIR / "charls_sde_structural_parameters.json", "w", encoding="utf-8") as f:
        json.dump(struct_payload, f, indent=2)

    # Save PyTorch model checkpoint for Phase 3 G-Computation exact structural counterfactual rollouts!
    torch.save(
        {
            "state_dict": m7_full.state_dict(),
            "x0_all_m": x0_all_m,
            "u_all_m": u_all_m,
            "u_driver_names": u_driver_names,
        },
        DATA_DIR / "charls_m7_psychometric_sde_checkpoint.pt",
    )

    # =========================================================================
    # PAIRED VENOUS BLOOD (N = 7,438) & NHANES EXTERNAL CONSTRUCT VALIDATION (N = 3,014)
    # =========================================================================
    pb_mod_df = run_paired_blood_jacobian_moderation(ds)
    pb_mod_df.to_csv(OUT_DIR / "paired_blood_jacobian_moderation.csv", index=False)

    print("[NHANES External Check] Running cross-cohort psychometric construct validation (N = 3,014 older adults >= 60)...")
    nh_val_df = run_nhanes_external_construct_validation()
    nh_val_df.to_csv(OUT_DIR / "nhanes_cross_cohort_validation.csv", index=False)

    print("\n=== Split A (5-Fold Community-Grouped CV, N = 20,928) Benchmark Summary ===")
    print(
        cv_df[
            [
                "model", "co_primary_tobit_NLL", "co_primary_CRPS_norm",
                "PICP_90_coverage_pct", "AUROC_elevated_depression_cesd10_ge10",
                "AUROC_cognitive_impairment_tr20_le6", "holm_p_NLL_vs_6baselines",
            ]
        ].to_string(index=False)
    )
    print("\n=== Split B (Multi-Horizon Temporal Holdouts, N = 17,360) Benchmark Summary ===")
    print(
        temp_df[
            [
                "split", "model", "co_primary_tobit_NLL", "co_primary_CRPS_norm",
                "AUROC_elevated_depression_cesd10_ge10", "AUROC_cognitive_impairment_tr20_le6",
                "holm_p_NLL_vs_6baselines",
            ]
        ].to_string(index=False)
    )
    print(f"[Pillar 2 Part B DONE] Completed all benchmarks, paired-blood moderation & external validation in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
