#!/usr/bin/env python3
"""
Phase 0 — Reproducible NHANES (2011–2014) Multimodal Cohort Extraction & Quality Control
=========================================================================================
Target Project: PsyTwin-SDE (npj Digital Medicine Collection)
Governance Rules (CLAUDE.md & lab-constitution):
  1. STRICT READ-ONLY RAW DATA: Reads ${NHANES_RAW_DIR} strictly read-only; writes exclusively inside
     ./.
  2. SAS XPORT IEEE ZERO FIX: Masks IBM-370 subnormal zero artifacts (|x| < 1e-70 -> 0.0)
     and decodes byte strings cleanly.
  3. ZERO Brain-Kidney Axis & ZERO Raw DICOM Imaging.
  4. 100% Real Script-Computed Numbers.
"""

from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_NHANES = Path(os.environ.get("NHANES_RAW_DIR", str(PROJECT_ROOT / "data_raw" / "NHANES")))
OUT_DATA = PROJECT_ROOT / "data_processed"
OUT_AUDIT = PROJECT_ROOT / "results" / "phase0_cohort_audit"


def read_xpt_clean(rel_path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read an NHANES SAS XPORT (.xpt) file strictly read-only and fix IBM-370 zero artifacts."""
    fpath = RAW_NHANES / rel_path
    if not fpath.exists():
        raise FileNotFoundError(f"Expected NHANES raw file not found: {fpath}")
    df = pd.read_sas(fpath, format="xport")
    if usecols is not None:
        present = [c for c in usecols if c in df.columns]
        df = df[present].copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].mask(df[num_cols].abs() < 1e-70, 0.0)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].apply(
            lambda v: v.decode("latin1").strip() if isinstance(v, (bytes, bytearray)) else (str(v).strip() if pd.notna(v) else np.nan)
        )
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].round().astype("int64")
    return df


def load_cycle_pair(rel_g: str, rel_h: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenate 2011-2012 (_G) and 2013-2014 (_H) NHANES modules."""
    df_g = read_xpt_clean(rel_g, usecols=usecols)
    df_h = read_xpt_clean(rel_h, usecols=usecols)
    return pd.concat([df_g, df_h], axis=0, ignore_index=True)


def parse_hms_to_seconds(val: object) -> float:
    """Convert 'HH:MM:SS' string to seconds since midnight."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    parts = s.split(":")
    if len(parts) != 3:
        return np.nan
    try:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        return np.nan


def compute_participant_circadian_metrics(paxhr: pd.DataFrame, paxday: pd.DataFrame, paxhd: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct exact 24h clock hours from PAXHD.PAXFTIME + PAXHR.PAXSSNHP / 80 Hz,
    clean hourly wrist triaxial MIMS + ambient light (lux) + sleep/wake state trajectories,
    and compute participant-level QC & non-parametric circadian rhythm metrics (M10, L5, RA, IS, IV).
    """
    t0 = time.time()
    hd = paxhd[["SEQN", "PAXSTS", "PAXFDAY", "PAXLDAY", "PAXFTIME", "PAXHAND"]].copy()
    hd["ftime_sec"] = hd["PAXFTIME"].apply(parse_hms_to_seconds)

    # Clean PAXDAY for day-level valid wear auditing
    day = paxday.copy()
    day["day_num"] = pd.to_numeric(day["PAXDAYD"], errors="coerce")
    day["weekday"] = pd.to_numeric(day["PAXDAYWD"], errors="coerce")
    day["wear_min_day"] = day["PAXWWMD"].fillna(0.0) + day["PAXSWMD"].fillna(0.0)
    day["valid_day_16h"] = ((day["PAXQFD"].fillna(1.0) == 0.0) & (day["PAXVMD"].fillna(0.0) >= 960.0) & (day["wear_min_day"] >= 960.0)).astype(int)
    day["valid_day_10h"] = ((day["PAXQFD"].fillna(1.0) == 0.0) & (day["PAXVMD"].fillna(0.0) >= 600.0) & (day["wear_min_day"] >= 600.0)).astype(int)

    day_agg = (
        day.groupby("SEQN", as_index=False)
        .agg(
            n_pax_days_total=("day_num", "count"),
            n_pax_valid_days_16h=("valid_day_16h", "sum"),
            n_pax_valid_days_10h=("valid_day_10h", "sum"),
            pax_total_wear_hours=("wear_min_day", lambda s: float(s.sum()) / 60.0),
        )
    )

    # Clean PAXHR (hourly triaxial MIMS + ambient light lux + wake/sleep minutes)
    hr = paxhr.merge(hd[["SEQN", "PAXSTS", "ftime_sec"]], on="SEQN", how="left")
    hr["day_num"] = pd.to_numeric(hr["PAXDAYH"], errors="coerce").fillna(-1).astype("int16")
    hr["weekday"] = pd.to_numeric(hr["PAXDAYWH"], errors="coerce").fillna(-1).astype("int16")

    # Exact elapsed seconds from start of Day 1 recording (80 Hz sampling -> 288,000 samples/hour)
    # PAXHR bins align to wall-clock hour boundaries, so (ftime_sec + elapsed_sec) // 3600 gives unique contiguous hour indices
    elapsed_sec = hr["PAXSSNHP"].fillna(0.0) / 80.0
    clock_sec = hr["ftime_sec"].fillna(0.0) + elapsed_sec
    hr["seq_hour_idx"] = np.floor((clock_sec + 1.0) / 3600.0).astype("int32")
    hr["clock_hour"] = (hr["seq_hour_idx"] % 24).astype("int8")

    # Normalize hourly sums by valid minutes (PAXVMH) and wear minutes
    hr["wear_min"] = hr["PAXWWMH"].fillna(0.0) + hr["PAXSWMH"].fillna(0.0)
    denom_vm = hr["PAXVMH"].where(hr["PAXVMH"] > 0, np.nan)
    denom_wear = hr["wear_min"].where(hr["wear_min"] > 0, np.nan)

    hr["mims_per_min"] = np.where(hr["PAXMTSH"] >= 0, hr["PAXMTSH"] / denom_vm, np.nan)
    hr["mims_per_wear_min"] = np.where(hr["PAXMTSH"] >= 0, hr["PAXMTSH"] / denom_wear, np.nan)
    hr["lux_per_min"] = np.where(hr["PAXLXSH"] >= 0, hr["PAXLXSH"] / denom_vm, np.nan)
    hr["sleep_frac_wear"] = hr["PAXSWMH"] / denom_wear
    hr["wake_frac_wear"] = hr["PAXWWMH"] / denom_wear
    hr["valid_hour_flag"] = (
        (hr["PAXQFH"].fillna(1.0) == 0.0)
        & (hr["PAXVMH"].fillna(0.0) >= 45.0)
        & (hr["wear_min"] >= 30.0)
        & hr["mims_per_min"].notna()
    ).astype("int8")

    # Merge valid_day_16h flag onto hourly rows for clean analytical subsetting
    day_flags = day[["SEQN", "day_num", "valid_day_16h", "valid_day_10h"]].drop_duplicates(["SEQN", "day_num"])
    hr = hr.merge(day_flags, on=["SEQN", "day_num"], how="left")
    hr["valid_day_16h"] = hr["valid_day_16h"].fillna(0).astype("int8")
    hr["valid_day_10h"] = hr["valid_day_10h"].fillna(0).astype("int8")

    keep_hr_cols = [
        "SEQN",
        "day_num",
        "weekday",
        "seq_hour_idx",
        "clock_hour",
        "PAXTMH",
        "PAXVMH",
        "PAXWWMH",
        "PAXSWMH",
        "PAXNWMH",
        "PAXUMH",
        "wear_min",
        "PAXMTSH",
        "mims_per_min",
        "mims_per_wear_min",
        "PAXLXSH",
        "lux_per_min",
        "sleep_frac_wear",
        "wake_frac_wear",
        "PAXQFH",
        "valid_hour_flag",
        "valid_day_16h",
        "valid_day_10h",
    ]
    hr_clean = hr[keep_hr_cols].sort_values(["SEQN", "seq_hour_idx"]).reset_index(drop=True)

    # Compute fast vectorized participant-level circadian rhythm summaries on valid hours
    vhr = hr_clean[hr_clean["valid_hour_flag"] == 1].copy()
    vhr["log_mims"] = np.log1p(vhr["mims_per_min"].clip(lower=0.0))
    vhr["log_lux"] = np.log1p(vhr["lux_per_min"].clip(lower=0.0))

    # 24-hour clock profile per participant (for M10, L5, RA, IS, cosinor)
    prof24 = (
        vhr.groupby(["SEQN", "clock_hour"], as_index=False)
        .agg(
            mims_h=("mims_per_min", "mean"),
            lux_h=("lux_per_min", "mean"),
            sleep_h=("sleep_frac_wear", "mean"),
            n_obs_h=("mims_per_min", "count"),
        )
    )
    piv_mims = prof24.pivot(index="SEQN", columns="clock_hour", values="mims_h")
    piv_mims = piv_mims.reindex(columns=list(range(24)))

    # Circular 10-hour (M10) and 5-hour (L5) rolling averages across the 24h profile (computed strictly on valid_24 rows -> zero empty-slice warnings)
    arr24 = piv_mims.to_numpy(dtype=float)  # (N_subj, 24)
    n_bins_present = np.sum(~np.isnan(arr24), axis=1)
    valid_24 = n_bins_present >= 20

    m10 = np.full(arr24.shape[0], np.nan)
    l5 = np.full(arr24.shape[0], np.nan)
    mesor = np.full(arr24.shape[0], np.nan)
    cosinor_amp = np.full(arr24.shape[0], np.nan)
    cosinor_acrophase_hr = np.full(arr24.shape[0], np.nan)

    if np.any(valid_24):
        arr_v = arr24[valid_24]
        arr_v_circ = np.concatenate([arr_v, arr_v[:, :10]], axis=1)
        m10_v = np.full(arr_v.shape[0], np.nan)
        l5_v = np.full(arr_v.shape[0], np.nan)
        for start_h in range(24):
            w10 = np.nanmean(arr_v_circ[:, start_h : start_h + 10], axis=1)
            w5 = np.nanmean(arr_v_circ[:, start_h : start_h + 5], axis=1)
            m10_v = np.where(np.isnan(m10_v) | (w10 > m10_v), w10, m10_v)
            l5_v = np.where(np.isnan(l5_v) | (w5 < l5_v), w5, l5_v)
        m10[valid_24] = m10_v
        l5[valid_24] = l5_v

        # Cosinor 24h fundamental harmonic on 24h mean profile
        hours_rad = 2.0 * np.pi * np.arange(24) / 24.0
        cos_vec = np.cos(hours_rad)
        sin_vec = np.sin(hours_rad)
        mesor_v = np.nanmean(arr_v, axis=1)
        centered_v = arr_v - mesor_v[:, None]
        c_coef = 2.0 * np.nanmean(centered_v * cos_vec[None, :], axis=1)
        s_coef = 2.0 * np.nanmean(centered_v * sin_vec[None, :], axis=1)
        mesor[valid_24] = mesor_v
        cosinor_amp[valid_24] = np.sqrt(c_coef**2 + s_coef**2)
        acrophase_rad = np.mod(np.arctan2(s_coef, c_coef), 2.0 * np.pi)
        cosinor_acrophase_hr[valid_24] = acrophase_rad * 24.0 / (2.0 * np.pi)

    ra = np.where((m10 + l5) > 0, (m10 - l5) / (m10 + l5), np.nan)

    prof_metrics = pd.DataFrame(
        {
            "SEQN": piv_mims.index.values,
            "pax_24h_bins_covered": n_bins_present,
            "pax_M10": m10,
            "pax_L5": l5,
            "pax_RA": ra,
            "pax_cosinor_mesor": mesor,
            "pax_cosinor_amp": cosinor_amp,
            "pax_cosinor_acrophase_hr": cosinor_acrophase_hr,
        }
    )

    # Interdaily Stability (IS), Intradaily Variability (IV), and Photic-Motor Coupling across hourly series
    # Vectorized groupby computations
    vhr_overall = (
        vhr.groupby("SEQN", as_index=False)
        .agg(
            n_valid_hours=("mims_per_min", "count"),
            pax_mean_mims_min=("mims_per_min", "mean"),
            pax_sd_mims_min=("mims_per_min", "std"),
            pax_mean_lux_min=("lux_per_min", "mean"),
            pax_sleep_frac=("sleep_frac_wear", "mean"),
            mean_log_mims=("log_mims", "mean"),
            mean_log_lux=("log_lux", "mean"),
        )
    )
    vhr_overall["pax_est_sleep_hours_day"] = 24.0 * vhr_overall["pax_sleep_frac"]

    # IS numerator: (N / p) * sum_h (x_bar_h - x_bar)^2 / sum_i (x_i - x_bar)^2
    vhr = vhr.merge(vhr_overall[["SEQN", "pax_mean_mims_min", "n_valid_hours", "mean_log_mims", "mean_log_lux"]], on="SEQN", how="left")
    vhr["sq_dev_total"] = (vhr["mims_per_min"] - vhr["pax_mean_mims_min"]) ** 2

    # Contiguous 1-hour differences for Intradaily Variability (IV)
    vhr["prev_seq_hour"] = vhr.groupby("SEQN")["seq_hour_idx"].shift(1)
    vhr["prev_mims"] = vhr.groupby("SEQN")["mims_per_min"].shift(1)
    vhr["is_contig"] = (vhr["seq_hour_idx"] - vhr["prev_seq_hour"]) == 1
    vhr["sq_diff1"] = np.where(vhr["is_contig"], (vhr["mims_per_min"] - vhr["prev_mims"]) ** 2, np.nan)

    # Photic-motor covariance components
    vhr["dm"] = vhr["log_mims"] - vhr["mean_log_mims"]
    vhr["dl"] = vhr["log_lux"] - vhr["mean_log_lux"]
    vhr["dm_dl"] = vhr["dm"] * vhr["dl"]
    vhr["dm2"] = vhr["dm"] ** 2
    vhr["dl2"] = vhr["dl"] ** 2

    ts_agg = (
        vhr.groupby("SEQN", as_index=False)
        .agg(
            ss_total=("sq_dev_total", "sum"),
            ms_diff1=("sq_diff1", "mean"),
            n_contig_pairs=("sq_diff1", "count"),
            cov_ml=("dm_dl", "sum"),
            ss_m=("dm2", "sum"),
            ss_l=("dl2", "sum"),
        )
    )
    ts_agg = ts_agg.merge(vhr_overall, on="SEQN", how="left")
    denom_var = ts_agg["ss_total"] / ts_agg["n_valid_hours"].where(ts_agg["n_valid_hours"] > 0, np.nan)
    ts_agg["pax_IV"] = np.where((denom_var > 0) & (ts_agg["n_contig_pairs"] >= 48), ts_agg["ms_diff1"] / denom_var, np.nan)
    ts_agg["pax_photic_motor_r"] = np.where(
        (ts_agg["ss_m"] > 0) & (ts_agg["ss_l"] > 0) & (ts_agg["n_valid_hours"] >= 48),
        ts_agg["cov_ml"] / np.sqrt(ts_agg["ss_m"] * ts_agg["ss_l"]),
        np.nan,
    )

    # IS from prof24 variance over overall variance
    prof24_m = prof24.merge(vhr_overall[["SEQN", "pax_mean_mims_min"]], on="SEQN", how="left")
    prof24_m["sq_dev_h"] = (prof24_m["mims_h"] - prof24_m["pax_mean_mims_min"]) ** 2
    is_num = prof24_m.groupby("SEQN", as_index=False).agg(ms_between_h=("sq_dev_h", "mean"), n_h=("sq_dev_h", "count"))
    ts_agg = ts_agg.merge(is_num, on="SEQN", how="left")
    ts_agg["pax_IS"] = np.where(
        (denom_var > 0) & (ts_agg["n_h"] >= 20) & (ts_agg["n_valid_hours"] >= 72),
        (ts_agg["ms_between_h"] / denom_var).clip(0.0, 1.0),
        np.nan,
    )

    pax_subj = hd[["SEQN", "PAXSTS", "PAXHAND"]].merge(day_agg, on="SEQN", how="left")
    pax_subj = pax_subj.merge(prof_metrics, on="SEQN", how="left")
    pax_subj = pax_subj.merge(
        ts_agg[
            [
                "SEQN",
                "n_valid_hours",
                "pax_mean_mims_min",
                "pax_sd_mims_min",
                "pax_mean_lux_min",
                "pax_est_sleep_hours_day",
                "pax_IS",
                "pax_IV",
                "pax_photic_motor_r",
            ]
        ],
        on="SEQN",
        how="left",
    )
    pax_subj["pax_has_data"] = (pax_subj["PAXSTS"] == 1.0).astype(int)
    pax_subj["pax_eligible_primary_4d16h"] = (
        (pax_subj["PAXSTS"] == 1.0)
        & (pax_subj["n_pax_valid_days_16h"].fillna(0) >= 4)
        & (pax_subj["pax_24h_bins_covered"].fillna(0) == 24)
    ).astype(int)
    pax_subj["pax_eligible_sens_3d16h"] = (
        (pax_subj["PAXSTS"] == 1.0)
        & (pax_subj["n_pax_valid_days_16h"].fillna(0) >= 3)
        & (pax_subj["pax_24h_bins_covered"].fillna(0) >= 22)
    ).astype(int)
    pax_subj["pax_eligible_sens_4d10h"] = (
        (pax_subj["PAXSTS"] == 1.0) & (pax_subj["n_pax_valid_days_10h"].fillna(0) >= 4)
    ).astype(int)

    print(f"[NHANES PAX] Processed {len(hr_clean):,} hourly epochs across {len(pax_subj):,} PAXHD records in {time.time() - t0:.1f}s")
    return pax_subj, hr_clean


def summarize_psychometric_scale(s: pd.Series, name: str, theo_min: float, theo_max: float, higher_is_better: bool) -> Dict[str, object]:
    """Compute empirical censoring (floor/ceiling), distribution Moments, and normality diagnostics."""
    valid = s.dropna().astype(float)
    n = int(len(valid))
    if n == 0:
        return {"scale": name, "n": 0}
    emp_min = float(valid.min())
    emp_max = float(valid.max())
    floor_n = int((valid <= (theo_min if not np.isnan(theo_min) else emp_min)).sum())
    ceil_n = int((valid >= (theo_max if not np.isnan(theo_max) else emp_max)).sum()) if not np.isnan(theo_max) else 0
    q25, q50, q75 = np.percentile(valid, [25, 50, 75])
    return {
        "scale": name,
        "direction": "higher_better" if higher_is_better else "higher_worse",
        "n_valid": n,
        "theoretical_min": theo_min,
        "theoretical_max": theo_max,
        "empirical_min": round(emp_min, 3),
        "empirical_max": round(emp_max, 3),
        "floor_n": floor_n,
        "floor_pct": round(100.0 * floor_n / n, 3),
        "ceiling_n": ceil_n,
        "ceiling_pct": round(100.0 * ceil_n / n, 3) if not np.isnan(theo_max) else 0.0,
        "mean": round(float(valid.mean()), 3),
        "sd": round(float(valid.std(ddof=1)), 3),
        "median": round(float(q50), 3),
        "iqr": round(float(q75 - q25), 3),
        "q25": round(float(q25), 3),
        "q75": round(float(q75), 3),
        "skewness": round(float(stats.skew(valid, bias=False)), 3),
        "excess_kurtosis": round(float(stats.kurtosis(valid, bias=False)), 3),
    }


def format_mean_sd(s: pd.Series) -> str:
    v = s.dropna()
    if len(v) == 0:
        return "NA"
    return f"{v.mean():.2f} ({v.std(ddof=1):.2f})"


def format_median_iqr(s: pd.Series) -> str:
    v = s.dropna()
    if len(v) == 0:
        return "NA"
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return f"{q50:.2f} [{q25:.2f}, {q75:.2f}]"


def format_n_pct(cond: pd.Series) -> str:
    v = cond.dropna()
    if len(v) == 0:
        return "NA"
    cnt = int((v == 1).sum())
    return f"{cnt:,} ({100.0 * cnt / len(v):.1f}%)"


def main() -> None:
    t_start = time.time()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)

    print("=== Phase 0: NHANES 2011–2014 Multimodal Extraction & QC ===")

    # 1. Demographics (DEMO_G + DEMO_H)
    demo_cols = [
        "SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR",
        "RIDRETH1", "RIDRETH3", "DMDEDUC2", "DMDMARTL", "INDFMPIR",
        "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
    ]
    demo_g = read_xpt_clean("2011-2012/Demographics/DEMO_G.xpt", demo_cols)
    demo_g["cycle"] = "2011-2012"
    demo_h = read_xpt_clean("2013-2014/Demographics/DEMO_H.xpt", demo_cols)
    demo_h["cycle"] = "2013-2014"
    df = pd.concat([demo_g, demo_h], axis=0, ignore_index=True)

    df["age_years"] = df["RIDAGEYR"].astype(float)
    df["female"] = np.where(df["RIAGENDR"].isin([1.0, 2.0]), (df["RIAGENDR"] == 2.0).astype(float), np.nan)
    df["mec_examined"] = (df["RIDSTATR"] == 2.0).astype(int)
    df["wtmec4yr"] = 0.5 * df["WTMEC2YR"].fillna(0.0)

    race_map = {
        1.0: "Mexican_American",
        2.0: "Other_Hispanic",
        3.0: "Non_Hispanic_White",
        4.0: "Non_Hispanic_Black",
        6.0: "Non_Hispanic_Asian",
        7.0: "Other_Multi",
    }
    df["race_ethnicity"] = df["RIDRETH3"].map(race_map).fillna("Unknown")

    df["educ_adult_code"] = df["DMDEDUC2"].where(df["DMDEDUC2"].isin([1.0, 2.0, 3.0, 4.0, 5.0]), np.nan)
    educ_map = {
        1.0: "Less_than_9th",
        2.0: "9th_to_11th",
        3.0: "High_School_GED",
        4.0: "Some_College_AA",
        5.0: "College_Graduate_or_Above",
    }
    df["education_cat"] = df["educ_adult_code"].map(educ_map)
    df["low_education_lt_hs"] = np.where(df["educ_adult_code"].notna(), (df["educ_adult_code"] <= 2.0).astype(float), np.nan)
    df["pir"] = df["INDFMPIR"]

    # 2. Serum Neurofilament Light Chain (sNfL: 2013–2014 SSSNFL_H.xpt)
    snfl = read_xpt_clean("2013-2014/Laboratory/SSSNFL_H.xpt")
    snfl = snfl.rename(
        columns={
            "SSSNFL": "snfl_pgml",
            "WTSSNH2Y": "wtssnh2y",
            "SSNFLH": "snfl_above_uloq",
            "SSNFLL": "snfl_below_lloq",
        }
    )
    snfl["log_snfl"] = np.where(snfl["snfl_pgml"] > 0, np.log(snfl["snfl_pgml"]), np.nan)
    snfl["log10_snfl"] = np.where(snfl["snfl_pgml"] > 0, np.log10(snfl["snfl_pgml"]), np.nan)
    df = df.merge(snfl[["SEQN", "snfl_pgml", "log_snfl", "log10_snfl", "wtssnh2y", "snfl_above_uloq", "snfl_below_lloq"]], on="SEQN", how="left")

    # 3. Zero-Ceiling 3-Domain Neurocognitive Battery (CFQ_G + CFQ_H, adults >=60)
    cfq_cols = ["SEQN", "CFASTAT", "CFDCST1", "CFDCST2", "CFDCST3", "CFDCSR", "CFDAST", "CFDDS"]
    cfq = load_cycle_pair("2011-2012/Questionnaire/CFQ_G.xpt", "2013-2014/Questionnaire/CFQ_H.xpt", cfq_cols)
    cfq = cfq.rename(
        columns={
            "CFASTAT": "cfq_status",
            "CFDCST1": "cerad_t1",
            "CFDCST2": "cerad_t2",
            "CFDCST3": "cerad_t3",
            "CFDCSR": "cerad_dr",
            "CFDAST": "animal_fluency",
            "CFDDS": "dsst",
        }
    )
    cfq["cerad_ir_sum"] = np.where(
        cfq[["cerad_t1", "cerad_t2", "cerad_t3"]].notna().all(axis=1),
        cfq["cerad_t1"] + cfq["cerad_t2"] + cfq["cerad_t3"],
        np.nan,
    )
    cfq["cerad_total_40"] = np.where(
        cfq["cerad_ir_sum"].notna() & cfq["cerad_dr"].notna(),
        cfq["cerad_ir_sum"] + cfq["cerad_dr"],
        np.nan,
    )
    for col in ["cerad_ir_sum", "cerad_dr", "animal_fluency", "dsst"]:
        mu = cfq[col].mean()
        sd = cfq[col].std(ddof=1)
        cfq[f"z_{col}"] = (cfq[col] - mu) / sd
    cfq["z_memory_domain"] = 0.5 * (cfq["z_cerad_ir_sum"] + cfq["z_cerad_dr"])
    cfq["global_cognition_z"] = np.where(
        cfq[["z_memory_domain", "z_animal_fluency", "z_dsst"]].notna().all(axis=1),
        (cfq["z_memory_domain"] + cfq["z_animal_fluency"] + cfq["z_dsst"]) / 3.0,
        np.nan,
    )
    cfq["cfq_complete_3domain"] = cfq["global_cognition_z"].notna().astype(int)
    df = df.merge(cfq, on="SEQN", how="left")

    # 4. Depression PHQ-9 (DPQ_G + DPQ_H)
    dpq_items = [f"DPQ0{i}0" for i in range(1, 10)]
    dpq = load_cycle_pair("2011-2012/Questionnaire/DPQ_G.xpt", "2013-2014/Questionnaire/DPQ_H.xpt", ["SEQN"] + dpq_items)
    for c in dpq_items:
        dpq[c] = dpq[c].where(dpq[c].isin([0.0, 1.0, 2.0, 3.0]), np.nan)
    n_valid_dpq = dpq[dpq_items].notna().sum(axis=1)
    dpq["phq9_total"] = np.where(n_valid_dpq == 9, dpq[dpq_items].sum(axis=1), np.nan)
    dpq["phq9_prorated"] = np.where(n_valid_dpq >= 7, dpq[dpq_items].mean(axis=1) * 9.0, np.nan)
    nosleep_items = [c for c in dpq_items if c != "DPQ030"]
    dpq["phq8_nosleep"] = np.where(dpq[nosleep_items].notna().sum(axis=1) == 8, dpq[nosleep_items].sum(axis=1), np.nan)
    dpq["dpq_sleep_item"] = dpq["DPQ030"]
    dpq["phq9_modsev_ge10"] = np.where(dpq["phq9_total"].notna(), (dpq["phq9_total"] >= 10.0).astype(float), np.nan)
    df = df.merge(dpq[["SEQN", "phq9_total", "phq9_prorated", "phq8_nosleep", "dpq_sleep_item", "phq9_modsev_ge10"]], on="SEQN", how="left")

    # 5. Self-Reported Sleep (SLQ_G + SLQ_H)
    slq = load_cycle_pair("2011-2012/Questionnaire/SLQ_G.xpt", "2013-2014/Questionnaire/SLQ_H.xpt", ["SEQN", "SLD010H", "SLQ050", "SLQ060"])
    slq["sleep_hours_self"] = slq["SLD010H"].where((slq["SLD010H"] >= 2.0) & (slq["SLD010H"] <= 24.0), np.nan)
    slq["sleep_trouble_doc"] = np.where(slq["SLQ050"].isin([1.0, 2.0]), (slq["SLQ050"] == 1.0).astype(float), np.nan)
    slq["sleep_disorder_doc"] = np.where(slq["SLQ060"].isin([1.0, 2.0]), (slq["SLQ060"] == 1.0).astype(float), np.nan)
    df = df.merge(slq[["SEQN", "sleep_hours_self", "sleep_trouble_doc", "sleep_disorder_doc"]], on="SEQN", how="left")

    # 6. Physical Examination & Laboratory Covariates
    bmx = load_cycle_pair("2011-2012/Examination/BMX_G.xpt", "2013-2014/Examination/BMX_H.xpt", ["SEQN", "BMXBMI", "BMXWAIST"])
    bmx = bmx.rename(columns={"BMXBMI": "bmi", "BMXWAIST": "waist_cm"})
    df = df.merge(bmx, on="SEQN", how="left")

    bpx = load_cycle_pair(
        "2011-2012/Examination/BPX_G.xpt",
        "2013-2014/Examination/BPX_H.xpt",
        ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    )
    sy_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in bpx.columns]
    di_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in bpx.columns]
    for c in sy_cols:
        bpx[c] = bpx[c].where(bpx[c] > 0, np.nan)
    for c in di_cols:
        bpx[c] = bpx[c].where(bpx[c] > 0, np.nan)
    bpx["sbp_mean"] = bpx[sy_cols].mean(axis=1)
    bpx["dbp_mean"] = bpx[di_cols].mean(axis=1)
    df = df.merge(bpx[["SEQN", "sbp_mean", "dbp_mean"]], on="SEQN", how="left")

    ghb = load_cycle_pair("2011-2012/Laboratory/GHB_G.xpt", "2013-2014/Laboratory/GHB_H.xpt", ["SEQN", "LBXGH"]).rename(columns={"LBXGH": "hba1c_pct"})
    df = df.merge(ghb, on="SEQN", how="left")

    tchol = load_cycle_pair("2011-2012/Laboratory/TCHOL_G.xpt", "2013-2014/Laboratory/TCHOL_H.xpt", ["SEQN", "LBXTC"]).rename(columns={"LBXTC": "total_chol_mgdl"})
    df = df.merge(tchol, on="SEQN", how="left")

    hdl = load_cycle_pair("2011-2012/Laboratory/HDL_G.xpt", "2013-2014/Laboratory/HDL_H.xpt", ["SEQN", "LBDHDD"]).rename(columns={"LBDHDD": "hdl_chol_mgdl"})
    df = df.merge(hdl, on="SEQN", how="left")

    biopro = load_cycle_pair(
        "2011-2012/Laboratory/BIOPRO_G.xpt",
        "2013-2014/Laboratory/BIOPRO_H.xpt",
        ["SEQN", "LBXSGL", "LBXSTR", "LBXSUA", "LBXSAL", "LBXSCR"],
    ).rename(
        columns={
            "LBXSGL": "glucose_mgdl",
            "LBXSTR": "triglycerides_mgdl",
            "LBXSUA": "uric_acid_mgdl",
            "LBXSAL": "albumin_gdl",
            "LBXSCR": "serum_creatinine_mgdl",  # strictly systemic sNfL clearance covariate
        }
    )
    df = df.merge(biopro, on="SEQN", how="left")

    cbc = load_cycle_pair(
        "2011-2012/Laboratory/CBC_G.xpt",
        "2013-2014/Laboratory/CBC_H.xpt",
        ["SEQN", "LBXWBCSI", "LBDNENO", "LBDLYMNO", "LBXHGB"],
    ).rename(
        columns={
            "LBXWBCSI": "wbc_10e9L",
            "LBDNENO": "neutrophils_10e9L",
            "LBDLYMNO": "lymphocytes_10e9L",
            "LBXHGB": "hemoglobin_gdl",
        }
    )
    cbc["nlr"] = np.where(cbc["lymphocytes_10e9L"] > 0, cbc["neutrophils_10e9L"] / cbc["lymphocytes_10e9L"], np.nan)
    df = df.merge(cbc, on="SEQN", how="left")

    # Lifestyle & Comorbidities (SMQ, ALQ, DIQ, MCQ)
    smq = load_cycle_pair("2011-2012/Questionnaire/SMQ_G.xpt", "2013-2014/Questionnaire/SMQ_H.xpt", ["SEQN", "SMQ020", "SMQ040"])
    smq["ever_smoked_100"] = np.where(smq["SMQ020"].isin([1.0, 2.0]), (smq["SMQ020"] == 1.0).astype(float), np.nan)
    smq["current_smoker"] = np.where(
        smq["SMQ020"] == 2.0,
        0.0,
        np.where(smq["SMQ040"].isin([1.0, 2.0]), 1.0, np.where(smq["SMQ040"] == 3.0, 0.0, np.nan)),
    )
    df = df.merge(smq[["SEQN", "ever_smoked_100", "current_smoker"]], on="SEQN", how="left")

    diq = load_cycle_pair("2011-2012/Questionnaire/DIQ_G.xpt", "2013-2014/Questionnaire/DIQ_H.xpt", ["SEQN", "DIQ010"])
    diq["diabetes_doc"] = np.where(diq["DIQ010"].isin([1.0, 2.0, 3.0]), (diq["DIQ010"] == 1.0).astype(float), np.nan)
    df = df.merge(diq[["SEQN", "diabetes_doc"]], on="SEQN", how="left")

    mcq = load_cycle_pair(
        "2011-2012/Questionnaire/MCQ_G.xpt",
        "2013-2014/Questionnaire/MCQ_H.xpt",
        ["SEQN", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"],
    )
    for raw_c, new_c in [("MCQ160B", "chf_doc"), ("MCQ160C", "chd_doc"), ("MCQ160E", "mi_doc"), ("MCQ160F", "stroke_doc")]:
        mcq[new_c] = np.where(mcq[raw_c].isin([1.0, 2.0]), (mcq[raw_c] == 1.0).astype(float), np.nan)
    mcq["cvd_any_doc"] = np.where(
        mcq[["chf_doc", "chd_doc", "mi_doc", "stroke_doc"]].notna().any(axis=1),
        (mcq[["chf_doc", "chd_doc", "mi_doc", "stroke_doc"]].max(axis=1) == 1.0).astype(float),
        np.nan,
    )
    df = df.merge(mcq[["SEQN", "chf_doc", "chd_doc", "mi_doc", "stroke_doc", "cvd_any_doc"]], on="SEQN", how="left")

    # Extended Pre-Phase-2 Confounder Lock (BPQ, ALQ, PAQ, DLQ/AUQ)
    bpq = load_cycle_pair(
        "2011-2012/Questionnaire/BPQ_G.xpt",
        "2013-2014/Questionnaire/BPQ_H.xpt",
        ["SEQN", "BPQ020", "BPQ040A", "BPQ050A", "BPQ080"],
    )
    bpq["hypertension_doc"] = np.where(bpq["BPQ020"].isin([1.0, 2.0]), (bpq["BPQ020"] == 1.0).astype(float), np.nan)
    bpq["bp_med_current"] = np.where(
        bpq["BPQ050A"] == 1.0,
        1.0,
        np.where((bpq["BPQ020"] == 2.0) | (bpq["BPQ040A"] == 2.0) | (bpq["BPQ050A"] == 2.0), 0.0, np.nan),
    )
    bpq["high_chol_doc"] = np.where(bpq["BPQ080"].isin([1.0, 2.0]), (bpq["BPQ080"] == 1.0).astype(float), np.nan)
    df = df.merge(bpq[["SEQN", "hypertension_doc", "bp_med_current", "high_chol_doc"]], on="SEQN", how="left")

    alq = load_cycle_pair("2011-2012/Questionnaire/ALQ_G.xpt", "2013-2014/Questionnaire/ALQ_H.xpt", ["SEQN", "ALQ101"])
    alq["alcohol_12yr"] = np.where(alq["ALQ101"].isin([1.0, 2.0]), (alq["ALQ101"] == 1.0).astype(float), np.nan)
    df = df.merge(alq[["SEQN", "alcohol_12yr"]], on="SEQN", how="left")

    paq = load_cycle_pair(
        "2011-2012/Questionnaire/PAQ_G.xpt",
        "2013-2014/Questionnaire/PAQ_H.xpt",
        ["SEQN", "PAQ605", "PAQ620", "PAQ650", "PAQ665", "PAD680"],
    )
    pa_items = ["PAQ605", "PAQ620", "PAQ650", "PAQ665"]
    for pc in pa_items:
        paq[pc] = paq[pc].where(paq[pc].isin([1.0, 2.0]), np.nan)
    paq["modvig_pa_any"] = np.where(
        paq[pa_items].notna().any(axis=1),
        (paq[pa_items].min(axis=1) == 1.0).astype(float),
        np.nan,
    )
    paq["leisure_modvig_pa"] = np.where(
        paq[["PAQ650", "PAQ665"]].notna().any(axis=1),
        (paq[["PAQ650", "PAQ665"]].min(axis=1) == 1.0).astype(float),
        np.nan,
    )
    paq["sedentary_min_day"] = paq["PAD680"].where((paq["PAD680"] >= 0.0) & (paq["PAD680"] <= 1440.0), np.nan)
    df = df.merge(paq[["SEQN", "modvig_pa_any", "leisure_modvig_pa", "sedentary_min_day"]], on="SEQN", how="left")

    dlq_h = read_xpt_clean("2013-2014/Questionnaire/DLQ_H.xpt", ["SEQN", "DLQ010", "DLQ020"])
    dlq_h["hearing_diff_dlq"] = np.where(dlq_h["DLQ010"].isin([1.0, 2.0]), (dlq_h["DLQ010"] == 1.0).astype(float), np.nan)
    dlq_h["vision_difficulty"] = np.where(dlq_h["DLQ020"].isin([1.0, 2.0]), (dlq_h["DLQ020"] == 1.0).astype(float), np.nan)
    auq_g = read_xpt_clean("2011-2012/Questionnaire/AUQ_G.xpt", ["SEQN", "AUQ054"])
    auq_g["hearing_diff_auq"] = np.where(auq_g["AUQ054"].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (auq_g["AUQ054"] >= 4.0).astype(float), np.nan)
    df = df.merge(dlq_h[["SEQN", "hearing_diff_dlq", "vision_difficulty"]], on="SEQN", how="left")
    df = df.merge(auq_g[["SEQN", "hearing_diff_auq"]], on="SEQN", how="left")
    df["hearing_difficulty"] = df["hearing_diff_dlq"].combine_first(df["hearing_diff_auq"])
    df = df.drop(columns=["hearing_diff_dlq", "hearing_diff_auq"])

    # 7. Multi-Day 24h Triaxial Actigraphy + Ambient Light (PAXHD + PAXDAY + PAXHR)
    paxhd = load_cycle_pair("2011-2012/Examination/PAXHD_G.xpt", "2013-2014/Examination/PAXHD_H.xpt")
    paxday = load_cycle_pair("2011-2012/Examination/PAXDAY_G.xpt", "2013-2014/Examination/PAXDAY_H.xpt")
    paxhr = load_cycle_pair("2011-2012/Examination/PAXHR_G.xpt", "2013-2014/Examination/PAXHR_H.xpt")

    pax_subj, paxhr_clean = compute_participant_circadian_metrics(paxhr, paxday, paxhd)
    df = df.merge(pax_subj, on="SEQN", how="left")
    df["pax_has_data"] = df["pax_has_data"].fillna(0).astype(int)
    df["pax_eligible_primary_4d16h"] = df["pax_eligible_primary_4d16h"].fillna(0).astype(int)
    df["pax_eligible_sens_3d16h"] = df["pax_eligible_sens_3d16h"].fillna(0).astype(int)
    df["pax_eligible_sens_4d10h"] = df["pax_eligible_sens_4d10h"].fillna(0).astype(int)

    # Save cleaned parquets
    part_parquet = OUT_DATA / "nhanes_2011_2014_participants.parquet"
    paxhr_parquet = OUT_DATA / "nhanes_2011_2014_paxhr_clean.parquet"
    df.to_parquet(part_parquet, index=False)
    paxhr_clean.to_parquet(paxhr_parquet, index=False)
    print(f"[NHANES SAVE] Saved {part_parquet.name} ({len(df):,} rows, {df.shape[1]} cols) & {paxhr_parquet.name} ({len(paxhr_clean):,} rows)")

    # 8. Exact Multi-Arm CONSORT Flow Counts
    adults20 = df[df["age_years"] >= 20.0]
    older60 = df[df["age_years"] >= 60.0]
    c2013_adults = df[(df["cycle"] == "2013-2014") & (df["age_years"] >= 20.0)]

    consort_rows = [
        ("01_total_enrolled_2011_2014", "Total NHANES 2011–2014 enrolled participants (DEMO_G + DEMO_H)", int(len(df))),
        ("01a_cycle_2011_2012", "  2011–2012 cycle (DEMO_G)", int((df["cycle"] == "2011-2012").sum())),
        ("01b_cycle_2013_2014", "  2013–2014 cycle (DEMO_H)", int((df["cycle"] == "2013-2014").sum())),
        ("02_mec_examined_all_ages", "Mobile Examination Center (MEC) examined (RIDSTATR == 2)", int((df["mec_examined"] == 1).sum())),
        ("03_adults_age_ge20", "Adults aged >= 20 years", int(len(adults20))),
        ("04_adults_mec_examined_ge20", "Adults aged >= 20 years examined at MEC", int((adults20["mec_examined"] == 1).sum())),
        ("05_adults_pax_device_returned", "Adults >= 20 with usable wrist actigraphy file (PAXSTS == 1)", int((adults20["pax_has_data"] == 1).sum())),
        ("06_adults_pax_valid_4d16h", "Adults >= 20 meeting primary 24h circadian QC (>= 4 valid days >= 16h wear & full 24h clock coverage)", int((adults20["pax_eligible_primary_4d16h"] == 1).sum())),
        ("06a_adults_pax_valid_3d16h_sens", "  Sensitivity: Adults >= 20 with >= 3 valid days >= 16h wear", int((adults20["pax_eligible_sens_3d16h"] == 1).sum())),
        ("06b_adults_pax_valid_4d10h_sens", "  Sensitivity: Adults >= 20 with >= 4 valid days >= 10h wear", int((adults20["pax_eligible_sens_4d10h"] == 1).sum())),
        ("07_adults_pax_4d16h_and_phq9", "Arm A (24h Circadian -> Affective Load): Adults >= 20 with primary valid PAX + complete PHQ-9", int(((adults20["pax_eligible_primary_4d16h"] == 1) & adults20["phq9_total"].notna()).sum())),
        ("08_snfl_total_measured_2013_2014", "Arm B (Neuroaxonal Injury Subcohort, 2013–2014 SSSNFL_H): Participants with measured serum sNfL (pg/mL)", int(df["snfl_pgml"].notna().sum())),
        ("08a_snfl_and_pax_device", "  sNfL participants with PAX actigraphy recording (PAXSTS == 1)", int((df["snfl_pgml"].notna() & (df["pax_has_data"] == 1)).sum())),
        ("08b_snfl_and_pax_4d16h", "  sNfL participants meeting primary 24h circadian PAX QC (>= 4 valid 16h days)", int((df["snfl_pgml"].notna() & (df["pax_eligible_primary_4d16h"] == 1)).sum())),
        ("08c_snfl_and_pax_4d10h_sens", "  sNfL participants meeting >= 4 valid 10h wear days (sensitivity)", int((df["snfl_pgml"].notna() & (df["pax_eligible_sens_4d10h"] == 1)).sum())),
        ("08d_snfl_pax_4d16h_and_phq9", "  Primary sNfL DML Mediation Analytic Sample (valid 4d16h PAX + sNfL + complete PHQ-9)", int((df["snfl_pgml"].notna() & (df["pax_eligible_primary_4d16h"] == 1) & df["phq9_total"].notna()).sum())),
        ("09_older_adults_age_ge60", "Older adults aged >= 60 years eligible for CFQ neurocognitive battery (2011–2014)", int(len(older60))),
        ("09a_cfq_dsst_measured", "  Older adults >= 60 with valid DSST processing speed (CFDDS)", int(older60["dsst"].notna().sum())),
        ("09b_cfq_animal_fluency_measured", "  Older adults >= 60 with valid Animal Fluency executive score (CFDAST)", int(older60["animal_fluency"].notna().sum())),
        ("09c_cfq_cerad_ir_measured", "  Older adults >= 60 with complete 3-trial CERAD immediate learning (CFDCST1-3)", int(older60["cerad_ir_sum"].notna().sum())),
        ("09d_cfq_cerad_dr_measured", "  Older adults >= 60 with valid CERAD delayed recall (CFDCSR)", int(older60["cerad_dr"].notna().sum())),
        ("09e_cfq_complete_3domain", "  Older adults >= 60 with complete 3-domain zero-ceiling cognition composite", int((older60["cfq_complete_3domain"] == 1).sum())),
        ("09f_cfq_dsst_and_pax_4d16h", "  Arm C (24h Circadian -> Zero-Ceiling Cognition): DSST + primary valid 4d16h PAX", int((older60["dsst"].notna() & (older60["pax_eligible_primary_4d16h"] == 1)).sum())),
        ("09g_cfq_3domain_pax_4d16h_phq9", "  Complete 3-domain cognition + primary valid 4d16h PAX + complete PHQ-9", int(((older60["cfq_complete_3domain"] == 1) & (older60["pax_eligible_primary_4d16h"] == 1) & older60["phq9_total"].notna()).sum())),
        ("10_triple_bridge_snfl_cfq_dsst", "Arm D (2013–2014 Ages 60–75 Triple Bridge): Measured sNfL + valid DSST", int((df["snfl_pgml"].notna() & df["dsst"].notna()).sum())),
        ("10a_triple_bridge_snfl_cfq_pax_4d16h", "  Triple Cross-Scale Bridge: Measured sNfL + valid DSST + primary valid 4d16h PAX", int((df["snfl_pgml"].notna() & df["dsst"].notna() & (df["pax_eligible_primary_4d16h"] == 1)).sum())),
        ("10b_triple_bridge_snfl_3dom_pax_phq9", "  Quadruple Complete Bridge: Measured sNfL + complete 3-domain cognition + valid 4d16h PAX + PHQ-9", int((df["snfl_pgml"].notna() & (df["cfq_complete_3domain"] == 1) & (df["pax_eligible_primary_4d16h"] == 1) & df["phq9_total"].notna()).sum())),
    ]
    consort_df = pd.DataFrame(consort_rows, columns=["step_id", "description", "n_participants"])
    consort_df.to_csv(OUT_AUDIT / "nhanes_consort_flow.csv", index=False)
    with open(OUT_AUDIT / "nhanes_consort_flow.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cohort": "NHANES_2011_2014",
                "raw_resolution_note": "PAXHR_G + PAXHR_H official CDC hourly triaxial MIMS + ambient light (lux) + wake/sleep minutes across 7–9 recording days (~168–192 contiguous 1h epochs/person)",
                "steps": consort_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    # 9. Variable-by-Variable Missingness Audit across Key Subcohorts
    audit_vars = [
        "age_years", "female", "educ_adult_code", "pir", "bmi", "sbp_mean", "dbp_mean",
        "hba1c_pct", "total_chol_mgdl", "hdl_chol_mgdl", "glucose_mgdl", "triglycerides_mgdl",
        "serum_creatinine_mgdl", "wbc_10e9L", "nlr", "hemoglobin_gdl",
        "ever_smoked_100", "current_smoker", "alcohol_12yr", "modvig_pa_any", "sedentary_min_day",
        "hypertension_doc", "bp_med_current", "high_chol_doc", "diabetes_doc", "cvd_any_doc", "hearing_difficulty",
        "sleep_hours_self", "sleep_trouble_doc", "phq9_total", "phq8_nosleep",
        "pax_mean_mims_min", "pax_RA", "pax_IS", "pax_IV", "pax_photic_motor_r",
        "snfl_pgml", "cerad_ir_sum", "cerad_dr", "animal_fluency", "dsst", "global_cognition_z",
    ]
    subcohorts = {
        "Adults_ge20_All": adults20,
        "ArmA_Adults_PAX4d16h": adults20[adults20["pax_eligible_primary_4d16h"] == 1],
        "ArmB_sNfL_Subcohort": df[df["snfl_pgml"].notna()],
        "ArmC_Older60_DSST": older60[older60["dsst"].notna()],
    }
    miss_rows = []
    for v in audit_vars:
        row: Dict[str, object] = {"variable": v}
        for cname, sub in subcohorts.items():
            n_tot = len(sub)
            n_miss = int(sub[v].isna().sum())
            row[f"{cname}_N"] = n_tot
            row[f"{cname}_nonnull"] = n_tot - n_miss
            row[f"{cname}_miss_pct"] = round(100.0 * n_miss / n_tot, 2) if n_tot > 0 else np.nan
        miss_rows.append(row)
    pd.DataFrame(miss_rows).to_csv(OUT_AUDIT / "nhanes_missingness_audit.csv", index=False)

    # 10. Psychometric Measurement-Fidelity Audit (Zero-Ceiling Proof vs Lin & Grewal 2026)
    psych_rows = [
        summarize_psychometric_scale(older60["dsst"], "NHANES_DSST_ProcessingSpeed", 0.0, 133.0, True),
        summarize_psychometric_scale(older60["animal_fluency"], "NHANES_AnimalFluency_Executive", 0.0, np.nan, True),
        summarize_psychometric_scale(older60["cerad_ir_sum"], "NHANES_CERAD_ImmediateLearning_3TrialSum", 0.0, 30.0, True),
        summarize_psychometric_scale(older60["cerad_dr"], "NHANES_CERAD_DelayedRecall", 0.0, 10.0, True),
        summarize_psychometric_scale(older60["cerad_total_40"], "NHANES_CERAD_Total_40", 0.0, 40.0, True),
        summarize_psychometric_scale(older60["global_cognition_z"], "NHANES_Global_3Domain_Cognition_Z", np.nan, np.nan, True),
        summarize_psychometric_scale(adults20["phq9_total"], "NHANES_PHQ9_Adults20Plus", 0.0, 27.0, False),
        summarize_psychometric_scale(older60["phq9_total"], "NHANES_PHQ9_Older60Plus", 0.0, 27.0, False),
    ]
    pd.DataFrame(psych_rows).to_csv(OUT_AUDIT / "nhanes_psychometric_fidelity_audit.csv", index=False)

    # 11. sNfL Distribution & Age-Stratum Audit (N = 2,071)
    snfl_sub = df[df["snfl_pgml"].notna()].copy()
    snfl_sub["age_band"] = pd.cut(snfl_sub["age_years"], bins=[19, 39, 59, 75], labels=["20-39", "40-59", "60-75"])
    snfl_audit_rows = []
    for label, grp in [("All_sNfL", snfl_sub)] + [(f"Age_{k}", g) for k, g in snfl_sub.groupby("age_band", observed=False)] + [
        ("Female", snfl_sub[snfl_sub["female"] == 1.0]),
        ("Male", snfl_sub[snfl_sub["female"] == 0.0]),
    ]:
        vals = grp["snfl_pgml"].dropna()
        q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90])
        snfl_audit_rows.append(
            {
                "stratum": label,
                "n": int(len(vals)),
                "mean_pgml": round(float(vals.mean()), 2),
                "sd_pgml": round(float(vals.std(ddof=1)), 2),
                "geom_mean_pgml": round(float(np.exp(np.log(vals).mean())), 2),
                "p10": round(float(q10), 2),
                "p25": round(float(q25), 2),
                "median_p50": round(float(q50), 2),
                "p75": round(float(q75), 2),
                "p90": round(float(q90), 2),
                "above_uloq_n": int((grp["snfl_above_uloq"] == 1.0).sum()),
                "below_lloq_n": int((grp["snfl_below_lloq"] == 1.0).sum()),
            }
        )
    pd.DataFrame(snfl_audit_rows).to_csv(OUT_AUDIT / "nhanes_snfl_distribution_audit.csv", index=False)

    # 12. Actigraphy QC Summary Table
    pax_qc_sub = adults20[adults20["pax_eligible_primary_4d16h"] == 1]
    pax_metrics_list = [
        "n_pax_valid_days_16h", "n_valid_hours", "pax_mean_mims_min", "pax_mean_lux_min",
        "pax_est_sleep_hours_day", "pax_M10", "pax_L5", "pax_RA", "pax_IS", "pax_IV",
        "pax_cosinor_mesor", "pax_cosinor_amp", "pax_cosinor_acrophase_hr", "pax_photic_motor_r",
    ]
    pax_qc_rows = []
    for m in pax_metrics_list:
        vals = pax_qc_sub[m].dropna()
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        pax_qc_rows.append(
            {
                "metric": m,
                "n": int(len(vals)),
                "mean": round(float(vals.mean()), 4),
                "sd": round(float(vals.std(ddof=1)), 4),
                "median": round(float(q50), 4),
                "q25": round(float(q25), 4),
                "q75": round(float(q75), 4),
            }
        )
    pd.DataFrame(pax_qc_rows).to_csv(OUT_AUDIT / "nhanes_actigraphy_qc_summary.csv", index=False)

    # 13. Publication Baseline Table 1 across Analytic Arms
    t1_cohorts = {
        "ArmA_Adults_ValidPAX": adults20[adults20["pax_eligible_primary_4d16h"] == 1],
        "ArmB_sNfL_BiologicalMediation": df[df["snfl_pgml"].notna()],
        "ArmC_Older60_3DomainCognition_DSST": older60[older60["dsst"].notna()],
        "ArmD_TripleBridge_sNfL_DSST_PAX": df[df["snfl_pgml"].notna() & df["dsst"].notna() & (df["pax_eligible_primary_4d16h"] == 1)],
    }
    t1_rows = []
    t1_rows.append({"Characteristic": "Sample size, N", **{k: f"{len(v):,}" for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Age, years, mean (SD)", **{k: format_mean_sd(v["age_years"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Female, n (%)", **{k: format_n_pct(v["female"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Low education (< high school), n (%)", **{k: format_n_pct(v["low_education_lt_hs"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Income-to-poverty ratio (PIR), mean (SD)", **{k: format_mean_sd(v["pir"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "BMI, kg/m^2, mean (SD)", **{k: format_mean_sd(v["bmi"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Systolic BP, mmHg, mean (SD)", **{k: format_mean_sd(v["sbp_mean"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "HbA1c, %, mean (SD)", **{k: format_mean_sd(v["hba1c_pct"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Total cholesterol, mg/dL, mean (SD)", **{k: format_mean_sd(v["total_chol_mgdl"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "WBC count, 10^9/L, mean (SD)", **{k: format_mean_sd(v["wbc_10e9L"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Neutrophil-to-lymphocyte ratio (NLR), median [IQR]", **{k: format_median_iqr(v["nlr"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Diagnosed hypertension, n (%)", **{k: format_n_pct(v["hypertension_doc"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Antihypertensive medication use, n (%)", **{k: format_n_pct(v["bp_med_current"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Diagnosed diabetes, n (%)", **{k: format_n_pct(v["diabetes_doc"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "History of cardiovascular disease/stroke, n (%)", **{k: format_n_pct(v["cvd_any_doc"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Current smoker, n (%)", **{k: format_n_pct(v["current_smoker"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Alcohol (>=12 drinks/year), n (%)", **{k: format_n_pct(v["alcohol_12yr"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Moderate-to-vigorous physical activity, n (%)", **{k: format_n_pct(v["modvig_pa_any"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Valid 16h actigraphy days, mean (SD)", **{k: format_mean_sd(v["n_pax_valid_days_16h"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "24h Relative Amplitude (RA), mean (SD)", **{k: format_mean_sd(v["pax_RA"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Interdaily Stability (IS), mean (SD)", **{k: format_mean_sd(v["pax_IS"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Intradaily Variability (IV), mean (SD)", **{k: format_mean_sd(v["pax_IV"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Photic-motor coupling r, mean (SD)", **{k: format_mean_sd(v["pax_photic_motor_r"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Serum sNfL, pg/mL, median [IQR]", **{k: format_median_iqr(v["snfl_pgml"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "PHQ-9 depressive score (0–27), median [IQR]", **{k: format_median_iqr(v["phq9_total"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Moderate-to-severe depression (PHQ-9 >= 10), n (%)", **{k: format_n_pct(v["phq9_modsev_ge10"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "DSST processing speed, mean (SD)", **{k: format_mean_sd(v["dsst"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "Animal Fluency score, mean (SD)", **{k: format_mean_sd(v["animal_fluency"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "CERAD 3-trial immediate recall (0–30), mean (SD)", **{k: format_mean_sd(v["cerad_ir_sum"]) for k, v in t1_cohorts.items()}})
    t1_rows.append({"Characteristic": "CERAD delayed recall (0–10), mean (SD)", **{k: format_mean_sd(v["cerad_dr"]) for k, v in t1_cohorts.items()}})

    pd.DataFrame(t1_rows).to_csv(OUT_AUDIT / "nhanes_baseline_table1.csv", index=False)

    print("=== NHANES Phase 0 CONSORT Summary ===")
    for _, r in consort_df.iterrows():
        print(f"  {r['step_id']}: {r['n_participants']:,} — {r['description']}")
    print(f"[NHANES DONE] Completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
