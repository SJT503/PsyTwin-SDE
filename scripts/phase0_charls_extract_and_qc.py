#!/usr/bin/env python3
"""
Phase 0 — Reproducible CHARLS (2011–2020, 5 Waves + 2011 & 2015 Paired Venous Blood) Extraction & QC
====================================================================================================
Target Project: PsyTwin-SDE (npj Digital Medicine Collection)
Governance Rules (CLAUDE.md & lab-constitution):
  1. STRICT READ-ONLY RAW DATA: Streams .zip and .rar files from ${CHARLS_RAW_DIR} strictly in-memory;
     writes all outputs exclusively inside ./.
  2. EXACT 11->12 CHAR ID HARMONIZATION: Harmonizes Wave 1 11-char IDs via x[:9] + '0' + x[9:]
     yielding exact N = 7,648 longitudinal paired venous blood participants (2011 & 2015).
  3. GATEWAY-PARITY WAVE 5 SCORING: Applies official Gateway to Global Aging Data H_CHARLS_D_do_file.do
     formulas to Wave 5 (2020) so all 5 waves (2011, 2013, 2015, 2018, 2020) share identical scales.
  4. ZERO Brain-Kidney Axis & ZERO Raw DICOM Imaging & 100% Real Script-Computed Numbers.
"""

from __future__ import annotations

import io
import json
import subprocess
import time
import zipfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CHARLS = Path(os.environ.get("CHARLS_RAW_DIR", str(PROJECT_ROOT / "data_raw" / "CHARLS")))
OUT_DATA = PROJECT_ROOT / "data_processed"
OUT_AUDIT = PROJECT_ROOT / "results" / "phase0_cohort_audit"


def harmonize_charls_id(val: object) -> str:
    """Convert CHARLS Wave 1 11-char ID to standard 12-char ID (insert '0' after 9-char householdID_w1)."""
    if pd.isna(val):
        return ""
    if isinstance(val, (bytes, bytearray)):
        s = val.decode("latin1", errors="ignore").strip()
    else:
        s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 11:
        return s[:9] + "0" + s[9:]
    return s


def read_stata_from_zip(rel_zip: str, dta_name: Optional[str] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a Stata .dta file from a read-only .zip archive entirely in memory."""
    zpath = RAW_CHARLS / rel_zip
    if not zpath.exists():
        raise FileNotFoundError(f"Expected CHARLS archive not found: {zpath}")
    with zipfile.ZipFile(zpath, "r") as zf:
        if dta_name is None:
            dtas = [n for n in zf.namelist() if n.lower().endswith(".dta")]
            if not dtas:
                raise ValueError(f"No .dta inside {zpath}")
            dta_name = dtas[0]
        raw_bytes = zf.read(dta_name)
    buf = io.BytesIO(raw_bytes)
    if columns is not None:
        sr = pd.io.stata.StataReader(buf)
        avail = set(sr.variable_labels().keys())
        present = [c for c in columns if c in avail]
        buf.seek(0)
        df = pd.read_stata(buf, columns=present, convert_categoricals=False)
    else:
        df = pd.read_stata(buf, convert_categoricals=False)
    if "ID" in df.columns:
        df["ID"] = df["ID"].map(harmonize_charls_id)
        df = df[df["ID"] != ""].drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
    return df


def read_stata_from_rar(rel_rar: str, inner_dta: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Stream a Stata .dta file from a read-only .rar archive via system bsdtar directly into memory."""
    rpath = RAW_CHARLS / rel_rar
    if not rpath.exists():
        raise FileNotFoundError(f"Expected CHARLS RAR archive not found: {rpath}")
    raw_bytes = subprocess.check_output(["tar", "-xOf", str(rpath), inner_dta])
    buf = io.BytesIO(raw_bytes)
    if columns is not None:
        sr = pd.io.stata.StataReader(buf)
        avail = set(sr.variable_labels().keys())
        present = [c for c in columns if c in avail]
        buf.seek(0)
        df = pd.read_stata(buf, columns=present, convert_categoricals=False)
    else:
        df = pd.read_stata(buf, convert_categoricals=False)
    if "ID" in df.columns:
        df["ID"] = df["ID"].map(harmonize_charls_id)
        df = df[df["ID"] != ""].drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
    return df


def extract_paired_venous_blood() -> pd.DataFrame:
    """
    Extract and harmonize CHARLS 2011 (Wave 1, N=11,847) and 2015 (Wave 3, N=13,420) venous blood panels
    using the verified 11->12-char ID rule, yielding exact N=7,648 longitudinal paired participants.
    """
    b1 = read_stata_from_zip("2011-Wave1/201_血检数据(29 04 2014).zip", "Blood_20140429.dta")
    b3 = read_stata_from_zip("2015-Wave3/2088_血检数据（更新于20 06 2019）.zip", "Blood.dta")

    b1_map = {
        "ID": "ID",
        "bloodweight": "blood_weight_2011",
        "qc1_va003": "fasting_2011",
        "newcrp": "crp_mgl_2011",
        "newhba1c": "hba1c_pct_2011",
        "newcho": "tchol_mgdl_2011",
        "newhdl": "hdl_mgdl_2011",
        "newldl": "ldl_mgdl_2011",
        "newtg": "tg_mgdl_2011",
        "newglu": "glucose_mgdl_2011",
        "newua": "uric_acid_mgdl_2011",
        "cystatinc": "cystatin_c_mgl_2011",
        "newcrea": "creatinine_mgdl_2011",
        "newbun": "bun_mgdl_2011",
        "qc1_vb002": "wbc_10e9L_2011",
        "qc1_vb004": "hgb_gdl_2011",
        "qc1_vb005": "hct_pct_2011",
        "qc1_vb006": "mcv_fl_2011",
        "qc1_vb009": "plt_10e9L_2011",
    }
    b3_map = {
        "ID": "ID",
        "Blood_weight": "blood_weight_2015",
        "bl_fasting": "fasting_2015",
        "bl_crp": "crp_mgl_2015",
        "bl_hbalc": "hba1c_pct_2015",
        "bl_cho": "tchol_mgdl_2015",
        "bl_hdl": "hdl_mgdl_2015",
        "bl_ldl": "ldl_mgdl_2015",
        "bl_tg": "tg_mgdl_2015",
        "bl_glu": "glucose_mgdl_2015",
        "bl_ua": "uric_acid_mgdl_2015",
        "bl_cysc": "cystatin_c_mgl_2015",
        "bl_crea": "creatinine_mgdl_2015",
        "bl_bun": "bun_mgdl_2015",
        "bl_wbc": "wbc_10e9L_2015",
        "bl_hgb": "hgb_gdl_2015",
        "bl_hct": "hct_pct_2015",
        "bl_mcv": "mcv_fl_2015",
        "bl_plt": "plt_10e9L_2015",
    }

    b1_c = b1[[k for k in b1_map if k in b1.columns]].rename(columns=b1_map).copy()
    b1_c["in_blood_2011"] = 1
    b3_c = b3[[k for k in b3_map if k in b3.columns]].rename(columns=b3_map).copy()
    b3_c["in_blood_2015"] = 1

    blood = pd.merge(b1_c, b3_c, on="ID", how="outer")
    blood["in_blood_2011"] = blood["in_blood_2011"].fillna(0).astype(int)
    blood["in_blood_2015"] = blood["in_blood_2015"].fillna(0).astype(int)
    blood["is_paired_2011_2015"] = ((blood["in_blood_2011"] == 1) & (blood["in_blood_2015"] == 1)).astype(int)

    analytes = [
        "crp_mgl", "hba1c_pct", "tchol_mgdl", "hdl_mgdl", "ldl_mgdl", "tg_mgdl",
        "glucose_mgdl", "uric_acid_mgdl", "cystatin_c_mgl", "creatinine_mgdl", "bun_mgdl",
        "wbc_10e9L", "hgb_gdl", "hct_pct", "mcv_fl", "plt_10e9L",
    ]
    for a in analytes:
        c11, c15 = f"{a}_2011", f"{a}_2015"
        blood[c11] = pd.to_numeric(blood[c11], errors="coerce")
        blood[c15] = pd.to_numeric(blood[c15], errors="coerce")
        blood[c11] = blood[c11].where(blood[c11] > 0, np.nan)
        blood[c15] = blood[c15].where(blood[c15] > 0, np.nan)
        blood[f"delta_{a}"] = blood[c15] - blood[c11]

    blood["log_crp_2011"] = np.log(blood["crp_mgl_2011"])
    blood["log_crp_2015"] = np.log(blood["crp_mgl_2015"])
    blood["delta_log_crp"] = blood["log_crp_2015"] - blood["log_crp_2011"]

    n11 = int(blood["in_blood_2011"].sum())
    n15 = int(blood["in_blood_2015"].sum())
    npair = int(blood["is_paired_2011_2015"].sum())
    nunion = int(len(blood))
    print(f"[CHARLS BLOOD] 2011 N={n11:,} | 2015 N={n15:,} | Union N={nunion:,} | Paired 2011&2015 N={npair:,}")
    assert npair == 7648, f"Expected exact N=7,648 paired blood overlap, got {npair}"
    assert nunion == 17619, f"Expected exact N=17,619 blood union, got {nunion}"
    return blood


def extract_wave1_to_4_raw_supplements() -> Dict[int, pd.DataFrame]:
    """
    Extract modifiable intervention targets not fully disaggregated in Harmonized CHARLS
    (nighttime sleep hours, afternoon nap minutes, distance/near eyesight, glasses, hearing,
    hearing aid, internet use, social activities) from raw Waves 1–4 Health_Status_and_Functioning.
    """
    raw_specs = [
        (1, "rar", "2011-Wave1/109_健康状况与功能(03 12 2013更新).rar", "health_status_and_functioning.dta"),
        (2, "zip", "2013-Wave2/311_健康状况与功能（更新于 18 11 2015）.zip", None),
        (3, "zip", "2015-Wave3/1146_健康状况与功能（更新于11 10 2017）.zip", None),
        (4, "zip", "2018-Wave4/2097_健康状况与功能.zip", None),
    ]
    req_cols = [
        "ID", "da032", "da033", "da034", "da038", "da038_w4", "da039", "da049", "da050",
        "da056s1", "da056s2", "da056s3", "da056s4", "da056s5", "da056s6", "da056s7", "da056s8", "da056s10", "da056s12",
        "da056_s1", "da056_s2", "da056_s3", "da056_s4", "da056_s5", "da056_s6", "da056_s7", "da056_s8", "da056_s10", "da056_s12",
        "da056_w4_2",
    ]
    supps: Dict[int, pd.DataFrame] = {}
    for w, kind, rel_p, inner in raw_specs:
        if kind == "rar":
            d = read_stata_from_rar(rel_p, inner, columns=req_cols)
        else:
            d = read_stata_from_zip(rel_p, inner, columns=req_cols)

        out = pd.DataFrame({"ID": d["ID"]})
        # Sleep hours (da049) & nap minutes (da050)
        sh = pd.to_numeric(d.get("da049", np.nan), errors="coerce")
        out["sleep_hours"] = sh.where((sh > 0.0) & (sh <= 24.0), np.nan)
        nap = pd.to_numeric(d.get("da050", np.nan), errors="coerce")
        out["nap_min"] = nap.where((nap >= 0.0) & (nap <= 360.0), np.nan)

        # Vision: da032 glasses (1=Yes regularly, 4=Sometimes, 3=No, 2=Legally blind),
        #         da033 distance (1=Exc..5=Poor), da034 near (1=Exc..5=Poor)
        g = pd.to_numeric(d.get("da032", np.nan), errors="coerce")
        out["glasses_worn"] = np.where(g.isin([1.0, 2.0, 3.0, 4.0]), g.isin([1.0, 4.0]).astype(float), np.nan)
        v_dist = pd.to_numeric(d.get("da033", np.nan), errors="coerce")
        v_dist = v_dist.where(v_dist.isin([1.0, 2.0, 3.0, 4.0, 5.0]), np.nan)
        v_near = pd.to_numeric(d.get("da034", np.nan), errors="coerce")
        v_near = v_near.where(v_near.isin([1.0, 2.0, 3.0, 4.0, 5.0]), np.nan)
        out["vision_dist_score_1_5"] = v_dist
        out["vision_near_score_1_5"] = v_near
        out["poor_vision"] = np.where(
            v_dist.notna() | v_near.notna() | (g == 2.0),
            ((v_dist == 5.0) | (v_near == 5.0) | (g == 2.0)).astype(float),
            np.nan,
        )

        # Hearing: da038 / da038_w4 hearing aid (1=Yes, 2=No), da039 hearing (1=Exc..5=Poor)
        ha_col = "da038_w4" if "da038_w4" in d.columns else "da038"
        ha = pd.to_numeric(d.get(ha_col, np.nan), errors="coerce")
        out["hearing_aid_worn"] = np.where(ha.isin([1.0, 2.0]), (ha == 1.0).astype(float), np.nan)
        hr_s = pd.to_numeric(d.get("da039", np.nan), errors="coerce")
        hr_s = hr_s.where(hr_s.isin([1.0, 2.0, 3.0, 4.0, 5.0]), np.nan)
        out["hearing_score_1_5"] = hr_s
        out["poor_hearing"] = np.where(hr_s.notna(), (hr_s == 5.0).astype(float), np.nan)

        # Internet use: da056s10 (W1-W3) or da056_s10 / da056_w4_2 (W4)
        if "da056s10" in d.columns:
            inet = pd.to_numeric(d["da056s10"], errors="coerce")
            none_c = pd.to_numeric(d.get("da056s12", np.nan), errors="coerce")
            any_ans = d[[c for c in d.columns if c.startswith("da056s")]].notna().any(axis=1)
            out["internet_use_raw"] = np.where(any_ans, ((inet == 10.0) | (inet == 1.0)).astype(float), np.nan)
        elif "da056_s10" in d.columns:
            inet = pd.to_numeric(d["da056_s10"], errors="coerce")
            any_ans = d[[c for c in d.columns if c.startswith("da056_s")]].notna().any(axis=1)
            used = (inet == 10.0) | (inet == 1.0)
            if "da056_w4_2" in d.columns:
                w42 = pd.to_numeric(d["da056_w4_2"], errors="coerce")
                used = used | (w42 == 1.0)
            out["internet_use_raw"] = np.where(any_ans, used.astype(float), np.nan)
        else:
            out["internet_use_raw"] = np.nan

        supps[w] = out
        print(f"[CHARLS SUPP W{w}] Loaded {len(out):,} rows (sleep non-null={out['sleep_hours'].notna().sum():,})")
    return supps


def extract_wave5_harmonized() -> Tuple[pd.DataFrame, set]:
    """
    Extract CHARLS Wave 5 (2020) and apply exact Gateway to Global Aging Data H_CHARLS_D_do_file.do
    scoring formulas for cognitive domains, CES-D10, and modifiable intervention targets.
    """
    h5 = read_stata_from_zip("2020-Wave5/2203_健康状况与功能.zip", "Health_Status_and_Functioning.dta")
    d5 = read_stata_from_zip(
        "2020-Wave5/2201_基本信息.zip",
        "Demographic_Background.dta",
        columns=["ID", "ba001", "ba002", "ba003_1", "ba008", "ba009", "ba010", "ba010_1", "ba011", "xrpartner", "xrage", "xrgender", "zrbirthyear", "zredu", "zrgender"],
    )
    s5 = read_stata_from_zip("2020-Wave5/2209_样本信息.zip", "Sample_Infor.dta", columns=["ID", "iyear", "imonth", "died"])
    w5_wt = read_stata_from_zip("2020-Wave5/2210_样本权重.zip", "Weights.dta", columns=["ID", "INDV_weight_ad2"])

    w5_died_ids = set(s5.loc[pd.to_numeric(s5.get("died", np.nan), errors="coerce") == 1.0, "ID"].tolist())

    m = h5.copy()
    for extra_df in (d5, s5, w5_wt):
        keep_extra = ["ID"] + [c for c in extra_df.columns if c != "ID" and c not in m.columns]
        m = m.merge(extra_df[keep_extra], on="ID", how="outer" if extra_df is d5 else "left")
    # Exclude Wave 5 exit-only deceased records from active Wave 5 interview rows (tracked separately in w5_died_ids)
    m = m[pd.to_numeric(m.get("died", 0.0), errors="coerce").fillna(0.0) != 1.0].reset_index(drop=True)

    out = pd.DataFrame({"ID": m["ID"]})
    out["inw"] = 1
    out["wave"] = 5
    out["iwy"] = pd.to_numeric(m.get("iyear", np.nan), errors="coerce").fillna(2020.0)
    out["iwm"] = pd.to_numeric(m.get("imonth", np.nan), errors="coerce").fillna(8.0)
    out["wtrespb"] = pd.to_numeric(m.get("INDV_weight_ad2", np.nan), errors="coerce")

    # Age & demographics fallback for Wave-5-only entrants
    s_nan = pd.Series(np.nan, index=m.index)
    byear = pd.to_numeric(m.get("ba003_1", s_nan), errors="coerce").where(lambda s: (s >= 1900) & (s <= 2005))
    zbyear = pd.to_numeric(m.get("zrbirthyear", s_nan), errors="coerce").where(lambda s: (s >= 1900) & (s <= 2005))
    byear = byear.fillna(zbyear)
    xrage = pd.to_numeric(m.get("xrage", s_nan), errors="coerce").where(lambda s: (s >= 18) & (s <= 115))
    out["w5_agey"] = xrage.fillna(out["iwy"] - byear)
    out["w5_byear"] = byear

    gen = pd.to_numeric(m.get("ba001", s_nan), errors="coerce").where(lambda s: s.isin([1.0, 2.0]))
    gen = gen.fillna(pd.to_numeric(m.get("xrgender", s_nan), errors="coerce").where(lambda s: s.isin([1.0, 2.0])))
    gen = gen.fillna(pd.to_numeric(m.get("zrgender", s_nan), errors="coerce").where(lambda s: s.isin([1.0, 2.0])))
    out["w5_ragender"] = gen
    out["w5_educ_raw"] = pd.to_numeric(m.get("ba010", s_nan), errors="coerce").fillna(pd.to_numeric(m.get("zredu", s_nan), errors="coerce"))
    # ba008: 1 Village, 2 City/Town; ba009: 1 Agricultural hukou
    ba008 = pd.to_numeric(m.get("ba008", s_nan), errors="coerce")
    out["rural_village"] = np.where(ba008.isin([1.0, 2.0]), (ba008 == 1.0).astype(float), np.nan)
    ba009 = pd.to_numeric(m.get("ba009", s_nan), errors="coerce")
    out["rural_hukou"] = np.where(ba009.isin([1.0, 2.0, 3.0, 4.0]), (ba009 == 1.0).astype(float), np.nan)

    # 1. Orientation (0-4: Year dc001, Date dc003, Day-of-week dc004, Month dc005; excludes Season dc002 per Gateway do-file)
    # Gateway Stata rule: 1=Correct (1), Error=0, Don't Know (997) treated as 0 when >=1 item answered, NaN if all 4 missing/refused
    o_cols = ["dc001", "dc003", "dc004", "dc005"]
    o_corr = pd.DataFrame({c: (m[c] == 1.0).astype(float) for c in o_cols})
    o_valid_any = pd.DataFrame({c: m[c].isin([1.0, 2.0, 997.0]) for c in o_cols}).any(axis=1)
    proxy_flag = pd.to_numeric(m.get("proxy", np.nan), errors="coerce") == 1.0
    out["orient"] = np.where(o_valid_any & (~proxy_flag), o_corr.sum(axis=1), np.nan)

    # 2. Serial 7s (0-5: step-relative subtraction per Gateway H_CHARLS_D_do_file.do)
    s_ans = [pd.to_numeric(m[f"dc007_{k}_1"], errors="coerce") for k in range(1, 6)]
    any_s_num = pd.concat([s.notna() for s in s_ans], axis=1).any(axis=1)
    ser7 = np.zeros(len(m), dtype=float)
    ser7 += (s_ans[0] == 93.0).astype(float)
    for k in range(1, 5):
        ser7 += ((s_ans[k] == (s_ans[k - 1] - 7.0)) & s_ans[k].notna() & s_ans[k - 1].notna()).astype(float)
    out["ser7"] = np.where(any_s_num & (~proxy_flag), ser7, np.nan)

    # 3. Pentagon Figure Drawing (0-1: dc009 1=Correct -> 1, 2=Error -> 0; 3=Not Assessed / 997 / 999 -> NaN)
    dc009 = pd.to_numeric(m["dc009"], errors="coerce")
    out["draw"] = np.where(dc009 == 1.0, 1.0, np.where(dc009 == 2.0, 0.0, np.nan))

    # 4. Immediate Word Recall (0-10: Trial 1 dc012_s1..10 per Gateway Wave 4 parity; plus 3-trial sum 0-30)
    def score_word_trial(prefix: str) -> pd.Series:
        hits = pd.DataFrame({f"{prefix}_s{i}": (pd.to_numeric(m[f"{prefix}_s{i}"], errors="coerce") == float(i)).astype(float) for i in range(1, 11)})
        any_word = pd.DataFrame({f"{prefix}_s{i}": pd.to_numeric(m[f"{prefix}_s{i}"], errors="coerce") == float(i) for i in range(1, 11)}).any(axis=1)
        none_above = pd.to_numeric(m.get(f"{prefix}_s11", np.nan), errors="coerce") == 11.0
        valid_trial = (any_word | none_above) & (~proxy_flag)
        return pd.Series(np.where(valid_trial, hits.sum(axis=1), np.nan), index=m.index)

    t1 = score_word_trial("dc012")
    t2 = score_word_trial("dc013")
    t3 = score_word_trial("dc014")
    out["imrc"] = t1
    out["w5_imrc_3trial_sum"] = np.where(t1.notna() & t2.notna() & t3.notna(), t1 + t2 + t3, np.nan)

    # 5. Delayed Word Recall (0-10: dc028_s1..10) & Total Episodic Memory tr20 (0-20)
    out["dlrc"] = score_word_trial("dc028")
    out["tr20"] = np.where(out["imrc"].notna() & out["dlrc"].notna(), out["imrc"] + out["dlrc"], np.nan)

    # 6. CES-D10 Depressive Symptoms (0-30: dc016..dc025, dc020 hopeful & dc023 happy reverse-coded)
    cesd_cols = [f"dc0{i}" for i in range(16, 26)]
    cesd_scaled = pd.DataFrame(index=m.index)
    for c in cesd_cols:
        v = pd.to_numeric(m[c], errors="coerce")
        v = v.where(v.isin([1.0, 2.0, 3.0, 4.0]), np.nan)
        if c in ("dc020", "dc023"):
            cesd_scaled[c] = 4.0 - v
        else:
            cesd_scaled[c] = v - 1.0
    n_miss_cesd = cesd_scaled.isna().sum(axis=1)
    out["cesd10m"] = np.where(~proxy_flag, n_miss_cesd, 10)
    # Gateway H_CHARLS_D_do_file.do parity: rowtotal when 0 <= cesd10m <= 9
    out["cesd10"] = np.where((n_miss_cesd <= 9) & (~proxy_flag), cesd_scaled.sum(axis=1, skipna=True), np.nan)
    out["cesd10_strict_le2m"] = np.where((n_miss_cesd <= 2) & (~proxy_flag), cesd_scaled.mean(axis=1) * 10.0, np.nan)
    nosleep_c = [c for c in cesd_cols if c != "dc022"]
    out["cesd9_nosleep"] = np.where((cesd_scaled[nosleep_c].isna().sum(axis=1) <= 1) & (~proxy_flag), cesd_scaled[nosleep_c].mean(axis=1) * 9.0, np.nan)

    # 7. Sleep, Nap, Social, Internet, Physical Activity & Sensory Interviewer Flags in Wave 5
    sh5 = pd.to_numeric(m["da030"], errors="coerce")
    out["sleep_hours"] = sh5.where((sh5 > 0.0) & (sh5 <= 24.0), np.nan)
    nap5 = pd.to_numeric(m["da031"], errors="coerce")
    out["nap_min"] = nap5.where((nap5 >= 0.0) & (nap5 <= 360.0), np.nan)

    da040 = pd.to_numeric(m["da040"], errors="coerce")
    out["internet_use"] = np.where(da040.isin([1.0, 2.0]), (da040 == 1.0).astype(float), np.nan)

    soc_any = pd.DataFrame({f"da038_s{i}": pd.to_numeric(m[f"da038_s{i}"], errors="coerce") == float(i) for i in range(1, 9)}).any(axis=1)
    soc_none = pd.to_numeric(m["da038_s9"], errors="coerce") == 9.0
    out["social_active"] = np.where(soc_any | soc_none, soc_any.astype(float), np.nan)

    pa_v = pd.to_numeric(m["da032_1_"], errors="coerce")
    pa_m = pd.to_numeric(m["da032_2_"], errors="coerce")
    out["phys_active_modvig"] = np.where(
        pa_v.isin([1.0, 2.0]) | pa_m.isin([1.0, 2.0]),
        ((pa_v == 1.0) | (pa_m == 1.0)).astype(float),
        np.nan,
    )

    # Interviewer sensory observation flags in Wave 5 (dc029_s1..s3)
    s_obs_valid = m["dc029_s8"].notna() | m["dc029_s1"].notna()
    out["poor_vision"] = np.where(s_obs_valid, (pd.to_numeric(m["dc029_s1"], errors="coerce") == 1.0).astype(float), np.nan)
    out["poor_hearing"] = np.where(s_obs_valid, (pd.to_numeric(m["dc029_s2"], errors="coerce") == 2.0).astype(float), np.nan)
    out["hearing_aid_worn"] = np.where(s_obs_valid, (pd.to_numeric(m["dc029_s3"], errors="coerce") == 3.0).astype(float), np.nan)
    out["glasses_worn"] = np.nan

    # 8. Extended Wave 5 Lifestyle, Marital, Chronic Comorbidities & Functional ADL/IADL Parity
    ba011 = pd.to_numeric(m.get("ba011", s_nan), errors="coerce")
    xrpart = pd.to_numeric(m.get("xrpartner", s_nan), errors="coerce")
    out["unmarried_or_widowed"] = np.where(
        ba011.isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ba011.isin([3.0, 4.0, 5.0, 6.0]).astype(float),
        np.where(xrpart.isin([1.0, 2.0]), (xrpart == 2.0).astype(float), np.nan),
    )

    da046 = pd.to_numeric(m.get("da046", s_nan), errors="coerce")
    da047 = pd.to_numeric(m.get("da047", s_nan), errors="coerce")
    out["smoken"] = np.where(
        da047 == 1.0,
        1.0,
        np.where((da046 == 2.0) | da047.isin([2.0, 3.0]), 0.0, np.nan),
    )
    out["smokev"] = np.where(
        (da046 == 1.0) | da047.isin([1.0, 2.0]),
        1.0,
        np.where((da046 == 2.0) | (da047 == 3.0), 0.0, np.nan),
    )
    da051 = pd.to_numeric(m.get("da051", s_nan), errors="coerce")
    out["drinkl"] = np.where(da051.isin([1.0, 2.0, 3.0]), da051.isin([1.0, 2.0]).astype(float), np.nan)

    # Wave 5 chronic diagnoses (1=hibpe, 2=dyslipe, 3=diabe, 7=hearte, 8=stroke, 12=memrye; ZERO organ-9 export)
    w5_dx_specs = [("hibpe", 1), ("dyslipe", 2), ("diabe", 3), ("hearte", 7), ("stroke", 8), ("memrye", 12)]
    for col_name, idx_k in w5_dx_specs:
        c2 = pd.to_numeric(m.get(f"da002_{idx_k}_", s_nan), errors="coerce")
        c3 = pd.to_numeric(m.get(f"da003_{idx_k}_", s_nan), errors="coerce")
        has_dx = c2.isin([1.0, 2.0, 3.0]) | (c3 == 1.0)
        no_dx = (c3 == 2.0) & (~c2.isin([1.0, 2.0, 3.0]))
        out[col_name] = np.where(has_dx, 1.0, np.where(no_dx, 0.0, np.nan))
        out[f"w5_disputed_{col_name}"] = (c2 == 99.0).astype(int)

    # Wave 5 ADL (4-item adla_c: dressing db001, bathing db003, eating db005, bed db007; 6-item adlab_c adds toilet db009, continence db011)
    adl6_cols = ["db001", "db003", "db005", "db007", "db009", "db011"]
    adl4_cols = ["db001", "db003", "db005", "db007"]
    adl_mat = pd.DataFrame({c: pd.to_numeric(m.get(c, s_nan), errors="coerce") for c in adl6_cols})
    adl_diff = pd.DataFrame({c: np.where(adl_mat[c].isin([1.0, 2.0, 3.0, 4.0]), (adl_mat[c] >= 2.0).astype(float), np.nan) for c in adl6_cols})
    out["adla_c"] = np.where(adl_diff[adl4_cols].notna().all(axis=1), adl_diff[adl4_cols].sum(axis=1), np.nan)
    out["adlab_c"] = np.where(adl_diff[adl6_cols].notna().all(axis=1), adl_diff[adl6_cols].sum(axis=1), np.nan)

    iadl5_cols = ["db012", "db014", "db016", "db020", "db022"]
    iadl_mat = pd.DataFrame({c: pd.to_numeric(m.get(c, s_nan), errors="coerce") for c in iadl5_cols})
    iadl_diff = pd.DataFrame({c: np.where(iadl_mat[c].isin([1.0, 2.0, 3.0, 4.0]), (iadl_mat[c] >= 2.0).astype(float), np.nan) for c in iadl5_cols})
    out["iadlza"] = np.where(iadl_diff.notna().all(axis=1), iadl_diff.sum(axis=1), np.nan)

    print(f"[CHARLS W5] Harmonized {len(out):,} active Wave 5 rows + {len(w5_died_ids):,} W5 mortality exits (tr20 non-null={out['tr20'].notna().sum():,}, cesd10 non-null={out['cesd10'].notna().sum():,})")
    return out, w5_died_ids


def build_charls_5wave_panel_and_audits() -> None:
    t0 = time.time()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)

    print("=== Phase 0: CHARLS 2011–2020 (5 Waves + Paired Venous Blood) Extraction & QC ===")

    # 1. Paired Venous Blood (2011 & 2015)
    blood = extract_paired_venous_blood()
    blood_parquet = OUT_DATA / "charls_paired_blood_2011_2015.parquet"
    blood.to_parquet(blood_parquet, index=False)

    # 2. Harmonized CHARLS Waves 1–4 (N = 25,586)
    base_cols = [
        "ID", "ID_w1", "householdID", "communityID", "ragender", "raidbyear", "raeduc_c", "raeducl",
        "inw1", "inw2", "inw3", "inw4",
    ]
    wave_templates = [
        "r{w}iwstat", "r{w}iwy", "r{w}iwm", "r{w}agey", "r{w}wtrespb",
        "h{w}rural", "r{w}rural2", "r{w}mstat", "hh{w}cperc",
        "r{w}smoken", "r{w}smokev", "r{w}drinkl",
        "r{w}imrc", "r{w}dlrc", "r{w}tr20", "r{w}ser7", "r{w}orient", "r{w}draw",
        "r{w}cesd10", "r{w}cesd10m", "r{w}sleeprl",
        "r{w}socwk", "r{w}vgact_c", "r{w}mdact_c", "r{w}ltact_c",
        "r{w}hibpe", "r{w}diabe", "r{w}hearte", "r{w}stroke", "r{w}dyslipe", "r{w}memrye",
        "r{w}rxhibp_c", "r{w}rxdiab_c", "r{w}rxheart_c", "r{w}rxdyslip_c",
        "r{w}adla_c", "r{w}adlab_c", "r{w}iadlza",
        "r{w}systo", "r{w}diasto", "r{w}mbmi",
    ]
    req_harm_cols = list(base_cols)
    for w in [1, 2, 3, 4]:
        for tmpl in wave_templates:
            req_harm_cols.append(tmpl.format(w=w))

    harm = read_stata_from_zip("Harmonized/1140_Harmonized Dataset.zip", "H_CHARLS_D_Data.dta", columns=req_harm_cols)
    print(f"[CHARLS HARM W1-W4] Loaded {len(harm):,} participants ({harm.shape[1]} selected cols)")

    # 3. Raw W1–W4 Health Supplements & Wave 5 Harmonized Module
    supps = extract_wave1_to_4_raw_supplements()
    w5, w5_died_ids = extract_wave5_harmonized()

    # Longitudinal mortality exit tracking (Waves 2–4 iwstat in {5, 6} + Wave 5 Sample_Infor died == 1) for IPCW attrition
    mort_df = pd.DataFrame({"ID": harm["ID"]})
    for w_m in [2, 3, 4]:
        iw_s = pd.to_numeric(harm.get(f"r{w_m}iwstat", np.nan), errors="coerce")
        mort_df[f"died_by_w{w_m}"] = iw_s.isin([5.0, 6.0]).astype(int)
    mort_df["died_w1_w4"] = mort_df[["died_by_w2", "died_by_w3", "died_by_w4"]].max(axis=1)
    mort_df["first_death_wave_w14"] = np.where(
        mort_df["died_by_w2"] == 1,
        2.0,
        np.where(mort_df["died_by_w3"] == 1, 3.0, np.where(mort_df["died_by_w4"] == 1, 4.0, np.nan)),
    )

    # 4. Build Person-Level Static Table across All 5 Waves (W1–W5 Union)
    static_w14 = harm[["ID", "householdID", "communityID", "ragender", "raidbyear", "raeduc_c", "raeducl", "inw1", "inw2", "inw3", "inw4"]].copy()
    static_w14["in_harmonized_w1_w4"] = 1
    w5_demo = w5[["ID", "inw", "w5_ragender", "w5_byear", "w5_educ_raw", "rural_village", "rural_hukou"]].rename(columns={"inw": "inw5"})

    persons = pd.merge(static_w14, w5_demo, on="ID", how="outer")
    persons["in_harmonized_w1_w4"] = persons["in_harmonized_w1_w4"].fillna(0).astype(int)
    for w in [1, 2, 3, 4, 5]:
        persons[f"inw{w}"] = persons[f"inw{w}"].fillna(0).astype(int)

    persons["ragender"] = persons["ragender"].fillna(persons["w5_ragender"])
    persons["female"] = np.where(persons["ragender"].isin([1.0, 2.0]), (persons["ragender"] == 2.0).astype(float), np.nan)
    persons["birth_year"] = persons["raidbyear"].fillna(persons["w5_byear"])
    persons["raeduc_c"] = persons["raeduc_c"].fillna(persons["w5_educ_raw"])

    # Fairness & Literacy Strata (raeduc_c: 1=No formal educ illiterate, 2-3=incomplete primary/sishu, 4=elementary, 5=middle, 6+=high school+)
    persons["illiterate_no_formal"] = np.where(persons["raeduc_c"].notna(), (persons["raeduc_c"] == 1.0).astype(float), np.nan)
    persons["low_educ_le_elementary"] = np.where(persons["raeduc_c"].notna(), (persons["raeduc_c"] <= 4.0).astype(float), np.nan)
    educ_3cat = np.where(
        persons["raeduc_c"] == 1.0,
        "1_Illiterate_NoFormal",
        np.where(
            persons["raeduc_c"].isin([2.0, 3.0, 4.0]),
            "2_Primary_or_Sishu",
            np.where(persons["raeduc_c"] >= 5.0, "3_Secondary_or_Higher", "Unknown"),
        ),
    )
    persons["education_3stratum"] = educ_3cat

    # Baseline rural village & rural hukou (first observed across waves)
    r_vil = harm["h1rural"].combine_first(harm["h2rural"]).combine_first(harm["h3rural"]).combine_first(harm["h4rural"])
    r_huk = harm["r1rural2"].combine_first(harm["r2rural2"]).combine_first(harm["r3rural2"]).combine_first(harm["r4rural2"])
    r_map = pd.DataFrame({"ID": harm["ID"], "base_rural_vil": r_vil, "base_rural_huk": r_huk})
    persons = persons.merge(r_map, on="ID", how="left")
    persons["rural_village_baseline"] = persons["base_rural_vil"].combine_first(persons["rural_village"])
    persons["rural_hukou_baseline"] = persons["base_rural_huk"].combine_first(persons["rural_hukou"])
    persons = persons.merge(mort_df[["ID", "died_w1_w4", "first_death_wave_w14"]], on="ID", how="left")
    persons["died_w5_exit"] = persons["ID"].isin(w5_died_ids).astype(int)
    persons["died_followup"] = ((persons["died_w1_w4"].fillna(0) == 1) | (persons["died_w5_exit"] == 1)).astype(int)
    persons["first_death_wave"] = np.where(
        persons["first_death_wave_w14"].notna(),
        persons["first_death_wave_w14"],
        np.where(persons["died_w5_exit"] == 1, 5.0, np.nan),
    )

    # 5. Construct Long 5-Wave Panel (person x wave for active wave rows inw == 1)
    default_wave_years = {1: 2011.5, 2: 2013.5, 3: 2015.5, 4: 2018.5, 5: 2020.6}
    panel_chunks: List[pd.DataFrame] = []

    for w in [1, 2, 3, 4]:
        sub = harm[harm[f"inw{w}"] == 1.0].copy()
        pw = pd.DataFrame({"ID": sub["ID"], "wave": w, "inw": 1})
        pw["iwy"] = sub.get(f"r{w}iwy", np.nan)
        pw["iwm"] = sub.get(f"r{w}iwm", np.nan)
        dec_yr = pw["iwy"] + (pw["iwm"].fillna(7.0) - 0.5) / 12.0
        pw["calendar_year_decimal"] = dec_yr.fillna(default_wave_years[w])
        pw["time_years_from_2011"] = pw["calendar_year_decimal"] - 2011.5
        pw["age_years"] = sub.get(f"r{w}agey", np.nan)
        pw["wtrespb"] = sub.get(f"r{w}wtrespb", np.nan)
        pw["rural_village"] = sub.get(f"h{w}rural", np.nan)
        pw["rural_hukou"] = sub.get(f"r{w}rural2", np.nan)

        mstat = pd.to_numeric(sub.get(f"r{w}mstat", pd.Series(np.nan, index=sub.index)), errors="coerce")
        pw["unmarried_or_widowed"] = np.where(
            mstat.isin([1.0, 3.0, 4.0, 5.0, 7.0, 8.0]),
            mstat.isin([4.0, 5.0, 7.0, 8.0]).astype(float),
            np.nan,
        )
        cperc = pd.to_numeric(sub.get(f"hh{w}cperc", pd.Series(np.nan, index=sub.index)), errors="coerce")
        cperc = cperc.where(cperc >= 0.0, np.nan)
        pw["log_hh_cperc"] = np.log1p(cperc)
        pw["smoken"] = pd.to_numeric(sub.get(f"r{w}smoken", pd.Series(np.nan, index=sub.index)), errors="coerce")
        pw["smokev"] = pd.to_numeric(sub.get(f"r{w}smokev", pd.Series(np.nan, index=sub.index)), errors="coerce")
        pw["drinkl"] = pd.to_numeric(sub.get(f"r{w}drinkl", pd.Series(np.nan, index=sub.index)), errors="coerce")

        for cog_c in ["imrc", "dlrc", "tr20", "ser7", "orient", "draw"]:
            pw[cog_c] = sub.get(f"r{w}{cog_c}", np.nan)
        pw["w5_imrc_3trial_sum"] = np.nan

        pw["cesd10"] = sub.get(f"r{w}cesd10", np.nan)
        pw["cesd10m"] = sub.get(f"r{w}cesd10m", np.nan)
        slp_item = sub.get(f"r{w}sleeprl", np.nan)
        slp_scaled = np.where(slp_item.isin([1.0, 2.0, 3.0, 4.0]), slp_item - 1.0, np.nan)
        pw["cesd9_nosleep"] = np.where(pw["cesd10"].notna() & (~np.isnan(slp_scaled)), pw["cesd10"] - slp_scaled, np.nan)

        pw["social_active_harm"] = sub.get(f"r{w}socwk", np.nan)
        vg = sub.get(f"r{w}vgact_c", pd.Series(np.nan, index=sub.index))
        md = sub.get(f"r{w}mdact_c", pd.Series(np.nan, index=sub.index))
        pw["phys_active_modvig"] = np.where(vg.notna() | md.notna(), ((vg == 1.0) | (md == 1.0)).astype(float), np.nan)

        for clin in ["hibpe", "diabe", "hearte", "stroke", "dyslipe", "memrye", "rxhibp_c", "rxdiab_c", "rxheart_c", "rxdyslip_c", "adla_c", "adlab_c"]:
            pw[clin] = sub.get(f"r{w}{clin}", np.nan)
            if clin in ("hibpe", "diabe", "hearte", "stroke", "dyslipe", "memrye"):
                pw[f"w5_disputed_{clin}"] = 0
        pw["iadlza"] = sub[f"r{w}iadlza"] if f"r{w}iadlza" in sub.columns else np.nan
        pw["sbp_mmhg"] = sub[f"r{w}systo"] if f"r{w}systo" in sub.columns else np.nan
        pw["dbp_mmhg"] = sub[f"r{w}diasto"] if f"r{w}diasto" in sub.columns else np.nan
        pw["bmi_kgm2"] = sub[f"r{w}mbmi"] if f"r{w}mbmi" in sub.columns else np.nan

        # Merge raw W1-W4 sleep/nap/sensory/internet supplement
        pw = pw.merge(supps[w], on="ID", how="left")
        pw["social_active"] = pw["social_active_harm"]
        pw["internet_use"] = pw["internet_use_raw"]
        pw = pw.drop(columns=["social_active_harm", "internet_use_raw"])
        panel_chunks.append(pw)

    # Append Wave 5 chunk aligned to exact same schema
    pw5 = pd.DataFrame({"ID": w5["ID"], "wave": 5, "inw": 1})
    pw5["iwy"] = w5["iwy"]
    pw5["iwm"] = w5["iwm"]
    pw5["calendar_year_decimal"] = (w5["iwy"] + (w5["iwm"].fillna(8.0) - 0.5) / 12.0).fillna(2020.6)
    pw5["time_years_from_2011"] = pw5["calendar_year_decimal"] - 2011.5
    pw5["age_years"] = w5["w5_agey"]
    pw5["wtrespb"] = w5["wtrespb"]
    pw5["rural_village"] = w5["rural_village"]
    pw5["rural_hukou"] = w5["rural_hukou"]
    pw5["unmarried_or_widowed"] = w5["unmarried_or_widowed"]
    pw5["log_hh_cperc"] = np.nan
    pw5["smoken"] = w5["smoken"]
    pw5["smokev"] = w5["smokev"]
    pw5["drinkl"] = w5["drinkl"]
    for cog_c in ["imrc", "dlrc", "tr20", "ser7", "orient", "draw", "w5_imrc_3trial_sum"]:
        pw5[cog_c] = w5[cog_c]
    pw5["cesd10"] = w5["cesd10"]
    pw5["cesd10m"] = w5["cesd10m"]
    pw5["cesd9_nosleep"] = w5["cesd9_nosleep"]
    pw5["phys_active_modvig"] = w5["phys_active_modvig"]
    for clin in ["hibpe", "diabe", "hearte", "stroke", "dyslipe", "memrye"]:
        pw5[clin] = w5[clin]
        pw5[f"w5_disputed_{clin}"] = w5[f"w5_disputed_{clin}"]
    for clin_na in ["rxhibp_c", "rxdiab_c", "rxheart_c", "rxdyslip_c", "sbp_mmhg", "dbp_mmhg", "bmi_kgm2"]:
        pw5[clin_na] = np.nan
    for adl_c in ["adla_c", "adlab_c", "iadlza"]:
        pw5[adl_c] = w5[adl_c]
    for supp_c in ["sleep_hours", "nap_min", "glasses_worn", "poor_vision", "hearing_aid_worn", "poor_hearing", "social_active", "internet_use"]:
        pw5[supp_c] = w5[supp_c]
    pw5["vision_dist_score_1_5"] = np.nan
    pw5["vision_near_score_1_5"] = np.nan
    pw5["hearing_score_1_5"] = np.nan
    panel_chunks.append(pw5)

    # Align dtypes & column order explicitly across all 5 chunks to avoid any pandas FutureWarning
    canon_cols = list(panel_chunks[0].columns)
    aligned_chunks: List[pd.DataFrame] = []
    for chk in panel_chunks:
        chk = chk.reindex(columns=canon_cols).copy()
        for c in canon_cols:
            if c == "ID":
                chk[c] = chk[c].astype(str)
            elif c in ("wave", "inw") or c.startswith("w5_disputed_"):
                chk[c] = pd.to_numeric(chk[c], errors="coerce").fillna(0).astype("int64")
            else:
                chk[c] = pd.to_numeric(chk[c], errors="coerce").astype("float64")
        aligned_chunks.append(chk)

    panel = pd.concat(aligned_chunks, axis=0, ignore_index=True)
    panel = panel.sort_values(["ID", "wave"]).reset_index(drop=True)

    # Carry forward cumulative ever-diagnosed chronic conditions into Wave 5 (Gateway parity, respecting Wave 5 disputed==1)
    for clin in ["hibpe", "diabe", "hearte", "stroke", "dyslipe", "memrye"]:
        prior_max = panel.groupby("ID")[clin].cummax()
        is_w5 = panel["wave"] == 5
        not_disputed = panel[f"w5_disputed_{clin}"] == 0
        panel[clin] = np.where(is_w5 & not_disputed & (prior_max == 1.0), 1.0, panel[clin])
        panel = panel.drop(columns=[f"w5_disputed_{clin}"])
    # Forward-fill household log per-capita consumption into Wave 5 where available for static/longitudinal SES control
    panel["log_hh_cperc"] = panel.groupby("ID")["log_hh_cperc"].ffill().bfill()

    # Merge person-level static covariates onto panel
    static_keep = [
        "ID", "householdID", "communityID", "female", "birth_year",
        "raeduc_c", "raeducl", "illiterate_no_formal", "low_educ_le_elementary", "education_3stratum",
        "rural_village_baseline", "rural_hukou_baseline", "in_harmonized_w1_w4",
        "died_followup", "first_death_wave",
    ]
    panel = panel.merge(persons[static_keep], on="ID", how="left")
    panel["age_years"] = panel["age_years"].fillna(np.floor(panel["calendar_year_decimal"]) - panel["birth_year"])
    panel["rural_village"] = panel["rural_village"].fillna(panel["rural_village_baseline"])
    panel["rural_hukou"] = panel["rural_hukou"].fillna(panel["rural_hukou_baseline"])

    # Derived cognitive composites & modifiable intervention indicators
    panel["mental_intact_10"] = np.where(
        panel[["ser7", "orient", "draw"]].notna().all(axis=1),
        panel["ser7"] + panel["orient"] + panel["draw"],
        np.nan,
    )
    panel["global_cog_30"] = np.where(
        panel["tr20"].notna() & panel["mental_intact_10"].notna(),
        panel["tr20"] + panel["mental_intact_10"],
        np.nan,
    )
    panel["optimal_sleep_6_to_85h"] = np.where(
        panel["sleep_hours"].notna(),
        ((panel["sleep_hours"] >= 6.0) & (panel["sleep_hours"] <= 8.5)).astype(float),
        np.nan,
    )
    panel["dual_sensory_impairment"] = np.where(
        panel["poor_vision"].notna() & panel["poor_hearing"].notna(),
        ((panel["poor_vision"] == 1.0) & (panel["poor_hearing"] == 1.0)).astype(float),
        np.nan,
    )
    panel["any_sensory_impairment"] = np.where(
        panel["poor_vision"].notna() | panel["poor_hearing"].notna(),
        ((panel["poor_vision"] == 1.0) | (panel["poor_hearing"] == 1.0)).astype(float),
        np.nan,
    )

    # Retest Practice-Effect Counters (number of PRIOR completed tests before current wave!)
    panel["has_cog_obs"] = panel[["imrc", "dlrc", "ser7", "orient", "draw"]].notna().any(axis=1).astype(int)
    panel["has_tr20_obs"] = panel["tr20"].notna().astype(int)
    panel["has_cesd_obs"] = panel["cesd10"].notna().astype(int)
    panel["k_prior_cog_tests"] = panel.groupby("ID")["has_cog_obs"].cumsum() - panel["has_cog_obs"]
    panel["k_prior_tr20_tests"] = panel.groupby("ID")["has_tr20_obs"].cumsum() - panel["has_tr20_obs"]
    panel["k_prior_cesd_tests"] = panel.groupby("ID")["has_cesd_obs"].cumsum() - panel["has_cesd_obs"]
    panel["first_wave_observed"] = panel.groupby("ID")["wave"].transform("min")
    panel["is_refreshment_entrant"] = (panel["first_wave_observed"] > 1).astype(int)

    # Merge wave-matched venous blood biomarkers onto Wave 1 (2011) and Wave 3 (2015) rows
    b_w1 = pd.DataFrame({"ID": blood.loc[blood["in_blood_2011"] == 1, "ID"], "wave": 1})
    b_w3 = pd.DataFrame({"ID": blood.loc[blood["in_blood_2015"] == 1, "ID"], "wave": 3})
    for a in ["crp_mgl", "hba1c_pct", "tchol_mgdl", "hdl_mgdl", "ldl_mgdl", "tg_mgdl", "glucose_mgdl", "uric_acid_mgdl", "cystatin_c_mgl", "creatinine_mgdl", "wbc_10e9L", "hgb_gdl"]:
        b_w1[f"blood_{a}"] = blood.loc[blood["in_blood_2011"] == 1, f"{a}_2011"].values
        b_w3[f"blood_{a}"] = blood.loc[blood["in_blood_2015"] == 1, f"{a}_2015"].values
    b_panel = pd.concat([b_w1, b_w3], axis=0, ignore_index=True)
    panel = panel.merge(b_panel, on=["ID", "wave"], how="left")
    panel["blood_log_crp"] = np.log(panel["blood_crp_mgl"])
    panel = panel.merge(blood[["ID", "in_blood_2011", "in_blood_2015", "is_paired_2011_2015"]], on="ID", how="left")
    for bc in ["in_blood_2011", "in_blood_2015", "is_paired_2011_2015"]:
        panel[bc] = panel[bc].fillna(0).astype(int)

    # Summarize person-level longitudinal coverage
    p_cov = (
        panel.groupby("ID", as_index=False)
        .agg(
            n_waves_interviewed=("wave", "count"),
            n_waves_tr20=("has_tr20_obs", "sum"),
            n_waves_any_cog=("has_cog_obs", "sum"),
            n_waves_global_cog30=("global_cog_30", lambda s: int(s.notna().sum())),
            n_waves_cesd10=("has_cesd_obs", "sum"),
            n_waves_coupled_tr20_cesd=("ID", lambda idx: 0),  # replaced below
            baseline_wave=("wave", "min"),
            baseline_age=("age_years", "first"),
            max_follow_years=("time_years_from_2011", lambda s: float(s.max() - s.min())),
        )
    )
    coupled_cnt = (
        panel.assign(both=((panel["tr20"].notna()) & (panel["cesd10"].notna())).astype(int))
        .groupby("ID", as_index=False)["both"]
        .sum()
        .rename(columns={"both": "n_waves_coupled_tr20_cesd"})
    )
    p_cov = p_cov.drop(columns=["n_waves_coupled_tr20_cesd"]).merge(coupled_cnt, on="ID", how="left")
    persons = persons.merge(p_cov, on="ID", how="left").merge(
        blood[["ID", "in_blood_2011", "in_blood_2015", "is_paired_2011_2015"]], on="ID", how="left"
    )
    for bc in ["in_blood_2011", "in_blood_2015", "is_paired_2011_2015"]:
        persons[bc] = persons[bc].fillna(0).astype(int)

    # Save cleaned parquets
    panel_pq = OUT_DATA / "charls_5wave_panel.parquet"
    persons_pq = OUT_DATA / "charls_participants_summary.parquet"
    panel.to_parquet(panel_pq, index=False)
    persons.to_parquet(persons_pq, index=False)
    print(f"[CHARLS SAVE] Saved {panel_pq.name} ({len(panel):,} person-waves), {persons_pq.name} ({len(persons):,} persons), {blood_parquet.name} ({len(blood):,} blood union)")

    # 6. Exact CONSORT Flow Audit (reporting both W1-W4 Harmonized and W1-W5 Full 9-Year Panel counts transparently)
    harm_only = persons[persons["in_harmonized_w1_w4"] == 1]
    p_ge45 = persons[persons["baseline_age"].fillna(0) >= 45.0]
    w14_tr20_cnt = harm[["r1tr20", "r2tr20", "r3tr20", "r4tr20"]].notna().sum(axis=1)
    w14_cesd_cnt = harm[["r1cesd10", "r2cesd10", "r3cesd10", "r4cesd10"]].notna().sum(axis=1)

    consort_rows = [
        ("01_total_5wave_union_2011_2020", "Total unique CHARLS participants across Waves 1–5 (2011–2020 union)", int(len(persons))),
        ("01a_harmonized_w1_w4_total", "  Harmonized CHARLS Waves 1–4 cohort (H_CHARLS_D_Data.dta)", int(len(harm))),
        ("01b_wave5_2020_total", "  Wave 5 (2020) respondents (Health_Status_and_Functioning.dta)", int(len(w5))),
        ("01c_wave5_overlap_with_w1_w4", "  Wave 5 respondents linked longitudinally to Waves 1–4", int((persons["in_harmonized_w1_w4"] == 1).astype(int).mul(persons["inw5"]).sum())),
        ("01d_wave5_new_entrants", "  New Wave 5 entrants not in Waves 1–4", int(((persons["in_harmonized_w1_w4"] == 0) & (persons["inw5"] == 1)).sum())),
        ("02_total_person_waves_w1_w5", "Total longitudinal person-wave interviews across Waves 1–5", int(len(panel))),
        ("02a_wave1_2011_n", "  Wave 1 (2011) active respondents", int((panel["wave"] == 1).sum())),
        ("02b_wave2_2013_n", "  Wave 2 (2013) active respondents", int((panel["wave"] == 2).sum())),
        ("02c_wave3_2015_n", "  Wave 3 (2015) active respondents", int((panel["wave"] == 3).sum())),
        ("02d_wave4_2018_n", "  Wave 4 (2018) active respondents", int((panel["wave"] == 4).sum())),
        ("02e_wave5_2020_n", "  Wave 5 (2020) active respondents", int((panel["wave"] == 5).sum())),
        ("03_participants_ge2_waves_interviewed_w1_w5", "Participants interviewed in >= 2 waves across Waves 1–5", int((persons["n_waves_interviewed"] >= 2).sum())),
        ("03a_participants_ge3_waves_interviewed_w1_w5", "  Participants interviewed in >= 3 waves across Waves 1–5", int((persons["n_waves_interviewed"] >= 3).sum())),
        ("03b_participants_ge4_waves_interviewed_w1_w5", "  Participants interviewed in >= 4 waves across Waves 1–5", int((persons["n_waves_interviewed"] >= 4).sum())),
        ("03c_participants_all5_waves_interviewed", "  Participants interviewed in all 5 waves (2011–2020)", int((persons["n_waves_interviewed"] == 5).sum())),
        ("04_w1_w4_tr20_ge2_manifest_check", "Episodic memory (tr20) >= 2 waves in Harmonized W1–W4 alone (manifest check)", int((w14_tr20_cnt >= 2).sum())),
        ("04a_w1_w4_tr20_ge3_manifest_check", "  Episodic memory (tr20) >= 3 waves in Harmonized W1–W4 alone (manifest check)", int((w14_tr20_cnt >= 3).sum())),
        ("05_w1_w5_tr20_ge2_full_panel", "Episodic memory (tr20) >= 2 waves across Full 9-Year W1–W5 Panel", int((persons["n_waves_tr20"] >= 2).sum())),
        ("05a_w1_w5_tr20_ge3_full_panel", "  Episodic memory (tr20) >= 3 waves across Full 9-Year W1–W5 Panel", int((persons["n_waves_tr20"] >= 3).sum())),
        ("05b_w1_w5_tr20_ge4_full_panel", "  Episodic memory (tr20) >= 4 waves across Full 9-Year W1–W5 Panel", int((persons["n_waves_tr20"] >= 4).sum())),
        ("06_w1_w5_cesd10_ge2_full_panel", "Depressive score (CES-D10) >= 2 waves across Full 9-Year W1–W5 Panel (W1–W4 alone = " + f"{int((w14_cesd_cnt >= 2).sum()):,})", int((persons["n_waves_cesd10"] >= 2).sum())),
        ("06a_w1_w5_cesd10_ge3_full_panel", "  Depressive score (CES-D10) >= 3 waves across Full 9-Year W1–W5 Panel", int((persons["n_waves_cesd10"] >= 3).sum())),
        ("06b_w1_w5_cesd10_ge4_full_panel", "  Depressive score (CES-D10) >= 4 waves across Full 9-Year W1–W5 Panel", int((persons["n_waves_cesd10"] >= 4).sum())),
        ("07_primary_sde_cohort_coupled_ge2_waves", "Primary Pillar 2 Coupled SDE Cohort: >= 2 waves with BOTH tr20 & CES-D10 observed (W1–W5)", int((persons["n_waves_coupled_tr20_cesd"] >= 2).sum())),
        ("07a_primary_sde_cohort_age_ge45", "  Coupled SDE Cohort aged >= 45 years at baseline (>= 2 coupled waves)", int(((persons["n_waves_coupled_tr20_cesd"] >= 2) & (persons["baseline_age"] >= 45.0)).sum())),
        ("07b_coupled_sde_ge3_waves", "  Coupled SDE Cohort with >= 3 waves of joint (tr20, CES-D10)", int((persons["n_waves_coupled_tr20_cesd"] >= 3).sum())),
        ("08_venous_blood_union_2011_2015", "Venous blood biomarker union (2011 Wave 1 N=11,847 U 2015 Wave 3 N=13,420)", int(len(blood))),
        ("08a_venous_blood_paired_2011_2015", "  Exact 2-wave longitudinal paired venous blood cohort (2011 & 2015, 11->12-char ID harmonized)", int(blood["is_paired_2011_2015"].sum())),
        ("08b_paired_blood_and_coupled_sde_ge2", "  Paired 2011 & 2015 venous blood + >= 2 coupled cognitive-depressive waves (W1–W5)", int(((persons["is_paired_2011_2015"] == 1) & (persons["n_waves_coupled_tr20_cesd"] >= 2)).sum())),
    ]
    consort_df = pd.DataFrame(consort_rows, columns=["step_id", "description", "n_participants"])
    consort_df.to_csv(OUT_AUDIT / "charls_consort_flow.csv", index=False)
    with open(OUT_AUDIT / "charls_consort_flow.json", "w", encoding="utf-8") as f:
        json.dump({"cohort": "CHARLS_2011_2020_5Wave_Plus_PairedBlood", "steps": consort_df.to_dict(orient="records")}, f, indent=2)

    # 7. Wave-by-Wave Missingness Audit
    wave_vars = [
        "age_years", "female", "raeduc_c", "rural_village", "unmarried_or_widowed", "log_hh_cperc",
        "smoken", "drinkl",
        "imrc", "dlrc", "tr20", "ser7", "orient", "draw", "mental_intact_10", "global_cog_30",
        "cesd10", "cesd9_nosleep",
        "sleep_hours", "nap_min", "poor_vision", "poor_hearing", "social_active", "internet_use", "phys_active_modvig",
        "hibpe", "diabe", "hearte", "stroke", "dyslipe", "memrye", "adla_c", "adlab_c", "iadlza",
        "blood_crp_mgl", "blood_hba1c_pct", "blood_tchol_mgdl", "blood_cystatin_c_mgl",
    ]
    wmiss_rows = []
    for v in wave_vars:
        row: Dict[str, object] = {"variable": v}
        for w in [1, 2, 3, 4, 5]:
            sub = panel[panel["wave"] == w]
            ntot = len(sub)
            nnon = int(sub[v].notna().sum())
            row[f"W{w}_N"] = ntot
            row[f"W{w}_nonnull"] = nnon
            row[f"W{w}_miss_pct"] = round(100.0 * (ntot - nnon) / ntot, 2) if ntot > 0 else np.nan
        wmiss_rows.append(row)
    pd.DataFrame(wmiss_rows).to_csv(OUT_AUDIT / "charls_wave_missingness_audit.csv", index=False)

    # 8. Psychometric Censoring (Floor/Ceiling) & Age-Adjusted Retest Practice-Effect Audit
    scale_bounds = [
        ("orient", 0.0, 4.0),
        ("draw", 0.0, 1.0),
        ("ser7", 0.0, 5.0),
        ("imrc", 0.0, 10.0),
        ("dlrc", 0.0, 10.0),
        ("tr20", 0.0, 20.0),
        ("mental_intact_10", 0.0, 10.0),
        ("global_cog_30", 0.0, 30.0),
        ("cesd10", 0.0, 30.0),
    ]
    censor_rows = []
    for sc, lo, hi in scale_bounds:
        # Estimate age/sex/education/urbanicity/wave-adjusted logarithmic retest effect in Waves 2-5
        w25 = panel[
            (panel["wave"] >= 2)
            & panel[sc].notna()
            & panel["age_years"].notna()
            & panel["female"].notna()
            & panel["raeduc_c"].notna()
            & panel["rural_village"].notna()
        ].copy()
        k_col = "k_prior_cesd_tests" if sc == "cesd10" else ("k_prior_tr20_tests" if sc == "tr20" else "k_prior_cog_tests")
        w25["log1p_k"] = np.log1p(w25[k_col].astype(float))
        retest_beta, retest_se, retest_p = np.nan, np.nan, np.nan
        if len(w25) > 500 and w25["log1p_k"].std() > 0:
            fit = smf.ols(f"{sc} ~ log1p_k + age_years + I(age_years**2) + female + rural_village + C(education_3stratum) + C(wave)", data=w25).fit()
            retest_beta = float(fit.params.get("log1p_k", np.nan))
            retest_se = float(fit.bse.get("log1p_k", np.nan))
            retest_p = float(fit.pvalues.get("log1p_k", np.nan))

        for w_label, sub in [("All_W1_W5", panel)] + [(f"W{w}", panel[panel["wave"] == w]) for w in [1, 2, 3, 4, 5]]:
            vals = sub[sc].dropna().astype(float)
            n = len(vals)
            if n == 0:
                continue
            fl_n = int((vals <= lo).sum())
            ce_n = int((vals >= hi).sum())
            censor_rows.append(
                {
                    "scale": sc,
                    "wave": w_label,
                    "n_valid": n,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "floor_n": fl_n,
                    "floor_pct": round(100.0 * fl_n / n, 2),
                    "ceiling_n": ce_n,
                    "ceiling_pct": round(100.0 * ce_n / n, 2),
                    "total_boundary_censored_pct": round(100.0 * (fl_n + ce_n) / n, 2),
                    "mean": round(float(vals.mean()), 3),
                    "sd": round(float(vals.std(ddof=1)), 3),
                    "median": round(float(np.median(vals)), 3),
                    "skewness": round(float(stats.skew(vals, bias=False)), 3),
                    "retest_log1p_k_beta_w2_w5": round(retest_beta, 4) if w_label == "All_W1_W5" else np.nan,
                    "retest_log1p_k_se": round(retest_se, 4) if w_label == "All_W1_W5" else np.nan,
                    "retest_log1p_k_p": float(f"{retest_p:.3e}") if (w_label == "All_W1_W5" and not np.isnan(retest_p)) else np.nan,
                }
            )
    pd.DataFrame(censor_rows).to_csv(OUT_AUDIT / "charls_psychometric_censoring_audit.csv", index=False)

    # 9. Paired Venous Blood Longitudinal Audit (2011 vs 2015 across exact N = 7,648)
    paired = blood[blood["is_paired_2011_2015"] == 1].copy()
    blood_audit_rows = []
    for a in [
        "crp_mgl", "hba1c_pct", "tchol_mgdl", "hdl_mgdl", "ldl_mgdl", "tg_mgdl",
        "glucose_mgdl", "uric_acid_mgdl", "cystatin_c_mgl", "creatinine_mgdl", "bun_mgdl",
        "wbc_10e9L", "hgb_gdl", "plt_10e9L",
    ]:
        both = paired[[f"{a}_2011", f"{a}_2015", f"delta_{a}"]].dropna()
        n_b = len(both)
        t_stat, p_val = stats.ttest_rel(both[f"{a}_2015"], both[f"{a}_2011"])
        blood_audit_rows.append(
            {
                "analyte": a,
                "n_paired_complete": int(n_b),
                "mean_2011": round(float(both[f"{a}_2011"].mean()), 3),
                "sd_2011": round(float(both[f"{a}_2011"].std(ddof=1)), 3),
                "median_2011": round(float(both[f"{a}_2011"].median()), 3),
                "mean_2015": round(float(both[f"{a}_2015"].mean()), 3),
                "sd_2015": round(float(both[f"{a}_2015"].std(ddof=1)), 3),
                "median_2015": round(float(both[f"{a}_2015"].median()), 3),
                "mean_delta_4yr": round(float(both[f"delta_{a}"].mean()), 4),
                "sd_delta_4yr": round(float(both[f"delta_{a}"].std(ddof=1)), 4),
                "paired_t_stat": round(float(t_stat), 3),
                "paired_p_value": float(f"{p_val:.3e}"),
            }
        )
    pd.DataFrame(blood_audit_rows).to_csv(OUT_AUDIT / "charls_paired_blood_audit.csv", index=False)

    # 10. Baseline Table 1 (Baseline First-Observed Wave across All Participants, Primary Coupled SDE Cohort, and Paired Blood Cohort)
    base_obs = panel.sort_values(["ID", "wave"]).groupby("ID", as_index=False).first()
    base_obs = base_obs.merge(
        persons[["ID", "n_waves_interviewed", "n_waves_coupled_tr20_cesd"]],
        on="ID",
        how="left",
    )

    def fmt_msd(s: pd.Series) -> str:
        v = s.dropna()
        return f"{v.mean():.2f} ({v.std(ddof=1):.2f})" if len(v) else "NA"

    def fmt_miqr(s: pd.Series) -> str:
        v = s.dropna()
        if not len(v):
            return "NA"
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        return f"{q50:.2f} [{q25:.2f}, {q75:.2f}]"

    def fmt_np(s: pd.Series) -> str:
        v = s.dropna()
        if not len(v):
            return "NA"
        c = int((v == 1.0).sum())
        return f"{c:,} ({100.0 * c / len(v):.1f}%)"

    t1_arms = {
        "All_CHARLS_W1_W5_Union": base_obs,
        "Primary_Coupled_SDE_ge2Waves": base_obs[base_obs["n_waves_coupled_tr20_cesd"] >= 2],
        "Paired_Venous_Blood_2011_2015": base_obs[base_obs["is_paired_2011_2015"] == 1],
    }
    t1_rows = [
        {"Characteristic": "Participants, N", **{k: f"{len(v):,}" for k, v in t1_arms.items()}},
        {"Characteristic": "Waves interviewed (1–5), mean (SD)", **{k: fmt_msd(v["n_waves_interviewed"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Baseline age, years, mean (SD)", **{k: fmt_msd(v["age_years"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Female, n (%)", **{k: fmt_np(v["female"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Rural village residence, n (%)", **{k: fmt_np(v["rural_village"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Agricultural/rural hukou, n (%)", **{k: fmt_np(v["rural_hukou"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Illiterate / no formal schooling, n (%)", **{k: fmt_np(v["illiterate_no_formal"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Low education (<= elementary school), n (%)", **{k: fmt_np(v["low_educ_le_elementary"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Unmarried / widowed / separated, n (%)", **{k: fmt_np(v["unmarried_or_widowed"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Current smoker, n (%)", **{k: fmt_np(v["smoken"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Alcohol consumption (past year), n (%)", **{k: fmt_np(v["drinkl"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Died during 9-year follow-up (IPCW exit), n (%)", **{k: fmt_np(v["died_followup"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Episodic memory (tr20, 0–20), mean (SD)", **{k: fmt_msd(v["tr20"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Immediate word recall (0–10), mean (SD)", **{k: fmt_msd(v["imrc"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Delayed word recall (0–10), mean (SD)", **{k: fmt_msd(v["dlrc"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Serial 7s working memory (0–5), mean (SD)", **{k: fmt_msd(v["ser7"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Date orientation (0–4), mean (SD)", **{k: fmt_msd(v["orient"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Figure drawing intact (0–1), n (%)", **{k: fmt_np(v["draw"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Global cognition composite (0–30), mean (SD)", **{k: fmt_msd(v["global_cog_30"]) for k, v in t1_arms.items()}},
        {"Characteristic": "CES-D10 depressive score (0–30), median [IQR]", **{k: fmt_miqr(v["cesd10"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Nighttime sleep duration, hours, mean (SD)", **{k: fmt_msd(v["sleep_hours"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Optimal sleep (6.0–8.5 h), n (%)", **{k: fmt_np(v["optimal_sleep_6_to_85h"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Poor vision, n (%)", **{k: fmt_np(v["poor_vision"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Poor hearing, n (%)", **{k: fmt_np(v["poor_hearing"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Dual sensory impairment, n (%)", **{k: fmt_np(v["dual_sensory_impairment"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Social activity participation, n (%)", **{k: fmt_np(v["social_active"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Internet / digital engagement, n (%)", **{k: fmt_np(v["internet_use"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Hypertension history, n (%)", **{k: fmt_np(v["hibpe"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Diabetes history, n (%)", **{k: fmt_np(v["diabe"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Heart disease history, n (%)", **{k: fmt_np(v["hearte"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Stroke history, n (%)", **{k: fmt_np(v["stroke"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Baseline hs-CRP (2011), mg/L, median [IQR]", **{k: fmt_miqr(v["blood_crp_mgl"]) for k, v in t1_arms.items()}},
        {"Characteristic": "Baseline HbA1c (2011), %, mean (SD)", **{k: fmt_msd(v["blood_hba1c_pct"]) for k, v in t1_arms.items()}},
    ]
    pd.DataFrame(t1_rows).to_csv(OUT_AUDIT / "charls_baseline_table1.csv", index=False)

    print("=== CHARLS Phase 0 CONSORT Summary ===")
    for _, r in consort_df.iterrows():
        print(f"  {r['step_id']}: {r['n_participants']:,} — {r['description']}")
    print(f"[CHARLS DONE] Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    build_charls_5wave_panel_and_audits()
