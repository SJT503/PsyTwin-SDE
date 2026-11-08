#!/usr/bin/env python3
"""
Phase 0 — Independent Verification & Governance Audit Gate
==========================================================
Independently audits all Phase 0 derived parquets (data_processed/) and CONSORT/QC/Baseline
deliverables (results/phase0_cohort_audit/) against CLAUDE.md, config/DATA_MANIFEST.json,
and lab-constitution iron rules before freezing Phase 1 SAP (docs/SAP_v1_FROZEN.md).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_NHANES = Path(os.environ.get("NHANES_RAW_DIR", str(PROJECT_ROOT / "data_raw" / "NHANES")))
RAW_CHARLS = Path(os.environ.get("CHARLS_RAW_DIR", str(PROJECT_ROOT / "data_raw" / "CHARLS")))
DATA_DIR = PROJECT_ROOT / "data_processed"
AUDIT_DIR = PROJECT_ROOT / "results" / "phase0_cohort_audit"


def check_no_subnormal_sas_zeros(df: pd.DataFrame, label: str) -> int:
    num_cols = df.select_dtypes(include=[np.number]).columns
    leaked = 0
    for c in num_cols:
        arr = df[c].dropna().to_numpy(dtype=float)
        bad = int(np.sum((np.abs(arr) > 0.0) & (np.abs(arr) < 1e-65)))
        leaked += bad
    return leaked


def check_raw_readonly_untouched(raw_dir: Path, session_start_epoch: float) -> Dict[str, object]:
    recent_files: List[str] = []
    total_files = 0
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            total_files += 1
            fp = Path(root) / fn
            try:
                if fp.stat().st_mtime >= session_start_epoch:
                    recent_files.append(str(fp))
            except OSError:
                pass
    return {"raw_dir": str(raw_dir), "total_files_scanned": total_files, "modified_since_session_start": len(recent_files), "recent_files": recent_files[:10]}


def main() -> None:
    # Session start cutoff: 2 hours prior to current wall clock
    cutoff_epoch = time.time() - 7200.0
    checks: List[Dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print("=== Running Phase 0 Independent Audit Gate ===")

    # 1. Required files existence check
    req_parquets = [
        "nhanes_2011_2014_participants.parquet",
        "nhanes_2011_2014_paxhr_clean.parquet",
        "charls_5wave_panel.parquet",
        "charls_participants_summary.parquet",
        "charls_paired_blood_2011_2015.parquet",
    ]
    req_audits = [
        "nhanes_consort_flow.csv",
        "nhanes_consort_flow.json",
        "nhanes_missingness_audit.csv",
        "nhanes_psychometric_fidelity_audit.csv",
        "nhanes_snfl_distribution_audit.csv",
        "nhanes_actigraphy_qc_summary.csv",
        "nhanes_baseline_table1.csv",
        "charls_consort_flow.csv",
        "charls_consort_flow.json",
        "charls_wave_missingness_audit.csv",
        "charls_psychometric_censoring_audit.csv",
        "charls_paired_blood_audit.csv",
        "charls_baseline_table1.csv",
    ]
    missing_pq = [f for f in req_parquets if not (DATA_DIR / f).exists()]
    missing_au = [f for f in req_audits if not (AUDIT_DIR / f).exists()]
    record("01_required_parquets_exist", len(missing_pq) == 0, f"5/5 parquets present (missing={missing_pq})")
    record("02_required_audit_tables_exist", len(missing_au) == 0, f"13/13 audit tables present (missing={missing_au})")

    # 2. NHANES integrity & manifest reconciliation
    nh_p = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_participants.parquet")
    nh_hr = pd.read_parquet(DATA_DIR / "nhanes_2011_2014_paxhr_clean.parquet")

    record("03_nhanes_unique_seqn", nh_p["SEQN"].is_unique and len(nh_p) == 19931, f"N={len(nh_p):,} unique SEQN")
    n_snfl = int(nh_p["snfl_pgml"].notna().sum())
    record("04_nhanes_snfl_exact_2071", n_snfl == 2071, f"sNfL non-null N={n_snfl:,} (expected 2,071)")
    n_dsst = int(nh_p.loc[nh_p["age_years"] >= 60.0, "dsst"].notna().sum())
    record("05_nhanes_dsst_exact_3014", n_dsst == 3014, f"Older adults >=60 DSST N={n_dsst:,} (expected 3,014)")

    leak_p = check_no_subnormal_sas_zeros(nh_p, "nhanes_participants")
    leak_hr = check_no_subnormal_sas_zeros(nh_hr, "nhanes_paxhr")
    record("06_nhanes_zero_sas_ieee_subnormal_leak", (leak_p + leak_hr) == 0, f"leaked subnormal floats = {leak_p + leak_hr}")

    dup_hr = int(nh_hr.duplicated(subset=["SEQN", "seq_hour_idx"]).sum())
    valid_clock = bool(nh_hr["clock_hour"].isin(list(range(24))).all())
    record("07_nhanes_paxhr_clock_and_uniqueness", dup_hr == 0 and valid_clock, f"rows={len(nh_hr):,}, dup_(SEQN,seq_hour_idx)={dup_hr}, clock_hour_0_23={valid_clock}")

    # 3. CHARLS integrity & manifest reconciliation
    ch_panel = pd.read_parquet(DATA_DIR / "charls_5wave_panel.parquet")
    ch_pers = pd.read_parquet(DATA_DIR / "charls_participants_summary.parquet")
    ch_blood = pd.read_parquet(DATA_DIR / "charls_paired_blood_2011_2015.parquet")

    n_harm = int((ch_pers["in_harmonized_w1_w4"] == 1).sum())
    record("08_charls_harmonized_exact_25586", n_harm == 25586 and ch_pers["ID"].is_unique, f"Harmonized W1-W4 N={n_harm:,} (expected 25,586), total W1-W5 union N={len(ch_pers):,}")

    n_b_union = len(ch_blood)
    n_b_pair = int(ch_blood["is_paired_2011_2015"].sum())
    all_12char = bool((ch_blood["ID"].str.len() == 12).all() and (ch_pers["ID"].str.len() == 12).all())
    record("09_charls_blood_exact_7648_paired_17619_union", n_b_pair == 7648 and n_b_union == 17619 and all_12char, f"Paired N={n_b_pair:,} (expected 7,648), Union N={n_b_union:,} (expected 17,619), 12-char IDs={all_12char}")

    dup_pw = int(ch_panel.duplicated(subset=["ID", "wave"]).sum())
    bounds_ok = True
    bound_specs = [("imrc", 0, 10), ("dlrc", 0, 10), ("tr20", 0, 20), ("ser7", 0, 5), ("orient", 0, 4), ("draw", 0, 1), ("cesd10", 0, 30)]
    bound_details = []
    for col, lo, hi in bound_specs:
        s = ch_panel[col].dropna()
        ok = bool(((s >= lo) & (s <= hi)).all())
        bounds_ok = bounds_ok and ok
        bound_details.append(f"{col}:[{s.min():.0f},{s.max():.0f}]")
    record("10_charls_panel_uniqueness_and_scale_bounds", dup_pw == 0 and bounds_ok, f"person-waves={len(ch_panel):,}, dup_(ID,wave)={dup_pw}, bounds={' '.join(bound_details)}")

    # Verify retest counter == 0 at first observed cognitive wave
    first_cog = ch_panel[ch_panel["has_cog_obs"] == 1].sort_values(["ID", "wave"]).groupby("ID").first()
    retest_zero_at_entry = bool((first_cog["k_prior_cog_tests"] == 0).all())
    record("11_charls_retest_counter_exact", retest_zero_at_entry, f"first cognitive assessment k_prior_cog_tests==0 for all {len(first_cog):,} examinees: {retest_zero_at_entry}")

    # 4. Forbidden scope terms check (ZERO brain-kidney / AKI / DICOM)
    forbidden_tokens = ["aki", "acute_kidney", "brain_kidney", "kidney_disease", "mannitol", "dicom", "nifti"]
    all_cols = list(nh_p.columns) + list(nh_hr.columns) + list(ch_panel.columns) + list(ch_pers.columns) + list(ch_blood.columns)
    bad_cols = [c for c in all_cols if any(tok in c.lower() for tok in forbidden_tokens)]
    bad_text_hits: List[str] = []
    for fn in req_audits:
        txt = (AUDIT_DIR / fn).read_text(encoding="utf-8", errors="ignore").lower()
        for tok in forbidden_tokens:
            if tok in txt:
                bad_text_hits.append(f"{fn}:{tok}")
    record("12_zero_forbidden_scope_terms", len(bad_cols) == 0 and len(bad_text_hits) == 0, f"forbidden_cols={bad_cols}, forbidden_text_hits={bad_text_hits}")

    # 5. Strict read-only raw directory check
    nh_ro = check_raw_readonly_untouched(RAW_NHANES, cutoff_epoch)
    ch_ro = check_raw_readonly_untouched(RAW_CHARLS, cutoff_epoch)
    ro_ok = (nh_ro["modified_since_session_start"] == 0) and (ch_ro["modified_since_session_start"] == 0)
    record(
        "13_strict_readonly_raw_directories_untouched",
        ro_ok,
        f"${NHANES_RAW_DIR} scanned={nh_ro['total_files_scanned']:,} mod={nh_ro['modified_since_session_start']}; ${CHARLS_RAW_DIR} scanned={ch_ro['total_files_scanned']:,} mod={ch_ro['modified_since_session_start']}",
    )

    overall_pass = all(c["status"] == "PASS" for c in checks)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "n_checks_total": len(checks),
        "n_checks_passed": sum(1 for c in checks if c["status"] == "PASS"),
        "checks": checks,
    }
    with open(AUDIT_DIR / "PHASE0_VERIFICATION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Phase 0 Independent Verification & Governance Audit Certificate",
        "",
        f"- **Audit Timestamp (UTC)**: `{report['timestamp_utc']}`",
        f"- **Overall Gate Status**: **`{report['overall_status']}`** (`{report['n_checks_passed']}/{report['n_checks_total']}` checks passed)",
        "",
        "| Check ID | Status | Empirical Verification Detail |",
        "| :--- | :---: | :--- |",
    ]
    for c in checks:
        md_lines.append(f"| `{c['check']}` | **{c['status']}** | {c['detail']} |")
    (AUDIT_DIR / "PHASE0_VERIFICATION_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"=== Phase 0 Audit Gate Overall Status: {report['overall_status']} ({report['n_checks_passed']}/{report['n_checks_total']}) ===")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
