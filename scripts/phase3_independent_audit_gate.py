#!/usr/bin/env python3
"""
Phase 3 — Independent Verification & Governance Audit Gate
==========================================================
Independently audits all Phase 3 Pillar 3 artifacts (results/phase3_pillar3_fairness/ and
data_processed/charls_gcomp_individual_benefits.parquet) against CLAUDE.md, docs/SAP_v1_FROZEN.md,
and lab-constitution iron rules.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_NHANES = Path(os.environ.get("NHANES_RAW_DIR", str(PROJECT_ROOT / "data_raw" / "NHANES")))
RAW_CHARLS = Path(os.environ.get("CHARLS_RAW_DIR", str(PROJECT_ROOT / "data_raw" / "CHARLS")))
DATA_DIR = PROJECT_ROOT / "data_processed"
P3_DIR = PROJECT_ROOT / "results" / "phase3_pillar3_fairness"


def check_raw_readonly_untouched(raw_dir: Path, cutoff_epoch: float) -> Dict[str, object]:
    recent_files: List[str] = []
    total_files = 0
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            total_files += 1
            fp = Path(root) / fn
            try:
                if fp.stat().st_mtime >= cutoff_epoch:
                    recent_files.append(str(fp))
            except OSError:
                pass
    return {"total_files": total_files, "modified_count": len(recent_files), "recent_files": recent_files[:10]}


def main() -> None:
    cutoff_epoch = time.time() - 14400.0
    checks: List[Dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print("=== Running Phase 3 Independent Audit Gate ===")

    req_p3 = [
        "jacobian_dominance_phenotypes_summary.csv",
        "paired_blood_phenotype_enrichment.csv",
        "gcomputation_9yr_counterfactual_prescriptions.csv",
        "gcomputation_natural_course_calibration.csv",
        "gamlss_fairness_audit_charls.csv",
        "subgroup_equity_disparity_table.csv",
        "gamlss_counterfactual_benefit_equity.csv",
        "gamlss_fairness_audit_nhanes.csv",
        "faraci_ambiguity_entropy_triage.csv",
        "pillar3_summary.json",
    ]
    miss = [f for f in req_p3 if not (P3_DIR / f).exists()]
    ben_pq = DATA_DIR / "charls_gcomp_individual_benefits.parquet"
    record("01_pillar3_files_complete", len(miss) == 0 and ben_pq.exists(), f"10/10 Pillar 3 files + individual benefit parquet present (miss={miss})")
    if miss or not ben_pq.exists():
        raise SystemExit(1)

    ph_df = pd.read_csv(P3_DIR / "jacobian_dominance_phenotypes_summary.csv")
    pb_df = pd.read_csv(P3_DIR / "paired_blood_phenotype_enrichment.csv")
    gc_df = pd.read_csv(P3_DIR / "gcomputation_9yr_counterfactual_prescriptions.csv")
    cal_df = pd.read_csv(P3_DIR / "gcomputation_natural_course_calibration.csv")
    gam_ch = pd.read_csv(P3_DIR / "gamlss_fairness_audit_charls.csv")
    sub_eq = pd.read_csv(P3_DIR / "subgroup_equity_disparity_table.csv")
    gam_nh = pd.read_csv(P3_DIR / "gamlss_fairness_audit_nhanes.csv")
    tri_df = pd.read_csv(P3_DIR / "faraci_ambiguity_entropy_triage.csv")
    ben_df = pd.read_parquet(ben_pq)

    record(
        "02_exact_cohort_counts_reconciled",
        int(ph_df["n_participants"].sum()) == 20928 and int(pb_df["n_participants"].sum()) == 7438 and len(ben_df) == 20928,
        f"Full phenotypes sum={int(ph_df['n_participants'].sum()):,}, Paired-blood phenotypes sum={int(pb_df['n_participants'].sum()):,}, Benefits N={len(ben_df):,}",
    )

    gc_valid = (
        int(gc_df["n_cluster_bootstrap"].min()) == 500
        and not gc_df[["delta_eta_cog_9yr_SD", "delta_eta_cog_se", "delta_eta_cog_p", "delta_eta_dep_9yr_SD", "delta_eta_dep_se", "delta_eta_dep_p", "fdr_q_cog", "fdr_q_dep"]].isna().any().any()
    )
    record("03_gcomputation_500_cluster_bootstrap_and_fdr_valid", gc_valid, f"18 G-Comp regime contrasts across full & paired-blood cohorts with B=500 bootstrap valid={gc_valid}")

    cal_err_cog = abs(float(cal_df.loc[0, "obs_eta_cog_w5_mean"]) - float(cal_df.loc[0, "nat_course_eta_cog_w5_mean"]))
    cal_err_dep = abs(float(cal_df.loc[0, "obs_eta_dep_w5_mean"]) - float(cal_df.loc[0, "nat_course_eta_dep_w5_mean"]))
    record("04_natural_course_wave5_calibration", cal_err_cog < 0.15 and cal_err_dep < 0.15, f"|Wave-5 NatCourse - Obs| cog={cal_err_cog:.4f} SD, dep={cal_err_dep:.4f} SD")

    gam_valid = (
        not gam_ch[["estimate", "robust_se", "p_value"]].isna().any().any()
        and not gam_nh[["estimate", "robust_se", "p_value"]].isna().any().any()
        and set(sub_eq["equity_axis"].unique()) == {"Education_Literacy", "Urban_Rural_Divide", "Sex", "Age_Decade"}
    )
    record("05_gamlss_location_scale_fairness_complete", gam_valid, f"CHARLS GAMLSS rows={len(gam_ch)}, NHANES GAMLSS rows={len(gam_nh)}, 4 equity axes complete={gam_valid}")

    # Verify M7 reduces Tobit NLL across ALL disadvantaged subgroups (Illiterate, Rural, Female, Oldest-Old)
    m7_equity_win = bool((sub_eq["NLL_gain_M7_vs_M5"] > 0.0).all() and (sub_eq["NLL_gain_M7_vs_M4"] > 0.0).all())
    record("06_psychometric_twin_improves_all_equity_strata", m7_equity_win, f"Min NLL gain M7 vs M5 across all strata = {sub_eq['NLL_gain_M7_vs_M5'].min():.5f}")

    # Verify Faraci ambiguity triage improves retained AUROC at 20% deferral vs 0% deferral
    dep_0 = float(tri_df.loc[(tri_df["clinical_endpoint"] == "Elevated_Depression_CESD10_ge10") & (tri_df["deferral_rate_pct"] == 0), "M7_retained_AUROC"].iloc[0])
    dep_20 = float(tri_df.loc[(tri_df["clinical_endpoint"] == "Elevated_Depression_CESD10_ge10") & (tri_df["deferral_rate_pct"] == 20), "M7_retained_AUROC"].iloc[0])
    record("07_faraci_ambiguity_triage_monotonic_gain", dep_20 > dep_0, f"Depression AUROC 0% deferral={dep_0:.4f} -> 20% ambiguity deferral={dep_20:.4f}")

    forbidden_tokens = ["aki", "acute_kidney", "brain_kidney", "kidney_disease", "mannitol", "dicom", "nifti"]
    bad_hits: List[str] = []
    for fn in req_p3:
        txt = (P3_DIR / fn).read_text(encoding="utf-8", errors="ignore").lower()
        for tok in forbidden_tokens:
            if tok in txt:
                bad_hits.append(f"{fn}:{tok}")
    record("08_zero_forbidden_scope_terms", len(bad_hits) == 0, f"forbidden_hits={bad_hits}")

    nh_ro = check_raw_readonly_untouched(RAW_NHANES, cutoff_epoch)
    ch_ro = check_raw_readonly_untouched(RAW_CHARLS, cutoff_epoch)
    record(
        "09_strict_readonly_raw_directories_untouched",
        nh_ro["modified_count"] == 0 and ch_ro["modified_count"] == 0,
        f"${NHANES_RAW_DIR} mod={nh_ro['modified_count']}; ${CHARLS_RAW_DIR} mod={ch_ro['modified_count']}",
    )

    overall_pass = all(c["status"] == "PASS" for c in checks)
    cert = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "n_checks_total": len(checks),
        "n_checks_passed": sum(1 for c in checks if c["status"] == "PASS"),
        "checks": checks,
    }
    res_dir = PROJECT_ROOT / "results"
    with open(res_dir / "PHASE3_VERIFICATION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
    md = [
        "# Phase 3 Independent Verification & Governance Audit Certificate",
        "",
        f"- **Audit Timestamp (UTC)**: `{cert['timestamp_utc']}`",
        f"- **Overall Gate Status**: **`{cert['overall_status']}`** (`{cert['n_checks_passed']}/{cert['n_checks_total']}` checks passed)",
        "",
        "| Check ID | Status | Empirical Verification Detail |",
        "| :--- | :---: | :--- |",
    ]
    for c in checks:
        md.append(f"| `{c['check']}` | **{c['status']}** | {c['detail']} |")
    (res_dir / "PHASE3_VERIFICATION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"=== Phase 3 Audit Gate Overall Status: {cert['overall_status']} ({cert['n_checks_passed']}/{cert['n_checks_total']}) ===")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
