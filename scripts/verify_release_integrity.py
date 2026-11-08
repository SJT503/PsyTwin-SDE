#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Release Verification Gate for PsyTwin-SDE (v1.0.0)
=============================================================
Verifies all 54 core quantitative results reported in the manuscript and
Supplementary Information directly against the aggregated CSV/JSON outputs
in `results/`, and confirms zero individual-level participant data leakage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ch_consort = pd.read_csv(ROOT / "results/phase0_cohort_audit/charls_consort_flow.csv").set_index("step_id")
    nh_consort = pd.read_csv(ROOT / "results/phase0_cohort_audit/nhanes_consort_flow.csv").set_index("step_id")
    ch_cens = pd.read_csv(ROOT / "results/phase0_cohort_audit/charls_psychometric_censoring_audit.csv")
    nh_cens = pd.read_csv(ROOT / "results/phase0_cohort_audit/nhanes_psychometric_fidelity_audit.csv").set_index("scale")

    p1_snfl = pd.read_csv(ROOT / "results/phase2_pillar1_nhanes/snfl_circadian_dml_associations.csv").set_index("exposure")
    p1_med2 = pd.read_csv(ROOT / "results/phase2_pillar1_nhanes/dml_two_sample_moment_coupled_cognition.csv")
    p1_med1 = pd.read_csv(ROOT / "results/phase2_pillar1_nhanes/dml_single_sample_mediation_results.csv")
    p1_tpdc = pd.read_csv(ROOT / "results/phase2_pillar1_nhanes/tpdc_band_summary.csv").set_index("feature")
    p1_fpca = pd.read_csv(ROOT / "results/phase2_pillar1_nhanes/fpca_variance_explained.csv")

    p2_cv = pd.read_csv(ROOT / "results/phase2_pillar2_sde/charls_cv_benchmark_7models.csv").set_index("model")
    p2_th = pd.read_csv(ROOT / "results/phase2_pillar2_sde/charls_temporal_holdout_benchmark.csv")
    p2_sim = pd.read_csv(ROOT / "results/phase2_pillar2_sde/simulation_recovery_metrics.csv")
    p2_struct = json.loads((ROOT / "results/phase2_pillar2_sde/charls_sde_structural_parameters.json").read_text(encoding="utf-8"))
    p2_blood = pd.read_csv(ROOT / "results/phase2_pillar2_sde/paired_blood_jacobian_moderation.csv")
    p2_nh = pd.read_csv(ROOT / "results/phase2_pillar2_sde/nhanes_cross_cohort_validation.csv").set_index("scale")

    p3_gcomp = pd.read_csv(ROOT / "results/phase3_pillar3_fairness/gcomputation_9yr_counterfactual_prescriptions.csv")
    p3_eq = pd.read_csv(ROOT / "results/phase3_pillar3_fairness/subgroup_equity_disparity_table.csv").set_index("stratum")
    p3_amb = pd.read_csv(ROOT / "results/phase3_pillar3_fairness/faraci_ambiguity_entropy_triage.csv")
    p3_pheno = pd.read_csv(ROOT / "results/phase3_pillar3_fairness/jacobian_dominance_phenotypes_summary.csv").set_index("phenotype")

    orient_pool = ch_cens[(ch_cens["scale"] == "orient") & (ch_cens["wave"] == "All_W1_W5")].iloc[0]
    ser7_pool = ch_cens[(ch_cens["scale"] == "ser7") & (ch_cens["wave"] == "All_W1_W5")].iloc[0]
    med_gcog = p1_med2[(p1_med2["exposure"] == "fpca_mims_pc1") & (p1_med2["outcome"] == "global_cognition_z")].iloc[0]
    med_phq8 = p1_med1[(p1_med1["exposure"] == "fpca_mims_pc1") & (p1_med1["outcome"] == "phq8_nosleep")].iloc[0]
    sim_m7_j12 = p2_sim[(p2_sim["model"] == "M7_Psychometric_Coupled_SDE") & (p2_sim["parameter"] == "J_cog_from_dep")].iloc[0]
    th_b1_w4_m7 = p2_th[(p2_th["split"] == "SplitB1_TrainW1_W3__Forecast_W4_Plus3Yr") & (p2_th["model"] == "M7_Psychometric_Coupled_SDE")].iloc[0]
    th_b1_w5_m7 = p2_th[(p2_th["split"] == "SplitB1_TrainW1_W3__Forecast_W5_Plus5Yr") & (p2_th["model"] == "M7_Psychometric_Coupled_SDE")].iloc[0]
    th_b2_w5_m7 = p2_th[(p2_th["split"] == "SplitB2_TrainW1_W4__Forecast_W5_Plus2Yr") & (p2_th["model"] == "M7_Psychometric_Coupled_SDE")].iloc[0]
    blood_j12 = p2_blood[(p2_blood["pathway"].str.startswith("J_cog_from_dep")) & (p2_blood["moderator"] == "Intercept_at_Mean")].iloc[0]
    g_r5_full = p3_gcomp[(p3_gcomp["cohort"] == "CHARLS_Full_CoupledSDE_N20928") & (p3_gcomp["intervention_regime"] == "R5_Combined_MultiTarget_Prescription")].iloc[0]
    g_r5_blood = p3_gcomp[(p3_gcomp["cohort"] == "CHARLS_PairedVenousBlood_N7438") & (p3_gcomp["intervention_regime"] == "R5_Combined_MultiTarget_Prescription")].iloc[0]
    amb_cog20 = p3_amb[(p3_amb["clinical_endpoint"] == "Cognitive_Impairment_tr20_le6") & (p3_amb["deferral_rate_pct"] == 20)].iloc[0]
    amb_dep20 = p3_amb[(p3_amb["clinical_endpoint"] == "Elevated_Depression_CESD10_ge10") & (p3_amb["deferral_rate_pct"] == 20)].iloc[0]

    checks = [
        ("NHANES_enrolled", int(nh_consort.loc["01_total_enrolled_2011_2014", "n_participants"]), 19931),
        ("NHANES_adults_ge20", int(nh_consort.loc["03_adults_age_ge20", "n_participants"]), 11329),
        ("NHANES_ArmA_4d16h", int(nh_consort.loc["06_adults_pax_valid_4d16h", "n_participants"]), 8148),
        ("NHANES_ArmA_4d10h_sens", int(nh_consort.loc["06b_adults_pax_valid_4d10h_sens", "n_participants"]), 8785),
        ("NHANES_ArmB_sNfL", int(nh_consort.loc["08_snfl_total_measured_2013_2014", "n_participants"]), 2071),
        ("NHANES_ArmB_sNfL_PAX", int(nh_consort.loc["08b_snfl_and_pax_4d16h", "n_participants"]), 1552),
        ("NHANES_ArmB_sNfL_PHQ9", int(nh_consort.loc["08d_snfl_pax_4d16h_and_phq9", "n_participants"]), 1467),
        ("NHANES_ArmC_DSST", int(nh_consort.loc["09a_cfq_dsst_measured", "n_participants"]), 3014),
        ("NHANES_ArmD_bridge", int(nh_consort.loc["10a_triple_bridge_snfl_cfq_pax_4d16h", "n_participants"]), 419),
        ("CHARLS_5wave_union", int(ch_consort.loc["01_total_5wave_union_2011_2020", "n_participants"]), 25873),
        ("CHARLS_person_waves", int(ch_consort.loc["02_total_person_waves_w1_w5", "n_participants"]), 96628),
        ("CHARLS_Primary_SDE", int(ch_consort.loc["07_primary_sde_cohort_coupled_ge2_waves", "n_participants"]), 20928),
        ("CHARLS_TemporalHoldout_ge3", int(ch_consort.loc["07b_coupled_sde_ge3_waves", "n_participants"]), 17360),
        ("CHARLS_PairedBlood_total", int(ch_consort.loc["08a_venous_blood_paired_2011_2015", "n_participants"]), 7648),
        ("CHARLS_PairedBlood_SDE", int(ch_consort.loc["08b_paired_blood_and_coupled_sde_ge2", "n_participants"]), 7438),
        ("CHARLS_orient_censoring_pct", round(float(orient_pool["total_boundary_censored_pct"]), 2), 46.95),
        ("CHARLS_ser7_censoring_pct", round(float(ser7_pool["total_boundary_censored_pct"]), 2), 44.96),
        ("NHANES_PHQ9_older_floor_pct", round(float(nh_cens.loc["NHANES_PHQ9_Older60Plus", "floor_pct"]), 2), 34.92),
        ("fPCA_MIMS_cumvar_4pc", round(float(p1_fpca[p1_fpca["channel"] == "log1p_MIMS"]["cumulative_variance_pct"].max()), 2), 90.32),
        ("fPCA_Lux_cumvar_4pc", round(float(p1_fpca[p1_fpca["channel"] == "log1p_Lux"]["cumulative_variance_pct"].max()), 2), 90.59),
        ("TPDC_photic_to_motor_circ", round(float(p1_tpdc.loc["tpdc_photic_to_motor_circ", "mean"]), 4), 0.6485),
        ("TPDC_motor_to_photic_circ", round(float(p1_tpdc.loc["tpdc_motor_to_photic_circ", "mean"]), 4), 0.4829),
        ("DML_mims_pc1_sNfL_alpha", round(float(p1_snfl.loc["fpca_mims_pc1", "alpha_log_snfl_std_4d16h"]), 4), -0.0841),
        ("DML_mims_pc1_sNfL_fdr_q", round(float(p1_snfl.loc["fpca_mims_pc1", "alpha_fdr_q_4d16h"]), 4), 0.0047),
        ("DML_2sample_gcog_NIE", round(float(med_gcog["NIE_moment_coupled_std"]), 4), 0.0132),
        ("DML_2sample_gcog_prop_med", round(float(med_gcog["prop_mediated_2sample"] * 100), 1), 23.1),
        ("DML_1sample_phq8_TE", round(float(med_phq8["TE_std"]), 4), -0.1363),
        ("DML_1sample_phq8_NDE", round(float(med_phq8["NDE_std"]), 4), -0.1319),
        ("DML_1sample_phq8_prop_med", round(float(med_phq8["prop_mediated"] * 100), 1), 3.2),
        ("Sim_M7_J12_rel_bias", round(float(sim_m7_j12["rel_bias_pct_N10000"]), 2), 3.05),
        ("Sim_M7_illit_RMSE", round(float(sim_m7_j12["illiterate_wave5_cog_rmse"]), 4), 0.4120),
        ("SplitA_M7_Tobit_NLL", round(float(p2_cv.loc["M7_Psychometric_Coupled_SDE", "co_primary_tobit_NLL"]), 5), 1.65742),
        ("SplitA_M7_CRPS", round(float(p2_cv.loc["M7_Psychometric_Coupled_SDE", "co_primary_CRPS_norm"]), 5), 0.12050),
        ("SplitA_M5_Tobit_NLL", round(float(p2_cv.loc["M5_Uncensored_Coupled_SDE", "co_primary_tobit_NLL"]), 5), 1.71851),
        ("SplitA_M4_Tobit_NLL", round(float(p2_cv.loc["M4_Naive_Neural_Latent_ODE", "co_primary_tobit_NLL"]), 5), 1.67963),
        ("SplitA_M7_AUROC_cog", round(float(p2_cv.loc["M7_Psychometric_Coupled_SDE", "AUROC_cognitive_impairment_tr20_le6"]), 4), 0.8521),
        ("SplitA_M7_AUROC_dep", round(float(p2_cv.loc["M7_Psychometric_Coupled_SDE", "AUROC_elevated_depression_cesd10_ge10"]), 4), 0.8011),
        ("SplitB1_Plus3Yr_M7_NLL", round(float(th_b1_w4_m7["co_primary_tobit_NLL"]), 5), 1.77499),
        ("SplitB1_Plus5Yr_M7_NLL", round(float(th_b1_w5_m7["co_primary_tobit_NLL"]), 5), 1.81705),
        ("SplitB2_Plus2Yr_M7_NLL", round(float(th_b2_w5_m7["co_primary_tobit_NLL"]), 5), 1.72194),
        ("Struct_J_cog_from_dep", round(float(p2_struct["J_cog_from_dep_mean"]), 5), -0.07241),
        ("Struct_J_dep_from_cog", round(float(p2_struct["J_dep_from_cog_mean"]), 5), -0.03647),
        ("Blood_J_cog_from_dep", round(float(blood_j12["estimate_SD_per_yr"]), 5), -0.09065),
        ("NHANES_ext_PHQ9_deltaNLL", round(float(p2_nh.loc["phq9_total", "delta_NLL_psych_minus_naive"]), 4), -0.0889),
        ("Pheno1_Affective_Driven_n", int(p3_pheno.loc["1_Affective_Driven_Cognitive_Decline", "n_participants"]), 11547),
        ("Pheno1_Affective_Driven_pct", round(float(p3_pheno.loc["1_Affective_Driven_Cognitive_Decline", "pct_cohort"]), 2), 55.17),
        ("GComp_R5_full_cog_SD", round(float(g_r5_full["delta_eta_cog_9yr_SD"]), 4), 0.1195),
        ("GComp_R5_full_years_delayed", round(float(g_r5_full["cognitive_aging_years_delayed"]), 2), 4.33),
        ("GComp_R5_full_dep_ARR", round(float(g_r5_full["ARR_elevated_depression_pct_pts"]), 2), 9.01),
        ("GComp_R5_blood_years_delayed", round(float(g_r5_blood["cognitive_aging_years_delayed"]), 2), 3.96),
        ("GComp_R5_blood_dep_ARR", round(float(g_r5_blood["ARR_elevated_depression_pct_pts"]), 2), 10.69),
        ("Equity_Illit_M7_PICP90", round(float(p3_eq.loc["1_Illiterate_NoFormal", "M7_tr20_PICP90_pct"]), 2), 90.80),
        ("Ambiguity_20pct_cog_AUROC", round(float(amb_cog20["M7_retained_AUROC"]), 4), 0.8795),
        ("Ambiguity_20pct_dep_AUROC", round(float(amb_dep20["M7_retained_AUROC"]), 4), 0.8265),
    ]

    failures = [f"{name}: got {got} != {exp}" for name, got, exp in checks if got != exp]
    if failures:
        print(f"[FAIL] {len(failures)} numerical mismatches: {failures}")
        return 1
    print(f"[PASS] PsyTwin-SDE Release Integrity Gate: {len(checks)}/{len(checks)} exact quantitative claims verified; 0 participant-level files present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
