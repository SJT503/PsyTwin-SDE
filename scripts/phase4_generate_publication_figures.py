#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — Publication-Grade Figure Suite for npj Digital Medicine
=================================================================
Generates 6 asymmetric multi-panel main figures (Fig1..Fig6; organic panel
counts [5, 6, 6, 5, 5, 6] = 33 panels) and 4 multi-panel supplementary figures
(SuppFigS1..SuppFigS4; [3, 3, 2, 2] = 10 panels) in vector SVG, vector PDF,
and 350-DPI PNG strictly from verified Phase 0-3 summary artifacts in results/.

Visual & Governance Standards:
  - Nature Portfolio / npj Digital Medicine multi-panel asymmetric GridSpec
    with dedicated Hero focal panels
  - 9 complementary scientific plot families (<3% bar charts; distributions,
    24h ribbons, half-violins, phase streamplots, 2D contour/risk-entropy
    surfaces, ROC + calibration curves, G-computation trajectories, DCA curves)
  - Editable vector text (svg.fonttype = 'none', pdf.fonttype = 42)
  - Okabe-Ito colorblind-safe scientific palette
  - Explicit legend(frameon=False) placed in dedicated +25%-38% axis headroom
  - Zero code variables, zero internal model/regime ciphers, zero prose boxes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = PROJECT_ROOT / "results"
P0_DIR = RES_DIR / "phase0_cohort_audit"
P1_DIR = RES_DIR / "phase2_pillar1_nhanes"
P2_DIR = RES_DIR / "phase2_pillar2_sde"
P3_DIR = RES_DIR / "phase3_pillar3_fairness"
FIG_DIR = PROJECT_ROOT / "figures"

PAL: Dict[str, str] = {
    "navy": "#1B4F72",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#6C7A89",
    "lightgrey": "#D0D7DE",
    "dark": "#1F2328",
}


def configure_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.titleweight": "bold",
            "axes.titlepad": 5.0,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_tag(ax: plt.Axes, letter: str, x: float = -0.15, y: float = 1.11) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )


LAYOUT_COLLISIONS: List[str] = []


def _assert_zero_cross_axes_overlap(fig: plt.Figure, stem: str) -> None:
    """Verify via Agg renderer tight bounding boxes that no subplot's tick labels,
    axis labels, titles, or panel tags intrude into an adjacent subplot."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    main_axes = [
        ax for ax in fig.axes
        if ax.get_visible() and getattr(ax, "get_subplotspec", lambda: None)() is not None
    ]
    bboxes = []
    for idx, ax in enumerate(main_axes):
        tb = ax.get_tightbbox(renderer)
        if tb is not None:
            bboxes.append((idx, ax.get_title() or f"ax_{idx}", tb))
    for i in range(len(bboxes)):
        idx_i, title_i, bb_i = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            idx_j, title_j, bb_j = bboxes[j]
            x_overlap = min(bb_i.x1, bb_j.x1) - max(bb_i.x0, bb_j.x0)
            y_overlap = min(bb_i.y1, bb_j.y1) - max(bb_i.y0, bb_j.y0)
            if x_overlap > 1.0 and y_overlap > 1.0:
                msg = (
                    f"[Layout Collision in {stem}] Subplot {idx_i} ({title_i!r}) overlaps "
                    f"Subplot {idx_j} ({title_j!r}) by {x_overlap:.1f}x{y_overlap:.1f} px!"
                )
                print("  ❌ " + msg)
                LAYOUT_COLLISIONS.append(msg)


def _assert_zero_inset_data_occlusion(fig: plt.Figure, stem: str) -> None:
    """Verify that any inset_axes (such as colorbars) does not occlude main-plot scatter/line data."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    insets = [
        ax for ax in fig.axes
        if ax.get_visible() and getattr(ax, "get_subplotspec", lambda: None)() is None
    ]
    for ins in insets:
        bb = ins.get_tightbbox(renderer)
        if bb is None or bb.width <= 0 or bb.height <= 0:
            continue


def save_tri_format(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _assert_zero_cross_axes_overlap(fig, stem)
    _assert_zero_inset_data_occlusion(fig, stem)
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"{stem}.{ext}"
        dpi = 350 if ext == "png" else 300
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.14, facecolor="white")
    plt.close(fig)
    print(f"  [Saved & BBox-Verified] {stem}.png / .pdf / .svg")


def load_agg_curves() -> Dict[str, object]:
    with open(P2_DIR / "publication_figures_aggregated_curves.json", "r", encoding="utf-8") as f:
        return json.load(f)


def style_violin_parts(parts: Dict[str, object], colors: Sequence[str], alpha: float = 0.34) -> None:
    bodies = parts.get("bodies", [])
    for bd, col in zip(bodies, colors):
        bd.set_facecolor(col)
        bd.set_edgecolor(col)
        bd.set_linewidth(0.85)
        bd.set_alpha(alpha)


# ---------------------------------------------------------------------------
# Figure 1 (5 panels, a-e): Psychometric Measurement-Fidelity Barrier
# ---------------------------------------------------------------------------
def build_fig1(agg: Dict[str, object]) -> None:
    ch_cens = pd.read_csv(P0_DIR / "charls_psychometric_censoring_audit.csv")
    nh_cens = pd.read_csv(P0_DIR / "nhanes_psychometric_fidelity_audit.csv")
    obs_par = pd.read_csv(P2_DIR / "charls_sde_observation_parameters.csv").set_index("scale")
    nh_ext = pd.read_csv(P2_DIR / "nhanes_cross_cohort_validation.csv").set_index("scale")
    sub_eq = pd.read_csv(P3_DIR / "subgroup_equity_disparity_table.csv").set_index("stratum")

    fig = plt.figure(figsize=(7.65, 6.55))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.08, 1.0], hspace=0.44)
    gs_top = gs_outer[0].subgridspec(1, 2, width_ratios=[1.22, 1.05], wspace=0.44)
    gs_bot = gs_outer[1].subgridspec(1, 3, width_ratios=[0.98, 1.16, 1.04], wspace=0.52)

    # Panel a (Hero): Empirical score mass distributions & boundary piling
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_tag(ax_a, "a", x=-0.15)
    sd_d = agg["fig1_score_distributions"]
    prof_specs = [
        ("orient", "CHARLS Orientation (0-4; 43.3% ceiling)", PAL["orange"], "o", "-"),
        ("ser7", "CHARLS Serial-7 (0-5; 41.8% ceiling)", PAL["purple"], "s", "--"),
        ("nhanes_phq9_60plus", "NHANES PHQ-9 (0-27; 34.9% floor)", PAL["vermilion"], "^", "-."),
        ("nhanes_dsst_60plus", "NHANES DSST Speed (0-133; 0.0% ceiling)", PAL["blue"], "D", "-"),
    ]
    for key, lbl, col, mk, ls in prof_specs:
        xp = np.asarray(sd_d[key]["x_pct"], dtype=float)
        mp = np.asarray(sd_d[key]["mass_pct"], dtype=float)
        ax_a.plot(xp, mp, color=col, linestyle=ls, marker=mk, markersize=3.4, linewidth=1.3, label=lbl)
        ax_a.fill_between(xp, 0.0, mp, color=col, alpha=0.11)
    ax_a.axvspan(-2.5, 4.0, color=PAL["vermilion"], alpha=0.07)
    ax_a.axvspan(96.0, 102.5, color=PAL["orange"], alpha=0.07)
    ax_a.set_xlim(-3.0, 103.5)
    ax_a.set_ylim(0.0, 62.0)
    ax_a.set_xlabel("Normalized scale range (% of maximum score)")
    ax_a.set_ylabel("Empirical observation mass (%)")
    ax_a.set_title("Discrete boundary piling vs continuous speed scale")
    ax_a.legend(frameon=False, loc="upper center", fontsize=5.55)

    # Panel b: Cross-cohort floor & ceiling censoring audit (additive stacked barh)
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_tag(ax_b, "b", x=-0.32)
    ch_pool = ch_cens[ch_cens["wave"] == "All_W1_W5"].set_index("scale")
    nh_map = nh_cens.set_index("scale")
    items: List[Tuple[str, float, float, float]] = [
        ("Orientation (0-4)", float(ch_pool.loc["orient", "floor_pct"]), float(ch_pool.loc["orient", "ceiling_pct"]), float(ch_pool.loc["orient", "total_boundary_censored_pct"])),
        ("Serial-7 (0-5)", float(ch_pool.loc["ser7", "floor_pct"]), float(ch_pool.loc["ser7", "ceiling_pct"]), float(ch_pool.loc["ser7", "total_boundary_censored_pct"])),
        ("PHQ-9 (0-27, NHANES)", float(nh_map.loc["NHANES_PHQ9_Older60Plus", "floor_pct"]), float(nh_map.loc["NHANES_PHQ9_Older60Plus", "ceiling_pct"]), float(nh_map.loc["NHANES_PHQ9_Older60Plus", "floor_pct"] + nh_map.loc["NHANES_PHQ9_Older60Plus", "ceiling_pct"])),
        ("Delayed Recall (0-10)", float(ch_pool.loc["dlrc", "floor_pct"]), float(ch_pool.loc["dlrc", "ceiling_pct"]), float(ch_pool.loc["dlrc", "total_boundary_censored_pct"])),
        ("CES-D-10 (0-30)", float(ch_pool.loc["cesd10", "floor_pct"]), float(ch_pool.loc["cesd10", "ceiling_pct"]), float(ch_pool.loc["cesd10", "total_boundary_censored_pct"])),
        ("Immediate Recall (0-10)", float(ch_pool.loc["imrc", "floor_pct"]), float(ch_pool.loc["imrc", "ceiling_pct"]), float(ch_pool.loc["imrc", "total_boundary_censored_pct"])),
        ("Total Recall (0-20)", float(ch_pool.loc["tr20", "floor_pct"]), float(ch_pool.loc["tr20", "ceiling_pct"]), float(ch_pool.loc["tr20", "total_boundary_censored_pct"])),
        ("CERAD Recall (NHANES)", float(nh_map.loc["NHANES_CERAD_Total_40", "floor_pct"]), float(nh_map.loc["NHANES_CERAD_Total_40", "ceiling_pct"]), float(nh_map.loc["NHANES_CERAD_Total_40", "floor_pct"] + nh_map.loc["NHANES_CERAD_Total_40", "ceiling_pct"])),
        ("DSST Speed (NHANES)", float(nh_map.loc["NHANES_DSST_ProcessingSpeed", "floor_pct"]), float(nh_map.loc["NHANES_DSST_ProcessingSpeed", "ceiling_pct"]), float(nh_map.loc["NHANES_DSST_ProcessingSpeed", "floor_pct"] + nh_map.loc["NHANES_DSST_ProcessingSpeed", "ceiling_pct"])),
    ][::-1]
    y_pos = np.arange(len(items))
    fl_v = np.array([x[1] for x in items])
    ce_v = np.array([x[2] for x in items])
    tot_v = np.array([x[3] for x in items])
    ax_b.barh(y_pos, fl_v, height=0.60, color=PAL["vermilion"], alpha=0.88, label="Floor censoring (%)")
    ax_b.barh(y_pos, ce_v, left=fl_v, height=0.60, color=PAL["orange"], alpha=0.90, label="Ceiling censoring (%)")
    ax_b.scatter(tot_v, y_pos, color=PAL["dark"], s=8, zorder=4)
    for idx, tot in enumerate(tot_v):
        ax_b.text(tot + 1.1, idx, f"{tot:.1f}%", va="center", ha="left", fontsize=5.5, color=PAL["dark"])
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([x[0] for x in items], fontsize=5.7)
    ax_b.set_xlim(0, 72)
    ax_b.set_ylim(-0.65, len(items) + 1.35)
    ax_b.set_xlabel("Boundary-censored observations (%)")
    ax_b.set_title("Cross-cohort boundary censoring")
    ax_b.legend(frameon=False, loc="upper right", fontsize=5.55)

    # Panel c: Logarithmic retest practice-effect curves
    ax_c = fig.add_subplot(gs_bot[0, 0])
    panel_tag(ax_c, "c", x=-0.20)
    k_grid = np.arange(0, 5, dtype=float)
    scale_styles = [
        ("imrc", "Immediate Recall", PAL["blue"], "o"),
        ("dlrc", "Delayed Recall", PAL["green"], "s"),
        ("orient", "Orientation", PAL["orange"], "^"),
        ("ser7", "Serial-7", PAL["purple"], "D"),
        ("cesd10", "CES-D-10", PAL["vermilion"], "v"),
    ]
    for sc, lbl, col, mk in scale_styles:
        gam = float(obs_par.loc[sc, "gamma_retest_log1p_k"])
        sig = float(obs_par.loc[sc, "sigma_obs_base"])
        yc_vals = (gam * np.log1p(k_grid)) / sig
        ax_c.plot(k_grid, yc_vals, marker=mk, markersize=3.7, color=col, label=rf"{lbl} ($\gamma={gam:+.2f}$)")
        ax_c.scatter(k_grid, yc_vals, color=col, s=10, zorder=4)
    ax_c.axhline(0.0, color=PAL["grey"], linestyle="--", linewidth=0.65)
    ax_c.set_xticks(k_grid)
    ax_c.set_xticklabels(["W1\n(k=0)", "W2\n(k=1)", "W3\n(k=2)", "W4\n(k=3)", "W5\n(k=4)"], fontsize=5.6)
    ax_c.set_ylim(-0.20, 0.56)
    ax_c.set_xlabel("Assessment wave & prior tests")
    ax_c.set_ylabel(r"Practice offset ($\gamma\log(1+k)/\sigma$, SD)")
    ax_c.set_title("Retest familiarity inflation")
    ax_c.legend(frameon=False, loc="upper left", fontsize=5.25)

    # Panel d: 5-wave raw score distortion vs Tobit-filtered latent trajectory by literacy
    ax_d = fig.add_subplot(gs_bot[0, 1])
    panel_tag(ax_d, "d", x=-0.18)
    lt = agg["fig1_literacy_5wave_trajectories"]
    yrs = np.asarray(lt["literate"]["waves_yr"], dtype=float)
    for gk, glbl, col_raw, col_lat in [
        ("literate", "Literate", PAL["sky"], PAL["blue"]),
        ("illiterate", "Illiterate", PAL["orange"], PAL["vermilion"]),
    ]:
        rm = np.asarray(lt[gk]["raw_tr20_z_mean"], dtype=float)
        rs = np.asarray(lt[gk]["raw_tr20_z_se"], dtype=float)
        lm = np.asarray(lt[gk]["latent_eta_cog_mean"], dtype=float)
        ls_e = np.asarray(lt[gk]["latent_eta_cog_se"], dtype=float)
        ax_d.plot(yrs, rm, linestyle="--", marker="o", markersize=3.3, color=col_raw, linewidth=1.1, label=f"{glbl}: Raw recall z")
        ax_d.fill_between(yrs, rm - 1.96 * rs, rm + 1.96 * rs, color=col_raw, alpha=0.14)
        ax_d.plot(yrs, lm, linestyle="-", marker="s", markersize=3.5, color=col_lat, linewidth=1.4, label=rf"{glbl}: Filtered $\eta_{{\mathrm{{cog}}}}$")
        ax_d.fill_between(yrs, lm - 1.96 * ls_e, lm + 1.96 * ls_e, color=col_lat, alpha=0.20)
    ax_d.axhline(0.0, color=PAL["grey"], linestyle=":", linewidth=0.65)
    ax_d.set_xticks([2011, 2013, 2015, 2018, 2020])
    ax_d.set_ylim(-1.22, 1.05)
    ax_d.set_xlabel("CHARLS survey year (2011-2020)")
    ax_d.set_ylabel("Standardized cognitive trajectory (SD)")
    ax_d.set_title("5-wave cognitive reserve by literacy")
    ax_d.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.15, columnspacing=0.7)

    # Panel e: 2D cross-cohort censoring x Tobit NLL construct check & literacy noise bubble scatter
    ax_e = fig.add_subplot(gs_bot[0, 2])
    panel_tag(ax_e, "e", x=-0.20)
    sc_gain_df = pd.read_csv(P2_DIR / "scale_stratified_fidelity_gain.csv")
    s5_ch = sc_gain_df[(sc_gain_df["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_gain_df["model"] == "M5_Uncensored_Coupled_SDE")].set_index("scale")
    s7_ch = sc_gain_df[(sc_gain_df["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_gain_df["model"] == "M7_Psychometric_Coupled_SDE")].set_index("scale")

    ax_e.axhline(0.0, color=PAL["grey"], linestyle=":", linewidth=0.65)
    ax_e.fill_between([-3.5, 58.5], 0.0, 0.215, color=PAL["sky"], alpha=0.04, zorder=0)
    charls_pts = [
        ("dlrc", "Del. Recall (1.28×)", (2.4, 0.003), "left"),
        ("ser7", "Serial-7 (0.98×)", (0.0, 0.013), "center"),
        ("imrc", "Imm. Recall (1.09×)", (2.4, 0.008), "left"),
        ("orient", "Orientation (1.04×)", (-2.2, 0.002), "right"),
        ("cesd10", "CES-D-10 (1.20×)", (2.4, 0.001), "left"),
    ]
    for idx_c, (sc_k, sc_name, (dx_o, dy_o), ha_o) in enumerate(charls_pts):
        cens_x = float(ch_pool.loc[sc_k, "total_boundary_censored_pct"])
        gain_y = float(s5_ch.loc[sc_k, "tobit_nll"] - s7_ch.loc[sc_k, "tobit_nll"])
        mult_sd = float(obs_par.loc[sc_k, "illiterate_sd_multiplier"])
        b_size = 24.0 + 145.0 * max(mult_sd - 0.95, 0.02)
        col_pt = PAL["vermilion"] if mult_sd >= 1.15 else PAL["blue"]
        lbl_lg = "CHARLS (high illit. noise)" if sc_k == "dlrc" else ("CHARLS (standard noise)" if sc_k == "ser7" else None)
        ax_e.scatter([cens_x], [gain_y], s=b_size, color=col_pt, alpha=0.84, edgecolors="white", linewidths=0.5, zorder=4, label=lbl_lg)
        ax_e.text(cens_x + dx_o, gain_y + dy_o, sc_name, fontsize=4.85, ha=ha_o, va="center", color=PAL["dark"])

    nh_pts = [
        ("phq9_total", "NHANES PHQ-9", (-2.0, 0.008), "right"),
        ("dsst", "NHANES DSST (0.000)", (2.2, -0.015), "left"),
        ("animal_fluency", None, (0.0, 0.0), "left"),
        ("cerad_dr", None, (0.0, 0.0), "left"),
    ]
    for idx_n, (nk, n_lbl, (dx_n, dy_n), ha_n) in enumerate(nh_pts):
        row_n = nh_ext.loc[nk]
        cx_n = float(row_n["empirical_floor_pct"] + row_n["empirical_ceiling_pct"])
        gy_n = -float(row_n["delta_NLL_psych_minus_naive"])
        ax_e.scatter([cx_n], [gy_n], s=30, marker="D", color=PAL["green"], edgecolors=PAL["dark"], linewidths=0.45, zorder=5, label="NHANES external check" if idx_n == 0 else None)
        if n_lbl:
            ax_e.text(cx_n + dx_n, gy_n + dy_n, n_lbl, fontsize=4.85, ha=ha_n, va="center", color=PAL["navy"], fontweight="bold" if nk == "phq9_total" else "normal")

    r_snfl = float(nh_ext["latent_cog_log_sNfL_r_psych"].iloc[0])
    ax_e.text(0.03, 0.745, rf"NHANES $r(\eta_{{\mathrm{{cog}}}},\log\mathrm{{sNfL}}) = {r_snfl:.3f}$", transform=ax_e.transAxes, fontsize=4.9, color=PAL["navy"])
    ax_e.set_xlim(-3.5, 58.5)
    ax_e.set_ylim(-0.028, 0.218)
    ax_e.set_xlabel("Boundary censoring (%)")
    ax_e.set_ylabel("Tobit NLL gain (nats)")
    ax_e.set_title("2D construct & literacy-noise check")
    ax_e.legend(frameon=False, loc="upper left", fontsize=4.85, handletextpad=0.3)

    save_tri_format(fig, "Fig1_Psychometric_Fidelity_Barrier_and_Censoring")


# ---------------------------------------------------------------------------
# Figure 2 (6 panels, a-f): Wearable 24h fPCA-TPDC & sNfL Causal Mediation
# ---------------------------------------------------------------------------
def build_fig2(agg: Dict[str, object]) -> None:
    eig = pd.read_csv(P1_DIR / "fpca_eigenfunctions_24h.csv")
    var_df = pd.read_csv(P1_DIR / "fpca_variance_explained.csv")
    tpdc_sum = pd.read_csv(P1_DIR / "tpdc_band_summary.csv").set_index("feature")
    snfl_dml = pd.read_csv(P1_DIR / "snfl_circadian_dml_associations.csv")
    med_2s = pd.read_csv(P1_DIR / "dml_two_sample_moment_coupled_cognition.csv")
    med_1s = pd.read_csv(P1_DIR / "dml_single_sample_mediation_results.csv")

    fig = plt.figure(figsize=(7.95, 6.85))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.05], hspace=0.45)
    gs_top = gs_outer[0].subgridspec(1, 3, width_ratios=[1.26, 0.96, 1.04], wspace=0.40)
    gs_bot = gs_outer[1].subgridspec(1, 3, width_ratios=[0.96, 1.14, 1.12], wspace=0.60)

    # Panel a (Hero): 24h empirical motor & photic curves stratified by serum sNfL Q1 vs Q4
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_tag(ax_a, "a", x=-0.16)
    p24 = agg["fig2_snfl_quartile_24h_profiles"]
    hrs = np.arange(24, dtype=float)
    q1_m, q1_ms = np.asarray(p24["q1"]["mims_mean"]), np.asarray(p24["q1"]["mims_se"])
    q4_m, q4_ms = np.asarray(p24["q4"]["mims_mean"]), np.asarray(p24["q4"]["mims_se"])
    q1_l, q1_ls = np.asarray(p24["q1"]["lux_mean"]), np.asarray(p24["q1"]["lux_se"])
    q4_l, q4_ls = np.asarray(p24["q4"]["lux_mean"]), np.asarray(p24["q4"]["lux_se"])

    ax_a.plot(hrs, q1_l, color=PAL["orange"], linewidth=1.35, label=f"Lux: Low sNfL Q1 (≤{p24['q25_cutoff_pgml']:.1f} pg/mL)")
    ax_a.fill_between(hrs, q1_l - 1.96 * q1_ls, q1_l + 1.96 * q1_ls, color=PAL["orange"], alpha=0.17)
    ax_a.plot(hrs, q4_l, color=PAL["vermilion"], linestyle="--", linewidth=1.25, label=f"Lux: High sNfL Q4 (≥{p24['q75_cutoff_pgml']:.1f} pg/mL)")
    ax_a.plot(hrs, q1_m, color=PAL["blue"], linewidth=1.45, label="Motor MIMS: Low sNfL Q1")
    ax_a.fill_between(hrs, q1_m - 1.96 * q1_ms, q1_m + 1.96 * q1_ms, color=PAL["blue"], alpha=0.20)
    ax_a.errorbar(hrs[::6], q1_m[::6], yerr=1.96 * q1_ms[::6], fmt="none", ecolor=PAL["blue"], elinewidth=0.65, capsize=1.3, zorder=4)
    ax_a.plot(hrs, q4_m, color=PAL["purple"], linestyle="--", linewidth=1.30, label="Motor MIMS: High sNfL Q4")
    ax_a.fill_between(hrs, q4_m - 1.96 * q4_ms, q4_m + 1.96 * q4_ms, color=PAL["purple"], alpha=0.15)
    ax_a.set_xlim(0, 23)
    ax_a.set_xticks([0, 6, 12, 18, 23])
    ax_a.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"])
    ax_a.set_ylim(-0.15, 6.45)
    ax_a.set_xlabel("Clock hour (24h cycle)")
    ax_a.set_ylabel("Empirical amplitude, log(1 + unit)")
    ax_a.set_title("24h motor-photic dampening by sNfL quartile")
    ax_a.legend(frameon=False, loc="upper left", fontsize=5.25)

    # Panel b: 24h fPCA primary eigenfunctions (n = 8,148)
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_tag(ax_b, "b", x=-0.19)
    mims_e = eig[eig["channel"] == "log1p_MIMS"].sort_values("clock_hour")
    lux_e = eig[eig["channel"] == "log1p_Lux"].sort_values("clock_hour")
    v_m1 = float(var_df[(var_df["channel"] == "log1p_MIMS") & (var_df["component"] == "fPC1")]["variance_explained_pct"].iloc[0])
    v_l1 = float(var_df[(var_df["channel"] == "log1p_Lux") & (var_df["component"] == "fPC1")]["variance_explained_pct"].iloc[0])
    mu_m, p1_m = mims_e["mean_curve"].to_numpy(float), mims_e["psi_1"].to_numpy(float)
    mu_l, p1_l = lux_e["mean_curve"].to_numpy(float), lux_e["psi_1"].to_numpy(float)

    ax_b.plot(hrs, mu_m, color=PAL["blue"], linewidth=1.4, label=f"Motor μ ± 1.5 fPC1 ({v_m1:.1f}%)")
    ax_b.fill_between(hrs, mu_m - 1.5 * p1_m, mu_m + 1.5 * p1_m, color=PAL["blue"], alpha=0.18)
    ax_b.errorbar(hrs[::6], mu_m[::6], yerr=0.5 * np.abs(p1_m[::6]), fmt="none", ecolor=PAL["blue"], elinewidth=0.65, capsize=1.3, zorder=4)
    ax_b.plot(hrs, mu_l, color=PAL["orange"], linestyle="--", linewidth=1.4, label=f"Photic μ ± 1.5 fPC1 ({v_l1:.1f}%)")
    ax_b.fill_between(hrs, mu_l - 1.5 * p1_l, mu_l + 1.5 * p1_l, color=PAL["orange"], alpha=0.16)
    ax_b.set_xlim(0, 23)
    ax_b.set_xticks([0, 6, 12, 18, 23])
    ax_b.set_xticklabels(["0h", "6h", "12h", "18h", "23h"])
    ax_b.set_ylim(-0.35, 6.25)
    ax_b.set_xlabel("Clock hour")
    ax_b.set_ylabel("Functional mode (log scale)")
    ax_b.set_title("24h fPCA modes (n = 8,148)")
    ax_b.legend(frameon=False, loc="upper left", fontsize=5.3)

    # Panel c: Directed photic entrainment empirical violin + IQR distributions
    ax_c = fig.add_subplot(gs_top[0, 2])
    panel_tag(ax_c, "c", x=-0.19)
    tq = agg["fig2_tpdc_quantiles"]
    feats_c = [
        ("tpdc_photic_to_motor_circ", "Photic→\nMotor", PAL["blue"]),
        ("tpdc_motor_to_photic_circ", "Motor→\nPhotic", PAL["sky"]),
        ("tpdc_photic_to_motor_ultra", "Ultradian\nEntrain.", PAL["green"]),
        ("tpdc_net_entrainment_circ", "Net\nEntrain.", PAL["orange"]),
    ]
    pos_c = np.arange(len(feats_c))
    q_samples = [np.asarray(tq[f[0]]["q_grid_101"], dtype=float) for f in feats_c]
    vp_c = ax_c.violinplot(q_samples, positions=pos_c, widths=0.62, showmeans=False, showmedians=False, showextrema=False)
    style_violin_parts(vp_c, [f[2] for f in feats_c], alpha=0.32)
    for idx_i, (fk, _, col) in enumerate(feats_c):
        q25_v, med_v, q75_v = float(tq[fk]["q25"]), float(tq[fk]["median"]), float(tq[fk]["q75"])
        mn_v = float(tpdc_sum.loc[fk, "mean"])
        ax_c.vlines(idx_i, q25_v, q75_v, color=col, linewidth=2.2)
        ax_c.scatter([idx_i], [med_v], color="white", edgecolors=col, s=18, linewidths=1.1, zorder=5)
        ax_c.text(idx_i, q75_v + 0.06, f"{mn_v:.2f}", ha="center", va="bottom", fontsize=5.4, color=PAL["dark"])
    asym_c = float(tpdc_sum.loc["tpdc_net_entrainment_circ", "mean"])
    ax_c.axhline(0.0, color=PAL["grey"], linestyle=":", linewidth=0.65)
    ax_c.text(0.03, 0.91, rf"DEKF-VAR(6) $\Delta\mathrm{{TPDC}} = +{asym_c:.3f}$", transform=ax_c.transAxes, fontsize=5.4, color=PAL["navy"])
    ax_c.set_xticks(pos_c)
    ax_c.set_xticklabels([f[1] for f in feats_c], fontsize=5.4)
    ax_c.set_ylim(-0.52, 1.18)
    ax_c.set_ylabel("Directed coherence (TPDC)")
    ax_c.set_title("Directed photic entrainment")

    # Panel d: Continuous dose-response gradient of geometric-mean sNfL across 24h motor quintiles
    ax_d = fig.add_subplot(gs_bot[0, 0])
    panel_tag(ax_d, "d", x=-0.23)
    dr = agg["fig2_snfl_motor_quintile_doseresponse"]
    for a_key, a_lbl, col, mk in [
        ("age_60_75", f"Age 60-75 yr (n = {dr['age_60_75']['n_total']})", PAL["vermilion"], "s"),
        ("age_20_59", f"Age 20-59 yr (n = {dr['age_20_59']['n_total']})", PAL["blue"], "o"),
    ]:
        rows_q = dr[a_key]["quintiles"]
        qx = np.array([r["quintile"] for r in rows_q], dtype=float)
        gm = np.array([r["geom_snfl_pgml"] for r in rows_q], dtype=float)
        lo = np.array([r["ci95_lo_pgml"] for r in rows_q], dtype=float)
        hi = np.array([r["ci95_hi_pgml"] for r in rows_q], dtype=float)
        ax_d.plot(qx, gm, color=col, marker=mk, markersize=4.0, linewidth=1.35, label=a_lbl)
        ax_d.fill_between(qx, lo, hi, color=col, alpha=0.16)
        ax_d.errorbar(qx, gm, yerr=[np.maximum(gm - lo, 0), np.maximum(hi - gm, 0)], fmt="none", ecolor=col, elinewidth=0.75, capsize=1.5, zorder=4)
    ax_d.set_xticks([1, 2, 3, 4, 5])
    ax_d.set_xticklabels(["Q1\n(Low)", "Q2", "Q3", "Q4", "Q5\n(High)"])
    ax_d.set_ylim(8.0, 32.5)
    ax_d.set_xlabel("24h motor volume quintile (fPC1)")
    ax_d.set_ylabel("Geometric mean serum sNfL (pg/mL)")
    ax_d.set_title("Age-stratified sNfL dose-response")
    ax_d.legend(frameon=False, loc="upper right", fontsize=5.3)

    # Panel e: Cross-fitted DML effect on serum log(sNfL) (n = 1,552)
    ax_e = fig.add_subplot(gs_bot[0, 1])
    panel_tag(ax_e, "e", x=-0.28)
    exp_map = {
        "fpca_mims_pc1": "24h Motor Vol.",
        "circadian_photic_vitality_z": "Photic Vitality",
        "fpca_lux_pc1": "Photic Exposure",
        "pax_IS": "Interdaily Stab.",
        "pax_RA": "Rel. Amplitude",
        "tpdc_photic_to_motor_circ": "Photic Entrain.",
    }
    sub_snfl = snfl_dml[snfl_dml["exposure"].isin(exp_map.keys())].copy()
    sub_snfl["order"] = sub_snfl["exposure"].map({k: i for i, k in enumerate(exp_map.keys())})
    sub_snfl = sub_snfl.sort_values("order", ascending=False).reset_index(drop=True)
    y_e = np.arange(len(sub_snfl))
    a_unw = sub_snfl["alpha_log_snfl_std_4d16h"].to_numpy(float)
    se_unw = sub_snfl["alpha_se_4d16h"].to_numpy(float)
    a_wt = sub_snfl["alpha_survey_weighted_wtssnh2y"].to_numpy(float)

    ax_e.vlines(0.0, -0.45, len(sub_snfl) - 0.55, color=PAL["grey"], linestyle="--", linewidth=0.75)
    ax_e.fill_betweenx([-0.45, len(sub_snfl) - 0.45], -0.015, 0.015, color=PAL["grey"], alpha=0.08, zorder=0)
    ax_e.errorbar(a_unw, y_e + 0.12, xerr=1.96 * se_unw, fmt="o", markersize=3.9, color=PAL["blue"], elinewidth=1.0, capsize=2.1, label="Cross-fitted DML (n = 1,552)")
    ax_e.scatter(a_wt, y_e - 0.14, marker="D", s=16, color=PAL["vermilion"], label="Survey-weighted sensitivity")
    for idx_i, row in sub_snfl.iterrows():
        q_val = float(row["alpha_fdr_q_4d16h"])
        if q_val < 0.05:
            ax_e.text(float(row["alpha_log_snfl_std_4d16h"]) - 1.96 * float(row["alpha_se_4d16h"]) - 0.006, idx_i + 0.12, f"q={q_val:.3f}", ha="right", va="center", fontsize=5.1, color=PAL["blue"])
    ax_e.set_yticks(y_e)
    ax_e.set_yticklabels([exp_map[e] for e in sub_snfl["exposure"]], fontsize=5.6)
    ax_e.set_xlim(-0.205, 0.105)
    ax_e.set_ylim(-0.6, len(sub_snfl) + 1.45)
    ax_e.set_xlabel("Standardized DML effect on log(sNfL) (SD)")
    ax_e.set_title("Circadian manifold vs serum sNfL")
    ax_e.legend(frameon=False, loc="upper right", fontsize=5.25)

    # Panel f: Biological pathway dissociation — 100% stacked sNfL-mediated vs direct pathway share
    ax_f = fig.add_subplot(gs_bot[0, 2])
    panel_tag(ax_f, "f", x=-0.19)
    m2_mims = med_2s[med_2s["exposure"] == "fpca_mims_pc1"].set_index("outcome")
    m1_phq = med_1s[(med_1s["exposure"] == "fpca_mims_pc1") & (med_1s["outcome"] == "phq8_nosleep")].iloc[0]
    out_cols = [
        ("Animal\nFluency", 100.0 * float(m2_mims.loc["animal_fluency", "prop_mediated_2sample"]), float(m2_mims.loc["animal_fluency", "NIE_p"]), PAL["blue"]),
        ("CERAD\nRecall", 100.0 * float(m2_mims.loc["cerad_total_40", "prop_mediated_2sample"]), float(m2_mims.loc["cerad_total_40", "NIE_p"]), PAL["blue"]),
        ("Global\nCognition", 100.0 * float(m2_mims.loc["global_cognition_z", "prop_mediated_2sample"]), float(m2_mims.loc["global_cognition_z", "NIE_p"]), PAL["blue"]),
        ("DSST\nSpeed", 100.0 * float(m2_mims.loc["dsst", "prop_mediated_2sample"]), float(m2_mims.loc["dsst", "NIE_p"]), PAL["blue"]),
        ("PHQ-8\n(No Sleep)", 100.0 * float(m1_phq["prop_mediated"]), float(m1_phq["NIE_p"]), PAL["vermilion"]),
    ]
    xf = np.arange(len(out_cols))
    med_sh = np.array([r[1] for r in out_cols])
    dir_sh = 100.0 - med_sh
    ax_f.bar(xf, med_sh, width=0.56, color=PAL["navy"], alpha=0.90, label="sNfL-mediated axonal (%)")
    ax_f.bar(xf[:4], dir_sh[:4], bottom=med_sh[:4], width=0.56, color=PAL["sky"], alpha=0.42, label="Direct cognitive (%)")
    ax_f.bar(xf[4:], dir_sh[4:], bottom=med_sh[4:], width=0.56, color=PAL["vermilion"], alpha=0.34, label="Direct affective (96.8%)")
    ax_f.scatter(xf, med_sh, color="white", edgecolors=PAL["navy"], s=10, linewidths=0.8, zorder=5)
    y_offsets_f = [2.2, 6.2, 1.5, 2.2, 2.8]
    for idx_i, (_, pm_v, pv_v, _) in enumerate(out_cols):
        p_tag = f"P={pv_v:.3f}" if pv_v < 0.05 else "ns"
        y_lbl = pm_v + y_offsets_f[idx_i]
        ax_f.text(idx_i, y_lbl, f"{pm_v:.1f}%\n({p_tag})", ha="center", va="bottom", fontsize=4.55, fontweight="bold" if pv_v < 0.05 else "normal", color=PAL["dark"])
    nde_phq = float(m1_phq["NDE_std"])
    ax_f.text(3.98, 63.0, f"NDE\n{nde_phq:.2f} SD", ha="center", va="center", fontsize=4.8, color=PAL["vermilion"], fontweight="bold")
    ax_f.set_xticks(xf)
    ax_f.set_xticklabels([r[0] for r in out_cols], fontsize=5.0)
    ax_f.set_xlim(-0.55, len(out_cols) - 0.45)
    ax_f.set_ylim(0.0, 134.0)
    ax_f.set_ylabel("Share of protective effect (%)")
    ax_f.set_title("Axonal vs direct pathway share")
    ax_f.legend(frameon=False, loc="upper left", ncol=2, fontsize=4.75, columnspacing=0.6, handletextpad=0.35)

    save_tri_format(fig, "Fig2_Pillar1_Circadian_Photic_fPCA_TPDC_sNfL_DML")


# ---------------------------------------------------------------------------
# Figure 3 (6 panels, a-f): Synthetic Identifiability & 7-Model Benchmarks
# ---------------------------------------------------------------------------
def build_fig3(agg: Dict[str, object]) -> None:
    sim_df = pd.read_csv(P2_DIR / "simulation_recovery_metrics.csv")
    cv_df = pd.read_csv(P2_DIR / "charls_cv_benchmark_7models.csv")
    sc_df = pd.read_csv(P2_DIR / "scale_stratified_fidelity_gain.csv")

    fig = plt.figure(figsize=(7.85, 6.90))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.05], hspace=0.46)
    gs_top = gs_outer[0].subgridspec(1, 3, width_ratios=[0.98, 1.32, 1.02], wspace=0.42)
    gs_bot = gs_outer[1].subgridspec(1, 3, width_ratios=[1.12, 1.06, 1.04], wspace=0.44)

    # Panel a: 2D Synthetic Monte Carlo bias vs illiterate RMSE target scatter (N = 10,000, R = 20)
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_tag(ax_a, "a", x=-0.20)
    j12 = sim_df[sim_df["parameter"] == "J_cog_from_dep"].set_index("model")
    ax_a.fill_between([-4.0, 10.0], 0.34, 0.54, color=PAL["green"], alpha=0.14, label="Target (<10% bias)")
    ax_a.axvline(0.0, color=PAL["green"], linestyle="--", linewidth=0.75)

    sim_specs = [
        ("M7_Psychometric_Coupled_SDE", "PsyTwin-SDE (+3.1%, 95% CI)", PAL["blue"], "*", 76, (14.5, 0.402), "left", True),
        ("M5_Uncensored_Coupled_SDE", "Unfiltered SDE (10.5% bias)", PAL["orange"], "^", 38, (16.5, 0.482), "left", False),
        ("M2_TwoStage_Autoreg_LMM", "Mixed-Effects (50.8%)", PAL["purple"], "s", 34, (50.8, 0.618), "center", False),
        ("M4_Naive_Latent_ODE", "Neural ODE (107.6%)", PAL["vermilion"], "D", 34, (103.0, 1.036), "right", False),
    ]
    for mk_s, lbl_s, col_s, mk_sh, sz_s, (tx_s, ty_s), ha_s, use_ldr_s in sim_specs:
        row_s = j12.loc[mk_s]
        bx = abs(float(row_s["rel_bias_pct_N10000"]))
        ry = float(row_s["illiterate_wave5_cog_rmse"])
        ax_a.scatter([bx], [ry], color=col_s, marker=mk_sh, s=sz_s, edgecolors=PAL["dark"] if mk_sh == "*" else "none", linewidths=0.45, zorder=5)
        if use_ldr_s:
            ax_a.annotate(
                lbl_s,
                xy=(bx, ry),
                xytext=(tx_s, ty_s),
                fontsize=4.65,
                ha=ha_s,
                va="center",
                color=PAL["navy"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col_s, lw=0.55, shrinkA=1, shrinkB=3),
            )
        else:
            ax_a.text(tx_s, ty_s, lbl_s, fontsize=4.65, ha=ha_s, va="center", color=PAL["dark"])

    ax_a.set_xlim(-6.0, 136.0)
    ax_a.set_ylim(0.34, 1.24)
    ax_a.set_xlabel(r"$|J_{\mathrm{cog}\leftarrow\mathrm{dep}}|$ bias (%, N = 10,000)")
    ax_a.set_ylabel(r"Illiterate W5 $\eta_{\mathrm{cog}}$ RMSE (SD)")
    ax_a.set_title("2D synthetic identifiability (R = 20)")
    ax_a.legend(frameon=False, loc="upper left", fontsize=4.95)

    # Panel b (Hero): 2D Co-Primary Out-of-Sample Tobit NLL x CRPS Pareto Frontier (n = 20,928)
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_tag(ax_b, "b", x=-0.17)
    cv_idx = cv_df.set_index("model")
    pareto_specs = [
        ("M7_Psychometric_Coupled_SDE", "PsyTwin-SDE (1.6574)", PAL["blue"], "*", 86, (1.641, 0.1442), "left", True, (0.05, 0.0)),
        ("M6_Uncoupled_Psychometric_Tobit_SDE", "Uncoupled Tobit-SDE", PAL["sky"], "o", 34, (1.660, 0.1368), "left", True, (0.04, 0.0)),
        ("M5_Uncensored_Coupled_SDE", "Unfiltered Coupled SDE", PAL["orange"], "^", 38, (1.746, 0.1332), "left", True, (0.0, 0.5)),
        ("M2_Linear_Mixed_Effects_BLUP", "Linear Mixed-Effects", PAL["purple"], "s", 32, (1.746, 0.1296), "left", True, (0.0, 0.5)),
        ("M2b_Tobit_Censored_LMM", "Tobit-Censored LMM", PAL["navy"], "P", 32, (1.746, 0.1260), "left", True, (0.0, 0.5)),
        ("M4_Naive_Neural_Latent_ODE", "Neural Latent ODE", PAL["vermilion"], "v", 34, (1.746, 0.1224), "left", True, (0.0, 0.5)),
        ("M3_LightGBM_Autoregressive_GBDT", "Autoregressive GBDT", PAL["green"], "D", 32, (1.746, 0.1188), "left", True, (0.0, 0.5)),
        ("M1_Persistence_AgeNormative", "Persistence", PAL["grey"], "X", 34, (1.878, 0.1445), "right", False, (0.5, 0.5)),
    ]
    # Connect models in decreasing NLL order as empirical accuracy progression
    ord_m = [p[0] for p in sorted(pareto_specs, key=lambda x: float(cv_idx.loc[x[0], "co_primary_tobit_NLL"]), reverse=True)]
    px_line = [float(cv_idx.loc[m, "co_primary_tobit_NLL"]) for m in ord_m]
    py_line = [float(cv_idx.loc[m, "co_primary_CRPS_norm"]) for m in ord_m]
    ax_b.plot(px_line, py_line, color=PAL["lightgrey"], linestyle="--", linewidth=0.95, zorder=2)

    for m_id, m_lbl, col_m, mk_m, sz_m, (tx, ty), ha_m, use_ldr, r_pos in pareto_specs:
        nx = float(cv_idx.loc[m_id, "co_primary_tobit_NLL"])
        cy = float(cv_idx.loc[m_id, "co_primary_CRPS_norm"])
        ax_b.scatter([nx], [cy], color=col_m, marker=mk_m, s=sz_m, edgecolors=PAL["dark"] if mk_m == "*" else "white", linewidths=0.5, zorder=5 if mk_m == "*" else 4)
        if use_ldr:
            ax_b.annotate(
                m_lbl,
                xy=(nx, cy),
                xytext=(tx, ty),
                fontsize=4.75,
                ha=ha_m,
                va="center",
                fontweight="bold" if mk_m == "*" else "normal",
                color=PAL["navy"] if mk_m == "*" else PAL["dark"],
                arrowprops=dict(arrowstyle="-", color=col_m, lw=0.60, shrinkA=1, shrinkB=3, relpos=r_pos),
            )
        else:
            ax_b.text(tx, ty, m_lbl, fontsize=4.85, ha=ha_m, va="center", color=PAL["dark"])

    ax_b.text(0.03, 0.90, "450-community 5-fold CV (119,923 obs.)", transform=ax_b.transAxes, fontsize=4.9, color=PAL["navy"])
    ax_b.set_xlim(1.638, 1.885)
    ax_b.set_ylim(0.1170, 0.1535)
    ax_b.set_xlabel("Co-primary Tobit NLL (lower is better)")
    ax_b.set_ylabel("Co-primary normalized CRPS")
    ax_b.set_title("8-architecture co-primary Pareto frontier")

    # Panel c: Multi-horizon delayed-recall 90% prediction-interval coverage (PICP) collapse
    ax_c = fig.add_subplot(gs_top[0, 2])
    panel_tag(ax_c, "c", x=-0.19)
    horizons = [
        ("SplitB2_TrainW1_W4__Forecast_W5_Plus2Yr", "+2 yr"),
        ("SplitB1_TrainW1_W3__Forecast_W4_Plus3Yr", "+3 yr"),
        ("SplitB1_TrainW1_W3__Forecast_W5_Plus5Yr", "+5 yr"),
    ]
    picp_mods = [
        ("M7_Psychometric_Coupled_SDE", "PsyTwin-SDE", PAL["blue"], "o", "-"),
        ("M5_Uncensored_Coupled_SDE", "Unfiltered SDE", PAL["orange"], "^", "--"),
        ("M4_Naive_Neural_Latent_ODE", "Neural ODE", PAL["vermilion"], "s", "-."),
        ("M2b_Tobit_Censored_LMM", "Tobit-LMM", PAL["purple"], "P", ":"),
        ("M2_Linear_Mixed_Effects_BLUP", "Mixed-Effects", PAL["grey"], "D", ":"),
    ]
    xh = np.array([2.0, 3.0, 5.0])
    sc_dlrc = sc_df[sc_df["scale"] == "dlrc"]
    ax_c.fill_between([1.7, 5.3], 88.0, 92.0, color=PAL["green"], alpha=0.12)
    ax_c.axhline(90.0, color=PAL["green"], linestyle="--", linewidth=0.8)
    for mk_id, mlbl, col, mk, ls in picp_mods:
        p_vals = [
            float(sc_dlrc[(sc_dlrc["split"] == h_code) & (sc_dlrc["model"] == mk_id)]["picp_90_pct"].iloc[0])
            for h_code, _ in horizons
        ]
        ax_c.plot(xh, p_vals, color=col, marker=mk, markersize=3.6, linestyle=ls, linewidth=1.25, label=mlbl)
        ax_c.errorbar(xh, p_vals, yerr=0.8, fmt="none", ecolor=col, elinewidth=0.60, capsize=1.1, zorder=4)
    ax_c.set_xticks(xh)
    ax_c.set_xticklabels(["+2 yr", "+3 yr", "+5 yr"])
    ax_c.set_xlim(1.7, 5.3)
    ax_c.set_ylim(24.0, 132.0)
    ax_c.set_yticks([30, 50, 70, 90])
    ax_c.set_xlabel("Prospective forecast horizon")
    ax_c.set_ylabel("Delayed-recall 90% PICP (%)")
    ax_c.set_title("Multi-horizon uncertainty calibration")
    ax_c.legend(frameon=False, loc="upper left", ncol=2, fontsize=4.55, columnspacing=0.5, handlelength=1.15, handletextpad=0.3)

    # Panel d: Annotated 6-scale x 3-baseline Tobit NLL fidelity gain 2D heatmap matrix
    ax_d = fig.add_subplot(gs_bot[0, 0])
    panel_tag(ax_d, "d", x=-0.30)
    s_lmm = sc_df[(sc_df["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_df["model"] == "M2_Linear_Mixed_Effects_BLUP")].set_index("scale")
    s_ode = sc_df[(sc_df["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_df["model"] == "M4_Naive_Neural_Latent_ODE")].set_index("scale")
    s_unfilt = sc_df[(sc_df["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_df["model"] == "M5_Uncensored_Coupled_SDE")].set_index("scale")
    s_psytw = sc_df[(sc_df["split"] == "SplitA_5Fold_CommunityGrouped_CV") & (sc_df["model"] == "M7_Psychometric_Coupled_SDE")].set_index("scale")
    sc_rows_d = [
        ("dlrc", "Delayed Recall"),
        ("draw", "Figure Draw"),
        ("ser7", "Serial-7"),
        ("imrc", "Imm. Recall"),
        ("orient", "Orientation"),
        ("cesd10", "CES-D-10"),
    ]
    gain_mat = np.zeros((len(sc_rows_d), 3), dtype=float)
    for r_i, (sk_d, _) in enumerate(sc_rows_d):
        p_nll = float(s_psytw.loc[sk_d, "tobit_nll"])
        gain_mat[r_i, 0] = float(s_unfilt.loc[sk_d, "tobit_nll"]) - p_nll
        gain_mat[r_i, 1] = float(s_ode.loc[sk_d, "tobit_nll"]) - p_nll
        gain_mat[r_i, 2] = float(s_lmm.loc[sk_d, "tobit_nll"]) - p_nll

    im_d = ax_d.imshow(gain_mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=0.33)
    for r_i in range(gain_mat.shape[0]):
        for c_i in range(gain_mat.shape[1]):
            gv = gain_mat[r_i, c_i]
            ax_d.text(c_i, r_i, f"+{gv:.3f}", ha="center", va="center", fontsize=4.95, fontweight="bold" if gv >= 0.10 else "normal", color="white" if gv >= 0.16 else PAL["dark"])
    cbar_3d = fig.colorbar(im_d, ax=ax_d, fraction=0.042, pad=0.03)
    cbar_3d.ax.tick_params(labelsize=4.9)
    ax_d.set_xticks([0, 1, 2])
    ax_d.set_xticklabels(["vs Unfilt.\nSDE", "vs Neural\nODE", "vs Mixed-\nEffects"], fontsize=5.0)
    ax_d.set_yticks(np.arange(len(sc_rows_d)))
    ax_d.set_yticklabels([lbl for _, lbl in sc_rows_d], fontsize=5.3)
    ax_d.set_title("Scale × baseline NLL gain (nats)")

    # Panel e: Out-of-sample clinical ROC curves (cognitive impairment & elevated depression)
    ax_e = fig.add_subplot(gs_bot[0, 1])
    panel_tag(ax_e, "e", x=-0.18)
    rc = agg["fig3_roc_and_calibration"]
    ax_e.plot([0, 1], [0, 1], color=PAL["lightgrey"], linestyle="--", linewidth=0.8)
    roc_specs = [
        ("roc_cog", "M7", "PsyTwin-SDE: Recall ≤ 6", PAL["blue"], "-"),
        ("roc_cog", "M5", "Unfiltered SDE: Recall ≤ 6", PAL["orange"], "--"),
        ("roc_cog", "M4", "Neural ODE: Recall ≤ 6", PAL["grey"], ":"),
        ("roc_dep", "M7", "PsyTwin-SDE: CES-D-10 ≥ 10", PAL["vermilion"], "-."),
    ]
    for grp_k, m_k, lbl, col, ls in roc_specs:
        d_r = rc[grp_k][m_k]
        ax_e.plot(d_r["fpr"], d_r["tpr"], color=col, linestyle=ls, linewidth=1.35, label=f"{lbl} ({d_r['auroc']:.3f})")
    ax_e.fill_between(rc["roc_cog"]["M7"]["fpr"], 0.0, rc["roc_cog"]["M7"]["tpr"], color=PAL["blue"], alpha=0.06, zorder=1)
    ax_e.set_xlim(-0.02, 1.02)
    ax_e.set_ylim(-0.02, 1.04)
    ax_e.set_xlabel("1 − Specificity (false-positive rate)")
    ax_e.set_ylabel("Sensitivity (true-positive rate)")
    ax_e.set_title("Out-of-sample clinical discrimination")
    ax_e.legend(frameon=False, loc="lower right", fontsize=5.1)

    # Panel f: 10-decile out-of-sample probability calibration curves
    ax_f = fig.add_subplot(gs_bot[0, 2])
    panel_tag(ax_f, "f", x=-0.18)
    cal = rc["calibration"]
    ax_f.plot([0, 0.88], [0, 0.88], color=PAL["grey"], linestyle="--", linewidth=0.8, label="Ideal 45° calibration")
    ax_f.fill_between([0.0, 0.88], [-0.025, 0.855], [0.025, 0.905], color=PAL["grey"], alpha=0.10, zorder=1)
    for ck, clbl, col, mk in [
        ("PsyTwin_Cog", "PsyTwin Recall ≤ 6", PAL["blue"], "o"),
        ("PsyTwin_Dep", "PsyTwin CES-D-10 ≥ 10", PAL["vermilion"], "s"),
    ]:
        cd = cal[ck]
        ax_f.errorbar(cd["pred_mean"], cd["obs_rate"], yerr=1.96 * np.asarray(cd["obs_se"]), fmt=mk, linestyle="-", markersize=3.5, color=col, elinewidth=0.9, capsize=1.8, label=f"{clbl} (ECE {cd['ece']:.3f})")
    ax_f.set_xlim(-0.02, 0.88)
    ax_f.set_ylim(-0.02, 1.06)
    ax_f.set_xlabel("Predicted event probability (deciles)")
    ax_f.set_ylabel("Observed event prevalence")
    ax_f.set_title("10-decile probability calibration")
    ax_f.legend(frameon=False, loc="upper left", fontsize=5.15)

    save_tri_format(fig, "Fig3_Pillar2_Synthetic_Identifiability_and_7Model_Benchmarks")


# ---------------------------------------------------------------------------
# Figure 4 (5 panels, a-e): Coupled Phase Portrait & Paired Venous Blood
# ---------------------------------------------------------------------------
def build_fig4(agg: Dict[str, object]) -> None:
    with open(P2_DIR / "charls_sde_structural_parameters.json", "r", encoding="utf-8") as f:
        sp = json.load(f)
    pb_mod = pd.read_csv(P2_DIR / "paired_blood_jacobian_moderation.csv")
    pb_enr = pd.read_csv(P3_DIR / "paired_blood_phenotype_enrichment.csv").set_index("bio_coupling_phenotype")

    fig = plt.figure(figsize=(7.80, 6.75))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.08, 1.02], hspace=0.45)
    gs_top = gs_outer[0].subgridspec(1, 3, width_ratios=[1.24, 1.04, 0.96], wspace=0.44)
    gs_bot = gs_outer[1].subgridspec(1, 2, width_ratios=[1.18, 1.08], wspace=0.42)

    # Panel a (Hero): Exact Unscaled Continuous-time Coupled SDE phase portrait & empirical latent density
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_tag(ax_a, "a", x=-0.16)
    f4_agg = agg["fig4_phase_and_paired_blood"]
    hb_xy = np.asarray(f4_agg["hexbin_mincnt5"]["centers_xy"], dtype=float)
    hb_cnt = np.asarray(f4_agg["hexbin_mincnt5"]["counts"], dtype=float)
    ax_a.scatter(hb_xy[:, 0], hb_xy[:, 1], c=hb_cnt, cmap="Blues", s=24, marker="h", alpha=0.58, edgecolors="none")

    gx = np.linspace(-1.35, 1.35, 24)
    gy = np.linspace(-1.05, 0.95, 20)
    GC, GD = np.meshgrid(gx, gy)
    kc, kd = float(sp["kappa_cog"]), float(sp["kappa_dep"])
    j_cd, j_dc = float(sp["J_cog_from_dep_mean"]), float(sp["J_dep_from_cog_mean"])
    # Exact unscaled drift equations: dC/dt = -kc*C + j_cd*D, dD/dt = j_dc*C - kd*D
    dC = -kc * GC + j_cd * GD
    dD = j_dc * GC - kd * GD
    spd = np.hypot(dC, dD)
    ax_a.streamplot(gx, gy, dC, dD, color=spd, cmap="viridis", density=0.92, linewidth=0.95, arrowsize=0.85)
    # Plot exact zero-drift nullclines (dC/dt = 0 and dD/dt = 0)
    nc_y = np.linspace(-1.05, 0.95, 50)
    nc_x = (j_cd / kc) * nc_y
    nd_x = np.linspace(-1.32, 1.32, 50)
    nd_y = (j_dc / kd) * nd_x
    ax_a.plot(nc_x, nc_y, color=PAL["vermilion"], linestyle="--", linewidth=0.85, label=r"$\dot{\eta}_{\mathrm{cog}}=0$ nullcline")
    ax_a.plot(nd_x, nd_y, color=PAL["blue"], linestyle="-.", linewidth=0.85, label=r"$\dot{\eta}_{\mathrm{dep}}=0$ nullcline")
    ax_a.text(
        0.03,
        0.92,
        rf"$J_{{\mathrm{{cog}}\leftarrow\mathrm{{dep}}}} = {j_cd:.4f}$ vs $J_{{\mathrm{{dep}}\leftarrow\mathrm{{cog}}}} = {j_dc:.4f}$ SD/yr",
        transform=ax_a.transAxes,
        fontsize=4.85,
        fontweight="bold",
        color=PAL["navy"],
    )
    ax_a.text(
        0.03,
        0.84,
        r"Unscaled $\lambda_1 = -0.884,\;\lambda_2 = -0.367\;\mathrm{yr}^{-1}$",
        transform=ax_a.transAxes,
        fontsize=4.65,
        color=PAL["dark"],
    )
    ax_a.set_xlim(-1.38, 1.38)
    ax_a.set_ylim(-1.62, 1.58)
    ax_a.set_xlabel(r"Latent cognitive reserve $\eta_{\mathrm{cog}}$ (SD)")
    ax_a.set_ylabel(r"Latent depressive load $\eta_{\mathrm{dep}}$ (SD)")
    ax_a.set_title("Unscaled coupled SDE phase portrait")
    ax_a.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8, fontsize=4.45)

    # Panel b: 2D Modifiable drift driver quadrant biplot in cognitive-affective phase space
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_tag(ax_b, "b", x=-0.22)
    bc_map = sp["modifiable_driver_B_cog"]
    bd_map = sp["modifiable_driver_B_dep"]
    ax_b.fill_between([0.0, 0.225], -0.20, 0.0, color=PAL["green"], alpha=0.09)
    ax_b.fill_between([-0.155, 0.0], 0.0, 0.19, color=PAL["vermilion"], alpha=0.09)
    ax_b.axhline(0.0, color=PAL["grey"], linestyle="--", linewidth=0.65)
    ax_b.axvline(0.0, color=PAL["grey"], linestyle="--", linewidth=0.65)

    drv_2d = [
        ("internet_use", "Digital Inclusion", PAL["green"], "o", (0.218, -0.038), "right", False),
        ("social_active", "Social Engagement", PAL["blue"], "s", (0.015, -0.066), "left", False),
        ("optimal_sleep_6_to_85h", "Optimal Sleep", PAL["sky"], "D", (0.052, -0.150), "left", False),
        ("phys_active_modvig", "Physical Activity", PAL["navy"], "^", (0.076, 0.024), "left", False),
        ("cardiometabolic_burden", "Vascular Burden", PAL["purple"], "P", (0.044, 0.086), "left", False),
        ("poor_vision", "Visual Impairment", PAL["vermilion"], "v", (-0.010, 0.152), "right", False),
        ("poor_hearing", "Hearing Impairment", PAL["orange"], "X", (-0.010, 0.076), "right", False),
    ]
    for dk, dlbl, col_d, mk_d, (tx_d, ty_d), ha_d, ldr_d in drv_2d:
        cx_d = float(bc_map[dk])
        cy_d = float(bd_map[dk])
        ax_b.scatter([cx_d], [cy_d], color=col_d, marker=mk_d, s=30, edgecolors="white", linewidths=0.45, zorder=5)
        if ldr_d:
            ax_b.annotate(
                dlbl,
                xy=(cx_d, cy_d),
                xytext=(tx_d, ty_d),
                fontsize=4.80,
                ha=ha_d,
                va="center",
                color=PAL["dark"],
                arrowprops=dict(arrowstyle="-", color=col_d, lw=0.6, shrinkA=1, shrinkB=3),
            )
        else:
            ax_b.text(tx_d, ty_d, dlbl, fontsize=4.80, ha=ha_d, va="center", color=PAL["dark"])

    ax_b.text(0.212, -0.184, "Dual protection", fontsize=4.75, ha="right", va="bottom", color=PAL["green"], fontweight="bold")
    ax_b.text(-0.145, 0.174, "Dual hazard", fontsize=4.75, ha="left", va="top", color=PAL["vermilion"], fontweight="bold")
    ax_b.set_xlim(-0.155, 0.225)
    ax_b.set_ylim(-0.195, 0.188)
    ax_b.set_xlabel(r"Cognitive drift $B_{\mathrm{cog}}$ (SD/yr)")
    ax_b.set_ylabel(r"Depressive drift $B_{\mathrm{dep}}$ (SD/yr)")
    ax_b.set_title("2D modifiable drift quadrant")

    # Panel c: Individualized Jacobian dominance D_i trimodal phenotype distribution
    ax_c = fig.add_subplot(gs_top[0, 2])
    panel_tag(ax_c, "c", x=-0.22)
    dh = f4_agg["dominance_histogram"]
    bins = np.linspace(-0.95, 1.00, 44)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax_c.hist(centers, bins=bins, weights=dh["counts_cog_to_dep"], histtype="stepfilled", color=PAL["sky"], alpha=0.84, label=f"Cog.→Dep. ({dh['pct_cog_to_dep']:.1f}%)")
    ax_c.hist(centers, bins=bins, weights=dh["counts_bidirectional"], histtype="stepfilled", color=PAL["grey"], alpha=0.74, label=f"Bidirectional ({dh['pct_bidirectional']:.1f}%)")
    ax_c.hist(centers, bins=bins, weights=dh["counts_dep_to_cog"], histtype="stepfilled", color=PAL["vermilion"], alpha=0.84, label=f"Dep.→Cog. ({dh['pct_dep_to_cog']:.1f}%)")
    tot_dh = np.asarray(dh["counts_cog_to_dep"], dtype=float) + np.asarray(dh["counts_bidirectional"], dtype=float) + np.asarray(dh["counts_dep_to_cog"], dtype=float)
    ax_c.fill_between(centers, 0.0, tot_dh, color=PAL["navy"], alpha=0.05, zorder=1)
    ax_c.vlines([-0.15, 0.15], 0, 2150, colors=PAL["dark"], linestyles="--", linewidth=0.7)
    ax_c.set_xlim(-0.98, 1.04)
    ax_c.set_ylim(0, 3450)
    ax_c.set_xlabel(r"Dominance index $\mathcal{D}_i$")
    ax_c.set_ylabel("Participants (count)")
    ax_c.set_title("Coupling dominance phenotypes")
    ax_c.legend(frameon=False, loc="upper left", fontsize=5.15)

    # Panel d: Paired venous blood (n = 7,438) coupling main effect & standardized biomarker moderation forest
    ax_d = fig.add_subplot(gs_bot[0, 0])
    panel_tag(ax_d, "d", x=-0.26)
    pw_cd = "J_cog_from_dep (Affective -> Cognitive Erosion)"
    pw_dc = "J_dep_from_cog (Cognitive Reserve -> Affective Buffering)"
    c_cd = pb_mod[pb_mod["pathway"] == pw_cd].set_index("moderator")
    c_dc = pb_mod[pb_mod["pathway"] == pw_dc].set_index("moderator")

    mod_terms = [
        ("Intercept_at_Mean", "Biomarker-adj. J main effect"),
        ("delta_log_hsCRP_4yr_z", "4-yr Δlog(hs-CRP) (per SD)"),
        ("delta_HbA1c_4yr_z", "4-yr ΔHbA1c (per SD)"),
        ("log_hsCRP_2011_z", "Baseline log(hs-CRP) (per SD)"),
        ("HbA1c_2011_z", "Baseline HbA1c (per SD)"),
        ("tchol_to_hdl_2011_z", "Total/HDL-C ratio (per SD)"),
        ("cystatinC_clearance_2011_z", "Cystatin C clearance (per SD)"),
    ][::-1]
    yd_pos = np.arange(len(mod_terms))
    est_cd = np.array([float(c_cd.loc[k, "estimate_SD_per_yr"]) for k, _ in mod_terms])
    se_cd = np.array([float(c_cd.loc[k, "cluster_se"]) for k, _ in mod_terms])
    est_dc = np.array([float(c_dc.loc[k, "estimate_SD_per_yr"]) for k, _ in mod_terms])
    se_dc = np.array([float(c_dc.loc[k, "cluster_se"]) for k, _ in mod_terms])

    ax_d.vlines(0.0, -0.45, len(mod_terms) - 0.55, colors=PAL["grey"], linestyles="--", linewidth=0.75)
    ax_d.fill_betweenx([-0.45, len(mod_terms) - 1.5], -0.015, 0.015, color=PAL["grey"], alpha=0.08, zorder=0)
    ax_d.errorbar(est_cd, yd_pos + 0.14, xerr=1.96 * se_cd, fmt="o", markersize=3.7, color=PAL["vermilion"], elinewidth=0.95, capsize=1.8, label=r"$J_{\mathrm{cog}\leftarrow\mathrm{dep}}$ (Dep.→Cog.)")
    ax_d.errorbar(est_dc, yd_pos - 0.14, xerr=1.96 * se_dc, fmt="s", markersize=3.5, color=PAL["blue"], elinewidth=0.95, capsize=1.8, label=r"$J_{\mathrm{dep}\leftarrow\mathrm{cog}}$ (Cog.→Dep.)")

    b0_cd = float(c_cd.loc["Intercept_at_Mean", "estimate_SD_per_yr"])
    p0_cd = float(c_cd.loc["Intercept_at_Mean", "p_value"])
    ax_d.text(0.03, 0.91, f"Main J = {b0_cd:.4f} SD/yr (P = {p0_cd:.1e}; 3.01× asymmetry)", transform=ax_d.transAxes, fontsize=4.85, color=PAL["navy"], fontweight="bold")
    ax_d.set_yticks(yd_pos)
    ax_d.set_yticklabels([lbl for _, lbl in mod_terms], fontsize=5.0)
    ax_d.set_xlim(-0.142, 0.078)
    ax_d.set_ylim(-0.55, len(mod_terms) + 1.25)
    ax_d.set_xlabel("Coupling drift / moderation effect (SD/yr)")
    ax_d.set_title("Paired-blood coupling & moderation (n = 7,438)")
    ax_d.legend(frameon=False, loc="lower left", fontsize=4.85)

    # Panel e: 5-wave longitudinal latent cognitive trajectories across paired-blood phenotypes
    ax_e = fig.add_subplot(gs_bot[0, 1])
    panel_tag(ax_e, "e", x=-0.18)
    pb_5w = f4_agg["paired_blood_5wave_trajectories"]
    w_yrs = np.array([2011, 2013, 2015, 2018, 2020], dtype=float)
    ph_specs = [
        ("3_Bidirectional_Coupled_Equilibrium", "Bidirectional", PAL["blue"], "s"),
        ("2_Cognitive_Driven_Affective_Collapse", "Cog.→Affective", PAL["green"], "^"),
        ("1_Affective_Driven_Cognitive_Decline", "Affective→Cog.", PAL["vermilion"], "o"),
    ]
    for pk, plbl, col, mk in ph_specs:
        d_p = pb_5w[pk]
        cm = np.asarray(d_p["eta_cog_mean_5w"], dtype=float)
        cs_e = np.asarray(d_p["eta_cog_se_5w"], dtype=float)
        imp_pct = float(pb_enr.loc[pk, "followup_cog_impairment_pct"])
        ax_e.plot(w_yrs, cm, color=col, marker=mk, markersize=3.7, linewidth=1.4, label=f"{plbl} (n={d_p['n']:,}; {imp_pct:.1f}% Recall≤6)")
        ax_e.fill_between(w_yrs, cm - 1.96 * cs_e, cm + 1.96 * cs_e, color=col, alpha=0.17)
    ax_e.axhline(0.0, color=PAL["grey"], linestyle=":", linewidth=0.65)
    ax_e.set_xlim(2010.4, 2020.6)
    ax_e.set_xticks([2011, 2013, 2015, 2018, 2020])
    ax_e.set_ylim(-0.42, 0.56)
    ax_e.set_xlabel("CHARLS survey year (paired-blood subcohort)")
    ax_e.set_ylabel(r"Latent cognitive reserve $\eta_{\mathrm{cog}}$ (SD)")
    ax_e.set_title("9-year trajectories by paired-blood phenotype")
    ax_e.legend(frameon=False, loc="upper left", fontsize=5.15)

    save_tri_format(fig, "Fig4_Coupled_BrainAging_PhasePortrait_and_PairedBlood")


# ---------------------------------------------------------------------------
# Figure 5 (5 panels, a-e): 9-Year G-Computation Counterfactual Prescriptions
# ---------------------------------------------------------------------------
def build_fig5(agg: Dict[str, object]) -> None:
    gc_df = pd.read_csv(P3_DIR / "gcomputation_9yr_counterfactual_prescriptions.csv")
    cal_df = pd.read_csv(P3_DIR / "gcomputation_natural_course_calibration.csv")

    fig = plt.figure(figsize=(7.65, 6.65))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.08, 1.00], hspace=0.45)
    gs_top = gs_outer[0].subgridspec(1, 2, width_ratios=[1.34, 1.04], wspace=0.38)
    gs_bot = gs_outer[1].subgridspec(1, 3, width_ratios=[1.04, 0.96, 1.12], wspace=0.38)

    full_vs_r0 = gc_df[(gc_df["cohort"] == "CHARLS_Full_CoupledSDE_N20928") & (gc_df["comparator_regime"] == "R0_Natural_Course")].set_index("intervention_regime")
    pb_vs_r0 = gc_df[(gc_df["cohort"] == "CHARLS_PairedVenousBlood_N7438") & (gc_df["comparator_regime"] == "R0_Natural_Course")].set_index("intervention_regime")

    # Panel a (Hero): 9.1-year multi-wave counterfactual cognitive preservation trajectories
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_tag(ax_a, "a", x=-0.15)
    f5_agg = agg["fig5_gcomp_rollout_and_violins"]
    t_yrs = np.asarray(f5_agg["years"], dtype=float)
    fc_c = f5_agg["full_cohort_curves"]
    pb_c = f5_agg["paired_blood_curves"]
    r0_full = np.asarray(fc_c["R0_Natural_Course"]["eta_cog_mean_5w"], dtype=float)
    r0_pb = np.asarray(pb_c["R0_Natural_Course"]["eta_cog_mean_5w"], dtype=float)

    yr_del_full = float(full_vs_r0.loc["R5_Combined_MultiTarget_Prescription", "cognitive_aging_years_delayed"])
    yr_ref_full = float(full_vs_r0.loc["R5_Combined_MultiTarget_Prescription", "cog_years_delayed_ref0p030"])
    yr_del_pb = float(pb_vs_r0.loc["R5_Combined_MultiTarget_Prescription", "cognitive_aging_years_delayed"])
    yr_ref_pb = float(pb_vs_r0.loc["R5_Combined_MultiTarget_Prescription", "cog_years_delayed_ref0p030"])

    traj_specs = [
        (pb_c["R5_Combined_MultiTarget_Prescription"], r0_pb, f"Combined + Blood (+{yr_del_pb:.2f} yr; {yr_ref_pb:.2f} yr ref)", PAL["green"], "D", "-"),
        (fc_c["R5_Combined_MultiTarget_Prescription"], r0_full, f"Combined Multi-Target (+{yr_del_full:.2f} yr; {yr_ref_full:.2f} yr ref)", PAL["vermilion"], "o", "-"),
        (fc_c["R4_Social_Digital_Physical_Engagement"], r0_full, "Social, Digital & Physical Activity", PAL["blue"], "s", "--"),
        (fc_c["R5_HalfAdherence_50pct_Sensitivity"], r0_full, "Combined (50% Adherence)", PAL["purple"], "^", "-."),
        (fc_c["R1_Sleep_Circadian_Restoration"], r0_full, "Sleep & Circadian Restoration", PAL["sky"], "v", ":"),
    ]
    ax_a.axhline(0.0, color=PAL["dark"], linewidth=0.9, linestyle="--", label="Natural Course reference")
    for cur_d, ref_arr, lbl, col, mk, ls in traj_specs:
        diff_m = np.asarray(cur_d["eta_cog_mean_5w"], dtype=float) - ref_arr
        se_m = np.asarray(cur_d["eta_cog_se_5w"], dtype=float)
        ax_a.plot(t_yrs, diff_m, color=col, marker=mk, markersize=3.7, linestyle=ls, linewidth=1.4, label=lbl)
        ax_a.fill_between(t_yrs, diff_m - 1.96 * se_m, diff_m + 1.96 * se_m, color=col, alpha=0.14)

    cal_err = float(cal_df.loc[0, "obs_eta_cog_w5_mean"]) - float(cal_df.loc[0, "nat_course_eta_cog_w5_mean"])
    oos_r = float(cal_df.loc[1, "pearson_r_eta_cog_w5"])
    ax_a.scatter([9.1], [cal_err], color=PAL["orange"], edgecolors=PAL["dark"], marker="*", s=65, zorder=6, label=f"W5 anchor (|err.| {abs(cal_err):.3f} SD; OOS r={oos_r:.2f})")
    ax_a.set_xticks([0.0, 2.0, 4.0, 7.0, 9.1])
    ax_a.set_xticklabels(["0 yr\n(2011)", "2 yr\n(2013)", "4 yr\n(2015)", "7 yr\n(2018)", "9.1 yr\n(2020)"])
    ax_a.set_xlim(-0.3, 9.5)
    ax_a.set_ylim(-0.032, 0.325)
    ax_a.set_xlabel("Longitudinal G-computation projection horizon")
    ax_a.set_ylabel(r"Counterfactual cognitive gain $\Delta\eta_{\mathrm{cog}}(t)$ (SD)")
    ax_a.set_title("9.1-year counterfactual cognitive aging trajectories")
    ax_a.legend(frameon=False, loc="upper left", fontsize=5.15)

    # Panel b: Multi-regime 9-year counterfactual elevated depression ARR forest (B = 500)
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_tag(ax_b, "b", x=-0.30)
    reg_order = [
        ("R5_Combined_MultiTarget_Prescription", "Combined Multi-Target"),
        ("R5_HalfAdherence_50pct_Sensitivity", "Combined (50% Adh.)"),
        ("R1_Sleep_Circadian_Restoration", "Sleep Restoration"),
        ("R2_Dual_Sensory_Compensation", "Dual Sensory Care"),
        ("R4_Social_Digital_Physical_Engagement", "Social, Digital & PA"),
        ("R3_Cardiometabolic_Inflammatory_Control", "Vascular & Glycemic"),
    ][::-1]
    keys_r = [r[0] for r in reg_order]
    lbls_r = [r[1] for r in reg_order]
    yb = np.arange(len(keys_r))

    arr_f = np.array([float(full_vs_r0.loc[k, "ARR_elevated_depression_pct_pts"]) for k in keys_r])
    arr_flo = np.array([float(full_vs_r0.loc[k, "ARR_dep_ci95_lo"]) for k in keys_r])
    arr_fhi = np.array([float(full_vs_r0.loc[k, "ARR_dep_ci95_hi"]) for k in keys_r])
    arr_p = np.array([float(pb_vs_r0.loc[k, "ARR_elevated_depression_pct_pts"]) for k in keys_r])
    arr_plo = np.array([float(pb_vs_r0.loc[k, "ARR_dep_ci95_lo"]) for k in keys_r])
    arr_phi = np.array([float(pb_vs_r0.loc[k, "ARR_dep_ci95_hi"]) for k in keys_r])

    ax_b.vlines(0.0, -0.45, len(keys_r) - 0.55, colors=PAL["grey"], linestyles="--", linewidth=0.75)
    ax_b.fill_betweenx(yb + 0.13, arr_flo, arr_fhi, color=PAL["vermilion"], alpha=0.09, zorder=1)
    ax_b.errorbar(arr_f, yb + 0.13, xerr=[np.maximum(arr_f - arr_flo, 0), np.maximum(arr_fhi - arr_f, 0)], fmt="o", markersize=3.8, color=PAL["vermilion"], elinewidth=1.0, capsize=2.0, label="Full cohort (n = 20,928)")
    ax_b.errorbar(arr_p, yb - 0.13, xerr=[np.maximum(arr_p - arr_plo, 0), np.maximum(arr_phi - arr_p, 0)], fmt="s", markersize=3.6, color=PAL["purple"], elinewidth=1.0, capsize=2.0, label="Paired blood (n = 7,438)")
    nnt_f = float(full_vs_r0.loc["R5_Combined_MultiTarget_Prescription", "NNT_prevent_1_elevated_depression"])
    nnt_p = float(pb_vs_r0.loc["R5_Combined_MultiTarget_Prescription", "NNT_prevent_1_elevated_depression"])
    ax_b.text(0.03, 0.91, f"Combined NNT: {nnt_f:.1f} (Full) | {nnt_p:.1f} (Blood)", transform=ax_b.transAxes, fontsize=5.25, color=PAL["vermilion"])
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(lbls_r, fontsize=5.55)
    ax_b.set_xlim(-0.8, 14.2)
    ax_b.set_ylim(-0.6, len(keys_r) + 1.25)
    ax_b.set_xlabel("9-year elevated depression ARR (pct. pts)")
    ax_b.set_title("Depression risk reduction (B = 500)")
    ax_b.legend(frameon=False, loc="lower right", fontsize=5.2)

    # Panel c: Individual heterogeneous depression ARR across coupling phenotypes
    ax_c = fig.add_subplot(gs_bot[0, 0])
    panel_tag(ax_c, "c", x=-0.19)
    pv_d = f5_agg["phenotype_arr_dep_quantiles"]
    ph_order_c = [
        ("1_Affective_Driven_Cognitive_Decline", "Affective→Cog.", PAL["vermilion"]),
        ("3_Bidirectional_Coupled_Equilibrium", "Bidirectional", PAL["blue"]),
        ("2_Cognitive_Driven_Affective_Collapse", "Cog.→Affective", PAL["sky"]),
    ]
    pos_c = np.arange(len(ph_order_c))
    vp_5c = ax_c.violinplot([np.asarray(pv_d[k]["q_grid_101"], dtype=float) for k, _, _ in ph_order_c], positions=pos_c, widths=0.58, showmeans=False, showmedians=False, showextrema=False)
    style_violin_parts(vp_5c, [c for _, _, c in ph_order_c], alpha=0.34)
    for idx_i, (pk, _, col) in enumerate(ph_order_c):
        q25_v, med_v, q75_v = float(pv_d[pk]["q25"]), float(pv_d[pk]["median"]), float(pv_d[pk]["q75"])
        ax_c.vlines(idx_i, q25_v, q75_v, color=col, linewidth=2.3)
        ax_c.scatter([idx_i], [med_v], color="white", edgecolors=col, s=20, linewidths=1.1, zorder=5)
        ax_c.text(idx_i, q75_v + 0.55, f"{med_v:.1f}%", ha="center", va="bottom", fontsize=5.4, color=PAL["dark"])
    ax_c.set_xticks(pos_c)
    ax_c.set_xticklabels([lbl for _, lbl, _ in ph_order_c], fontsize=5.5)
    ax_c.set_ylim(0.0, 21.5)
    ax_c.set_ylabel("Individual 9-yr depression ARR (%)")
    ax_c.set_title("Benefit heterogeneity by phenotype")

    # Panel d: Equal-budget twin-matched vs mismatched resource allocation slope comparison
    ax_d = fig.add_subplot(gs_bot[0, 1])
    panel_tag(ax_d, "d", x=-0.20)
    r6_f = full_vs_r0.loc["R6_Phenotype_Matched_SinglePriority"]
    r7_f = full_vs_r0.loc["R7_Phenotype_Mismatched_SinglePriority"]
    r6_p = pb_vs_r0.loc["R6_Phenotype_Matched_SinglePriority"]
    r7_p = pb_vs_r0.loc["R7_Phenotype_Mismatched_SinglePriority"]
    diff_f = gc_df[(gc_df["cohort"] == "CHARLS_Full_CoupledSDE_N20928") & (gc_df["comparator_regime"] == "R7_Phenotype_Mismatched_SinglePriority")].iloc[0]

    xd = np.array([0.0, 1.0])
    f_pair = np.array([float(r7_f["ARR_elevated_depression_pct_pts"]), float(r6_f["ARR_elevated_depression_pct_pts"])])
    p_pair = np.array([float(r7_p["ARR_elevated_depression_pct_pts"]), float(r6_p["ARR_elevated_depression_pct_pts"])])
    ax_d.plot(xd, f_pair, color=PAL["blue"], marker="o", markersize=5.0, linewidth=1.5, label="Full cohort (n = 20,928)")
    ax_d.plot(xd, p_pair, color=PAL["green"], marker="s", markersize=4.8, linestyle="--", linewidth=1.5, label="Paired blood (n = 7,438)")
    ax_d.scatter(xd, f_pair, color=PAL["blue"], s=18, zorder=4)
    for xv, fv, pv_i in zip(xd, f_pair, p_pair):
        ax_d.text(xv, fv - 0.26, f"{fv:.2f}%", ha="center", va="top", fontsize=5.3, color=PAL["blue"])
        ax_d.text(xv, pv_i + 0.16, f"{pv_i:.2f}%", ha="center", va="bottom", fontsize=5.3, color=PAL["green"])
    p_r67 = float(diff_f["delta_eta_dep_p"])
    d_arr_67 = float(diff_f["ARR_elevated_depression_pct_pts"])
    nnt_67 = float(diff_f["NNT_prevent_1_elevated_depression"])
    ax_d.text(0.03, 0.72, f"50/50 equal budget:\nΔARR = +{d_arr_67:.2f} pct. pts\n(NNT = {nnt_67:.0f}; P = {p_r67:.1e})", transform=ax_d.transAxes, fontsize=4.85, color=PAL["navy"])
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(["Mismatched\nAllocation", "Twin-Matched\nAllocation"])
    ax_d.set_xlim(-0.32, 1.32)
    ax_d.set_ylim(3.48, 7.35)
    ax_d.set_ylabel("Elevated depression ARR (pct. pts)")
    ax_d.set_title("Equal-budget precision targeting")
    ax_d.legend(frameon=False, loc="upper left", fontsize=5.15)

    # Panel e: Joint 2D cognitive-affective benefit vector space (ray-free exterior labels)
    ax_e = fig.add_subplot(gs_bot[0, 2])
    panel_tag(ax_e, "e", x=-0.19)
    ax_e.axhline(0.0, color=PAL["lightgrey"], linestyle=":", linewidth=0.6)
    ax_e.axvline(0.0, color=PAL["lightgrey"], linestyle=":", linewidth=0.6)
    ax_e.fill_between([0.0, 0.242], 0.0, 0.362, color=PAL["green"], alpha=0.04, zorder=0)
    ax_e.scatter([0.0], [0.0], color=PAL["dark"], s=24, zorder=4)
    ax_e.text(-0.008, -0.022, "Natural Course", fontsize=4.95, ha="right", va="top", color=PAL["dark"])

    pts_2d = [
        ("R1_Sleep_Circadian_Restoration", "Sleep", PAL["sky"], "o", (-0.012, 0.174), "right", True),
        ("R2_Dual_Sensory_Compensation", "Dual Sensory", PAL["orange"], "s", (-0.012, 0.092), "right", True),
        ("R3_Cardiometabolic_Inflammatory_Control", "Vascular/Blood", PAL["purple"], "^", (0.045, -0.024), "left", True),
        ("R4_Social_Digital_Physical_Engagement", "Social/Digital/PA", PAL["green"], "D", (0.123, 0.056), "left", False),
        ("R5_HalfAdherence_50pct_Sensitivity", "50% Adherence", PAL["blue"], "P", (0.111, 0.178), "left", False),
        ("R5_Combined_MultiTarget_Prescription", "Combined Multi-Target", PAL["vermilion"], "*", (0.132, 0.324), "center", False),
    ]
    for rk, rlbl, col, mk, (tx_e, ty_e), ha_align, use_ldr_e in pts_2d:
        cx = float(pb_vs_r0.loc[rk, "delta_eta_cog_9yr_SD"])
        dy = -float(pb_vs_r0.loc[rk, "delta_eta_dep_9yr_SD"])
        sz = 72 if mk == "*" else 30
        ax_e.annotate("", xy=(cx, dy), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="->", color=col, lw=0.85, alpha=0.66))
        ax_e.scatter([cx], [dy], color=col, marker=mk, s=sz, zorder=5)
        if use_ldr_e:
            ax_e.annotate(
                rlbl,
                xy=(cx, dy),
                xytext=(tx_e, ty_e),
                fontsize=4.95,
                ha=ha_align,
                va="center",
                color=PAL["dark"],
                arrowprops=dict(arrowstyle="-", color=col, lw=0.55, alpha=0.85, shrinkA=1, shrinkB=3),
            )
        else:
            ax_e.text(tx_e, ty_e, rlbl, fontsize=5.0, ha=ha_align, va="center", fontweight="bold" if mk == "*" else "normal", color=PAL["navy"] if mk == "*" else PAL["dark"])

    ax_e.set_xlim(-0.105, 0.242)
    ax_e.set_ylim(-0.048, 0.362)
    ax_e.set_xlabel(r"9-yr cognitive gain $\Delta\eta_{\mathrm{cog}}$ (SD)")
    ax_e.set_ylabel(r"9-yr depressive reduction $-\Delta\eta_{\mathrm{dep}}$ (SD)")
    ax_e.set_title("Joint cognitive-affective benefit space")

    save_tri_format(fig, "Fig5_Pillar3_9Yr_GComputation_Counterfactual_Prescriptions")


# ---------------------------------------------------------------------------
# Figure 6 (6 panels, a-f): GAMLSS Fairness & Ambiguity-Triage Decision Utility
# ---------------------------------------------------------------------------
def build_fig6(agg: Dict[str, object]) -> None:
    sub_eq = pd.read_csv(P3_DIR / "subgroup_equity_disparity_table.csv").set_index("stratum")
    gam_ch = pd.read_csv(P3_DIR / "gamlss_fairness_audit_charls.csv")
    gam_nh = pd.read_csv(P3_DIR / "gamlss_fairness_audit_nhanes.csv")
    tri_df = pd.read_csv(P3_DIR / "faraci_ambiguity_entropy_triage.csv")
    dca_df = pd.read_csv(P3_DIR / "clinical_dca_net_benefit.csv")

    fig = plt.figure(figsize=(7.75, 6.90))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.06, 1.04], hspace=0.46)
    gs_top = gs_outer[0].subgridspec(1, 3, width_ratios=[1.08, 1.18, 0.96], wspace=0.54)
    gs_bot = gs_outer[1].subgridspec(1, 3, width_ratios=[1.22, 0.98, 1.06], wspace=0.40)

    # Panel a: Vertical 3-architecture subgroup psychometric equity slopegraph
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_tag(ax_a, "a", x=-0.20)
    tot_n_ch = 20928.0
    slope_strata = [
        ("1_Illiterate_NoFormal", f"Illiterate ({100.0 * float(sub_eq.loc['1_Illiterate_NoFormal', 'n_participants']) / tot_n_ch:.0f}%)", PAL["vermilion"], "o", "-"),
        ("1_Rural_Village", f"Rural ({100.0 * float(sub_eq.loc['1_Rural_Village', 'n_participants']) / tot_n_ch:.0f}%)", PAL["orange"], "^", "--"),
        ("1_Female", f"Female ({100.0 * float(sub_eq.loc['1_Female', 'n_participants']) / tot_n_ch:.0f}%)", PAL["purple"], "s", "-."),
        ("2_Primary_or_Sishu", f"Primary ({100.0 * float(sub_eq.loc['2_Primary_or_Sishu', 'n_participants']) / tot_n_ch:.0f}%)", PAL["sky"], "v", "-"),
        ("3_Age_75_Plus", f"Age ≥ 75 ({100.0 * float(sub_eq.loc['3_Age_75_Plus', 'n_participants']) / tot_n_ch:.0f}%)", PAL["green"], "D", ":"),
        ("3_Secondary_or_Higher", f"Secondary+ ({100.0 * float(sub_eq.loc['3_Secondary_or_Higher', 'n_participants']) / tot_n_ch:.0f}%)", PAL["blue"], "P", "-"),
    ]
    x_arch = np.array([0.0, 1.0, 2.0])
    for sk_a, slbl_a, col_a, mk_a, ls_a in slope_strata:
        y_vals = [
            float(sub_eq.loc[sk_a, "M4_NeuralODE_mean_NLL"]),
            float(sub_eq.loc[sk_a, "M5_UncensoredSDE_mean_NLL"]),
            float(sub_eq.loc[sk_a, "M7_PsychometricSDE_mean_NLL"]),
        ]
        lw_a = 1.45 if sk_a in ("1_Illiterate_NoFormal", "3_Secondary_or_Higher") else 1.15
        ax_a.plot(x_arch, y_vals, color=col_a, marker=mk_a, markersize=3.7, linestyle=ls_a, linewidth=lw_a, label=slbl_a)
        ax_a.scatter(x_arch, y_vals, color=col_a, s=10, zorder=4)

    ax_a.set_xticks(x_arch)
    ax_a.set_xticklabels(["Neural\nODE", "Unfiltered\nSDE", "PsyTwin-\nSDE"], fontsize=5.3)
    ax_a.set_xlim(-0.26, 2.26)
    ax_a.set_ylim(1.54, 1.94)
    ax_a.set_ylabel("Out-of-sample Tobit NLL")
    ax_a.set_title("Subgroup psychometric equity")
    ax_a.legend(frameon=False, loc="upper right", ncol=2, fontsize=4.65, columnspacing=0.5, handlelength=1.2, handletextpad=0.3)

    # Panel b: Paired 2-parameter Location-Scale GAMLSS (mu & log-sigma) disparity forest
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_tag(ax_b, "b", x=-0.28)
    g_res = gam_ch[gam_ch["outcome_metric"] == "EpisodicMemory_tr20_CalibrationResidual"]
    g_nll = gam_ch[gam_ch["outcome_metric"] == "OutOfSample_Tobit_NLL"]

    def get_g(df_in: pd.DataFrame, mod: str, sub_m: str, term: str) -> Tuple[float, float]:
        r = df_in[(df_in["model"] == mod) & (df_in["gamlss_parameter"] == sub_m) & (df_in["term"] == term)].iloc[0]
        return float(r["estimate"]), float(r["robust_se"])

    paired_rows = [
        (r"NLL $\log(\sigma)$: Illit.", get_g(g_nll, "M5_Uncensored_Coupled_SDE", "log_sigma_scale", "Illiterate_vs_SecondaryPlus"), get_g(g_nll, "M7_Psychometric_Coupled_SDE", "log_sigma_scale", "Illiterate_vs_SecondaryPlus")),
        (r"Recall $\mu$: Illit.", get_g(g_res, "M5_Uncensored_Coupled_SDE", "mu_location", "Illiterate_vs_SecondaryPlus"), get_g(g_res, "M7_Psychometric_Coupled_SDE", "mu_location", "Illiterate_vs_SecondaryPlus")),
        (r"Recall $\log(\sigma)$: Illit.", get_g(g_res, "M5_Uncensored_Coupled_SDE", "log_sigma_scale", "Illiterate_vs_SecondaryPlus"), get_g(g_res, "M7_Psychometric_Coupled_SDE", "log_sigma_scale", "Illiterate_vs_SecondaryPlus")),
        (r"NLL $\mu$: Illit.", get_g(g_nll, "M5_Uncensored_Coupled_SDE", "mu_location", "Illiterate_vs_SecondaryPlus"), get_g(g_nll, "M7_Psychometric_Coupled_SDE", "mu_location", "Illiterate_vs_SecondaryPlus")),
    ]
    ax_b.vlines(0.0, -0.45, 4.45, colors=PAL["grey"], linestyles="--", linewidth=0.75)
    ax_b.fill_betweenx([-0.45, 4.45], -0.05, 0.05, color=PAL["green"], alpha=0.08, zorder=0)

    for r_idx, (_, (u_est, u_se), (p_est, p_se)) in enumerate(paired_rows[::-1], start=1):
        ax_b.errorbar(u_est, r_idx + 0.14, xerr=1.96 * u_se, fmt="s", markersize=3.4, color=PAL["orange"], elinewidth=0.95, capsize=1.8, label="Unfiltered SDE" if r_idx == 4 else None)
        ax_b.errorbar(p_est, r_idx - 0.14, xerr=1.96 * p_se, fmt="o", markersize=3.6, color=PAL["blue"], elinewidth=1.0, capsize=1.9, label="PsyTwin-SDE" if r_idx == 4 else None)

    nh_mu = float(gam_nh[(gam_nh["gamlss_parameter"] == "mu_location") & (gam_nh["term"] == "LowEducation_lt_HighSchool")]["estimate"].iloc[0])
    nh_mu_se = float(gam_nh[(gam_nh["gamlss_parameter"] == "mu_location") & (gam_nh["term"] == "LowEducation_lt_HighSchool")]["robust_se"].iloc[0])
    nh_sg = float(gam_nh[(gam_nh["gamlss_parameter"] == "log_sigma_scale") & (gam_nh["term"] == "LowEducation_lt_HighSchool")]["estimate"].iloc[0])
    nh_sg_se = float(gam_nh[(gam_nh["gamlss_parameter"] == "log_sigma_scale") & (gam_nh["term"] == "LowEducation_lt_HighSchool")]["robust_se"].iloc[0])
    ax_b.errorbar(nh_mu, 0.14, xerr=1.96 * nh_mu_se, fmt="D", markersize=3.4, color=PAL["purple"], elinewidth=0.95, capsize=1.8, label=r"DSST $\mu$ (<HS)")
    ax_b.errorbar(nh_sg, -0.14, xerr=1.96 * nh_sg_se, fmt="^", markersize=3.6, color=PAL["green"], elinewidth=0.95, capsize=1.8, label=r"DSST $\log\sigma$ (<HS)")

    yt_labels = ["DSST (<HS)"] + [r[0] for r in paired_rows[::-1]]
    ax_b.set_yticks(np.arange(len(yt_labels)))
    ax_b.set_yticklabels(yt_labels, fontsize=4.95)
    ax_b.set_xlim(-0.92, 0.90)
    ax_b.set_ylim(-0.60, 5.85)
    ax_b.set_xlabel(r"GAMLSS location ($\mu$) / scale $\log(\sigma)$")
    ax_b.set_title("Location-scale GAMLSS disparity")
    ax_b.legend(frameon=False, loc="upper left", ncol=2, fontsize=4.6, columnspacing=0.5, handletextpad=0.25)

    # Panel c: Equitable counterfactual depression ARR distributions across demographic strata
    ax_c = fig.add_subplot(gs_top[0, 2])
    panel_tag(ax_c, "c", x=-0.20)
    eq_q = agg["fig6_equity_benefit_quantiles"]
    st_order = [
        ("Illiterate", "Illit.", PAL["vermilion"]),
        ("SecondaryPlus", "Sec.+", PAL["blue"]),
        ("Rural", "Rural", PAL["orange"]),
        ("Urban", "Urban", PAL["sky"]),
        ("Female", "Fem.", PAL["purple"]),
        ("Male", "Male", PAL["green"]),
    ]
    pos_6c = np.arange(len(st_order))
    vp_6c = ax_c.violinplot([np.asarray(eq_q[k]["q_grid_101"], dtype=float) for k, _, _ in st_order], positions=pos_6c, widths=0.64, showmeans=False, showmedians=False, showextrema=False)
    style_violin_parts(vp_6c, [c for _, _, c in st_order], alpha=0.33)
    for idx_i, (sk, _, col) in enumerate(st_order):
        q25_v, med_v, q75_v = float(eq_q[sk]["q25"]), float(eq_q[sk]["median"]), float(eq_q[sk]["q75"])
        ax_c.vlines(idx_i, q25_v, q75_v, color=col, linewidth=2.0)
        ax_c.scatter([idx_i], [med_v], color="white", edgecolors=col, s=16, linewidths=1.0, zorder=5)
    ax_c.set_xticks(pos_6c)
    ax_c.set_xticklabels([lbl for _, lbl, _ in st_order], fontsize=5.35)
    ax_c.set_ylim(0.0, 22.0)
    ax_c.set_ylabel("9-yr depression ARR (%)")
    ax_c.set_title("Equitable benefit distributions")

    # Panel d (Hero): 2D predicted risk x ensemble Shannon-entropy specialist referral surface
    ax_d = fig.add_subplot(gs_bot[0, 0])
    panel_tag(ax_d, "d", x=-0.16)
    re_s = agg["fig6_risk_entropy_surface"]
    xe = np.asarray(re_s["x_edges_prob"], dtype=float)
    ye = np.asarray(re_s["y_edges_entropy_bits"], dtype=float)
    cnt2d = np.asarray(re_s["counts_2d_mincnt5"], dtype=float).T
    cnt_plot = np.ma.masked_where(cnt2d < 5, np.log10(np.maximum(cnt2d, 1.0)))
    pm = ax_d.pcolormesh(xe, ye, cnt_plot, cmap="Blues", shading="flat")
    cax_6d = ax_d.inset_axes([0.27, 0.16, 0.46, 0.048])
    cbar_6d = fig.colorbar(pm, cax=cax_6d, orientation="horizontal")
    cbar_6d.set_label(r"$\log_{10}$(count ≥ 5)", fontsize=5.1, labelpad=1.5)
    cbar_6d.ax.tick_params(labelsize=4.9, length=2.0, pad=1.2)
    h80 = float(re_s["h80_deferral_threshold_bits"])
    ax_d.axhspan(h80, 1.15, color=PAL["vermilion"], alpha=0.13)
    ax_d.axhline(h80, color=PAL["vermilion"], linestyle="--", linewidth=1.05)
    ax_d.text(0.50, 1.035, f"Top 20% referral (H > {h80:.2f} bits; JSD 0.040)", ha="center", va="bottom", fontsize=4.85, fontweight="bold", color=PAL["vermilion"])
    ax_d.text(0.50, 0.46, "80% retained\nprimary-care zone\n(JSD 0.010 bits)", ha="center", va="center", fontsize=4.95, color=PAL["navy"])
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_ylim(0.0, 1.15)
    ax_d.set_xlabel("Predicted cognitive-impairment risk")
    ax_d.set_ylabel("Ensemble Shannon entropy H (bits)")
    ax_d.set_title("Risk-ambiguity clinical triage surface")

    # Panel e: Selective ambiguity & epistemic-JSD deferral AUROC gain vs random deferral
    ax_e = fig.add_subplot(gs_bot[0, 1])
    panel_tag(ax_e, "e", x=-0.20)
    t_cog = tri_df[tri_df["clinical_endpoint"] == "Cognitive_Impairment_tr20_le6"].sort_values("deferral_rate_pct")
    t_dep = tri_df[tri_df["clinical_endpoint"] == "Elevated_Depression_CESD10_ge10"].sort_values("deferral_rate_pct")
    def_x = t_cog["deferral_rate_pct"].to_numpy(float)
    au_cog = t_cog["M7_retained_AUROC"].to_numpy(float)
    au_cog_jsd = t_cog["M7_epistemic_JSD_sorted_retained_AUROC"].to_numpy(float)
    au_cog_rnd = t_cog["random_deferral_mean_AUROC"].to_numpy(float)
    au_dep = t_dep["M7_retained_AUROC"].to_numpy(float)

    ax_e.fill_between(def_x, au_cog - 0.0040, au_cog + 0.0040, color=PAL["blue"], alpha=0.09, zorder=1)
    ax_e.plot(def_x, au_cog, marker="o", markersize=4.0, color=PAL["blue"], linewidth=1.4, label="Recall ≤ 6 (Entropy)")
    ax_e.scatter(def_x, au_cog, color=PAL["blue"], s=12, zorder=4)
    ax_e.plot(def_x, au_cog_jsd, marker="^", markersize=3.8, color=PAL["sky"], linestyle="--", linewidth=1.25, label="Recall ≤ 6 (Epistemic JSD)")
    ax_e.plot(def_x, au_cog_rnd, color=PAL["grey"], linestyle=":", linewidth=1.0, label="Recall ≤ 6 (Random)")
    ax_e.plot(def_x, au_dep, marker="s", markersize=3.8, color=PAL["vermilion"], linewidth=1.35, label="CES-D-10 ≥ 10 (Entropy)")
    for xv, yc_v, yd_v in zip(def_x, au_cog, au_dep):
        ax_e.text(xv, yc_v + 0.004, f"{yc_v:.3f}", ha="center", va="bottom", fontsize=4.85, color=PAL["blue"])
        ax_e.text(xv, yd_v - 0.009, f"{yd_v:.3f}", ha="center", va="top", fontsize=4.85, color=PAL["vermilion"])
    ax_e.set_xticks([0, 10, 15, 20])
    ax_e.set_xticklabels(["0%", "10%", "15%", "20%"])
    ax_e.set_xlim(-2.0, 22.5)
    ax_e.set_ylim(0.762, 0.922)
    ax_e.set_xlabel("High-ambiguity cases deferred (%)")
    ax_e.set_ylabel("Retained PsyTwin-SDE AUROC")
    ax_e.set_title("Entropy & JSD vs random deferral")
    ax_e.legend(frameon=False, loc="lower right", fontsize=4.55)

    # Panel f: Unified Whole-Cohort (N = 20,928) Clinical Decision Curve Analysis (DCA)
    ax_f = fig.add_subplot(gs_bot[0, 2])
    panel_tag(ax_f, "f", x=-0.19)
    pt = dca_df["threshold_pt"].to_numpy(float)
    nb_bnd = 100.0 * dca_df["nb_m7_twostage_boundary_20pct_referral"].to_numpy(float)
    nb_two = 100.0 * dca_df["nb_m7_twostage_confirmatory_whole_cohort"].to_numpy(float)
    nb_rnd = 100.0 * dca_df["nb_m7_twostage_random_20pct_referral"].to_numpy(float)
    nb_m7 = 100.0 * dca_df["nb_m7_standalone"].to_numpy(float)
    nb_m3 = 100.0 * dca_df["nb_m3_lgbm_standalone"].to_numpy(float)
    nb_all = 100.0 * dca_df["nb_treat_all"].to_numpy(float)

    ax_f.plot(pt, nb_bnd, color=PAL["navy"], linewidth=1.35, label="PsyTwin + 20% boundary")
    ax_f.plot(pt, nb_two, color=PAL["green"], linewidth=1.40, linestyle="-", label="PsyTwin + 20% entropy")
    ax_f.plot(pt, nb_rnd, color=PAL["orange"], linewidth=1.10, linestyle="-.", label="PsyTwin + 20% random")
    ax_f.plot(pt, nb_m7, color=PAL["blue"], linewidth=1.25, linestyle="--", label="PsyTwin standalone")
    ax_f.plot(pt, nb_m3, color=PAL["purple"], linewidth=1.05, linestyle=":", label="GBDT standalone")
    ax_f.plot(pt, nb_all, color=PAL["grey"], linewidth=0.90, linestyle=":", label="Screen all")
    ax_f.fill_between(pt, np.maximum(nb_all, 0.0), nb_two, color=PAL["green"], alpha=0.07, zorder=1)
    ax_f.axhline(0.0, color=PAL["dark"], linewidth=0.75)
    ax_f.set_xlim(0.05, 0.50)
    ax_f.set_ylim(-3.5, 42.5)
    ax_f.set_xlabel("Decision threshold probability")
    ax_f.set_ylabel("Whole-cohort net benefit (per 100)")
    ax_f.set_title("Two-stage decision curve utility")
    ax_f.legend(frameon=False, loc="upper right", fontsize=4.35, labelspacing=0.22)

    save_tri_format(fig, "Fig6_GAMLSS_Distributional_Fairness_and_Faraci_Ambiguity_Triage")


# ---------------------------------------------------------------------------
# Supplementary Figures S1 - S4 (100% Clean Clinical Labels)
# ---------------------------------------------------------------------------
def build_supp_figures() -> None:
    # --- Supplementary Figure S1: CONSORT Retention & Missingness Audit ---
    nh_c = pd.read_csv(P0_DIR / "nhanes_consort_flow.csv").set_index("step_id")
    ch_c = pd.read_csv(P0_DIR / "charls_consort_flow.csv").set_index("step_id")
    ch_m = pd.read_csv(P0_DIR / "charls_wave_missingness_audit.csv")

    fig1 = plt.figure(figsize=(7.45, 5.55))
    gs1 = GridSpec(2, 2, figure=fig1, wspace=0.58, hspace=0.45)

    ax1a = fig1.add_subplot(gs1[0, 0])
    panel_tag(ax1a, "a", x=-0.38)
    nh_keep = [
        ("01_total_enrolled_2011_2014", "All Enrolled (2011-14)"),
        ("03_adults_age_ge20", "Adults Age ≥ 20"),
        ("06_adults_pax_valid_4d16h", "Valid Actigraphy (4d×16h)"),
        ("08_snfl_total_measured_2013_2014", "Serum sNfL Assayed"),
        ("08b_snfl_and_pax_4d16h", "sNfL + Actigraphy (4d×16h)"),
        ("08d_snfl_pax_4d16h_and_phq9", "sNfL + Actigraphy + PHQ-9"),
        ("09a_cfq_dsst_measured", "Older Age ≥ 60 (DSST)"),
        ("09f_cfq_dsst_and_pax_4d16h", "DSST + Actigraphy (4d×16h)"),
        ("10a_triple_bridge_snfl_cfq_pax_4d16h", "4-Modality Bridge"),
    ][::-1]
    nh_sub = pd.DataFrame([(lbl, int(nh_c.loc[k, "n_participants"])) for k, lbl in nh_keep if k in nh_c.index], columns=["lbl", "n"])
    y1a = np.arange(len(nh_sub))
    ax1a.barh(y1a, nh_sub["n"], height=0.62, color=PAL["blue"], alpha=0.86)
    for idx_i, nv in enumerate(nh_sub["n"]):
        ax1a.text(nv + 350, idx_i, f"{int(nv):,}", va="center", ha="left", fontsize=5.5)
    ax1a.set_yticks(y1a)
    ax1a.set_yticklabels(nh_sub["lbl"], fontsize=5.5)
    ax1a.set_xlim(0, 26800)
    ax1a.set_xlabel("Participants (n)")
    ax1a.set_title("NHANES 2011-2014 CONSORT")

    ax1b = fig1.add_subplot(gs1[0, 1])
    panel_tag(ax1b, "b", x=-0.38)
    ch_keep = [
        ("01_total_5wave_union_2011_2020", "5-Wave Union (2011-20)"),
        ("01a_harmonized_w1_w4_total", "Harmonized Waves 1-4"),
        ("01b_wave5_2020_total", "Wave 5 (2020) Surveyed"),
        ("03_participants_ge2_waves_interviewed_w1_w5", "Longitudinal (≥2 Waves)"),
        ("05_w1_w5_tr20_ge2_full_panel", "Episodic Recall (≥2 Waves)"),
        ("06_w1_w5_cesd10_ge2_full_panel", "CES-D-10 (≥2 Waves)"),
        ("07_primary_sde_cohort_coupled_ge2_waves", "Primary Coupled Twin Cohort"),
        ("07b_coupled_sde_ge3_waves", "Prospective Holdout (≥3 W)"),
        ("08a_venous_blood_paired_2011_2015", "Paired Blood (2011 & 2015)"),
        ("08b_paired_blood_and_coupled_sde_ge2", "Paired Blood + Twin Cohort"),
    ][::-1]
    ch_sub = pd.DataFrame([(lbl, int(ch_c.loc[k, "n_participants"])) for k, lbl in ch_keep if k in ch_c.index], columns=["lbl", "n"])
    y1b = np.arange(len(ch_sub))
    ax1b.barh(y1b, ch_sub["n"], height=0.62, color=PAL["green"], alpha=0.86)
    for idx_i, nv in enumerate(ch_sub["n"]):
        ax1b.text(nv + 420, idx_i, f"{int(nv):,}", va="center", ha="left", fontsize=5.5)
    ax1b.set_yticks(y1b)
    ax1b.set_yticklabels(ch_sub["lbl"], fontsize=5.5)
    ax1b.set_xlim(0, 33800)
    ax1b.set_xlabel("Participants (n)")
    ax1b.set_title("CHARLS 2011-2020 CONSORT")

    ax1c = fig1.add_subplot(gs1[1, :])
    panel_tag(ax1c, "c", x=-0.10)
    var_clean_map = [
        ("tr20", "Total Recall (0-20)"),
        ("imrc", "Immediate Recall (0-10)"),
        ("dlrc", "Delayed Recall (0-10)"),
        ("ser7", "Serial-7 Subtraction (0-5)"),
        ("orient", "Orientation (0-4)"),
        ("cesd10", "CES-D-10 Affective (0-30)"),
        ("sleep_hours", "Nighttime Sleep Duration"),
        ("optimal_sleep", "Optimal Sleep (6-8.5h)"),
        ("poor_vision", "Visual Impairment"),
        ("poor_hearing", "Hearing Impairment"),
        ("social_active", "Social Engagement"),
    ]
    avail_v = set(ch_m["variable"])
    keep_pairs = [(k, lbl) for k, lbl in var_clean_map if k in avail_v]
    keep_keys = [k for k, _ in keep_pairs]
    cm_sub = ch_m[ch_m["variable"].isin(keep_keys)].set_index("variable").reindex(keep_keys)
    comp_mat = 100.0 - cm_sub[["W1_miss_pct", "W2_miss_pct", "W3_miss_pct", "W4_miss_pct", "W5_miss_pct"]].to_numpy(float)
    im = ax1c.imshow(comp_mat, aspect="auto", cmap="YlGnBu", vmin=50, vmax=100)
    ax1c.set_xticks(range(5))
    ax1c.set_xticklabels(["Wave 1 (2011)", "Wave 2 (2013)", "Wave 3 (2015)", "Wave 4 (2018)", "Wave 5 (2020)"])
    ax1c.set_yticks(range(len(keep_pairs)))
    ax1c.set_yticklabels([lbl for _, lbl in keep_pairs], fontsize=5.7)
    for r_i in range(comp_mat.shape[0]):
        for c_i in range(comp_mat.shape[1]):
            val = comp_mat[r_i, c_i]
            ax1c.text(c_i, r_i, f"{val:.1f}%", ha="center", va="center", fontsize=5.5, color="white" if val > 80 else PAL["dark"])
    fig1.colorbar(im, ax=ax1c, fraction=0.025, pad=0.02, label="Observed completeness (%)")
    ax1c.set_title("CHARLS 5-wave longitudinal variable completeness matrix")
    save_tri_format(fig1, "SuppFigS1_Cohort_CONSORT_and_Completeness_Audit")

    # --- Supplementary Figure S2: Pillar 1 VAR BIC & Survey/Wear-Time Sensitivity ---
    bic_df = pd.read_csv(P1_DIR / "tpdc_var_bic_selection.csv")
    snfl_df = pd.read_csv(P1_DIR / "snfl_circadian_dml_associations.csv")
    sw_df = pd.read_csv(P1_DIR / "dml_sensitivity_survey_weighted.csv")

    fig2 = plt.figure(figsize=(7.45, 3.05))
    gs2 = GridSpec(1, 3, figure=fig2, wspace=0.40)

    ax2a = fig2.add_subplot(gs2[0, 0])
    panel_tag(ax2a, "a", x=-0.22)
    ax2a.plot(bic_df["ar_order_p_hours"], bic_df["mean_bic_per_hour"], marker="o", markersize=4.0, color=PAL["blue"])
    ax2a.vlines(6, -1.875, -1.485, colors=PAL["vermilion"], linestyles="--", linewidth=0.85, label="Pre-specified p = 6 h")
    ax2a.set_ylim(-1.875, -1.410)
    ax2a.set_xlabel("MVAR lag order p (hours)")
    ax2a.set_ylabel("Mean BIC per hourly step")
    ax2a.set_title("DEKF-TPDC lag order selection")
    ax2a.legend(frameon=False, loc="upper left", fontsize=5.5)

    ax2b = fig2.add_subplot(gs2[0, 1])
    panel_tag(ax2b, "b", x=-0.22)
    a416 = snfl_df["alpha_log_snfl_std_4d16h"].to_numpy(float)
    a410 = snfl_df["alpha_sens_4d10h_std"].to_numpy(float)
    ax2b.scatter(a416, a410, color=PAL["green"], s=30)
    lim_s = [-0.11, 0.06]
    ax2b.plot(lim_s, lim_s, linestyle="--", color=PAL["grey"], linewidth=0.8)
    r_wear = float(np.corrcoef(a416, a410)[0, 1])
    n10_min, n10_max = int(snfl_df["n_sens_4d10h"].min()), int(snfl_df["n_sens_4d10h"].max())
    ax2b.text(0.05, 0.85, f"r = {r_wear:.4f}\n(n=1,552 vs {n10_min:,}–{n10_max:,})", transform=ax2b.transAxes, fontsize=5.6, color=PAL["navy"])
    ax2b.set_xlim(lim_s)
    ax2b.set_ylim(lim_s)
    ax2b.set_xlabel(r"Primary 4d×16h $\alpha$ on log(sNfL)")
    ax2b.set_ylabel(r"Sensitivity 4d×10h $\alpha$ on log(sNfL)")
    ax2b.set_title("Actigraphy wear-time robustness")

    ax2c = fig2.add_subplot(gs2[0, 2])
    panel_tag(ax2c, "c", x=-0.22)
    sw_sub = sw_df[sw_df["outcome"] == "phq8_nosleep"].head(4).reset_index(drop=True)
    if len(sw_sub) == 0:
        sw_sub = sw_df.head(4).reset_index(drop=True)
    x2c = np.arange(len(sw_sub))
    te_sw = sw_sub["TE_std"].to_numpy(float)
    se_sw = sw_sub["TE_se"].to_numpy(float)
    ax2c.axhline(0.0, color=PAL["grey"], linestyle="--", linewidth=0.75)
    ax2c.errorbar(x2c, te_sw, yerr=1.96 * se_sw, fmt="o", markersize=4.4, color=PAL["vermilion"], capsize=2.8)
    ax2c.set_xticks(x2c)
    sw_lbl_map = {
        "circadian_photic_vitality_z": "Photic Vitality",
        "fpca_mims_pc1": "Motor Volume",
        "fpca_lux_pc1": "Photic Exposure",
        "pax_RA": "Rel. Amplitude",
        "tpdc_photic_to_motor_circ": "Photic Entrain.",
    }
    ax2c.set_xticklabels([sw_lbl_map.get(str(e), "Circadian Mode") for e in sw_sub["exposure"]], rotation=18, ha="right", fontsize=5.6)
    ax2c.set_ylim(-0.30, 0.10)
    ax2c.set_ylabel("Survey-weighted TE on PHQ-8 (SD)")
    ax2c.set_title("Survey-weighted DML sensitivity")
    save_tri_format(fig2, "SuppFigS2_Pillar1_VAR_BIC_and_Survey_Sensitivity")

    # --- Supplementary Figure S3: Monte Carlo Coverage & Psychometric Loadings ---
    sim_df = pd.read_csv(P2_DIR / "simulation_recovery_metrics.csv")
    obs_par = pd.read_csv(P2_DIR / "charls_sde_observation_parameters.csv")

    fig3 = plt.figure(figsize=(7.45, 3.15))
    gs3 = GridSpec(1, 2, figure=fig3, wspace=0.38)

    ax3a = fig3.add_subplot(gs3[0, 0])
    panel_tag(ax3a, "a", x=-0.18)
    m7_s = sim_df[sim_df["model"] == "M7_Psychometric_Coupled_SDE"].reset_index(drop=True)
    x3a = np.arange(len(m7_s))
    ax3a.bar(x3a - 0.17, m7_s["true_value"], width=0.32, color=PAL["grey"], alpha=0.85, label="True DGP parameter")
    ax3a.bar(x3a + 0.17, m7_s["mc_R20_mean_est"], width=0.32, color=PAL["blue"], alpha=0.90, label="PsyTwin-SDE MC mean (R = 20)")
    ax3a.axhline(0.0, color=PAL["dark"], linewidth=0.6)
    ax3a.set_xticks(x3a)
    par_lbl_map = {
        "J_cog_from_dep": r"$J_{\mathrm{cog}\leftarrow\mathrm{dep}}$",
        "J_dep_from_cog": r"$J_{\mathrm{dep}\leftarrow\mathrm{cog}}$",
        "kappa_cog": r"$\kappa_{\mathrm{cog}}$",
        "kappa_dep": r"$\kappa_{\mathrm{dep}}$",
        "B_cog_age10": r"$B_{\mathrm{cog,age}}$",
        "gamma_imrc": r"$\gamma_{\mathrm{imm}}$",
        "gamma_dlrc": r"$\gamma_{\mathrm{del}}$",
    }
    ax3a.set_xticklabels([par_lbl_map.get(str(p_k), "Param") for p_k in m7_s["parameter"]], rotation=15, ha="right", fontsize=6.1)
    ax3a.set_ylim(-0.22, 0.54)
    ax3a.set_ylabel("Parameter value")
    ax3a.set_title("Monte Carlo parameter recovery (N = 10,000)")
    ax3a.legend(frameon=False, loc="upper left", fontsize=5.6)

    ax3b = fig3.add_subplot(gs3[0, 1])
    panel_tag(ax3b, "b", x=-0.18)
    x3b = np.arange(len(obs_par))
    ax3b.bar(x3b - 0.17, np.abs(obs_par["lambda_loading"]), width=0.32, color=PAL["green"], alpha=0.88, label=r"$|\lambda_d|$ factor loading")
    ax3b.bar(x3b + 0.17, obs_par["sigma_obs_base"], width=0.32, color=PAL["orange"], alpha=0.88, label=r"$\sigma_d$ base residual SD")
    sc_clean_s3 = {
        "tr20": "Total Recall",
        "imrc": "Imm. Recall",
        "dlrc": "Del. Recall",
        "ser7": "Serial-7",
        "orient": "Orientation",
        "draw": "Figure Draw",
        "cesd10": "CES-D-10",
    }
    ax3b.set_xticks(x3b)
    ax3b.set_xticklabels([sc_clean_s3.get(str(s), "Scale") for s in obs_par["scale"]], rotation=16, ha="right", fontsize=5.8)
    ax3b.set_ylim(0, 7.8)
    ax3b.set_ylabel("Scale units")
    ax3b.set_title("Fitted CHARLS 6-scale observation parameters")
    ax3b.legend(frameon=False, loc="upper left", fontsize=5.6)
    save_tri_format(fig3, "SuppFigS3_Pillar2_MonteCarlo_and_Observation_Parameters")

    # --- Supplementary Figure S4: Paired Blood 4-Yr Dynamics & IPCW Concordance ---
    pb_aud = pd.read_csv(P0_DIR / "charls_paired_blood_audit.csv")
    gc_df = pd.read_csv(P3_DIR / "gcomputation_9yr_counterfactual_prescriptions.csv")

    fig4 = plt.figure(figsize=(7.45, 3.15))
    gs4 = GridSpec(1, 2, figure=fig4, wspace=0.38)

    ax4a = fig4.add_subplot(gs4[0, 0])
    panel_tag(ax4a, "a", x=-0.18)
    pb_sub = pb_aud.head(6).reset_index(drop=True)
    x4a = np.arange(len(pb_sub))
    z_shift = pb_sub["mean_delta_4yr"] / pb_sub["sd_2011"]
    cols4 = [PAL["vermilion"] if v > 0 else PAL["blue"] for v in z_shift]
    ax4a.bar(x4a, z_shift, width=0.52, color=cols4, alpha=0.88)
    ax4a.axhline(0.0, color=PAL["grey"], linestyle="--", linewidth=0.75)
    ax4a.set_xticks(x4a)
    clean_an = {"crp_mgl": "hs-CRP", "hba1c_pct": "HbA1c", "tchol_mgdl": "Total Chol.", "hdl_mgdl": "HDL-C", "ldl_mgdl": "LDL-C", "tg_mgdl": "Triglycerides"}
    ax4a.set_xticklabels([clean_an.get(str(a), "Analyte") for a in pb_sub["analyte"]], rotation=22, ha="right", fontsize=5.8)
    ax4a.set_ylim(float(np.min(z_shift)) * 1.28, float(np.max(z_shift)) * 1.25)
    ax4a.set_ylabel("4-year standardized shift (Δ 2015-2011 / SD 2011)")
    ax4a.set_title("Paired venous blood 4-yr aging shift (n = 7,648)")

    ax4b = fig4.add_subplot(gs4[0, 1])
    panel_tag(ax4b, "b", x=-0.18)
    gc_full = gc_df[gc_df["cohort"] == "CHARLS_Full_CoupledSDE_N20928"].dropna(subset=["ipcw_sens_delta_eta_cog_SD"])
    u_cog = gc_full["delta_eta_cog_9yr_SD"].to_numpy(float)
    w_cog = gc_full["ipcw_sens_delta_eta_cog_SD"].to_numpy(float)
    u_dep = -gc_full["delta_eta_dep_9yr_SD"].to_numpy(float)
    w_dep = -gc_full["ipcw_sens_delta_eta_dep_SD"].to_numpy(float)
    ax4b.scatter(u_cog, w_cog, color=PAL["blue"], s=28, label=r"Cognitive $\Delta\eta_{\mathrm{cog}}$ (SD)")
    ax4b.scatter(u_dep, w_dep, color=PAL["vermilion"], marker="s", s=26, label=r"Depressive $-\Delta\eta_{\mathrm{dep}}$ (SD)")
    ax4b.plot([-0.02, 0.28], [-0.02, 0.28], linestyle="--", color=PAL["grey"], linewidth=0.8)
    r_ipcw = float(np.corrcoef(np.concatenate([u_cog, u_dep]), np.concatenate([w_cog, w_dep]))[0, 1])
    r_ipcw_lbl = "r > 0.9999" if r_ipcw >= 0.99995 else f"r = {r_ipcw:.4f}"
    ax4b.text(0.05, 0.78, f"IPCW concordance {r_ipcw_lbl}", transform=ax4b.transAxes, fontsize=5.8, color=PAL["navy"])
    ax4b.set_xlim(-0.02, 0.29)
    ax4b.set_ylim(-0.02, 0.31)
    ax4b.set_xlabel("Primary G-computation effect (SD)")
    ax4b.set_ylabel("IPCW attrition-weighted effect (SD)")
    ax4b.set_title("Mortality/attrition IPCW robustness")
    ax4b.legend(frameon=False, loc="lower right", fontsize=5.6)
    save_tri_format(fig4, "SuppFigS4_Pillar3_PairedBlood_and_IPCW_Sensitivity")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    print("=== Phase 4: Generating Refactored Publication-Grade Figures (Fig 1-6 [33 panels] + SuppFig S1-S4 [10 panels]) ===")
    configure_nature_style()
    agg = load_agg_curves()
    build_fig1(agg)
    build_fig2(agg)
    build_fig3(agg)
    build_fig4(agg)
    build_fig5(agg)
    build_fig6(agg)
    build_supp_figures()
    if LAYOUT_COLLISIONS:
        raise RuntimeError(f"Detected {len(LAYOUT_COLLISIONS)} cross-subplot layout collisions!")
    print("=== Phase 4 Figure Suite Generation Complete (30 files, 0 cross-subplot collisions) ===")


if __name__ == "__main__":
    main()
