# PsyTwin-SDE: Psychometric-Physics-Aware Cross-Scale Brain-Aging Digital Twin

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.14.4](https://img.shields.io/badge/Python-3.14.4-3776AB.svg)](requirements.txt)

Official open-source implementation and reproducible verification suite for:

> **Psychometric filtering and neuroaxonal biomarker profiling guide equitable brain-aging trajectory modeling**  
> Jiangtao Sheng*, Xueer Liu, Runmin Lin, Qiaochu Sun, Xiaoxuan Chen, Kangsheng Li, and Weiqiang Chen*  
> *Shantou University Medical College & First Affiliated Hospital of Shantou University Medical College, Shantou, Guangdong, China*  
> Correspondence: `jtsheng@stu.edu.cn` (J.S.); `wqchen@stu.edu.cn` (W.C.)

---

## Overview

`PsyTwin-SDE` resolves two coupled barriers in preventive brain-aging digital medicine across two independent national cohorts—the **US National Health and Nutrition Examination Survey (NHANES 2011–2014; $N = 8,148$ valid 24-hour actigraphy)** and the **9-year 5-wave China Health and Retirement Longitudinal Study (CHARLS 2011–2020; $N = 20,928$ longitudinal trajectories)**:

1. **Cross-Scale Wearable Circadian-Photic Entrainment & Serum `sNfL` Causal Mediation (Pillar 1 — NHANES)**:
   - Penalized 9-basis orthonormal periodic Fourier functional principal component analysis (`fPCA`, GCV $\lambda = 0.56234$) capturing $>90\%$ of 24-hour wrist motor (`log[1+MIMS]`) and ambient photic (`log[1+Lux]`) variance ($N = 8,148$).
   - State-space dual extended Kalman filter time-varying partial directed coherence (`DEKF-TPDC`, $p = 6\text{ h}$, $\rho = 0.985$) isolating asymmetric ambient photic-to-motor entrainment ($0.6485$ vs $0.4829$ reverse feedback).
   - Five-fold cross-fitted Neyman-orthogonal double machine learning (`DML`) and two-sample moment-coupled causal mediation demonstrating biological pathway dissociation between `sNfL`-mediated neurocognitive protection ($14.5\%\text{--}33.0\%$ mediated across executive fluency, episodic memory, processing speed, and global cognition) and direct non-axonal depressive stabilization ($3.2\%$ `sNfL`-mediated, $P = 0.178$).
2. **Psychometric-Coupled Continuous-Time SDE Digital Twin (`PsyTwin-SDE`, Pillar 2 — CHARLS & Synthetic Proof)**:
   - Analytically integrated coupled cognitive-affective Itô stochastic differential equation solved across irregular multi-year survey gaps via exact Putzer $2\times 2$ spectral decomposition and Van Loan $4\times 4$ block-matrix exponentials (zero Euler discretization error).
   - Two-sided truncated-Gaussian Kalman-Tobit measurement update jointly correcting floor/ceiling boundary censoring (up to $46.95\%$), logarithmic retest practice gains, and illiteracy-heteroscedastic measurement noise across $N = 20,928$ participants ($119,923$ held-out evaluations across 450 communities, including $N = 7,648$ with paired 2011 & 2015 venous blood assays).
3. **Nine-Year Counterfactual G-Computation, Location-Scale GAMLSS Fairness & Ambiguity Triage (Pillar 3)**:
   - Mundlak within-between separated longitudinal parametric G-computation ($B = 500$ community-clustered bootstrap replicates) evaluating multidomain sleep, dual sensory, cardiometabolic-inflammatory (`hs-CRP`, `HbA1c`), and social-digital prescriptions (`R0`–`R7`).
   - Two-parameter Gaussian location-scale `GAMLSS` distributional fairness auditing across literacy, rural-urban, sex, and older-age strata alongside multi-model Shannon-entropy clinical ambiguity triage.

---

## Repository Structure

```text
PsyTwin-SDE/
├── README.md                          # Architecture, installation, and reproduction guide
├── LICENSE                            # MIT Open-Source License
├── CITATION.cff                       # GitHub / Zenodo machine-readable citation metadata
├── .zenodo.json                       # Zenodo archival deposit metadata (v1.0.0)
├── requirements.txt                   # Exact pinned Python package dependencies
├── environment.yml                    # Conda / Mamba reproducible environment specification
├── config/
│   └── DATA_MANIFEST.json             # Cohort schemas, module identifiers, and ID harmonization rules
├── docs/
│   └── SAP_v1_FROZEN.md               # Pre-specified Statistical Analysis Plan (SAP v1.0)
├── scripts/
│   ├── psychometric_sde_core.py       # Core Van Loan coupled SDE & 2D Gauss-Hermite Tobit-Kalman library
│   ├── phase0_nhanes_extract_and_qc.py
│   ├── phase0_charls_extract_and_qc.py
│   ├── phase0_independent_audit_gate.py
│   ├── phase2_pillar1_nhanes_fpca_tpdc_dml.py
│   ├── phase2_pillar2_simulation_recovery.py
│   ├── phase2_pillar2_charls_psychometric_sde.py
│   ├── phase2_independent_audit_gate.py
│   ├── phase3_pillar3_gcomp_gamlss_fairness.py
│   ├── phase3_independent_audit_gate.py
│   ├── phase4_generate_publication_figures.py
│   └── verify_release_integrity.py    # Instant 54-claim numerical verification against summary tables
└── results/                           # Aggregated summary CSV/JSON verification tables (0 individual rows)
    ├── phase0_cohort_audit/
    ├── phase2_pillar1_nhanes/
    ├── phase2_pillar2_sde/
    ├── phase3_pillar3_fairness/
    └── phase5_manuscript_audit/
        └── verified_bibliography_5field.csv
```

---

## System Requirements & Installation

- **Operating System**: Linux, macOS, or Windows (64-bit)
- **Python**: `3.14.4` (compatible with Python `>= 3.11`)
- **Hardware**: Standard multi-core CPU workstation (`>= 16 GB` RAM recommended for full 5-wave CHARLS + NHANES 24h actigraphy extraction; instant summary verification and figure generation require `< 2 GB` RAM and finish in `< 30 seconds`).
- **Deterministic Random Seed**: `20270507` across all Monte Carlo, cross-fitting, cross-validation, and cluster-bootstrap modules.

```bash
pip install -r requirements.txt
```

---

## Quick Start 1: Instant Numerical Verification & Figure Reproduction (< 30 s)

In strict compliance with CDC/NCHS and Peking University CHARLS data-use agreements, **no individual-level participant records** are redistributed in this repository. All aggregated summary tables (`results/`) are included so reviewers and readers can immediately verify all 54 core manuscript statistics and regenerate all 10 publication figures (`Figures 1–6` and `Supplementary Figures S1–S4`) without downloading raw survey archives:

```bash
# 1. Verify all 54 manuscript quantitative claims against aggregated CSV/JSON artifacts
python scripts/verify_release_integrity.py

# 2. Regenerate all 6 main figures and 4 supplementary figures (PNG 600 DPI + vector PDF/SVG)
python scripts/phase4_generate_publication_figures.py
```

---

## Quick Start 2: Full End-to-End Reproduction from Public Raw Cohorts

1. **Download Official Public Raw Datasets**:
   - **US NHANES (2011–2012 `G` and 2013–2014 `H` cycles)**: Download SAS XPORT (`.XPT`) modules (`DEMO`, `PAXHR`, `PAXDAY`, `SSSNFL_H`, `CFQ`, `DPQ`, `BMX`, `BPX`, `GHB`, `TCHOL`, `HDL`, `BIOPRO`, `CBC`, `SMQ`, `ALQ`, `DIQ`, `MCQ`) from CDC/NCHS: https://wwwn.cdc.gov/nchs/nhanes/
   - **CHARLS (Waves 1–5, 2011–2020)**: Register and download `Harmonized CHARLS` (`H_CHARLS_D_Data.dta`), `Wave 5 (2020) Health Status and Functioning`, `2011 Wave 1 Blood`, and `2015 Wave 3 Blood` from Peking University: https://charls.pku.edu.cn/
2. **Configure Raw Data Environment Variables and Execute Pipeline**:

```bash
export NHANES_RAW_DIR="/path/to/NHANES"
export CHARLS_RAW_DIR="/path/to/CHARLS"

# Phase 0: Cohort extraction, 11->12-char paired blood ID harmonization (N=7,648), & censoring audit
python scripts/phase0_nhanes_extract_and_qc.py
python scripts/phase0_charls_extract_and_qc.py
python scripts/phase0_independent_audit_gate.py

# Phase 2 Pillar 1: Penalized Fourier fPCA, state-space DEKF-TPDC, & Neyman-orthogonal DML mediation
python scripts/phase2_pillar1_nhanes_fpca_tpdc_dml.py

# Phase 2 Pillar 2: Synthetic identifiability proof (N=10,000, R=20 MC) & 7-model CHARLS SDE benchmark
python scripts/phase2_pillar2_simulation_recovery.py
python scripts/phase2_pillar2_charls_psychometric_sde.py
python scripts/phase2_independent_audit_gate.py

# Phase 3 Pillar 3: 9-yr Mundlak G-computation (B=500 bootstrap), GAMLSS fairness, & entropy triage
python scripts/phase3_pillar3_gcomp_gamlss_fairness.py
python scripts/phase3_independent_audit_gate.py

# Phase 4: Publication figure suite & final release parity gate
python scripts/phase4_generate_publication_figures.py
python scripts/verify_release_integrity.py
```

---

## License & Citation

Released under the [MIT License](LICENSE). If you use `PsyTwin-SDE` or its Kalman-Tobit psychometric observation wrapper in your research, please cite the accompanying manuscript (see [`CITATION.cff`](CITATION.cff)).
