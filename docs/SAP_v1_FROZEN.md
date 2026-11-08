# FROZEN STATISTICAL & ALGORITHMIC ANALYSIS PLAN (SAP v1.0 — FROZEN)
## PsyTwin-SDE: Brain-Aging Psychometric-Aware Cross-Scale Digital Twin

- **Document ID**: `SAP_v1_FROZEN`
- **Freeze Timestamp**: `2026-10-30` (Frozen immediately upon `13/13 PASS` of `scripts/phase0_independent_audit_gate.py` prior to any Phase 2–3 outcome modeling)
- **Target Journal & Collection**: *npj Digital Medicine* — Guest-Edited Collection: *Integrative Multimodal Models for Brain and Health: Use of AI for Real-World Impact* (`https://www.nature.com/collections/bfccgegdjf`, deadline `2027-05-07`)
- **Governance & Scope Lock (`CLAUDE.md` & `lab-constitution`)**:
  - **Raw Data Access**: `STRICT_READ_ONLY` on `${NHANES_RAW_DIR}` (`1,599` files untouched) and `${CHARLS_RAW_DIR}` (`141` files untouched).
  - **Hard Exclusions**: ZERO brain–kidney axis / AKI endpoints (serum creatinine and cystatin C enter strictly as physiological clearance / systemic biological aging covariates, never as outcomes or organ-axis framing); ZERO raw DICOM imaging; ZERO overlap with `external acute-care cohorts` or `external neurological cohorts`.

---

## 1. Empirical Phase 0 Grounding & Frozen Analytic Cohorts

All sample sizes, censoring bounds, and variable distributions below are locked from reproducible Phase 0 execution (`scripts/phase0_nhanes_extract_and_qc.py` and `scripts/phase0_charls_extract_and_qc.py`, verified in `results/phase0_cohort_audit/PHASE0_VERIFICATION_REPORT.md`).

### 1.1 Cohort 1 — NHANES Cross-Scale Circadian Actigraphy, Serum `sNfL` & Zero-Ceiling 3-Domain Cognition (2011–2014)
- **Source Population**: `N = 19,931` enrolled (`2011–2012`: `9,756`; `2013–2014`: `10,175`); `N = 11,329` adults aged $\ge 20$ years (`10,907` MEC-examined).
- **Actigraphy & Ambient Light Resolution (`PAXHR_G` + `PAXHR_H`)**: `2,785,039` official CDC hourly wrist triaxial Monitor-Independent Movement Summary (`MIMS/min`), wrist ambient light (`lux/min`), and wake/sleep state epochs reconstructed to exact wall-clock hours $h \in \{0,\dots,23\}$ via `PAXHD.PAXFTIME + PAXHR.PAXSSNHP / 80 Hz` ($\Delta\tau = 1\text{ h}$, sampling rate $f_s = 1\text{ h}^{-1}$, Nyquist frequency $0.5\text{ cycles/h} = 12\text{ cycles/day}$). Among primary QC-passing adults ($\ge 4$ valid days with $\ge 16\text{ h/day}$ wear and full 24-hour clock coverage, `N = 8,148`), median valid recording length is `183.0` hours [IQR `175.0, 187.0`] across `6.25 ± 1.00` valid days.
- **Pre-Specified NHANES Analytic Subcohorts**:
  1. **Arm A (24h Circadian–Photic Manifold $\to$ Affective Load)**: `N = 8,148` adults ($\ge 20$ y) passing primary 4d16h `PAX` QC; `N = 7,506` with complete `PHQ-9` (`N = 7,508` with sleep-item-stripped `PHQ-8_nosleep`).
  2. **Arm B (24h Circadian–Photic Manifold $\to$ Serum `sNfL` Neuroaxonal Injury Mediation, 2013–2014)**: `N = 2,071` adults (ages 20–75 y) with measured serum `sNfL` (`SSSNFL_H`, median `12.30 pg/mL` [IQR `8.20, 19.10`]; ages 60–75 median `19.25 pg/mL` [IQR `14.50, 28.53`]); `N = 1,552` meeting primary 4d16h `PAX` QC (`N = 1,708` sensitivity at $\ge 4$d10h wear); `N = 1,467` with complete `PAX` + `sNfL` + `PHQ-9`.
  3. **Arm C (24h Circadian–Photic Manifold $\to$ Zero-Ceiling 3-Domain Cognition, 2011–2014 Ages $\ge 60$ y)**: `N = 3,014` with `DSST` processing speed (**`0.000%` ceiling**, `0.133%` floor, skewness `0.017`), `N = 3,110` with `Animal Fluency` (**`0.000%` ceiling**, `0.000%` floor), `N = 3,131` with `CERAD` 3-trial immediate learning (`0.096%` ceiling), `N = 3,126` with `CERAD` delayed recall, and `N = 2,934` with complete 3-domain composite (`global_cognition_z`); `N = 2,541` with valid 4d16h `PAX` + `DSST` (`N = 2,438` complete 3-domain + `PAX` + `PHQ-9`).
  4. **Arm D (Direct 4-Modality Individual-Level Cross-Scale Bridge, 2013–2014 Ages 60–75 y)**: `N = 506` with simultaneous `sNfL` + `DSST`; **`N = 419`** with simultaneous `valid 4d16h PAX + sNfL + DSST`; **`N = 410`** with all four modalities complete (`PAX + sNfL + 3-domain cognition + PHQ-9`).

### 1.2 Cohort 2 — CHARLS 9-Year 5-Wave Longitudinal Cognitive–Affective & Paired Venous Blood Cohort (2011–2020)
- **Source Population**: `N = 25,873` unique participants (`96,628` longitudinal person-wave interviews across Wave 1 [2011, `n = 17,708`], Wave 2 [2013, `n = 18,612`], Wave 3 [2015, `n = 21,097`], Wave 4 [2018, `n = 19,816`], and Wave 5 [2020, `n = 19,395`, harmonized to 100% Gateway to Global Aging Data `H_CHARLS_D_do_file.do` mathematical parity]).
- **Empirical Verification of the `Lin & Grewal 2026` Psychometric Fidelity Barrier in CHARLS**:
  - Severe upper-boundary ceiling censoring on coarse mental-status subtests: `orient` (`0–4`): **`43.32%` ceiling** (`49.78%` in W2); `ser7` (`0–5`): **`41.78%` ceiling** (`43.08%` in W1); `draw` (`0–1`): **`100.0%` binary boundary** (`58.20%` at `1`, `41.80%` at `0`); `mental_intact_10` (`0–10`): **`20.32%` ceiling**.
  - Substantial lower-boundary floor censoring on memory recall and depression: `dlrc` (`0–10`): **`16.17%` floor** (`21.93%` in W3); `imrc` (`0–10`): **`7.06%` floor**; `tr20` (`0–20`): **`4.53%` floor** (`0.05%` ceiling); `CES-D10` (`0–30`): **`7.88%` floor** (`0.16%` ceiling).
  - Significant logarithmic retest practice effect on episodic memory `tr20` across Waves 2–5 (`\beta_{\log(1+k)} = +0.2958` words, `SE = 0.0385`, $p = 1.51 \times 10^{-14}$, adjusted for quadratic age, sex, education, rurality, and calendar wave), identifiable because refreshment entrants ($k = 0$) and repeat examinees ($k \in \{1,\dots,4\}$) are observed concurrently at the same chronological ages.
- **Pre-Specified CHARLS Analytic Subcohorts**:
  1. **Primary Coupled SDE Digital Twin Cohort**: **`N = 20,928`** participants with $\ge 2$ waves of joint `(tr20, CES-D10)` (`N = 19,212` aged $\ge 45$ y at baseline; **`N = 17,360`** with $\ge 3$ coupled waves for multi-horizon out-of-time trajectory forecasting).
  2. **Longitudinal Paired Venous Blood Mechanistic & Drift-Modulation Subcohort**: **`N = 7,648`** participants with paired 2011 (Wave 1) and 2015 (Wave 3) venous blood panels (`hs-CRP`, `HbA1c`, `TC`, `HDL`, `LDL`, `TG`, `fasting glucose`, `uric acid`, `cystatin C`, `CBC`; **`N = 7,438`** overlapping the Primary Coupled SDE Cohort; `N = 17,619` single- or dual-wave blood union).

### 1.3 Pre-Phase-2 Gap-Check Hardening Lock (`查漏补缺` — Locked Prior to Phase 2 Modeling)
- **Extended Confounder & Wave-5 Parity Completion**:
  - `NHANES`: Locked `BPQ_G/H` (`hypertension_doc`, `bp_med_current`, `high_chol_doc`), `ALQ_G/H` (`alcohol_12yr`), `PAQ_G/H` (`modvig_pa_any`, `leisure_modvig_pa`, `sedentary_min_day`), and `DLQ_H`/`AUQ_G` (`hearing_difficulty`, `vision_difficulty`) into $\mathbf{W}_i$.
  - `CHARLS`: Completed full 5-wave parity (`2011–2020`) for `smoken`, `smokev`, `drinkl`, `unmarried_or_widowed` (`r1..r4mstat` + W5 `ba011`), `log_hh_cperc`, cumulative chronic comorbidities (`hibpe, diabe, hearte, stroke, dyslipe, memrye` carrying forward prior diagnoses unless disputed in W5 `da002_k_ == 99`), and functional limitation indices (`adla_c` `0–4`, `adlab_c` `0–6`, `iadlza` `0–5` from W5 `db001..db022`).
- **Competing Mortality & Longitudinal Attrition (`IPCW/IPAW` Sensitivity)**:
  - Extracted longitudinal death/exit ascertainment (`died_followup` and `first_death_wave` from `r2..r4iwstat in {5,6}` and Wave 5 `Sample_Infor.dta` `died == 1`) to fit stabilized Inverse-Probability-of-Continuation/Attrition Weights (`IPCW`) in Pillar 2 & Pillar 3 sensitivity analyses, ruling out healthy-survivor selection bias across the 9-year window.

---

## 2. Pillar 1 Specification — Wearable Circadian–Ultradian Manifold (`fPCA-TPDC`) & `sNfL` Double Machine Learning Causal Mediation (`NHANES`)

### 2.1 Functional Manifold Representation (`fPCA`) of 24h Triaxial Activity & Ambient Light
For each participant $i$ passing primary 4d16h `PAX` QC (`N = 8,148`), let $X_i^{\text{MIMS}}(h) = \log(1 + \overline{\text{MIMS}}_{i,h})$ and $X_i^{\text{Lux}}(h) = \log(1 + \overline{\text{Lux}}_{i,h})$ denote the 24-hour circadian profiles over $h \in [0, 24)$. Expanding onto an orthonormal periodic Fourier basis $\{\phi_m(h)\}_{m=1}^{M}$ ($M = 9$, capturing up to 4 harmonics/day) with roughness penalty $\lambda \int [X''(h)]^2 dh$ selected by generalized cross-validation yields Karhunen–Loève functional principal components:
$$X_i^{(c)}(h) = \mu^{(c)}(h) + \sum_{k=1}^{K} \xi_{i,k}^{(c)} \psi_k^{(c)}(h), \qquad c \in \{\text{MIMS}, \text{Lux}\}, \quad K = 4$$
alongside canonical non-parametric rhythm metrics (`RA`, `IS`, `IV`, `M10`, `L5`, sleep regularity, and cosinor `MESOR`/amplitude/acrophase).

### 2.2 Time-Resolved Partial Directed Coherence (`DEKF-TPDC`) on Contiguous Multi-Day Hourly Series
Adapting Guest Editor Muthuraman's `TPDC` framework (`PMC12602468`) to each participant's contiguous multi-day hourly bivariate time series $\mathbf{z}_{i,\tau} = [\tilde{x}_{i,\tau}^{\text{Lux}}, \tilde{x}_{i,\tau}^{\text{MIMS}}]^\top$ ($\tau = 1,\dots,T_i$, median $T_i = 183$ h, linear interpolation only across isolated 1–2 h non-wear gaps; segments $< 72$ contiguous hours excluded in sensitivity):
1. Estimate time-varying bivariate autoregressive coefficients $\mathbf{A}_r(\tau) \in \mathbb{R}^{2 \times 2}$ ($r = 1,\dots,p$, order $p = 6$ h selected by BIC covering ultradian lags) via a **Dual Extended Kalman Filter (DEKF)** tracking state and parameter evolution across the 7-day recording.
2. Transform $\mathbf{A}(\omega, \tau) = \mathbf{I}_2 - \sum_{r=1}^{p} \mathbf{A}_r(\tau) e^{-\mathrm{i} 2\pi \omega r}$ across normalized frequencies $\omega \in (0, 0.5]\text{ h}^{-1}$ to obtain **Time-Resolved Partial Directed Coherence** from channel $j$ to channel $l$:
   $$\pi_{l \leftarrow j}(\omega, \tau) = \frac{|A_{lj}(\omega, \tau)|}{\sqrt{\sum_{m=1}^{2} |A_{mj}(\omega, \tau)|^2}}$$
3. Aggregate participant-level directed oscillatory biomarkers across two pre-specified physiological frequency bands:
   - **Circadian Photic Entrainment Band** ($\Omega_{\text{circ}} = [1/28, 1/20]\text{ h}^{-1}$): mean directed photic-to-motor entrainment $\overline{\pi}_{\text{MIMS} \leftarrow \text{Lux}}(\Omega_{\text{circ}})$ and temporal entrainment instability $\text{SD}_\tau(\pi_{\text{MIMS} \leftarrow \text{Lux}}(\Omega_{\text{circ}}, \tau))$.
   - **Ultradian Fragmentation Band** ($\Omega_{\text{ultra}} = [1/12, 1/3]\text{ h}^{-1}$, 3–12 h periods): ultradian-to-circadian power ratio $R_{\text{ultra/circ}, i}$ and bidirectional photic–motor decoupling.

### 2.3 Cross-Fitted Double Machine Learning (DML) Causal Mediation via Serum `sNfL`
We test whether neuroaxonal injury ($M_i = \log \text{sNfL}_i$) mediates the effect of 24h circadian/photic breakdown ($A_i \in \{\xi_{i,1}^{\text{MIMS}}, \text{RA}_i, \text{IS}_i, \text{IV}_i, \overline{\pi}_{\text{MIMS} \leftarrow \text{Lux}}(\Omega_{\text{circ}})\}$) on:
- **Outcome Family Y1 (Affective Load, Arm B `N = 1,467`)**: `PHQ-9` total score and sleep-item-stripped `PHQ-8_nosleep` (primary unconfounded affective endpoint eliminating tautological overlap with actigraphic sleep!).
- **Outcome Family Y2 (Zero-Ceiling 3-Domain Cognition, Arm D Direct Quadruple Bridge `N = 419` / `410` + Two-Sample Moment-Coupled DML combining Arm B `N = 1,552` [$A \to M | \mathbf{W}$] and Arm C `N = 2,541` [$A \to Y | \mathbf{W}$])**: `DSST` processing speed (primary zero-ceiling cognitive endpoint), `Animal Fluency`, `CERAD` immediate/delayed recall, and `global_cognition_z`.

Confounder vector $\mathbf{W}_i$ (locked pre-exposure covariates): age (restricted cubic spline), sex, race/ethnicity, education, income-to-poverty ratio (`PIR`), `BMI`, systolic/diastolic BP, `HbA1c`, total/HDL cholesterol, systemic leukocyte inflammation (`WBC`, `NLR`), smoking, diabetes, cardiovascular/stroke history, NHANES survey cycle/stratum weights, and physiological serum creatinine clearance adjustment for `sNfL`.

Using 5-fold cross-fitted Neyman-orthogonal score functions (`Farbmacher et al., Econometrics J 2022`) with ensemble nuisance learners (` Ridge + Random Forest + LightGBM` stacked via cross-validated MSE), we estimate Total Effect ($\theta_{\text{TE}}$), Natural Indirect Effect via `sNfL` ($\theta_{\text{NIE}}$), Natural Direct Effect ($\theta_{\text{NDE}}$), mediated proportion, survey-weighted sensitivity (`WTSSNH2Y` / `WTMEC4YR`), and VanderWeele mediational **E-values**.

---

## 3. Pillar 2 Specification — Psychometric-Physics-Aware Coupled Cognitive–Affective State-Space Digital Twin (`CHARLS` 9-Year 5-Wave)

### 3.1 Latent Continuous-Time Coupled Neural-Drift SDE (Structural Layer)
For participant $i$ over continuous chronological time $t_{i,w} = \text{calendar\_year}_{i,w} - 2011.5$ ($w \in \{1,2,3,4,5\}$, spanning 9.1 years), let $\boldsymbol{\eta}_i(t) = [\eta_{i,\text{cog}}(t), \eta_{i,\text{dep}}(t)]^\top \in \mathbb{R}^2$ represent **true latent cognitive reserve** (higher = better) and **true latent affective load** (higher = worse depressive burden). Their continuous-time evolution obeys the coupled stochastic differential equation:
$$d\begin{bmatrix}\eta_{i,\text{cog}}(t) \\ \eta_{i,\text{dep}}(t)\end{bmatrix} = \left(\mathbf{J}_i \begin{bmatrix}\eta_{i,\text{cog}}(t) \\ \eta_{i,\text{dep}}(t)\end{bmatrix} + \mathbf{B}\mathbf{u}_i(t) + \mathbf{g}_{\phi}(\boldsymbol{\eta}_i(t), \mathbf{x}_i) + \mathbf{b}_i\right) dt + \boldsymbol{\Sigma}^{1/2} d\mathbf{W}_i(t)$$
where:
- $\mathbf{J}_i = \begin{bmatrix} -\kappa_{\text{cog}} & J_{\text{cog} \leftarrow \text{dep}, i} \\ J_{\text{dep} \leftarrow \text{cog}, i} & -\kappa_{\text{dep}} \end{bmatrix}$ is the **individualized directed coupling Jacobian**, parameterized via baseline & longitudinal biological/demographic moderators $\mathbf{z}_i$ (including paired venous blood inflammation/glycemic burden $\log\text{hs-CRP}, \text{HbA1c}, \Delta\log\text{hs-CRP}, \Delta\text{HbA1c}$ in the `N = 7,438` paired blood subcohort):
  $$J_{\text{cog} \leftarrow \text{dep}, i} = \alpha_{12,0} + \boldsymbol{\alpha}_{12}^\top \mathbf{z}_i, \qquad J_{\text{dep} \leftarrow \text{cog}, i} = \alpha_{21,0} + \boldsymbol{\alpha}_{21}^\top \mathbf{z}_i$$
- $\mathbf{u}_i(t)$ contains time-varying modifiable drivers (sleep adequacy, dual sensory impairment/compensation, social/digital engagement, physical activity, vascular-metabolic disease burden) and quadratic chronological aging drift.
- $\mathbf{g}_{\phi}$ is an $L_2$-regularized residual neural drift MLP ($\|\mathbf{g}_\phi\| \le \epsilon$, tested against $\mathbf{g}_\phi \equiv \mathbf{0}$ linear SDE ablation to avoid overfitting sparse 5-wave trajectories), $\mathbf{b}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega}_b)$ captures individual random drift/intercept heterogeneity, and $\boldsymbol{\Sigma}$ is the diffusion covariance.
- Exact analytical matrix-exponential discretization ($\mathbf{A}(\Delta t) = \exp(\mathbf{J}_i \Delta t)$) handles irregular inter-wave intervals ($\Delta t_{i,w} \in [1.5, 9.2]$ years) without Euler step-size error.

### 3.2 Psychometric-Error-Aware Observation Filter (Overcoming `Lin & Grewal 2026`)
Instead of treating raw bounded test scores $Y_{i,w,d} \in [L_d, U_d]$ as homoscedastic Gaussian readings, each observed scale $d \in \{\text{imrc }[0,10], \text{dlrc }[0,10], \text{ser7 }[0,5], \text{orient }[0,4], \text{draw }\{0,1\}, \text{cesd10 }[0,30]\}$ is generated through a **3-component Psychometric Observation Layer**:
1. **Logarithmic Retest Practice-Effect Offset**:
   $$r_{i,w,d} = \gamma_d \log(1 + k_{i,w,d})$$
   where $k_{i,w,d} \in \{0,1,2,3,4\}$ is the exact count of prior completed administrations for participant $i$ before wave $w$ (identifiable separately from aging via Waves 2–5 refreshment cohorts).
2. **Scale-Specific IRT Link & Heteroscedastic Measurement Variance**:
   $$Y_{i,w,d}^* = \nu_d + \lambda_d \eta_{i,\text{dom}(d)}(t_{i,w}) + \gamma_d \log(1 + k_{i,w,d}) + \epsilon_{i,w,d}, \qquad \epsilon_{i,w,d} \sim \mathcal{N}\!\left(0, \sigma_{d}^2 \exp(\delta_d^\top \mathbf{v}_{i,w})\right)$$
   where $\mathbf{v}_{i,w}$ includes illiteracy/education and interviewer sensory/interference flags (accounting for higher test noise in low-literacy examinees). For binary `draw` $\in \{0,1\}$, a 2-parameter Probit/Logistic IRT likelihood $P(Y_{i,w,\text{draw}}=1) = \Phi(\nu_{\text{draw}} + \lambda_{\text{draw}}\eta_{i,\text{cog}}(t_{i,w}) + \gamma_{\text{draw}}\log(1+k_{i,w}))$ is used.
3. **Two-Sided Tobit Boundary Censoring Likelihood**:
   For bounded integer/continuous scales $Y_{i,w,d} \in [L_d, U_d]$, the exact log-likelihood contribution at interior vs boundary observations is:
   $$\ell(Y_{i,w,d} \mid \mu_{i,w,d}, s_{i,w,d}) = \mathbb{I}(L_d < Y < U_d)\log\left[\frac{1}{s}\phi\!\left(\frac{Y-\mu}{s}\right)\right] + \mathbb{I}(Y \le L_d)\log\Phi\!\left(\frac{L_d + 0.5 - \mu}{s}\right) + \mathbb{I}(Y \ge U_d)\log\left[1 - \Phi\!\left(\frac{U_d - 0.5 - \mu}{s}\right)\right]$$
   incorporating the $\pm 0.5$ discrete continuity correction at boundaries $L_d$ and $U_d$. State filtering and parameter estimation use **Laplace / Gauss–Hermite Tobit-Kalman Marginal Maximum Likelihood & Variational Smoothing** across all observed subscales simultaneously.

### 3.3 Pre-Specified Simulation Recovery Proof & 6-Family Benchmark Protocol
1. **Simulation Identifiability Proof**: Generate $N = 10,000$ synthetic 5-wave trajectories with known ground-truth coupling $(J_{\text{cog}\leftarrow\text{dep}}^{\text{true}}, J_{\text{dep}\leftarrow\text{cog}}^{\text{true}})$, retest gains $\gamma_d^{\text{true}}$, and CHARLS-calibrated boundary censoring (`43.3%` `orient` ceiling, `41.8%` `ser7` ceiling, `16.2%` `dlrc` floor, `7.9%` `cesd10` floor). Quantify relative bias, RMSE, and 95% CI coverage of `Psychometric-Coupled-SDE` vs naïve uncensored SDE and LMM.
2. **Real-World Out-of-Sample & Out-of-Time Benchmarks (`CHARLS`, `N = 20,928` & `N = 17,360` $\ge 3$-wave subset)**:
   - **Split A (Community-Grouped 5-Fold Cross-Validation)**: Grouped strictly by `communityID` (`450` communities; zero household/spouse leakage), forecasting held-out final observed wave(s) from earlier waves.
   - **Split B (Multi-Horizon Temporal Holdout)**: Train on Waves 1–3 (`2011–2015`) $\to$ forecast Wave 4 (`2018`, +3 y) and Wave 5 (`2020`, +5 y); and train on Waves 1–4 (`2011–2018`) $\to$ forecast Wave 5 (`2020`, +2 y).
   - **External Cross-Cohort Construct Validation (`NHANES` Arm C `N = 3,014`)**: Apply the latent measurement model to `NHANES` contrasting zero-ceiling `DSST`/`Animal Fluency` vs bounded `PHQ-9` (`34.9%` floor).
   - **Benchmark Models (7 total)**: (M1) Persistence/Age-Normative; (M2) Linear Mixed-Effects (`LMM`); (M3) `LightGBM` autoregressive tabular GBDT (`Wan et al. 2026` baseline); (M4) `Naïve Neural/Latent ODE` (MSE loss, no Tobit/retest filter); (M5) `Uncensored Coupled SDE` (ablation of Tobit/IRT/retest filter); (M6) `Uncoupled Psychometric Tobit-SDE` ($\mathbf{J}_{12}=\mathbf{J}_{21}=0$ ablation); (M7) **`Psychometric-Coupled-SDE` (Proposed)**.
   - **Primary & Secondary Evaluation Metrics**: Censored Tobit Negative Log-Likelihood (`NLL`, co-primary), Continuous Ranked Probability Score (`CRPS`, co-primary), `AUROC` / `Brier` for incident clinically significant cognitive decline ($\Delta \eta_{\text{cog}} \le -0.5\text{ SD}$ or lowest quintile `tr20`/`global_cog_30`) and incident elevated depression (`CES-D10 >= 10`), 90% Prediction Interval empirical coverage, and domain-stratified `MAE` / Spearman $\rho$.

---

## 4. Pillar 3 Specification — Individualized Coupling Phenotypes, 9-Year Parametric Longitudinal `G-Computation` & `GAMLSS` Fairness Audit

### 4.1 Individualized Directed Jacobian Dominance Phenotyping
From each participant's empirical Bayes posterior trajectory and moderated Jacobian $\widehat{\mathbf{J}}_i$, compute the standardized directed coupling dominance index:
$$\mathcal{D}_i = \frac{|J_{\text{cog} \leftarrow \text{dep}, i}|\,\sigma_{\eta_{\text{dep}}} - |J_{\text{dep} \leftarrow \text{cog}, i}|\,\sigma_{\eta_{\text{cog}}}}{|J_{\text{cog} \leftarrow \text{dep}, i}|\,\sigma_{\eta_{\text{dep}}} + |J_{\text{dep} \leftarrow \text{cog}, i}|\,\sigma_{\eta_{\text{cog}}}} \in [-1, +1]$$
classifying participants into **(1) Affective-Driven Cognitive Decline** ($\mathcal{D}_i > +0.15$), **(2) Cognitive-Driven Affective Collapse** ($\mathcal{D}_i < -0.15$), and **(3) Bidirectional Coupled Equilibrium** ($|\mathcal{D}_i| \le 0.15$), and testing biological enrichment against 4-year `hs-CRP` and `HbA1c` surges in the `N = 7,438` paired venous blood cohort.

### 4.2 9-Year Parametric Longitudinal `G-Computation` Counterfactual Multi-Target Prescriptions
To avoid time-varying treatment–confounder feedback bias (where depressive load alters sleep and social activity, which subsequently alter cognition and future sleep), we implement **Parametric Longitudinal G-Computation** (`Robins 1986`) over the 9-year 5-wave horizon (`2011 -> 2020`, `N = 20,928` overall and `N = 7,438` paired-blood mechanistic subset) with `B = 500` cluster-bootstrap replicates (`communityID` resampling) comparing Observed Natural Course against 5 feasible modifiable regimes:
1. **Regime R1 (Sleep & Circadian Restoration)**: Shift abnormal nighttime sleep ($< 6.0\text{ h}$ or $> 8.5\text{ h}$, affecting `35.4%` at baseline) into optimal `6.5–8.0 h` + eliminate restless sleep fragmentation.
2. **Regime R2 (Dual Sensory Compensation)**: Compensate uncorrected poor vision (`30.9%` at baseline) and poor hearing (`10.8%`; dual sensory impairment `6.4%`).
3. **Regime R3 (Cardiometabolic–Inflammatory Control, Paired-Blood Subcohort `N = 7,438`)**: Maintain/reduce systemic inflammation (`hs-CRP <= 1.0 mg/L`) and glycemic load (`HbA1c < 5.7%`) + blood pressure control.
4. **Regime R4 (Social & Digital Cognitive Engagement)**: Sustain active social/community participation + structured digital/internet engagement (`7.6%` at baseline rising across W1–W5) + moderate physical activity.
5. **Regime R5 (Combined Multi-Target Precision Twin Prescription)**: Joint R1 + R2 + R3 + R4 vs Dominant-Driver-Matched Single-Domain Prescription (testing whether matching the intervention to the individual's Jacobian dominance phenotype $\mathcal{D}_i$ achieves disproportionate 9-year cognitive preservation and depression prevention per target modified!).

### 4.3 `GAMLSS` Distributional Fairness & Epistemic/Aleatoric Uncertainty Audit (`Faraci Alignment`)
Directly honoring Guest Editor Faraci's uncertainty and distributional equity paradigm (`PMC12816009`, `PMC11978946`), we fit **Generalized Additive Models for Location, Scale, and Shape (`GAMLSS`)** to both out-of-sample digital twin prediction residuals and 9-year counterfactual benefit distributions across 4 pre-specified equity axes:
- **Literacy / Educational Attainment**: `Illiterate / No Formal Schooling` (`24.3%` in CHARLS!) vs `Primary / Sishu` (`44.0%`) vs `Secondary or Higher` (`31.7%`); and `< High School` (`22.2%`) vs `>= High School` in NHANES.
- **Urban–Rural Divide**: `Rural Village` (`55.8%`) / `Agricultural Hukou` (`74.3%`) vs `Urban Community` (`44.2%`).
- **Sex**: `Female` (`51.8%`) vs `Male` (`48.2%`).
- **Age Decades**: `45–59`, `60–74`, `>= 75` years.

Specifically, we model both conditional location bias $\mu(\mathbf{s})$ and heteroscedastic uncertainty scale $\log \sigma(\mathbf{s})$ across strata $\mathbf{s}$, demonstrating quantitatively how naïve uncensored models inflate prediction error and misallocate counterfactual risk in rural/illiterate older adults due to boundary censoring and test-noise heteroscedasticity—and how the Psychometric-Aware Digital Twin restores cross-stratum calibration parity.

---

## 5. Multiplicity, Reproducibility & Integrity Gateways
1. **Multiple Testing Control**: Co-primary Pillar 1 (`sNfL` NIE on `DSST` and `PHQ-8_nosleep`) and Pillar 2 (`NLL` and `CRPS` improvement of `Psychometric-Coupled-SDE` over all 6 baselines) are controlled at family-wise $\alpha = 0.05$ via Holm–Bonferroni; secondary domain/biomarker panels use Benjamini–Hochberg `FDR q < 0.05`.
2. **100% Code-to-Artifact Traceability**: Every Phase 2 and Phase 3 table, statistic, p-value, and CI must be written by deterministic seeded scripts (`seed = 20270507`) in `scripts/` to `results/phase2_pillars1_2/` and `results/phase3_pillar3_fairness/`, audited by an independent verification script prior to Phase 4 figure generation and Phase 5 manuscript drafting.
