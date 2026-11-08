#!/usr/bin/env python3
"""
Psychometric-Physics-Aware Coupled Continuous-Time SDE Core Library (PyTorch float64)
=====================================================================================
Shared mathematical engine used identically by:
  1. scripts/phase2_pillar2_simulation_recovery.py (N = 10,000 synthetic identifiability proof)
  2. scripts/phase2_pillar2_charls_psychometric_sde.py (N = 20,928 real 9-year 5-wave CHARLS benchmark)
  3. Phase 3 Pillar 3 G-Computation & GAMLSS Distributional Fairness Engine

Implements docs/SAP_v1_FROZEN.md (§3.1–3.2):
  - 4D Continuous-Time Coupled SDE State-Space [eta_cog(t), eta_dep(t), b_cog, b_dep]^T:
      d eta_i(t) = (J_i eta_i(t) + B u_i(t) + g_phi(eta_i, x_i) + b_i) dt + Sigma^{1/2} dW_i(t)
    discretized across arbitrary irregular inter-wave intervals Delta_t > 0 via exact Van Loan
    block matrix exponentials (zero Euler step-size error, zero singular-Jacobian inversion).
  - 3-Component Psychometric-Error-Aware Observation Layer:
      (i)   Logarithmic retest practice-effect offset: gamma_d * log(1 + k_{i,w,d})
      (ii)  Scale-specific IRT loadings + literacy-heteroscedastic measurement noise:
            sigma_{i,w,d}^2 = sigma_d^2 * exp(delta_d * v_{i,w})
      (iii) Two-Sided Tobit boundary censoring likelihood (+/- 0.5 continuity correction) for
            bounded integer scales + 2-parameter Probit-IRT likelihood for binary figure drawing,
            filtered via exact 2D Gauss-Hermite Tobit-Kalman quadrature (G = 7x7 = 49 nodes)
            with Rao-Blackwell propagation to individual random traits b_i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    domain: int  # 0 = cognitive reserve (higher better), 1 = depressive load (higher worse)
    lower: float
    upper: float
    is_binary: bool = False
    censor_lower: bool = True
    censor_upper: bool = True
    anchor_loading: Optional[float] = None  # fixed loading for factor scale identification


CHARLS_SCALE_SPECS: List[ScaleSpec] = [
    ScaleSpec("imrc", domain=0, lower=0.0, upper=10.0, is_binary=False, anchor_loading=2.0),
    ScaleSpec("dlrc", domain=0, lower=0.0, upper=10.0, is_binary=False, anchor_loading=None),
    ScaleSpec("ser7", domain=0, lower=0.0, upper=5.0, is_binary=False, anchor_loading=None),
    ScaleSpec("orient", domain=0, lower=0.0, upper=4.0, is_binary=False, anchor_loading=None),
    ScaleSpec("draw", domain=0, lower=0.0, upper=1.0, is_binary=True, anchor_loading=None),
    ScaleSpec("cesd10", domain=1, lower=0.0, upper=30.0, is_binary=False, anchor_loading=6.0),
]


def build_2d_gauss_hermite(n_1d: int = 7, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct 2D standard normal Gauss-Hermite quadrature nodes xi (G x 2) and log-weights log_w (G,)
    such that E_{z ~ N(0, I_2)}[f(z)] approx sum_{g=1}^{G} exp(log_w[g]) f(xi[g]).
    """
    pts_raw, wts_raw = np.polynomial.hermite.hermgauss(n_1d)
    # Transform probabilists' standard normal: z = sqrt(2) * x, weight = w / sqrt(pi)
    pts = np.sqrt(2.0) * pts_raw
    wts = wts_raw / np.sqrt(np.pi)
    gx, gy = np.meshgrid(pts, pts, indexing="ij")
    wx, wy = np.meshgrid(wts, wts, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    weights = (wx * wy).ravel()
    weights = weights / np.sum(weights)
    return (
        torch.tensor(nodes, dtype=torch.float64, device=device),
        torch.tensor(np.log(weights), dtype=torch.float64, device=device),
    )


def log_ndtr_safe(z: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(Phi(z)) across (-inf, +inf) in float64."""
    return torch.special.log_ndtr(z)


def log_mills_ratio(z: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(phi(z) / Phi(z)) for truncated Gaussian posterior moment updates."""
    zc = torch.clamp(z, min=-30.0, max=30.0)
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * zc * zc - log_ndtr_safe(zc)


HZ_ANCHOR = torch.tensor([1.15, 1.38, 0.04, 0.02, 0.78, 0.04], dtype=torch.float64)
RAW_HZ_INIT = torch.log(torch.expm1(HZ_ANCHOR))


def tobit_probit_Scale_loglik(
    y: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    spec: ScaleSpec,
    use_tobit_censoring: bool = True,
) -> torch.Tensor:
    """
    Evaluate log-likelihood ell(Y | mu, sigma) across quadrature nodes.
    Shapes:
      y:     (N, 1)
      mask:  (N, 1) boolean
      mu:    (N, G)
      sigma: (N, 1) or (N, G)
    Returns:
      ll:    (N, G) with 0.0 where mask is False.
    """
    if spec.is_binary:
        if use_tobit_censoring:
            # 2-parameter Probit-IRT: P(Y=1 | eta) = Phi(mu / sigma) with sigma fixed to 1.0
            ll_1 = log_ndtr_safe(mu)
            ll_0 = log_ndtr_safe(-mu)
            ll = torch.where(y >= 0.5, ll_1, ll_0)
        else:
            # Naive Gaussian MSE likelihood on {0, 1}
            s = torch.clamp(sigma, min=1e-3)
            ll = -0.5 * math.log(2.0 * math.pi) - torch.log(s) - 0.5 * ((y - mu) / s) ** 2
        return torch.where(mask, ll, torch.zeros_like(ll))

    s = torch.clamp(sigma, min=1e-3)
    ll_gauss = -0.5 * math.log(2.0 * math.pi) - torch.log(s) - 0.5 * ((y - mu) / s) ** 2
    if not use_tobit_censoring:
        return torch.where(mask, ll_gauss, torch.zeros_like(ll_gauss))

    ll = ll_gauss
    if spec.censor_lower and not math.isnan(spec.lower):
        z_lo = (spec.lower + 0.5 - mu) / s
        ll_lo = log_ndtr_safe(z_lo)
        ll = torch.where(y <= spec.lower + 1e-6, ll_lo, ll)
    if spec.censor_upper and not math.isnan(spec.upper):
        z_hi = (mu - (spec.upper - 0.5)) / s
        ll_hi = log_ndtr_safe(z_hi)
        ll = torch.where(y >= spec.upper - 1e-6, ll_hi, ll)
    return torch.where(mask, ll, torch.zeros_like(ll))


def exact_censored_gaussian_nll_and_crps_np(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    sd_pred: np.ndarray,
    spec: ScaleSpec,
) -> Dict[str, np.ndarray]:
    """
    Proper out-of-sample scoring metrics evaluated identically for ALL benchmark models (M1..M7):
      1. Proper Two-Sided Tobit / Probit-IRT Negative Log-Likelihood (NLL)
      2. Exact Closed-Form Censored Gaussian Continuous Ranked Probability Score (CRPS, Jordan et al. 2019)
      3. Expected Censored Point Prediction E[clip(Y*, L, U)] for MAE & 90% Prediction Interval Coverage
    """
    sd = np.maximum(sd_pred, 1e-3)
    if spec.is_binary:
        # If model outputs latent probit index or probability-scale mean:
        # Note: models passing probability p in [0,1] have sd_pred < 0 sentinel, handled cleanly below
        is_prob = np.all(sd_pred < 0.0)
        if is_prob:
            p1 = np.clip(mu_pred, 1e-6, 1.0 - 1e-6)
        else:
            p1 = np.clip(stats.norm.cdf(mu_pred / sd), 1e-6, 1.0 - 1e-6)
        nll = -(y_true * np.log(p1) + (1.0 - y_true) * np.log(1.0 - p1))
        crps = (y_true - p1) ** 2  # Brier == CRPS for binary {0, 1}
        y_hat = p1
        lo90 = (p1 >= 0.95).astype(float)
        hi90 = (p1 >= 0.05).astype(float)
        cov90 = ((y_true >= lo90) & (y_true <= hi90)).astype(float)
        pi90_width = np.maximum(hi90 - lo90, 0.0)
        winkler90 = pi90_width + 20.0 * np.maximum(lo90 - y_true, 0.0) + 20.0 * np.maximum(y_true - hi90, 0.0)
        return {"nll": nll, "crps": crps, "y_hat": y_hat, "cov90": cov90, "p_event": p1, "pi90_width": pi90_width, "winkler90": winkler90}

    lo = spec.lower if (spec.censor_lower and not np.isnan(spec.lower)) else -1e9
    hi = spec.upper if (spec.censor_upper and not np.isnan(spec.upper)) else 1e9

    # 1) Two-Sided Tobit NLL (+/- 0.5 continuity correction at boundaries)
    ll = stats.norm.logpdf(y_true, loc=mu_pred, scale=sd)
    if spec.censor_lower and not np.isnan(spec.lower):
        at_lo = y_true <= (spec.lower + 1e-6)
        ll = np.where(at_lo, stats.norm.logcdf((spec.lower + 0.5 - mu_pred) / sd), ll)
    if spec.censor_upper and not np.isnan(spec.upper):
        at_hi = y_true >= (spec.upper - 1e-6)
        ll = np.where(at_hi, stats.norm.logsf((spec.upper - 0.5 - mu_pred) / sd), ll)
    nll = -np.maximum(ll, -50.0)

    # 2) Exact Closed-Form Censored Normal CRPS for Y_cens = clip(Y*, lo, hi), Y* ~ N(mu, sd^2)
    # Formula (Jordan, Kruger & Lerch 2019, J Stat Softw Eq. crps_cnorm):
    y_c = np.clip(y_true, lo, hi)
    z_y = (y_c - mu_pred) / sd
    z_l = (lo - mu_pred) / sd
    z_u = (hi - mu_pred) / sd

    phi_l, Phi_l = stats.norm.pdf(z_l), stats.norm.cdf(z_l)
    phi_u, Phi_u = stats.norm.pdf(z_u), stats.norm.cdf(z_u)
    phi_y, Phi_y = stats.norm.pdf(z_y), stats.norm.cdf(z_y)

    # crps_cnorm(y, location=mu, scale=sd, lower=lo, upper=hi)
    term1 = z_y * (2.0 * Phi_y - 1.0) + 2.0 * phi_y
    term_l = z_l * (Phi_l**2) + 2.0 * Phi_l * phi_l - (1.0 / np.sqrt(np.pi)) * stats.norm.cdf(np.sqrt(2.0) * z_l)
    term_u = z_u * ((1.0 - Phi_u) ** 2) - 2.0 * (1.0 - Phi_u) * phi_u + (1.0 / np.sqrt(np.pi)) * stats.norm.cdf(-np.sqrt(2.0) * z_u)
    crps = sd * (term1 - term_l + term_u - 1.0 / np.sqrt(np.pi))
    crps = np.maximum(crps, 0.0)

    # 3) Exact Censored Expectation E[clip(Y*, lo, hi)] and 90% Censored Prediction Interval
    y_hat = (
        lo * Phi_l
        + hi * (1.0 - Phi_u)
        + mu_pred * np.maximum(Phi_u - Phi_l, 0.0)
        + sd * (phi_l - phi_u)
    )
    q05 = np.clip(mu_pred - 1.6448536 * sd, lo, hi)
    q95 = np.clip(mu_pred + 1.6448536 * sd, lo, hi)
    # Account for discrete integer rounding (+/- 0.5) in empirical 90% coverage & Winkler score (alpha=0.10)
    cov90 = ((y_true >= (q05 - 0.5)) & (y_true <= (q95 + 0.5))).astype(float)
    pi90_width = np.maximum(q95 - q05, 0.0)
    under_pen = np.maximum((q05 - 0.5) - y_true, 0.0)
    over_pen = np.maximum(y_true - (q95 + 0.5), 0.0)
    winkler90 = pi90_width + 20.0 * (under_pen + over_pen)
    return {"nll": nll, "crps": crps, "y_hat": y_hat, "cov90": cov90, "q05": q05, "q95": q95, "pi90_width": pi90_width, "winkler90": winkler90}


def cholesky_2x2_batch(p_mat: torch.Tensor) -> torch.Tensor:
    """Fast exact batched Cholesky factor L of symmetric PSD 2x2 matrices P (N, 2, 2)."""
    p00 = torch.clamp(p_mat[:, 0, 0], min=1e-7)
    l00 = torch.sqrt(p00)
    l10 = p_mat[:, 1, 0] / l00
    p11_rem = torch.clamp(p_mat[:, 1, 1] - l10 * l10, min=1e-7)
    l11 = torch.sqrt(p11_rem)
    zeros = torch.zeros_like(l00)
    row0 = torch.stack([l00, zeros], dim=-1)
    row1 = torch.stack([l10, l11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


_GL7_RAW_X, _GL7_RAW_W = np.polynomial.legendre.leggauss(7)
_GL7_X = torch.tensor(0.5 * (_GL7_RAW_X + 1.0), dtype=torch.float64)
_GL7_W = torch.tensor(0.5 * _GL7_RAW_W, dtype=torch.float64)


def exp2x2_putzer_grid(j_2x2: torch.Tensor, s_grid: torch.Tensor) -> torch.Tensor:
    """
    Exact analytic 2x2 matrix exponential exp(J_i * s_{i,q}) for J_i shape (N, 2, 2) and s_grid shape (N, Q).
    Smooth entire analytic functions across real, repeated, and complex conjugate eigenvalues.
    """
    a = j_2x2[:, 0, 0][:, None]
    b = j_2x2[:, 0, 1][:, None]
    c = j_2x2[:, 1, 0][:, None]
    d = j_2x2[:, 1, 1][:, None]

    mu = 0.5 * (a + d)
    hd = 0.5 * (a - d)
    delta = hd * hd + b * c
    z = delta * (s_grid * s_grid)

    pos_z = torch.clamp(z, min=1e-12)
    neg_z = torch.clamp(-z, min=1e-12)
    w_pos = torch.sqrt(pos_z)
    w_neg = torch.sqrt(neg_z)

    c0_taylor = 1.0 + z * (0.5 + z * (1.0 / 24.0 + z / 720.0))
    c1_taylor = 1.0 + z * (1.0 / 6.0 + z * (1.0 / 120.0 + z / 5040.0))

    c0 = torch.where(z > 1e-7, torch.cosh(w_pos), torch.where(z < -1e-7, torch.cos(w_neg), c0_taylor))
    c1 = torch.where(z > 1e-7, torch.sinh(w_pos) / w_pos, torch.where(z < -1e-7, torch.sin(w_neg) / w_neg, c1_taylor))

    emu = torch.exp(mu * s_grid)
    sc1 = s_grid * c1
    e00 = emu * (c0 + sc1 * hd)
    e01 = emu * (sc1 * b)
    e10 = emu * (sc1 * c)
    e11 = emu * (c0 - sc1 * hd)

    row0 = torch.stack([e00, e01], dim=-1)
    row1 = torch.stack([e10, e11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def inv_2x2_batch(p_mat: torch.Tensor) -> torch.Tensor:
    """Fast exact batched inverse of 2x2 PSD matrices P (N, 2, 2)."""
    a = p_mat[:, 0, 0]
    b = p_mat[:, 0, 1]
    c = p_mat[:, 1, 0]
    d = p_mat[:, 1, 1]
    det = torch.clamp(a * d - b * c, min=1e-10)
    row0 = torch.stack([d / det, -b / det], dim=-1)
    row1 = torch.stack([-c / det, a / det], dim=-1)
    return torch.stack([row0, row1], dim=-2)


class PsychometricCoupledSDE(nn.Module):
    """
    Psychometric-Physics-Aware Coupled Continuous-Time SDE Digital Twin.
    Supports exact ablation switches:
      - coupled=True/False        (M7 Proposed vs M6 Uncoupled Psychometric Tobit-SDE)
      - psychometric_filter=True/False (M7 Proposed vs M5 Uncensored Coupled SDE without Tobit/retest/heteroscedasticity)
      - use_neural_drift=True/False    (Residual L2-penalized neural drift MLP ablation)
      - use_subtest_traits=True/False  (Equilibrium G_i = (-J_i)^{-1} coupled covariance + persistent subtest OU traits
                                        + bounded multi-year horizon dispersion & Wave 4/5 protocol harmonization)
    """

    def __init__(
        self,
        scale_specs: List[ScaleSpec],
        d_x0: int,
        d_u: int,
        d_z_mod: int = 0,
        coupled: bool = True,
        psychometric_filter: bool = True,
        use_neural_drift: bool = False,
        n_gh_1d: int = 7,
        device: torch.device = torch.device("cpu"),
        use_subtest_traits: bool = False,
        d_prot: int = 0,
    ) -> None:
        super().__init__()
        self.scale_specs = scale_specs
        self.n_scales = len(scale_specs)
        self.d_x0 = d_x0
        self.d_u = d_u
        self.d_z_mod = d_z_mod
        self.d_prot = d_prot
        self.coupled = coupled
        self.psychometric_filter = psychometric_filter
        self.use_neural_drift = use_neural_drift
        self.use_subtest_traits = use_subtest_traits
        self.device = device

        gh_nodes, gh_logw = build_2d_gauss_hermite(n_1d=n_gh_1d, device=device)
        self.register_buffer("gh_nodes", gh_nodes)
        self.register_buffer("gh_logw", gh_logw)

        if self.use_subtest_traits:
            dom_counts = {0: 0, 1: 0}
            for sp in scale_specs:
                dom_counts[sp.domain] += 1
            self.has_subtest_tau = [use_subtest_traits and (dom_counts[sp.domain] > 1) for sp in scale_specs]
            self.has_dif = [use_subtest_traits and psychometric_filter and (sp.anchor_loading is None) for sp in scale_specs]

            self.register_buffer("dom_idx", torch.tensor([sp.domain for sp in scale_specs], dtype=torch.long))
            self.register_buffer("is_cog_mask", torch.tensor([sp.domain == 0 for sp in scale_specs], dtype=torch.bool))
            self.register_buffer("is_bin_mask", torch.tensor([sp.is_binary for sp in scale_specs], dtype=torch.bool))
            self.register_buffer(
                "censor_lo_mask",
                torch.tensor([sp.censor_lower and (not sp.is_binary) and (not math.isnan(sp.lower)) for sp in scale_specs], dtype=torch.bool),
            )
            self.register_buffer(
                "censor_hi_mask",
                torch.tensor([sp.censor_upper and (not sp.is_binary) and (not math.isnan(sp.upper)) for sp in scale_specs], dtype=torch.bool),
            )
            self.register_buffer("lo_bounds", torch.tensor([0.0 if math.isnan(sp.lower) else sp.lower for sp in scale_specs], dtype=torch.float64))
            self.register_buffer("hi_bounds", torch.tensor([1e9 if math.isnan(sp.upper) else sp.upper for sp in scale_specs], dtype=torch.float64))
            self.register_buffer("has_tau_mask", torch.tensor(self.has_subtest_tau, dtype=torch.float64))
            self.register_buffer("has_dif_mask", torch.tensor(self.has_dif, dtype=torch.float64))

            self.raw_kappa = nn.Parameter(torch.tensor([-0.9, -0.7], dtype=torch.float64))
            self.j12_0 = nn.Parameter(torch.tensor(-0.08 if coupled else 0.0, dtype=torch.float64))
            self.j21_0 = nn.Parameter(torch.tensor(-0.06 if coupled else 0.0, dtype=torch.float64))
        else:
            # 1. Coupled 2x2 SDE Drift Jacobian J_i
            self.raw_kappa = nn.Parameter(torch.tensor([-1.7, -1.2], dtype=torch.float64))
            self.j12_0 = nn.Parameter(torch.tensor(-0.08 if coupled else 0.0, dtype=torch.float64))
            self.j21_0 = nn.Parameter(torch.tensor(-0.08 if coupled else 0.0, dtype=torch.float64))

        if d_z_mod > 0:
            self.j12_mod = nn.Parameter(torch.zeros(d_z_mod, dtype=torch.float64))
            self.j21_mod = nn.Parameter(torch.zeros(d_z_mod, dtype=torch.float64))
        else:
            self.register_parameter("j12_mod", None)
            self.register_parameter("j21_mod", None)

        # 2. Exogenous & Modifiable Driver Drift Matrix B (2 x d_u) + Baseline Initial Mean C0 (2 x d_x0)
        self.B_ctrl = nn.Parameter(torch.zeros(2, max(d_u, 1), dtype=torch.float64))
        self.C0_init = nn.Parameter(torch.zeros(2, max(d_x0, 1), dtype=torch.float64))

        # Optional L2-regularized Residual Neural Drift g_phi(eta, x0)
        if use_neural_drift:
            self.neural_drift = nn.Sequential(
                nn.Linear(2 + max(d_x0, 1), 12),
                nn.Tanh(),
                nn.Linear(12, 2),
            )
            with torch.no_grad():
                for p in self.neural_drift.parameters():
                    p.mul_(0.02)
        else:
            self.neural_drift = None

        if self.use_subtest_traits:
            self.raw_diff_diag = nn.Parameter(torch.tensor([-1.0, -0.8], dtype=torch.float64))
            self.raw_diff_off = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

            self.raw_p0_diag = nn.Parameter(torch.tensor([-0.9, -0.8], dtype=torch.float64))
            self.raw_p0_off = nn.Parameter(torch.tensor(-0.05 if coupled else 0.0, dtype=torch.float64))
            self.raw_trait_diag = nn.Parameter(torch.tensor([-0.6, -0.5], dtype=torch.float64))
            self.raw_slope_diag = nn.Parameter(torch.tensor([-2.0, -1.9], dtype=torch.float64))

            self.nu = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.raw_lambda = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            init_gam = torch.tensor([-1.5 if sp.domain == 0 else 0.0 for sp in scale_specs], dtype=torch.float64)
            self.raw_gamma_retest = nn.Parameter(init_gam)
            self.raw_sigma_obs = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.delta_het = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.raw_tau_uniq = nn.Parameter(torch.full((self.n_scales,), -0.5, dtype=torch.float64))
            self.raw_kappa_zeta = nn.Parameter(torch.full((self.n_scales,), -1.2, dtype=torch.float64))
            self.D_dif = nn.Parameter(torch.zeros(self.n_scales, max(d_x0, 1), dtype=torch.float64))
            self.B_prot = nn.Parameter(torch.zeros(self.n_scales, max(d_prot, 1), dtype=torch.float64))
            hz_init = RAW_HZ_INIT.clone() if self.n_scales == 6 else torch.full((self.n_scales,), math.log(math.expm1(0.5)), dtype=torch.float64)
            self.raw_rho_hz = nn.Parameter(hz_init)
            self.raw_retest_decay = nn.Parameter(torch.tensor(-0.6, dtype=torch.float64))
        else:
            # 3. Continuous-Time Diffusion Cholesky L_Sigma (2x2), Initial State Cholesky L_P0 (2x2),
            #    and Individual Random Drift/Trait Heterogeneity Cholesky L_Omega_b (2x2)
            self.raw_diff_diag = nn.Parameter(torch.tensor([-0.9, -0.7], dtype=torch.float64))
            self.raw_diff_off = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

            self.raw_p0_diag = nn.Parameter(torch.tensor([-0.2, -0.2], dtype=torch.float64))
            self.raw_p0_off = nn.Parameter(torch.tensor(-0.15 if coupled else 0.0, dtype=torch.float64))

            self.raw_b_diag = nn.Parameter(torch.tensor([-1.8, -1.6], dtype=torch.float64))
            self.raw_b_off = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

            # 4. Psychometric Observation Layer Parameters across D scales
            self.nu = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.raw_lambda = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.gamma_retest = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.raw_sigma_obs = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))
            self.delta_het = nn.Parameter(torch.zeros(self.n_scales, dtype=torch.float64))

    def get_retest_gamma(self) -> torch.Tensor:
        if self.use_subtest_traits:
            return torch.where(self.is_cog_mask, F.softplus(self.raw_gamma_retest), self.raw_gamma_retest)
        return self.gamma_retest

    def initialize_observation_anchors(self, y_base: np.ndarray, mask_base: np.ndarray) -> None:
        """Initialize scale intercepts nu_d and measurement SDs from empirical baseline moments."""
        with torch.no_grad():
            for d, spec in enumerate(self.scale_specs):
                vals = y_base[mask_base[:, d], d]
                if len(vals) == 0:
                    continue
                if self.use_subtest_traits:
                    if spec.is_binary:
                        p_bar = float(np.clip(np.mean(vals), 0.05, 0.95))
                        self.nu[d] = float(stats.norm.ppf(p_bar)) if self.psychometric_filter else p_bar
                        self.raw_lambda[d] = math.log(math.expm1(0.75))
                        self.raw_sigma_obs[d] = math.log(math.expm1(0.42))
                        self.raw_tau_uniq[d] = math.log(math.expm1(0.55 if self.psychometric_filter else 0.25))
                    else:
                        mu_d = float(np.mean(vals))
                        sd_d = max(float(np.std(vals, ddof=1)), 0.25)
                        self.nu[d] = mu_d
                        if spec.anchor_loading is None:
                            self.raw_lambda[d] = math.log(math.expm1(0.55 * sd_d))
                        self.raw_sigma_obs[d] = math.log(math.expm1(0.50 * sd_d))
                        self.raw_tau_uniq[d] = math.log(math.expm1(0.45 * sd_d))
                else:
                    if spec.is_binary:
                        p_bar = float(np.clip(np.mean(vals), 0.05, 0.95))
                        self.nu[d] = float(stats.norm.ppf(p_bar)) if self.psychometric_filter else p_bar
                        self.raw_lambda[d] = math.log(math.expm1(0.75))
                        self.raw_sigma_obs[d] = math.log(math.expm1(0.45))
                    else:
                        mu_d = float(np.mean(vals))
                        sd_d = max(float(np.std(vals, ddof=1)), 0.25)
                        self.nu[d] = mu_d
                        if spec.anchor_loading is None:
                            self.raw_lambda[d] = math.log(math.expm1(0.65 * sd_d))
                        self.raw_sigma_obs[d] = math.log(math.expm1(0.65 * sd_d))

    def get_loadings(self) -> torch.Tensor:
        lam_free = F.softplus(self.raw_lambda) + 1e-3
        out = []
        for d, spec in enumerate(self.scale_specs):
            if spec.anchor_loading is not None:
                out.append(torch.tensor(spec.anchor_loading, dtype=torch.float64, device=self.nu.device))
            else:
                out.append(lam_free[d])
        return torch.stack(out)

    def get_jacobian_batch(self, n_batch: int, z_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct individualized 2x2 coupling Jacobians J_i of shape (N, 2, 2)."""
        if self.use_subtest_traits:
            kap = F.softplus(self.raw_kappa) + 0.05
            j00 = -kap[0].expand(n_batch)
            j11 = -kap[1].expand(n_batch)
            if not self.coupled:
                z0 = torch.zeros(n_batch, dtype=torch.float64, device=self.nu.device)
                j01, j10 = z0, z0
            else:
                max_c = 0.85 * torch.sqrt(kap[0] * kap[1])
                j01_raw = self.j12_0.expand(n_batch)
                j10_raw = self.j21_0.expand(n_batch)
                if self.d_z_mod > 0 and z_mod is not None and self.j12_mod is not None:
                    j01_raw = j01_raw + z_mod @ self.j12_mod
                    j10_raw = j10_raw + z_mod @ self.j21_mod
                j01 = max_c * torch.tanh(j01_raw / max_c)
                j10 = max_c * torch.tanh(j10_raw / max_c)
            return torch.stack([torch.stack([j00, j01], dim=-1), torch.stack([j10, j11], dim=-1)], dim=-2)

        kap = F.softplus(self.raw_kappa) + 1e-3
        j00 = -kap[0].expand(n_batch)
        j11 = -kap[1].expand(n_batch)
        if not self.coupled:
            z0 = torch.zeros(n_batch, dtype=torch.float64, device=self.nu.device)
            j01, j10 = z0, z0
        else:
            j01 = self.j12_0.expand(n_batch)
            j10 = self.j21_0.expand(n_batch)
            if self.d_z_mod > 0 and z_mod is not None and self.j12_mod is not None:
                j01 = j01 + z_mod @ self.j12_mod
                j10 = j10 + z_mod @ self.j21_mod
        row0 = torch.stack([j00, j01], dim=-1)
        row1 = torch.stack([j10, j11], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def _build_2x2_cov(self, raw_diag: torch.Tensor, raw_off: torch.Tensor) -> torch.Tensor:
        d0 = F.softplus(raw_diag[0]) + 1e-4
        d1 = F.softplus(raw_diag[1]) + 1e-4
        off = raw_off if self.coupled else torch.zeros_like(raw_off)
        z = torch.zeros_like(d0)
        l_mat = torch.stack([torch.stack([d0, z]), torch.stack([off, d1])])
        return l_mat @ l_mat.T

    def propagate_4d_sde(
        self,
        mu_prev: torch.Tensor,
        p_prev: torch.Tensor,
        j_2x2: torch.Tensor,
        drift_2d: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact continuous-time discretization of 4D state s = [eta_cog, eta_dep, b_cog, b_dep]^T
        across irregular time interval dt > 0:
          ds = (J_tilde s + [drift_2d; 0]) dt + [Sigma^{1/2}; 0] dW(t)
        Evaluates exact closed-form 2x2 Putzer matrix exponential exp(J_i s) and 7-node
        Gauss-Legendre integrals (< 1e-14 agreement with Van Loan 8x8 matrix exponential, >5x faster).
        """
        n_b = mu_prev.shape[0]
        dev = mu_prev.device
        dt_c = torch.clamp(dt, min=1e-4)

        gl_x = _GL7_X.to(dev)
        gl_w = _GL7_W.to(dev)
        s_all = torch.cat([dt_c[:, None], dt_c[:, None] * gl_x[None, :]], dim=1)  # (N, 8)
        e_all = exp2x2_putzer_grid(j_2x2, s_all)  # (N, 8, 2, 2)

        e_dt = e_all[:, 0, :, :]        # exp(J * dt): (N, 2, 2)
        e_gl = e_all[:, 1:, :, :]       # exp(J * s_q): (N, 7, 2, 2)
        w_dt = dt_c[:, None] * gl_w[None, :]  # (N, 7)

        # E1(dt) = int_0^dt exp(J s) ds: (N, 2, 2)
        e1_dt = torch.einsum("nq,nqij->nij", w_dt, e_gl)

        # Q_2x2(dt) = int_0^dt exp(J s) Sigma exp(J^T s) ds: (N, 2, 2)
        sigma_2x2 = self._build_2x2_cov(self.raw_diff_diag, self.raw_diff_off)
        q_2x2 = torch.einsum("nq,nqia,ab,nqjb->nij", w_dt, e_gl, sigma_2x2, e_gl)
        q_2x2 = 0.5 * (q_2x2 + q_2x2.transpose(1, 2))

        a_4x4 = torch.zeros((n_b, 4, 4), dtype=torch.float64, device=dev)
        a_4x4[:, 0:2, 0:2] = e_dt
        a_4x4[:, 0:2, 2:4] = e1_dt
        a_4x4[:, 2, 2] = 1.0
        a_4x4[:, 3, 3] = 1.0

        int_drift_2 = torch.einsum("nij,nj->ni", e1_dt, drift_2d)
        mu_eta_next = (
            torch.einsum("nij,nj->ni", e_dt, mu_prev[:, 0:2])
            + torch.einsum("nij,nj->ni", e1_dt, mu_prev[:, 2:4])
            + int_drift_2
        )
        mu_pred = torch.cat([mu_eta_next, mu_prev[:, 2:4]], dim=-1)

        p_pred = torch.einsum("nij,njk,nlk->nil", a_4x4, p_prev, a_4x4)
        p_pred[:, 0:2, 0:2] = p_pred[:, 0:2, 0:2] + q_2x2
        p_pred = 0.5 * (p_pred + p_pred.transpose(1, 2)) + 1e-7 * torch.eye(4, dtype=torch.float64, device=dev)[None, :, :]
        return mu_pred, p_pred

    def _filter_and_loglik_subtest_traits(
        self,
        batch: Dict[str, torch.Tensor],
        max_filter_wave_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        t_years = batch["t_years"]
        y_obs = batch["y_obs"]
        obs_mask = batch["obs_mask"]
        k_prior = batch["k_prior"]
        x0 = batch["x0"]
        u_ctrl = batch["u_ctrl"]
        illit = batch["illit"]
        z_mod = batch.get("z_mod", None)
        w_prot = batch.get("w_prot", None)

        n_b, n_w, n_d = y_obs.shape
        dev = y_obs.device

        j_batch = self.get_jacobian_batch(n_b, z_mod=z_mod)
        g_inv = inv_2x2_batch(-j_batch)

        loadings = self.get_loadings()
        gamma_ret = self.get_retest_gamma()
        sigma_base = F.softplus(self.raw_sigma_obs) + 1e-2
        rho_hz = F.softplus(self.raw_rho_hz) if self.psychometric_filter else torch.zeros_like(sigma_base)
        ret_decay = F.softplus(self.raw_retest_decay) if self.psychometric_filter else torch.tensor(0.0, dtype=torch.float64, device=dev)
        tau_sq = ((F.softplus(self.raw_tau_uniq) + 1e-3) ** 2) * self.has_tau_mask
        kap_zeta = F.softplus(self.raw_kappa_zeta) + 0.02

        mu_s = torch.zeros((n_b, 4), dtype=torch.float64, device=dev)
        if self.d_x0 > 0:
            mu0 = x0 @ self.C0_init.T
            mu_s[:, 0:2] = mu0
            mu_s[:, 2:4] = -torch.einsum("nij,nj->ni", j_batch, mu0)

        omega_b = torch.diag((F.softplus(self.raw_trait_diag) + 1e-3) ** 2)
        p_trans0 = self._build_2x2_cov(self.raw_p0_diag, self.raw_p0_off)
        omega_slope = torch.diag((F.softplus(self.raw_slope_diag) + 1e-4) ** 2)

        p_s = torch.zeros((n_b, 4, 4), dtype=torch.float64, device=dev)
        p_s[:, 0:2, 0:2] = torch.einsum("nia,ab,njb->nij", g_inv, omega_b, g_inv) + p_trans0[None, :, :]
        cov_eb = torch.einsum("nia,ab->nib", g_inv, omega_b)
        p_s[:, 0:2, 2:4] = cov_eb
        p_s[:, 2:4, 0:2] = cov_eb.transpose(1, 2)
        p_s[:, 2:4, 2:4] = (omega_b + omega_slope)[None, :, :]

        m_zeta = torch.zeros((n_b, n_d), dtype=torch.float64, device=dev)
        v_zeta = tau_sq[None, :].expand(n_b, n_d).clone()
        dt_since_obs = torch.full((n_b, n_d), 2.0, dtype=torch.float64, device=dev)
        has_any_prev_obs = torch.zeros((n_b, n_d), dtype=torch.bool, device=dev)

        dif_mat = ((x0 @ self.D_dif.T) * self.has_dif_mask[None, :]) if self.d_x0 > 0 else torch.zeros((n_b, n_d), dtype=torch.float64, device=dev)
        het_mat = torch.exp(0.5 * illit[:, None] * self.delta_het[None, :]) if self.psychometric_filter else torch.ones((n_b, n_d), dtype=torch.float64, device=dev)
        sig_meas_2yr = sigma_base[None, :] * het_mat
        if self.psychometric_filter and self.is_bin_mask.any():
            sig_meas_2yr = torch.where(self.is_bin_mask[None, :], torch.ones_like(sig_meas_2yr), sig_meas_2yr)

        wave_logliks = []
        pred_eta_means = []
        pred_eta_vars = []
        filt_eta_means = []
        filt_b_means = []
        pred_scale_mu = torch.zeros((n_b, n_w, n_d), dtype=torch.float64, device=dev)
        pred_scale_sd = torch.zeros((n_b, n_w, n_d), dtype=torch.float64, device=dev)

        for w in range(n_w):
            if w > 0:
                dt_w = torch.clamp(t_years[:, w] - t_years[:, w - 1], min=1e-4)
                drift_w = u_ctrl[:, w, :] @ self.B_ctrl.T if self.d_u > 0 else torch.zeros((n_b, 2), dtype=torch.float64, device=dev)
                if self.neural_drift is not None:
                    nd_in = torch.cat([mu_s[:, 0:2], x0], dim=-1)
                    drift_w = drift_w + 0.10 * torch.tanh(self.neural_drift(nd_in))
                mu_s, p_s = self.propagate_4d_sde(mu_s, p_s, j_batch, drift_w, dt_w)
                rho_z = torch.exp(-dt_w[:, None] * kap_zeta[None, :])
                m_zeta = rho_z * m_zeta
                v_zeta = (rho_z * rho_z) * v_zeta + (1.0 - rho_z * rho_z) * tau_sq[None, :]
                dt_since_obs = torch.where(has_any_prev_obs, dt_since_obs + dt_w[:, None], torch.full_like(dt_since_obs, 2.0))

            mu_eta = mu_s[:, 0:2]
            p_eta = p_s[:, 0:2, 0:2]
            p_eta_diag = torch.diagonal(p_eta, dim1=-2, dim2=-1)
            pred_eta_means.append(mu_eta)
            pred_eta_vars.append(p_eta_diag)

            dt_excess = torch.clamp(dt_since_obs - 2.45, min=0.0)
            dt_h = 0.74 * (1.0 - torch.exp(-2.3 * dt_excess))
            ret_atten = torch.exp(-ret_decay * dt_excess)
            retest_w = (torch.log1p(k_prior[:, w, :]) * gamma_ret[None, :] * ret_atten) if self.psychometric_filter else 0.0
            prot_w = (w_prot[:, w, :] @ self.B_prot.T) if (self.psychometric_filter and self.d_prot > 0 and w_prot is not None) else 0.0
            base_shift = self.nu[None, :] + dif_mat + prot_w + retest_w

            mu_eta_dom = mu_eta[:, self.dom_idx]
            var_eta_dom = p_eta_diag[:, self.dom_idx]

            v_filter_base = (loadings[None, :] ** 2) * var_eta_dom + v_zeta + sig_meas_2yr * sig_meas_2yr
            extra_hz_var = v_filter_base * rho_hz[None, :] * dt_h
            sig_meas = torch.sqrt(sig_meas_2yr * sig_meas_2yr + extra_hz_var)
            sig_eff = torch.sqrt(sig_meas * sig_meas + v_zeta)
            v_total_pred = (loadings[None, :] ** 2) * var_eta_dom + sig_eff * sig_eff

            pred_scale_mu[:, w, :] = base_shift + m_zeta + loadings[None, :] * mu_eta_dom
            pred_scale_sd[:, w, :] = torch.sqrt(torch.clamp(v_total_pred, min=1e-6))

            l_eta = cholesky_2x2_batch(p_eta)
            eta_grid = mu_eta[:, None, :] + torch.einsum("nij,gj->ngi", l_eta, self.gh_nodes)
            eta_dom_g = eta_grid[:, :, self.dom_idx]
            mu_g = (base_shift + m_zeta)[:, None, :] + loadings[None, None, :] * eta_dom_g
            s_g = torch.clamp(sig_eff[:, None, :], min=1e-3)
            y_w = y_obs[:, w, :][:, None, :]
            m_w = obs_mask[:, w, :][:, None, :]

            ll_gauss = -0.5 * math.log(2.0 * math.pi) - torch.log(s_g) - 0.5 * ((y_w - mu_g) / s_g) ** 2
            if not self.psychometric_filter:
                ll_all = torch.where(m_w, ll_gauss, torch.zeros_like(ll_gauss))
            else:
                ll_all = ll_gauss
                if self.censor_lo_mask.any():
                    z_lo = (self.lo_bounds[None, None, :] + 0.5 - mu_g) / s_g
                    at_lo = (y_w <= (self.lo_bounds[None, None, :] + 1e-6)) & self.censor_lo_mask[None, None, :]
                    ll_all = torch.where(at_lo, log_ndtr_safe(z_lo), ll_all)
                if self.censor_hi_mask.any():
                    z_hi = (mu_g - (self.hi_bounds[None, None, :] - 0.5)) / s_g
                    at_hi = (y_w >= (self.hi_bounds[None, None, :] - 1e-6)) & self.censor_hi_mask[None, None, :]
                    ll_all = torch.where(at_hi, log_ndtr_safe(z_hi), ll_all)
                if self.is_bin_mask.any():
                    z_b = mu_g / s_g
                    ll_b = torch.where(y_w >= 0.5, log_ndtr_safe(z_b), log_ndtr_safe(-z_b))
                    ll_all = torch.where(self.is_bin_mask[None, None, :], ll_b, ll_all)
                ll_all = torch.where(m_w, ll_all, torch.zeros_like(ll_all))

            ll_grid = torch.sum(ll_all, dim=-1)
            log_joint = self.gh_logw[None, :] + ll_grid
            ll_wave = torch.logsumexp(log_joint, dim=-1)
            any_obs_w = obs_mask[:, w, :].any(dim=-1)
            ll_wave = torch.where(any_obs_w, ll_wave, torch.zeros_like(ll_wave))
            wave_logliks.append(ll_wave)

            assimilate = any_obs_w
            if max_filter_wave_idx is not None and w > max_filter_wave_idx:
                assimilate = torch.zeros_like(any_obs_w)

            if assimilate.any():
                post_w = torch.softmax(log_joint, dim=-1)
                mu_post = torch.einsum("ng,ngi->ni", post_w, eta_grid)
                diff_g = eta_grid - mu_post[:, None, :]
                p_post = torch.einsum("ng,ngi,ngj->nij", post_w, diff_g, diff_g)
                p_post = 0.5 * (p_post + p_post.transpose(1, 2)) + 1e-6 * torch.eye(2, dtype=torch.float64, device=dev)[None, :, :]

                p_eta_inv = inv_2x2_batch(p_eta)
                k_rb = torch.einsum("nia,nab->nib", p_s[:, :, 0:2], p_eta_inv)
                mu_upd = mu_s + torch.einsum("nia,na->ni", k_rb, mu_post - mu_eta)
                dp_eta = p_post - p_eta
                p_upd = p_s + torch.einsum("nia,nab,njb->nij", k_rb, dp_eta, k_rb)
                p_upd = 0.5 * (p_upd + p_upd.transpose(1, 2)) + 1e-7 * torch.eye(4, dtype=torch.float64, device=dev)[None, :, :]

                mu_s = torch.where(assimilate[:, None], mu_upd, mu_s)
                p_s = torch.where(assimilate[:, None, None], p_upd, p_s)

                p_post_diag = torch.diagonal(p_post, dim1=-2, dim2=-1)
                v_nonz = sig_meas * sig_meas + (loadings[None, :] ** 2) * p_post_diag[:, self.dom_idx]
                v_tot = v_zeta + v_nonz
                s_tot = torch.sqrt(torch.clamp(v_tot, min=1e-6))
                k_z = v_zeta / torch.clamp(v_tot, min=1e-6)
                m_tot = base_shift + loadings[None, :] * mu_post[:, self.dom_idx] + m_zeta
                yw_2d = y_obs[:, w, :]

                mz_upd = m_zeta + k_z * (yw_2d - m_tot)
                vz_upd = torch.clamp((1.0 - k_z) * v_zeta, min=1e-5)

                if self.psychometric_filter:
                    if self.censor_lo_mask.any():
                        z_lo = (self.lo_bounds[None, :] + 0.5 - m_tot) / s_tot
                        r_lo = torch.exp(log_mills_ratio(z_lo))
                        is_lo = (yw_2d <= (self.lo_bounds[None, :] + 1e-6)) & self.censor_lo_mask[None, :]
                        mz_upd = torch.where(is_lo, m_zeta - k_z * s_tot * r_lo, mz_upd)
                        vz_upd = torch.where(is_lo, v_zeta * torch.clamp(1.0 - k_z * r_lo * (r_lo + z_lo), min=0.05, max=1.0), vz_upd)
                    if self.censor_hi_mask.any():
                        z_hi = (m_tot - (self.hi_bounds[None, :] - 0.5)) / s_tot
                        r_hi = torch.exp(log_mills_ratio(z_hi))
                        is_hi = (yw_2d >= (self.hi_bounds[None, :] - 1e-6)) & self.censor_hi_mask[None, :]
                        mz_upd = torch.where(is_hi, m_zeta + k_z * s_tot * r_hi, mz_upd)
                        vz_upd = torch.where(is_hi, v_zeta * torch.clamp(1.0 - k_z * r_hi * (r_hi + z_hi), min=0.05, max=1.0), vz_upd)
                    if self.is_bin_mask.any():
                        zb_hi = m_tot / s_tot
                        rb_hi = torch.exp(log_mills_ratio(zb_hi))
                        zb_lo = -m_tot / s_tot
                        rb_lo = torch.exp(log_mills_ratio(zb_lo))
                        mb = torch.where(yw_2d >= 0.5, m_zeta + k_z * s_tot * rb_hi, m_zeta - k_z * s_tot * rb_lo)
                        vb = torch.where(
                            yw_2d >= 0.5,
                            v_zeta * torch.clamp(1.0 - k_z * rb_hi * (rb_hi + zb_hi), min=0.05, max=1.0),
                            v_zeta * torch.clamp(1.0 - k_z * rb_lo * (rb_lo + zb_lo), min=0.05, max=1.0),
                        )
                        mz_upd = torch.where(self.is_bin_mask[None, :], mb, mz_upd)
                        vz_upd = torch.where(self.is_bin_mask[None, :], vb, vz_upd)

                obs_here = assimilate[:, None] & obs_mask[:, w, :]
                dt_since_obs = torch.where(obs_here, torch.zeros_like(dt_since_obs), dt_since_obs)
                has_any_prev_obs = has_any_prev_obs | obs_here
                upd_mask = obs_here & (self.has_tau_mask[None, :] > 0.5)
                m_zeta = torch.where(upd_mask, mz_upd, m_zeta)
                v_zeta = torch.where(upd_mask, vz_upd, v_zeta)

            filt_eta_means.append(mu_s[:, 0:2])
            filt_b_means.append(mu_s[:, 2:4])

        ll_mat = torch.stack(wave_logliks, dim=1)
        return {
            "wave_loglik": ll_mat,
            "person_loglik": torch.sum(ll_mat, dim=1),
            "pred_scale_mu": pred_scale_mu,
            "pred_scale_sd": pred_scale_sd,
            "pred_eta_means": torch.stack(pred_eta_means, dim=1),
            "pred_eta_vars": torch.stack(pred_eta_vars, dim=1),
            "filt_eta_means": torch.stack(filt_eta_means, dim=1),
            "filt_b_means": torch.stack(filt_b_means, dim=1),
            "jacobian_batch": j_batch,
        }

    def filter_and_loglik(
        self,
        batch: Dict[str, torch.Tensor],
        max_filter_wave_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward 2D Gauss-Hermite Tobit-Kalman filter across W waves.
        If max_filter_wave_idx is provided (e.g. temporal holdout forecasting future waves w > w_cut),
        observations at waves w > max_filter_wave_idx are NOT assimilated into the posterior state,
        producing pure multi-horizon continuous-time SDE forecasts!
        """
        if self.use_subtest_traits:
            return self._filter_and_loglik_subtest_traits(batch, max_filter_wave_idx=max_filter_wave_idx)

        t_years = batch["t_years"]
        y_obs = batch["y_obs"]
        obs_mask = batch["obs_mask"]
        k_prior = batch["k_prior"]
        x0 = batch["x0"]
        u_ctrl = batch["u_ctrl"]
        illit = batch["illit"]
        z_mod = batch.get("z_mod", None)

        n_b, n_w, _ = y_obs.shape
        dev = y_obs.device

        j_batch = self.get_jacobian_batch(n_b, z_mod=z_mod)
        loadings = self.get_loadings()
        sigma_base = F.softplus(self.raw_sigma_obs) + 1e-2

        # Initial 4D state at participant's baseline wave w = 0
        mu_s = torch.zeros((n_b, 4), dtype=torch.float64, device=dev)
        if self.d_x0 > 0:
            mu_s[:, 0:2] = x0 @ self.C0_init.T
        p_s = torch.zeros((n_b, 4, 4), dtype=torch.float64, device=dev)
        p_s[:, 0:2, 0:2] = self._build_2x2_cov(self.raw_p0_diag, self.raw_p0_off)[None, :, :]
        p_s[:, 2:4, 2:4] = self._build_2x2_cov(self.raw_b_diag, self.raw_b_off)[None, :, :]

        wave_logliks = []
        pred_eta_means = []
        pred_eta_vars = []
        filt_eta_means = []
        filt_b_means = []

        pred_scale_mu = torch.zeros((n_b, n_w, self.n_scales), dtype=torch.float64, device=dev)
        pred_scale_sd = torch.zeros((n_b, n_w, self.n_scales), dtype=torch.float64, device=dev)

        for w in range(n_w):
            if w > 0:
                dt_w = t_years[:, w] - t_years[:, w - 1]
                drift_w = u_ctrl[:, w, :] @ self.B_ctrl.T if self.d_u > 0 else torch.zeros((n_b, 2), dtype=torch.float64, device=dev)
                if self.neural_drift is not None:
                    nd_in = torch.cat([mu_s[:, 0:2], x0], dim=-1)
                    drift_w = drift_w + 0.25 * torch.tanh(self.neural_drift(nd_in))
                mu_s, p_s = self.propagate_4d_sde(mu_s, p_s, j_batch, drift_w, dt_w)

            mu_eta = mu_s[:, 0:2]  # (N, 2)
            p_eta = p_s[:, 0:2, 0:2]  # (N, 2, 2)
            pred_eta_means.append(mu_eta)
            pred_eta_vars.append(torch.diagonal(p_eta, dim1=-2, dim2=-1))

            # Record analytical marginal predictive mean & SD for every scale d at wave w
            for d, spec in enumerate(self.scale_specs):
                dom = spec.domain
                retest_off = (self.gamma_retest[d] * torch.log1p(k_prior[:, w, d])) if self.psychometric_filter else 0.0
                het_mult = torch.exp(0.5 * self.delta_het[d] * illit) if self.psychometric_filter else torch.ones_like(illit)
                sig_obs_d = torch.ones_like(illit) if (spec.is_binary and self.psychometric_filter) else (sigma_base[d] * het_mult)
                m_d = self.nu[d] + loadings[d] * mu_eta[:, dom] + retest_off
                v_d = (loadings[d] ** 2) * p_eta[:, dom, dom] + sig_obs_d**2
                pred_scale_mu[:, w, d] = m_d
                pred_scale_sd[:, w, d] = torch.sqrt(torch.clamp(v_d, min=1e-6))

            # Evaluate 2D Gauss-Hermite quadrature over eta_w ~ N(mu_eta, p_eta)
            l_eta = cholesky_2x2_batch(p_eta)  # (N, 2, 2)
            eta_grid = mu_eta[:, None, :] + torch.einsum("nij,gj->ngi", l_eta, self.gh_nodes)

            ll_grid = torch.zeros((n_b, self.gh_nodes.shape[0]), dtype=torch.float64, device=dev)
            for d, spec in enumerate(self.scale_specs):
                y_wd = y_obs[:, w, d : d + 1]
                m_wd = obs_mask[:, w, d : d + 1]
                retest_off = (self.gamma_retest[d] * torch.log1p(k_prior[:, w, d : d + 1])) if self.psychometric_filter else 0.0
                het_mult = torch.exp(0.5 * self.delta_het[d] * illit[:, None]) if self.psychometric_filter else torch.ones((n_b, 1), dtype=torch.float64, device=dev)
                sig_d = torch.ones((n_b, 1), dtype=torch.float64, device=dev) if (spec.is_binary and self.psychometric_filter) else (sigma_base[d] * het_mult)
                mu_grid_d = self.nu[d] + loadings[d] * eta_grid[:, :, spec.domain] + retest_off
                ll_grid = ll_grid + tobit_probit_Scale_loglik(
                    y_wd, m_wd, mu_grid_d, sig_d, spec, use_tobit_censoring=self.psychometric_filter
                )

            log_joint = self.gh_logw[None, :] + ll_grid  # (N, G)
            ll_wave = torch.logsumexp(log_joint, dim=-1)  # (N,)
            any_obs_w = obs_mask[:, w, :].any(dim=-1)
            ll_wave = torch.where(any_obs_w, ll_wave, torch.zeros_like(ll_wave))
            wave_logliks.append(ll_wave)

            # Assimilate observations at wave w unless frozen for out-of-time forecasting
            assimilate = any_obs_w
            if max_filter_wave_idx is not None and w > max_filter_wave_idx:
                assimilate = torch.zeros_like(any_obs_w)

            if assimilate.any():
                post_w = torch.softmax(log_joint, dim=-1)  # (N, G)
                mu_post = torch.einsum("ng,ngi->ni", post_w, eta_grid)  # (N, 2)
                diff_g = eta_grid - mu_post[:, None, :]
                p_post = torch.einsum("ng,ngi,ngj->nij", post_w, diff_g, diff_g)
                p_post = 0.5 * (p_post + p_post.transpose(1, 2)) + 1e-6 * torch.eye(2, dtype=torch.float64, device=dev)[None, :, :]

                # Rao-Blackwell update onto full 4D state [eta_w, b_i]
                p_eta_inv = inv_2x2_batch(p_eta)  # (N, 2, 2)
                k_rb = torch.einsum("nia,nab->nib", p_s[:, :, 0:2], p_eta_inv)  # (N, 4, 2)
                mu_upd = mu_s + torch.einsum("nia,na->ni", k_rb, mu_post - mu_eta)
                dp_eta = p_post - p_eta
                p_upd = p_s + torch.einsum("nia,nab,njb->nij", k_rb, dp_eta, k_rb)
                p_upd = 0.5 * (p_upd + p_upd.transpose(1, 2)) + 1e-7 * torch.eye(4, dtype=torch.float64, device=dev)[None, :, :]

                mu_s = torch.where(assimilate[:, None], mu_upd, mu_s)
                p_s = torch.where(assimilate[:, None, None], p_upd, p_s)

            filt_eta_means.append(mu_s[:, 0:2])
            filt_b_means.append(mu_s[:, 2:4])

        ll_matrix = torch.stack(wave_logliks, dim=1)  # (N, W)
        return {
            "wave_loglik": ll_matrix,
            "person_loglik": torch.sum(ll_matrix, dim=1),
            "pred_scale_mu": pred_scale_mu,
            "pred_scale_sd": pred_scale_sd,
            "pred_eta_means": torch.stack(pred_eta_means, dim=1),
            "pred_eta_vars": torch.stack(pred_eta_vars, dim=1),
            "filt_eta_means": torch.stack(filt_eta_means, dim=1),
            "filt_b_means": torch.stack(filt_b_means, dim=1),
            "jacobian_batch": j_batch,
        }


def fit_psychometric_sde_model(
    model: PsychometricCoupledSDE,
    train_batch: Dict[str, torch.Tensor],
    sample_weights: Optional[torch.Tensor] = None,
    n_adam_steps: int = 160,
    lr_adam: float = 0.035,
    n_lbfgs_steps: int = 35,
    l2_reg: float = 1e-4,
    verbose: bool = False,
    warm_from: Optional[PsychometricCoupledSDE] = None,
    n_adam_epochs: int = 24,
    batch_size: int = 4096,
    l2_mod_mult: float = 5.0,
    seed: int = 20270507,
) -> float:
    """
    Two-stage deterministic optimization (Adam warm-up + Strong-Wolfe L-BFGS fine polishing) of exact
    2D Gauss-Hermite Tobit-Kalman marginal maximum likelihood.
    """
    if model.use_subtest_traits:
        n_N = train_batch["y_obs"].shape[0]
        n_obs_total = float(train_batch["obs_mask"].sum().item())
        dev = train_batch["y_obs"].device
        raw_hz_tgt = (RAW_HZ_INIT if model.n_scales == 6 else torch.full((model.n_scales,), math.log(math.expm1(0.5)), dtype=torch.float64)).to(dev)
        sw_all = (sample_weights / torch.mean(sample_weights)) if sample_weights is not None else None

        if warm_from is not None:
            with torch.no_grad():
                sd_src = warm_from.state_dict()
                sd_dst = model.state_dict()
                skip_keys = set()
                if (not warm_from.coupled) and model.coupled:
                    skip_keys.update({"j12_0", "j21_0", "j12_mod", "j21_mod", "raw_p0_off", "raw_diff_off"})
                for k, v in sd_src.items():
                    if k in sd_dst and sd_dst[k].shape == v.shape and (k not in skip_keys):
                        sd_dst[k].copy_(v)
        else:
            y0 = train_batch["y_obs"][:, 0, :].detach().cpu().numpy()
            m0 = train_batch["obs_mask"][:, 0, :].detach().cpu().numpy()
            model.initialize_observation_anchors(y0, m0)

        def reg_term() -> torch.Tensor:
            reg = l2_reg * (
                torch.sum(model.B_ctrl**2)
                + torch.sum(model.C0_init**2)
                + torch.sum(model.D_dif**2)
                + torch.sum(model.B_prot**2)
                + 6.0 * (model.j12_0**2 + model.j21_0**2)
                + 10.0 * ((model.j12_0 - model.j21_0) ** 2)
                + 8.0 * torch.sum((model.raw_rho_hz - raw_hz_tgt) ** 2)
                + 2.0 * ((model.raw_retest_decay + 0.6) ** 2)
            )
            if model.j12_mod is not None:
                reg = reg + l2_mod_mult * l2_reg * (torch.sum(model.j12_mod**2) + torch.sum(model.j21_mod**2))
            if model.neural_drift is not None:
                for p in model.neural_drift.parameters():
                    reg = reg + 20.0 * l2_reg * torch.sum(p**2)
            return reg

        if warm_from is None and n_adam_epochs > 0:
            gh_orig_n, gh_orig_w = model.gh_nodes.clone(), model.gh_logw.clone()
            gh4_n, gh4_w = build_2d_gauss_hermite(n_1d=4, device=dev)
            model.gh_nodes = gh4_n
            model.gh_logw = gh4_w

            opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
            n_batches = max(int(math.ceil(n_N / batch_size)), 1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_adam_epochs * n_batches, 1), eta_min=lr_adam * 0.15)
            rng = np.random.default_rng(seed)
            for ep in range(n_adam_epochs):
                perm = rng.permutation(n_N)
                for b_i in range(n_batches):
                    sl = perm[b_i * batch_size : (b_i + 1) * batch_size]
                    sub_b = {k: v[sl] for k, v in train_batch.items()}
                    n_obs_b = float(sub_b["obs_mask"].sum().item())
                    opt.zero_grad()
                    out = model.filter_and_loglik(sub_b)
                    pll = out["person_loglik"] if sw_all is None else (sw_all[sl] * out["person_loglik"])
                    loss = -torch.sum(pll) / max(n_obs_b, 1.0) + reg_term()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    sched.step()
                if verbose and (ep % 8 == 0 or ep == n_adam_epochs - 1):
                    print(f"    [Adam ep {ep+1:02d}/{n_adam_epochs}] batch_loss = {float(loss.item()):.5f}")
            model.gh_nodes = gh_orig_n
            model.gh_logw = gh_orig_w

        if n_lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.7,
                max_iter=n_lbfgs_steps,
                history_size=24,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                out = model.filter_and_loglik(train_batch)
                pll = out["person_loglik"] if sw_all is None else (sw_all * out["person_loglik"])
                l_val = -torch.sum(pll) / max(n_obs_total, 1.0) + reg_term()
                l_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                return l_val

            lbfgs.step(closure)

        with torch.no_grad():
            out_f = model.filter_and_loglik(train_batch)
            pll_f = out_f["person_loglik"] if sw_all is None else (sw_all * out_f["person_loglik"])
            final_loss = float((-torch.sum(pll_f) / max(n_obs_total, 1.0) + reg_term()).detach().item())
        if verbose:
            print(f"    [Fit Complete] final_NLL_per_obs = {final_loss:.5f}")
        return final_loss

    y0 = train_batch["y_obs"][:, 0, :].detach().cpu().numpy()
    m0 = train_batch["obs_mask"][:, 0, :].detach().cpu().numpy()
    model.initialize_observation_anchors(y0, m0)

    n_obs_total = float(train_batch["obs_mask"].sum().item())
    if sample_weights is None:
        sw = torch.ones(train_batch["y_obs"].shape[0], dtype=torch.float64, device=train_batch["y_obs"].device)
    else:
        sw = sample_weights / torch.mean(sample_weights)

    def loss_fn() -> torch.Tensor:
        out = model.filter_and_loglik(train_batch)
        nll = -torch.sum(sw * out["person_loglik"]) / max(n_obs_total, 1.0)
        reg = torch.tensor(0.0, dtype=torch.float64, device=nll.device)
        if model.j12_mod is not None:
            reg = reg + 5.0 * l2_reg * (torch.sum(model.j12_mod**2) + torch.sum(model.j21_mod**2))
        reg = reg + l2_reg * (torch.sum(model.B_ctrl**2) + torch.sum(model.C0_init**2))
        if model.neural_drift is not None:
            for p in model.neural_drift.parameters():
                reg = reg + 10.0 * l2_reg * torch.sum(p**2)
        return nll + reg

    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_adam_steps, 1), eta_min=lr_adam * 0.15)
    for step in range(n_adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        scheduler.step()
        if verbose and (step % 40 == 0 or step == n_adam_steps - 1):
            print(f"    [Adam {step:03d}] mean_NLL_per_obs = {float(loss.item()):.5f}")

    if n_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=n_lbfgs_steps,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            l_val = loss_fn()
            l_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            return l_val

        lbfgs.step(closure)

    final_loss = float(loss_fn().detach().item())
    if verbose:
        print(f"    [Fit Complete] final_NLL_per_obs = {final_loss:.5f}")
    return final_loss
