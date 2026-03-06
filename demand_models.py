"""demand_models.py

Demand models used across the TTI green/yellow reproducibility package.

User-driven design constraints
------------------------------
- Demand is IID and stationary.
- Demand is integer and bounded with a realistic maximum dmax (default <= 20).
- For each experiment, we keep the same mean across demand families for fair comparison.
- We preserve the CV levels requested in the experimental design (e.g., 0.4 / 0.8 / 1.2).
- We provide demand families that can generate calm periods and promotional peaks
  without introducing time dependence (achieved via IID mixtures).

New demand families (recommended)
--------------------------------
1) beta_binomial
   Bounded flexible dispersion via a Beta-Binomial distribution on {0,...,n}.
   Calibrated to match target mean and variance exactly (up to float tolerance).

2) mix_binomial_2regime
   IID mixture of two Binomial distributions (normal vs promo).
   Uses a single "delta" parameter so the mean constraint holds exactly, and
   solves delta by bisection to match target variance.

3) mix_binomial_3regime
   IID mixture of three Binomial distributions (calm, normal, promo).
   Fixes regime probabilities, anchors the middle regime at the overall mean,
   and solves a single delta to match target variance.

Legacy demand families (kept for backward compatibility)
--------------------------------------------------------
- gamma: integer demand from Gamma(mean, cv) via rounding.
- mix_poisson_3regime: three-regime mixture of truncated Poissons.

Important implementation note
-----------------------------
The policy-comparison scripts pre-generate a demand_stream and inject it into
all policies (BS, PIL, PPO) for fairness. The environment can also generate its
own IID demand stream for PPO training when demand_stream is not provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import numpy as np


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    s = float(np.sum(p))
    if (not np.isfinite(s)) or s <= 0.0:
        raise ValueError("pmf normalization failed (sum<=0 or not finite)")
    return p / s


def mean_from_pmf(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(np.dot(np.arange(p.size, dtype=np.float64), p))


def var_from_pmf(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    k = np.arange(p.size, dtype=np.float64)
    mu = float(np.dot(k, p))
    return float(np.dot((k - mu) ** 2, p))


def sample_from_pmf(*, rng: np.random.Generator, pmf: np.ndarray, size: int | Tuple[int, ...]) -> np.ndarray:
    pmf = np.asarray(pmf, dtype=np.float64).reshape(-1)
    support = np.arange(pmf.size, dtype=np.int64)
    return rng.choice(support, size=size, p=pmf).astype(int)


# ---------------------------------------------------------------------
# (Legacy) Gamma -> integer via rounding
# ---------------------------------------------------------------------

def gamma_shape_scale(mean_demand: float, coef_of_var: float) -> Tuple[float, float]:
    """Return (shape, scale) for Gamma(mean, cv)."""
    mean_demand = float(mean_demand)
    coef_of_var = float(coef_of_var)

    if mean_demand <= 0.0:
        return 1.0, 1.0

    if coef_of_var <= 1e-12:
        shape = 1e9
        scale = mean_demand / shape
        return float(shape), float(scale)

    shape = 1.0 / (coef_of_var ** 2)
    scale = mean_demand / shape
    return float(shape), float(scale)


def sample_gamma_int_stream(
    *,
    rng: np.random.Generator,
    mean_demand: float,
    coef_of_var: float,
    size: int,
) -> np.ndarray:
    """Sample integer demand by rounding a Gamma draw (>=0)."""
    shape, scale = gamma_shape_scale(float(mean_demand), float(coef_of_var))
    d = rng.gamma(shape, scale, size=int(size))
    return np.maximum(0, np.rint(d)).astype(int)


# ---------------------------------------------------------------------
# Binomial PMF
# ---------------------------------------------------------------------

def _log_binom_coeffs(n: int) -> np.ndarray:
    """Return log(C(n,k)) for k=0..n."""
    n = int(n)
    k = np.arange(n + 1, dtype=np.float64)
    # log C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    lg = np.vectorize(math.lgamma)
    return (math.lgamma(n + 1.0) - (lg(k + 1.0) + lg((n - k) + 1.0)))


def pmf_binomial_0_to_n(n: int, p: float) -> np.ndarray:
    """PMF of Binomial(n,p) on {0,...,n}."""
    n = int(n)
    p = float(p)
    if n < 0:
        raise ValueError("n must be >= 0")
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in [0,1]")

    if n == 0:
        return np.array([1.0], dtype=np.float64)

    # Stable log-space PMF
    k = np.arange(n + 1, dtype=np.float64)
    logC = _log_binom_coeffs(n)

    if p == 0.0:
        out = np.zeros(n + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    if p == 1.0:
        out = np.zeros(n + 1, dtype=np.float64)
        out[-1] = 1.0
        return out

    logp = math.log(p)
    logq = math.log(1.0 - p)
    logpmf = logC + k * logp + (float(n) - k) * logq

    # normalize for safety
    m = float(np.max(logpmf))
    pmf = np.exp(logpmf - m)
    return _normalize(pmf)


# ---------------------------------------------------------------------
# New demand family 1: Beta-Binomial calibrated to mean and CV
# ---------------------------------------------------------------------

def beta_binomial_ab_from_mean_cv(*, mean: float, cv: float, n: int) -> Tuple[float, float]:
    """Solve (a,b) for a Beta-Binomial(n,a,b) given target mean and CV.

    Mean:  E[D] = n * a/(a+b) = mean
    Var:   Var[D] = n p (1-p) * (t + n)/(t + 1),  where p=a/t, t=a+b

    This family cannot produce variance below the Binomial variance n p (1-p).
    If the requested variance is below that, we return a large t (binomial limit).
    """
    n = int(n)
    mean = float(mean)
    cv = float(cv)
    if n <= 0:
        raise ValueError("n must be >= 1")
    if mean < 0.0 or mean > float(n):
        raise ValueError("mean must be in [0,n]")
    if cv < 0.0:
        raise ValueError("cv must be >= 0")

    if mean == 0.0:
        return 1.0, 1e9

    p = mean / float(n)
    var_target = (cv * mean) ** 2
    base = float(n) * p * (1.0 - p)  # Binomial variance

    # Feasible range for Beta-Binomial variance: [base, base*n]
    var_min = base
    var_max = base * float(n)

    if var_target <= var_min * (1.0 + 1e-12):
        # Close to binomial: take large concentration
        t = 1e9
    elif var_target >= var_max * (1.0 - 1e-12):
        # At/above maximum: take tiny concentration
        t = 1e-9
    else:
        # Solve: var_target = base * (t+n)/(t+1)
        # => (var_target - base) t = base*n - var_target
        t = (base * float(n) - var_target) / (var_target - base)
        if t <= 0.0:
            # Numerical safety
            t = 1e-9

    a = max(1e-9, p * t)
    b = max(1e-9, (1.0 - p) * t)
    return float(a), float(b)


def pmf_beta_binomial_0_to_n(n: int, a: float, b: float) -> np.ndarray:
    """PMF of Beta-Binomial(n,a,b) on {0,...,n}."""
    n = int(n)
    a = float(a)
    b = float(b)
    if n < 0:
        raise ValueError("n must be >= 0")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a,b must be > 0")

    k = np.arange(n + 1, dtype=np.float64)
    lg = np.vectorize(math.lgamma)

    # log C(n,k)
    logC = math.lgamma(n + 1.0) - (lg(k + 1.0) + lg((n - k) + 1.0))

    # log Beta(k+a, n-k+b) - log Beta(a,b)
    log_num = lg(k + a) + lg((n - k) + b) - lg(float(n) + a + b)
    log_den = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    logpmf = logC + log_num - float(log_den)
    m = float(np.max(logpmf))
    pmf = np.exp(logpmf - m)
    return _normalize(pmf)


# ---------------------------------------------------------------------
# New demand family 2: 2-regime Binomial mixture calibrated to mean and CV
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MixBinom2Params:
    n: int = 20
    pi_high: float = 2.0 / 7.0


def _mix2_delta_bounds(*, pbar: float, pi: float) -> float:
    if pi <= 0.0 or pi >= 1.0:
        raise ValueError("pi_high must be in (0,1)")
    ub1 = pbar
    ub2 = (1.0 - pbar) * pi / (1.0 - pi)
    return float(max(0.0, min(ub1, ub2)))


def pmf_mix_binomial_2regime_calibrated(*, mean: float, cv: float, params: MixBinom2Params) -> np.ndarray:
    """Two-regime mixture calibrated to match mean and variance.

    We enforce the mean constraint exactly by parameterizing:
      p_low  = pbar - delta
      p_high = pbar + (1-pi)/pi * delta
    and solve delta to match the target variance.
    """
    n = int(params.n)
    pi = float(params.pi_high)
    mean = float(mean)
    cv = float(cv)

    if not (0.0 <= mean <= float(n)):
        raise ValueError("mean must be in [0,n]")
    if cv < 0.0:
        raise ValueError("cv must be >= 0")

    if mean == 0.0:
        out = np.zeros(n + 1, dtype=np.float64)
        out[0] = 1.0
        return out

    pbar = mean / float(n)
    sigma2_target = (cv * mean) ** 2

    dmax = _mix2_delta_bounds(pbar=pbar, pi=pi)

    def var_for_delta(delta: float) -> float:
        delta = float(delta)
        p_low = pbar - delta
        p_high = pbar + (1.0 - pi) / pi * delta
        m1 = float(n) * p_low
        m2 = float(n) * p_high
        v1 = float(n) * p_low * (1.0 - p_low)
        v2 = float(n) * p_high * (1.0 - p_high)
        return (1.0 - pi) * (v1 + (m1 - mean) ** 2) + pi * (v2 + (m2 - mean) ** 2)

    var0 = var_for_delta(0.0)
    var_max = var_for_delta(dmax)

    if sigma2_target <= var0 * (1.0 + 1e-12):
        delta_star = 0.0
    elif sigma2_target >= var_max * (1.0 - 1e-12):
        delta_star = dmax
    else:
        lo, hi = 0.0, dmax
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            vm = var_for_delta(mid)
            if vm < sigma2_target:
                lo = mid
            else:
                hi = mid
        delta_star = 0.5 * (lo + hi)

    p_low = pbar - delta_star
    p_high = pbar + (1.0 - pi) / pi * delta_star

    pmf = (1.0 - pi) * pmf_binomial_0_to_n(n, p_low) + pi * pmf_binomial_0_to_n(n, p_high)
    return _normalize(pmf)


# ---------------------------------------------------------------------
# New demand family 3: 3-regime Binomial mixture calibrated to mean and CV
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MixBinom3Params:
    n: int = 20
    pi_calm: float = 2.0 / 7.0
    pi_normal: float = 3.0 / 7.0
    pi_promo: float = 2.0 / 7.0


def pmf_mix_binomial_3regime_calibrated(*, mean: float, cv: float, params: MixBinom3Params) -> np.ndarray:
    """Three-regime mixture calibrated to match mean and variance.

    We fix regime probabilities (calm, normal, promo). We anchor the normal regime
    at pbar=mean/n, then parameterize the extremes with a single delta:
      p_promo = pbar + delta
      p_calm  = pbar - (pi_promo/pi_calm) * delta
    so the mean constraint holds exactly. We solve delta to match target variance.
    """
    n = int(params.n)
    pi1 = float(params.pi_calm)
    pi2 = float(params.pi_normal)
    pi3 = float(params.pi_promo)

    tot = pi1 + pi2 + pi3
    if tot <= 0:
        raise ValueError("mixture probabilities must sum to > 0")
    pi1, pi2, pi3 = pi1 / tot, pi2 / tot, pi3 / tot

    if pi1 <= 0.0 or pi3 <= 0.0:
        raise ValueError("pi_calm and pi_promo must be > 0")

    mean = float(mean)
    cv = float(cv)

    if not (0.0 <= mean <= float(n)):
        raise ValueError("mean must be in [0,n]")
    if cv < 0.0:
        raise ValueError("cv must be >= 0")

    if mean == 0.0:
        out = np.zeros(n + 1, dtype=np.float64)
        out[0] = 1.0
        return out

    pbar = mean / float(n)
    sigma2_target = (cv * mean) ** 2

    # delta bounds to keep probabilities in [0,1]
    dmax1 = 1.0 - pbar
    dmax2 = pbar * (pi1 / pi3)
    dmax = float(max(0.0, min(dmax1, dmax2)))

    def var_for_delta(delta: float) -> float:
        delta = float(delta)
        p_promo = pbar + delta
        p_calm = pbar - (pi3 / pi1) * delta
        p_norm = pbar

        m1 = float(n) * p_calm
        m2 = float(n) * p_norm
        m3 = float(n) * p_promo
        v1 = float(n) * p_calm * (1.0 - p_calm)
        v2 = float(n) * p_norm * (1.0 - p_norm)
        v3 = float(n) * p_promo * (1.0 - p_promo)

        return (
            pi1 * (v1 + (m1 - mean) ** 2)
            + pi2 * (v2 + (m2 - mean) ** 2)
            + pi3 * (v3 + (m3 - mean) ** 2)
        )

    var0 = var_for_delta(0.0)
    var_max = var_for_delta(dmax)

    if sigma2_target <= var0 * (1.0 + 1e-12):
        delta_star = 0.0
    elif sigma2_target >= var_max * (1.0 - 1e-12):
        delta_star = dmax
    else:
        lo, hi = 0.0, dmax
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            vm = var_for_delta(mid)
            if vm < sigma2_target:
                lo = mid
            else:
                hi = mid
        delta_star = 0.5 * (lo + hi)

    p_promo = pbar + delta_star
    p_calm = pbar - (pi3 / pi1) * delta_star
    p_norm = pbar

    pmf = (
        pi1 * pmf_binomial_0_to_n(n, p_calm)
        + pi2 * pmf_binomial_0_to_n(n, p_norm)
        + pi3 * pmf_binomial_0_to_n(n, p_promo)
    )
    return _normalize(pmf)


# ---------------------------------------------------------------------
# (Legacy) Truncated Poisson and 3-regime mixture
# ---------------------------------------------------------------------

def pmf_trunc_poisson_0_to_dmax(lam: float, dmax: int) -> np.ndarray:
    """PMF of Poisson(lam) truncated to {0,...,dmax} and renormalized."""
    lam = float(lam)
    dmax = int(dmax)
    if dmax < 0:
        raise ValueError("dmax must be >= 0")
    if lam < 0.0:
        raise ValueError("lam must be >= 0")

    p = np.empty(dmax + 1, dtype=np.float64)
    p[0] = math.exp(-lam)
    for k in range(dmax):
        p[k + 1] = p[k] * lam / float(k + 1)
    return _normalize(p)


def pmf_mix_poisson_3regime(
    dmax: int,
    mean_target: float,
    lam1: float,
    lam2: float,
    lam3: float,
    w1: float,
    w2: float,
) -> np.ndarray:
    """Three-regime mixture of truncated Poissons, calibrated to mean_target."""
    if w1 <= 0.0 or w2 <= 0.0:
        raise ValueError("w1 and w2 must be > 0")

    p1 = pmf_trunc_poisson_0_to_dmax(lam1, dmax)
    p2 = pmf_trunc_poisson_0_to_dmax(lam2, dmax)
    p3 = pmf_trunc_poisson_0_to_dmax(lam3, dmax)

    mu1 = mean_from_pmf(p1)
    mu2 = mean_from_pmf(p2)
    mu3 = mean_from_pmf(p3)

    den = (mu3 - float(mean_target))
    if abs(den) < 1e-14:
        raise ValueError("mu3 too close to mean_target; cannot solve for w3")

    num = float(w1) * (mu1 - float(mean_target)) + float(w2) * (mu2 - float(mean_target))
    w3 = - num / den
    if w3 <= 0.0:
        raise ValueError("Solved w3 is nonpositive; adjust lam1/lam2/lam3 or w1/w2")

    tot = float(w1) + float(w2) + float(w3)
    pi1 = float(w1) / tot
    pi2 = float(w2) / tot
    pi3 = float(w3) / tot

    p = pi1 * p1 + pi2 * p2 + pi3 * p3
    return _normalize(p)


@dataclass(frozen=True)
class MixPoisson3Params:
    dmax: int = 20
    lam1: float = 1.0
    lam2: Optional[float] = None
    lam3: float = 12.0
    w1: float = 0.2
    w2: float = 1.0

    def resolved(self, *, mean_target: float) -> "MixPoisson3Params":
        if self.lam2 is None:
            lam2 = float(mean_target)
        else:
            lam2_raw = float(self.lam2)
            lam2 = float(mean_target) if (not math.isfinite(lam2_raw)) else lam2_raw
        return MixPoisson3Params(
            dmax=int(self.dmax),
            lam1=float(self.lam1),
            lam2=float(lam2),
            lam3=float(self.lam3),
            w1=float(self.w1),
            w2=float(self.w2),
        )


_PMF_CACHE: Dict[Tuple[Any, ...], np.ndarray] = {}


def pmf_cached(
    *,
    demand_type: str,
    mean: float,
    cv: float,
    n: int,
    mix_pi_high: Optional[float] = None,
    mix_pi_calm: Optional[float] = None,
    mix_pi_normal: Optional[float] = None,
    mix_pi_promo: Optional[float] = None,
    # legacy poisson-mixture knobs
    mix_lam1: Optional[float] = None,
    mix_lam2: Optional[float] = None,
    mix_lam3: Optional[float] = None,
    mix_w1: Optional[float] = None,
    mix_w2: Optional[float] = None,
) -> np.ndarray:
    """Return cached PMF for bounded models.

    For demand_type='gamma', there is no PMF here (sampling is continuous+rounding).
    """
    dt = str(demand_type).strip().lower()
    mean = float(mean)
    cv = float(cv)
    n = int(n)

    key = (
        dt,
        mean,
        cv,
        n,
        mix_pi_high,
        mix_pi_calm,
        mix_pi_normal,
        mix_pi_promo,
        mix_lam1,
        mix_lam2,
        mix_lam3,
        mix_w1,
        mix_w2,
    )
    if key in _PMF_CACHE:
        return _PMF_CACHE[key]

    if dt in ("beta_binomial", "bbinom", "beta-binom"):
        a, b = beta_binomial_ab_from_mean_cv(mean=mean, cv=cv, n=n)
        pmf = pmf_beta_binomial_0_to_n(n, a, b)

    elif dt in ("mix_binomial_2regime", "mix_binom_2", "mix2"):
        pi = float(mix_pi_high) if mix_pi_high is not None else (2.0 / 7.0)
        pmf = pmf_mix_binomial_2regime_calibrated(mean=mean, cv=cv, params=MixBinom2Params(n=n, pi_high=pi))

    elif dt in ("mix_binomial_3regime", "mix_binom_3", "mix3"):
        pi_c = float(mix_pi_calm) if mix_pi_calm is not None else (2.0 / 7.0)
        pi_n = float(mix_pi_normal) if mix_pi_normal is not None else (3.0 / 7.0)
        pi_p = float(mix_pi_promo) if mix_pi_promo is not None else (2.0 / 7.0)
        pmf = pmf_mix_binomial_3regime_calibrated(mean=mean, cv=cv, params=MixBinom3Params(n=n, pi_calm=pi_c, pi_normal=pi_n, pi_promo=pi_p))

    elif dt in ("mix_poisson_3regime", "mix_poisson_3", "mix_poisson", "mix3_poisson"):
        mp = MixPoisson3Params(
            dmax=int(n),
            lam1=float(mix_lam1) if mix_lam1 is not None else 1.0,
            lam2=float(mix_lam2) if mix_lam2 is not None else None,
            lam3=float(mix_lam3) if mix_lam3 is not None else 12.0,
            w1=float(mix_w1) if mix_w1 is not None else 0.2,
            w2=float(mix_w2) if mix_w2 is not None else 1.0,
        ).resolved(mean_target=float(mean))
        pmf = pmf_mix_poisson_3regime(int(mp.dmax), float(mean), float(mp.lam1), float(mp.lam2), float(mp.lam3), float(mp.w1), float(mp.w2))

    else:
        raise ValueError(f"Unknown demand_type={demand_type!r}")

    _PMF_CACHE[key] = pmf
    return pmf


def sample_demand_stream(
    *,
    rng: np.random.Generator,
    mean: float,
    cv: float,
    horizon: int,
    demand_type: str,
    n: int,
    mix_pi_high: Optional[float] = None,
    mix_pi_calm: Optional[float] = None,
    mix_pi_normal: Optional[float] = None,
    mix_pi_promo: Optional[float] = None,
    # legacy poisson-mixture knobs
    mix_lam1: Optional[float] = None,
    mix_lam2: Optional[float] = None,
    mix_lam3: Optional[float] = None,
    mix_w1: Optional[float] = None,
    mix_w2: Optional[float] = None,
) -> np.ndarray:
    dt = str(demand_type).strip().lower()
    if dt in ("gamma", "gamma_round", "gamma_iid"):
        return sample_gamma_int_stream(rng=rng, mean_demand=float(mean), coef_of_var=float(cv), size=int(horizon))

    pmf = pmf_cached(
        demand_type=dt,
        mean=float(mean),
        cv=float(cv),
        n=int(n),
        mix_pi_high=mix_pi_high,
        mix_pi_calm=mix_pi_calm,
        mix_pi_normal=mix_pi_normal,
        mix_pi_promo=mix_pi_promo,
        mix_lam1=mix_lam1,
        mix_lam2=mix_lam2,
        mix_lam3=mix_lam3,
        mix_w1=mix_w1,
        mix_w2=mix_w2,
    )
    return sample_from_pmf(rng=rng, pmf=pmf, size=int(horizon))


