"""PIL_GY_policy.py

Projected Inventory Level (PIL) benchmark for the Green/Yellow (G/Y) inventory
environment.

This file implements a Bu et al.-style PIL policy adapted to the **two-stream**
Green/Yellow dynamics used in the TTI reproducibility package:

  - Total demand is split into green vs yellow demand using the environment's
    (paper-consistent) beta parameter.
  - Green and yellow demand are served **separately** from their respective
    inventories (no pooling / substitution).
  - Green items may deteriorate into yellow items after aging according to the
    environment's alpha parameter.
  - The correction term is computed on the *full* (G,Y,pipeline) state, not on
    an aggregated single-quality approximation.

Order quantity
--------------
For lost-sales systems, the Bu et al. PIL order quantity can be written as:

    q_t = ( S - IP_t + L * mu + C(s_t) )^+

where:
  - IP_t is the current inventory position (on-hand + pipeline) in *units*.
  - mu is the mean of total demand per period.
  - C(s_t) = E[ sum_{tau=0}^{L-1} ( outdating_tau - lost_tau ) | s_t ]
    computed over the lead-time horizon assuming **no new orders** inside the
    correction horizon (only existing pipeline arrivals).

We estimate C(s_t) either via Monte Carlo (PIL) or via a deterministic
mean-demand rollout (APIL).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from demand_models import pmf_cached


# ---------------------------------------------------------------------------
# Demand helpers
# ---------------------------------------------------------------------------
def _gamma_shape_scale(mean_demand: float, coef_of_var: float) -> Tuple[float, float]:
    """Return (shape, scale) for Gamma(mean, cv)."""
    mean_demand = float(mean_demand)
    coef_of_var = float(coef_of_var)

    if mean_demand <= 0:
        return 1.0, 1.0

    if coef_of_var <= 1e-12:
        # Nearly deterministic.
        shape = 1e9
        scale = mean_demand / shape
        return float(shape), float(scale)

    shape = 1.0 / (coef_of_var ** 2)
    scale = mean_demand / shape
    return float(shape), float(scale)


def _split_demand(total_demand: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split total demand into green/yellow using the *environment* rule.

    Environment rule (see InventoryEnvGY_Config.py):
        Dg = round(beta * D)
        Dy = D - Dg
        and clamp Dg to [0, D].
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    D = np.asarray(total_demand, dtype=np.int64)
    Dg = np.rint(beta * D).astype(np.int64)
    Dg = np.clip(Dg, 0, D)
    Dy = D - Dg
    return Dg, Dy


# ---------------------------------------------------------------------------
# Correction term: full G/Y dynamics
# ---------------------------------------------------------------------------

def _opt_float(x: Optional[float]) -> Optional[float]:
    """Convert to finite float else None (handles NaN from Excel)."""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


# ---------------------------------------------------------------------------
# Issuing helper (integer, consistent with env)
# ---------------------------------------------------------------------------
def _issue_from_buckets_int(
    buckets: np.ndarray,
    demand: int,
    *,
    issuing_policy: str,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Issue inventory from integer age buckets.

    Parameters
    ----------
    buckets:
        1D numpy array of nonnegative integers. Modified in place.
    demand:
        Nonnegative integer demand.
    issuing_policy:
        "fifo", "lifo", or "random".
    rng:
        RNG used only for issuing_policy="random".

    Returns
    -------
    served, remaining_demand
    """
    d = int(demand)
    if d <= 0:
        return 0, 0
    if buckets is None or buckets.size == 0:
        return 0, d

    pol = str(issuing_policy).strip().lower()
    if pol not in {"fifo", "lifo", "random"}:
        pol = "fifo"

    if pol == "fifo":
        remaining = d
        served = 0
        for i in range(int(buckets.size)):
            if remaining <= 0:
                break
            avail = int(buckets[i])
            if avail <= 0:
                continue
            take = avail if avail < remaining else remaining
            buckets[i] = int(avail - take)
            remaining -= take
            served += take
        return int(served), int(remaining)

    if pol == "lifo":
        remaining = d
        served = 0
        for i in range(int(buckets.size) - 1, -1, -1):
            if remaining <= 0:
                break
            avail = int(buckets[i])
            if avail <= 0:
                continue
            take = avail if avail < remaining else remaining
            buckets[i] = int(avail - take)
            remaining -= take
            served += take
        return int(served), int(remaining)

    # pol == "random": multivariate hypergeometric allocation (sequential)
    total = int(np.sum(buckets))
    if total <= 0:
        return 0, d

    served = int(d if d < total else total)
    remaining = int(d - served)
    if served <= 0:
        return 0, d

    remaining_sample = served
    remaining_total = total
    for i in range(int(buckets.size)):
        if remaining_sample <= 0:
            break
        good = int(buckets[i])
        if good <= 0:
            remaining_total -= good
            continue
        bad = int(remaining_total - good)
        if bad < 0:
            bad = 0
        draw = int(rng.hypergeometric(good, bad, remaining_sample))
        if draw > good:
            draw = good
        buckets[i] = int(good - draw)
        remaining_sample -= draw
        remaining_total -= good

    return int(served), int(remaining)


# ---------------------------------------------------------------------------
# Correction term: full G/Y dynamics, issuing-policy aware
# ---------------------------------------------------------------------------
def _correction_mc_fifo_gy(
    *,
    green_on_hand: np.ndarray,
    green_pipeline: np.ndarray,
    yellow_on_hand: np.ndarray,
    m: int,
    L: int,
    alpha: float,
    beta: float,
    mean_demand: float,
    coef_of_var: float,
    demand_type: str = "gamma",
    demand_dmax: int = 20,
    # Binomial-mixture knobs (optional)
    mix_pi_high: Optional[float] = None,
    mix_pi_calm: Optional[float] = None,
    mix_pi_normal: Optional[float] = None,
    mix_pi_promo: Optional[float] = None,
    # Legacy Poisson-mixture knobs (optional)
    mix_lam1: Optional[float] = None,
    mix_lam2: Optional[float] = None,
    mix_lam3: Optional[float] = None,
    mix_w1: Optional[float] = None,
    mix_w2: Optional[float] = None,
    n_mc: int = 1000,
    rng: np.random.Generator,
    issuing_policy: str = "fifo",
) -> float:
    """MC estimate of E[ sum_{t=0}^{L-1} (outdating_t - lost_t) ] under G/Y.

    This simulation matches the environment dynamics:
      demand -> issue (fifo/lifo/random) -> outdating -> shift/age -> binomial deterioration

    Notes
    -----
    - Uses integer state transitions (no fractional stock).
    - For issuing_policy="random", uses hypergeometric issuing (uniform among all on-hand units).
    """
    m = int(m)
    L = int(L)
    if L <= 0:
        return 0.0

    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = float(np.clip(beta, 0.0, 1.0))

    pmf = pmf_cached(
        demand_type=str(demand_type),
        mean=float(mean_demand),
        cv=float(coef_of_var),
        n=int(demand_dmax),
        mix_pi_high=_opt_float(mix_pi_high),
        mix_pi_calm=_opt_float(mix_pi_calm),
        mix_pi_normal=_opt_float(mix_pi_normal),
        mix_pi_promo=_opt_float(mix_pi_promo),
        mix_lam1=_opt_float(mix_lam1),
        mix_lam2=_opt_float(mix_lam2),
        mix_lam3=_opt_float(mix_lam3),
        mix_w1=_opt_float(mix_w1),
        mix_w2=_opt_float(mix_w2),
    )
    support = np.arange(int(len(pmf)), dtype=np.int64)
    D = rng.choice(support, size=(int(n_mc), int(L)), replace=True, p=pmf).astype(np.int64)

    outd_acc = 0.0
    lost_acc = 0.0

    for j in range(int(n_mc)):
        g_on = np.asarray(green_on_hand, dtype=np.int64).reshape(-1)[:m].copy()
        g_pipe = np.asarray(green_pipeline, dtype=np.int64).reshape(-1).copy()
        if g_pipe.size < max(L - 1, 0):
            g_pipe = np.pad(g_pipe, (0, max(L - 1, 0) - g_pipe.size), mode="constant")
        else:
            g_pipe = g_pipe[: max(L - 1, 0)]
        G = np.concatenate([g_on, g_pipe], axis=0).astype(np.int64)

        Y = np.asarray(yellow_on_hand, dtype=np.int64).reshape(-1)[: max(m - 1, 0)].copy()
        y_len = int(Y.size)

        for t in range(int(L)):
            Dt = int(D[j, t])
            Dg = int(np.rint(beta * Dt))
            if Dg < 0:
                Dg = 0
            if Dg > Dt:
                Dg = Dt
            Dy = int(Dt - Dg)

            _, rem_g = _issue_from_buckets_int(G[:m], Dg, issuing_policy=issuing_policy, rng=rng)
            _, rem_y = _issue_from_buckets_int(Y, Dy, issuing_policy=issuing_policy, rng=rng)

            outd_g = int(G[0]) if G.size > 0 else 0
            outd_y = int(Y[0]) if y_len > 0 else 0

            outd_acc += float(outd_g + outd_y)
            lost_acc += float(rem_g + rem_y)

            if G.size > 0:
                G = np.roll(G, -1)
                G[-1] = 0

            if y_len > 0:
                Y = np.roll(Y, -1)
                Y[-1] = 0

            if alpha > 0.0 and y_len > 0 and m > 1:
                k = int(min(m - 1, y_len))
                if k > 0:
                    n = G[:k].astype(np.int64)
                    det = rng.binomial(n=n, p=float(alpha)).astype(np.int64)
                    G[:k] -= det
                    Y[:k] += det

    return float((outd_acc - lost_acc) / float(max(1, int(n_mc))))


def _correction_deterministic_fifo_gy(
    *,
    green_on_hand: np.ndarray,
    green_pipeline: np.ndarray,
    yellow_on_hand: np.ndarray,
    m: int,
    L: int,
    alpha: float,
    beta: float,
    mean_demand: float,
    issuing_policy: str = "fifo",
) -> float:
    """Deterministic approximation of the correction term under mean demand.

    Respects issuing_policy:
      - fifo/lifo: deterministic issue order
      - random: expected random picking => proportional depletion

    Deterioration is applied in expectation: alpha * G.
    """
    m = int(m)
    L = int(L)
    if L <= 0:
        return 0.0

    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = float(np.clip(beta, 0.0, 1.0))
    mu = float(mean_demand)

    g_on = np.asarray(green_on_hand, dtype=np.float64).reshape(-1)[:m].copy()
    g_pipe = np.asarray(green_pipeline, dtype=np.float64).reshape(-1).copy()
    if g_pipe.size < max(L - 1, 0):
        g_pipe = np.pad(g_pipe, (0, max(L - 1, 0) - g_pipe.size), mode="constant")
    else:
        g_pipe = g_pipe[: max(L - 1, 0)]
    G = np.concatenate([g_on, g_pipe], axis=0)

    Y = np.asarray(yellow_on_hand, dtype=np.float64).reshape(-1)[: max(m - 1, 0)].copy()
    y_len = int(Y.size)

    pol = str(issuing_policy).strip().lower()
    if pol not in {"fifo", "lifo", "random"}:
        pol = "fifo"

    outd_total = 0.0
    lost_total = 0.0

    for _ in range(int(L)):
        dg = beta * mu
        dy = mu - dg

        need_g = float(dg)
        g_len = int(min(m, int(G.size)))
        if pol == "random":
            total_g = float(np.sum(G[:g_len])) if g_len > 0 else 0.0
            if total_g > 0.0:
                served = min(need_g, total_g)
                frac = served / total_g
                G[:g_len] = G[:g_len] * (1.0 - frac)
                need_g -= served
            lost_g = max(0.0, need_g)
        else:
            idxs_g = range(g_len - 1, -1, -1) if pol == "lifo" else range(g_len)
            for i in idxs_g:
                if need_g <= 0.0:
                    break
                take = min(float(G[i]), need_g)
                G[i] -= take
                need_g -= take
            lost_g = max(0.0, need_g)

        need_y = float(dy)
        if pol == "random":
            total_y = float(np.sum(Y)) if y_len > 0 else 0.0
            if total_y > 0.0:
                served = min(need_y, total_y)
                frac = served / total_y
                Y[:y_len] = Y[:y_len] * (1.0 - frac)
                need_y -= served
            lost_y = max(0.0, need_y)
        else:
            idxs_y = range(y_len - 1, -1, -1) if pol == "lifo" else range(y_len)
            for i in idxs_y:
                if need_y <= 0.0:
                    break
                take = min(float(Y[i]), need_y)
                Y[i] -= take
                need_y -= take
            lost_y = max(0.0, need_y)

        outd_g = float(G[0]) if G.size > 0 else 0.0
        outd_y = float(Y[0]) if y_len > 0 else 0.0
        outd_total += (outd_g + outd_y)
        lost_total += (lost_g + lost_y)

        if G.size > 0:
            G[0] = 0.0
            G = np.roll(G, -1)
            G[-1] = 0.0

        if y_len > 0:
            Y[0] = 0.0
            Y = np.roll(Y, -1)
            Y[-1] = 0.0

        if alpha > 0.0 and y_len > 0 and m > 1:
            k = int(min(m - 1, y_len))
            if k > 0 and G.size >= k:
                det = alpha * G[:k]
                G[:k] -= det
                Y[:k] += det

    return float(outd_total - lost_total)


def pil_order_quantity(
    *,
    S_pil: float,
    green_on_hand: np.ndarray,
    green_pipeline: np.ndarray,
    yellow_on_hand: np.ndarray,
    m: int,
    L: int,
    alpha: float,
    beta: float,
    mean_demand: float,
    coef_of_var: float,
    issuing_policy: str = "fifo",
    demand_type: str = "gamma",
    demand_dmax: int = 20,
    # Binomial-mixture knobs (optional)
    mix_pi_high: Optional[float] = None,
    mix_pi_calm: Optional[float] = None,
    mix_pi_normal: Optional[float] = None,
    mix_pi_promo: Optional[float] = None,
    # Legacy Poisson-mixture knobs (optional)
    mix_lam1: Optional[float] = None,
    mix_lam2: Optional[float] = None,
    mix_lam3: Optional[float] = None,
    mix_w1: Optional[float] = None,
    mix_w2: Optional[float] = None,
    n_mc: int = 1000,
    use_correction: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_order: Optional[int] = None,
) -> int:
    """Compute PIL order quantity (integer units) for the G/Y environment.

    Parameters
    ----------
    S_pil : float
        Tuned projected-inventory target.
    green_on_hand : array-like, shape (m,)
        Green on-hand buckets, oldest -> youngest.
    green_pipeline : array-like, shape (L-1,)
        Green pipeline buckets, head -> tail (arrivals).
    yellow_on_hand : array-like, shape (m-1,)
        Yellow on-hand buckets, oldest -> youngest.
    m, L : int
        Shelf-life and lead-time.
    alpha, beta : float
        Deterioration rate and (paper-consistent) green-demand share.
    mean_demand, coef_of_var : float
        Total-demand Gamma parameters.
    n_mc : int
        If > 0 and use_correction=True: Monte Carlo paths for C(s).
        If <= 0 and use_correction=True: deterministic APIL rollout.
    use_correction : bool
        If False: C(s) = 0.
    rng : np.random.Generator
        RNG for MC.
    max_order : int
        Optional cap on q.
    """
    m = int(m)
    L = int(L)
    if rng is None:
        rng = np.random.default_rng()

    g_on = np.asarray(green_on_hand, dtype=np.float64).reshape(-1)
    g_pipe = np.asarray(green_pipeline, dtype=np.float64).reshape(-1)
    y_on = np.asarray(yellow_on_hand, dtype=np.float64).reshape(-1)

    # Inventory position (units) = green on-hand + yellow on-hand + green pipeline.
    inv_pos_total = float(np.sum(g_on[:m]) + np.sum(g_pipe[: max(L - 1, 0)]) + np.sum(y_on[: max(m - 1, 0)]))

    corr = 0.0
    if bool(use_correction):
        if int(n_mc) > 0:
            corr = _correction_mc_fifo_gy(
                green_on_hand=g_on,
                green_pipeline=g_pipe,
                yellow_on_hand=y_on,
                m=m,
                L=L,
                alpha=float(alpha),
                beta=float(beta),
                mean_demand=float(mean_demand),
                coef_of_var=float(coef_of_var),
                demand_type=str(demand_type),
                demand_dmax=int(demand_dmax),
                mix_pi_high=mix_pi_high,
                mix_pi_calm=mix_pi_calm,
                mix_pi_normal=mix_pi_normal,
                mix_pi_promo=mix_pi_promo,
                mix_lam1=mix_lam1,
                mix_lam2=mix_lam2,
                mix_lam3=mix_lam3,
                mix_w1=mix_w1,
                mix_w2=mix_w2,
                n_mc=int(n_mc),
                rng=rng,
                issuing_policy=str(issuing_policy),
            )
        else:
            corr = _correction_deterministic_fifo_gy(
                green_on_hand=g_on,
                green_pipeline=g_pipe,
                yellow_on_hand=y_on,
                m=m,
                L=L,
                alpha=float(alpha),
                beta=float(beta),
                mean_demand=float(mean_demand),
                issuing_policy=str(issuing_policy),
            )

    # q = (S - IP + L*mu + corr)^+
    q = float(S_pil) - inv_pos_total + float(L) * float(mean_demand) + float(corr)
    if q < 0.0:
        q = 0.0
    if max_order is not None:
        q = min(q, float(max_order))
    return int(np.rint(q))


