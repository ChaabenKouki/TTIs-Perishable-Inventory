"""InventoryEnvGY_Config.py

Gymnasium environment for the TTI green/yellow (TTI) perishable inventory system.

Reference truth for dynamics: TTI.pdf ("System with time-temperature indicator..." section).

State representation used in code (paper-consistent):
- Green inventory x^G_t is tracked by remaining life buckets + pipeline:
    green_stock[0:m]      -> on-hand buckets with remaining life 1..m (FIFO issues from index 0)
    green_stock[m:]       -> pipeline positions (length L-1), where green_stock[m] arrives next period
- Yellow inventory x^Y_t is tracked by remaining life buckets:
    yellow_stock[0:m-1]   -> on-hand buckets with remaining life 1..m-1 (FIFO issues from index 0)

Within-period event sequence implemented in step():
1) Observe state at start of period t (on-hand age buckets + pipeline).
2) Place order q_t (bounded integer). It will arrive after lead time via pipeline shift.
3) Sample total demand D_t from the configured stationary demand model (iid), as specified by demand_type.
4) Split demand using paper definition (after mapping beta from Excel):
       D^G_t = round(beta_paper * D_t)
       D^Y_t = D_t - D^G_t
5) Satisfy green demand FIFO from green on-hand only; lost sales allowed.
6) Satisfy yellow demand FIFO from yellow on-hand only; lost sales allowed.
7) Compute period profit = revenue - ordering - holding - lost-sales-penalty - outdating.
   Holding cost is charged on post-service leftovers (including items that will outdate at end).
8) Outdating at end of period: expired_green = remaining in green bucket 0;
   expired_yellow = remaining in yellow bucket 0.
9) Age/pipeline shift one step forward:
       green_stock = roll(-1) ; green_stock[-1]=0 ; then inject q_t at tail position.
       yellow_stock = roll(-1) ; yellow_stock[-1]=0 (to remove expired bucket).
10) Deterioration (green->yellow) applied AFTER the shift:
       for buckets with remaining life 1..m-1 (indices 0..m-2):
           deteriorated_i ~ Binomial(n_i, alpha) units move from green to yellow at the same remaining-life index.

Notes
-----
- FIX-1 (beta mapping): configurations.xlsx stores beta_excel in [0,0.2] as the *yellow share*.
  Paper uses beta as the *green share*. Therefore:
      beta_paper = 1 - beta_excel

- FIX-2 (yellow length): yellow_stock has length m-1 (no "remaining life m" yellow bucket).

- FIX-3 (yellow expiration): after np.roll(yellow_stock, -1) we MUST clear the last bucket.

- FIX-6 (reward scaling): raw_profit is computed in monetary units, then the RL reward is
      reward = raw_profit / reward_scale.
  All CSV/plots should report raw_profit per period.

- FIX-8 (demand stream injection): for fair policy comparisons, you can pass a deterministic
  integer demand_stream (length >= episode_length) via config['demand_stream'].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover
    import gym
    from gym import spaces

import numpy as np

import math

from demand_models import sample_demand_stream


def _opt_float(x):
    """Convert to finite float, else None (handles NaN from Excel)."""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


class InventoryEnvGYConfig(gym.Env):
    """Green/Yellow perishable inventory environment (Gymnasium)."""

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # -----------------------------
        # Core parameters (from Excel)
        # -----------------------------
        self.m = int(config["m"])
        self.L = int(config["L"])
        # Support both Excel-style "Alpha" and code-style "alpha"
        self.Alpha = float(config.get("Alpha", config.get("alpha", 0.0)))
        self.Alpha = float(np.clip(self.Alpha, 0.0, 1.0))
        self.alpha = self.Alpha  # canonical internal name used in step()

        # Beta mapping (Excel stores yellow share)
        self.beta_excel = float(config.get("beta_excel", config.get("beta", 0.0)))
        self.beta_excel = float(np.clip(self.beta_excel, 0.0, 1.0))
        self.beta = 1.0 - self.beta_excel  # paper-consistent beta (green share)
        self.beta = float(np.clip(self.beta, 0.0, 1.0))

        # Demand parameters (iid, stationary; configured by Excel via demand_type)
        self.demand_type = str(config.get("demand_type", config.get("demand_model", ""))).strip().lower()
        if self.demand_type in ("", "nan", "none"):
            raise ValueError(
                "Config must provide a valid demand_type (or legacy demand_model). "
                "Demand must come from Excel (no silent default)."
            )
        if self.demand_type.startswith("gamma"):
            raise ValueError(
                f"demand_type={self.demand_type!r} is not allowed in this project. "
                "Use a bounded demand family from Excel (e.g., beta_binomial, mix_binomial_2regime, mix_binomial_3regime, mix_poisson_3regime)."
            )
        if self.demand_type in ("", "nan", "none"):
            raise ValueError("demand_type must be provided (non-empty).")
        if self.demand_type.startswith("gamma"):
            raise ValueError(
                f"demand_type='{self.demand_type}' is not allowed in this project. "
                "Use a bounded family (e.g., beta_binomial, mix_binomial_2regime, mix_binomial_3regime, mix_poisson_3regime)."
            )

        self.mean_demand = float(config["mean_demand"])
        self.coef_of_var = float(config["coef_of_var"])

        self.demand_dmax = int(config.get("demand_dmax", config.get("dmax", 65)))
        if self.demand_dmax < 0:
            raise ValueError("demand_dmax must be >= 0")

        # Optional regime / mixture parameters (NaN-safe)
        self.mix_pi_high = _opt_float(config.get("mix_pi_high", config.get("pi_high", None)))
        self.mix_pi_calm = _opt_float(config.get("mix_pi_calm", None))
        self.mix_pi_normal = _opt_float(config.get("mix_pi_normal", None))
        self.mix_pi_promo = _opt_float(config.get("mix_pi_promo", None))

        # Legacy Poisson-mixture knobs (optional; NaN-safe)
        self.mix_lam1 = _opt_float(config.get("mix_lam1", config.get("lam1", None)))
        self.mix_lam2 = _opt_float(config.get("mix_lam2", config.get("lam2", None)))
        self.mix_lam3 = _opt_float(config.get("mix_lam3", config.get("lam3", None)))
        self.mix_w1 = _opt_float(config.get("mix_w1", config.get("w1", None)))
        self.mix_w2 = _opt_float(config.get("mix_w2", config.get("w2", None)))

        # CV validation: required for bounded calibrations; tolerated for Poisson-mixture mode.
        if self.demand_type in ("mix_poisson_3regime", "mix_poisson_3", "mix3", "mix_poisson"):
            pass
        else:
            if self.coef_of_var <= 0:
                raise ValueError("coef_of_var must be > 0 for bounded demand calibration")

        # Economic parameters
        self.p1 = float(config["p1"])  # selling price green
        self.p2 = float(config["p2"])  # selling price yellow
        self.c = float(config["c"])    # unit ordering cost
        self.h = float(config["h"])    # holding cost per unit per period
        self.b1 = float(config["b1"])  # lost sales penalty green
        self.b2 = float(config["b2"])  # lost sales penalty yellow
        self.w = float(config["w"])    # outdating cost per unit

        # Optional costs for optional sensitivity
        self.fixed_order_cost = float(config.get("fixed_order_cost", 0.0))
        self.tti_unit_cost = float(config.get("tti_unit_cost", 0.0))

        # -----------------------------
        # Long-run settings
        # -----------------------------
        self.episode_length = int(config.get("episode_length", 2000))
        self.warm_up_fraction = float(config.get("warm_up_fraction", 0.10))
        self.warm_up_fraction = float(np.clip(self.warm_up_fraction, 0.0, 0.99))
        self.warm_up_period = int(round(self.warm_up_fraction * self.episode_length))

        # Reward scaling (RL stability): reward = raw_profit / reward_scale
        self.reward_scale = float(config.get("reward_scale", 100.0))
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be > 0")

        # Max order quantity (action bounds)
        self.max_order = int(config.get("max_order", 50))
        if self.max_order < 0:
            raise ValueError("max_order must be >= 0")

        # Sensor/TTI classification error: affects OBSERVATION only (not the true state)
        self.sensor_error_eps = float(config.get("sensor_error_eps", 0.0))
        # Optional: enrich observation with additional deterministic correction components
        # (still no reward shaping, purely state features).
        self.obs_extra_features = bool(config.get("obs_extra_features", False))
        self.sensor_error_eps = float(np.clip(self.sensor_error_eps, 0.0, 0.5))

        # Issuing policy (controls how demand is served from age buckets).
        # fifo: oldest first (index 0 upward)
        # lifo: youngest first (reverse)
        # random: discrete random pick among all on-hand items (multivariate hypergeometric).
        self.issuing_policy = str(config.get("issuing_policy", "fifo")).strip().lower()
        if self.issuing_policy not in ("fifo", "lifo", "random"):
            raise ValueError(
                f"Unknown issuing_policy={self.issuing_policy!r}. Must be one of: fifo, lifo, random."
            )


        # -----------------------------
        # RNG and optional demand stream injection
        # -----------------------------
        self._seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self._seed)

        self.demand_stream: Optional[np.ndarray] = None
        self._demand_idx: int = 0
        if "demand_stream" in config and config["demand_stream"] is not None:
            ds = np.asarray(config["demand_stream"], dtype=int)
            if ds.ndim != 1:
                raise ValueError("demand_stream must be 1D")
            self.demand_stream = ds

        # True if a fixed external stream is provided (benchmarking). When False
        # (training/evaluation), we regenerate a fresh stream at every reset so
        # each episode is stochastic.
        self._external_demand_stream = self.demand_stream is not None

        # Demand stream sampling is handled via demand_models.sample_demand_stream
        # using (demand_type, mean_demand, coef_of_var, demand_dmax, mix_*).

        # -----------------------------
        # State arrays
        # -----------------------------
        green_len = self.m + max(self.L - 1, 0)
        yellow_len = max(self.m - 1, 0)

        self.green_stock = np.zeros(green_len, dtype=np.int64)
        self.yellow_stock = np.zeros(yellow_len, dtype=np.int64)

        self.current_step = 0

        # Initial demands used as part of state (as in your original PPO setup)
        self.initial_green_demand = 0.0
        self.initial_yellow_demand = 0.0

        # Convenience cached values (updated every step)
        self.on_hand_green = 0.0
        self.pipeline_green = 0.0
        self.on_hand_yellow = 0.0
        self.inventory_position_total = 0.0

        # -----------------------------
        # Spaces
        # -----------------------------
        observation_length = int(self.green_stock.size + self.yellow_stock.size + 8 + (3 if self.obs_extra_features else 0))

        # Observation is float (allows sensor error mixing).
        self.observation_space = spaces.Box(
            low=-1e12,
            high=1e12,
            shape=(observation_length,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.max_order + 1)

    # -----------------------------
    # Gymnasium API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)

        self.current_step = 0
        self._demand_idx = 0

        # Clear state
        self.green_stock[:] = 0
        if self.yellow_stock.size:
            self.yellow_stock[:] = 0

        # Demand stream handling
        # - If an external stream was injected (benchmarking), keep it fixed across resets.
        # - Otherwise (training/evaluation), draw a *fresh* stream each reset so that
        #   PPO sees stochastic episodes and the evaluation std/CI is meaningful.
        if not self._external_demand_stream:
            self.demand_stream = sample_demand_stream(
                rng=self.rng,
                mean=self.mean_demand,
                cv=self.coef_of_var,
                horizon=int(self.episode_length),
                demand_type=self.demand_type,
                n=int(self.demand_dmax),
                mix_pi_high=self.mix_pi_high,
                mix_pi_calm=self.mix_pi_calm,
                mix_pi_normal=self.mix_pi_normal,
                mix_pi_promo=self.mix_pi_promo,
                mix_lam1=self.mix_lam1,
                mix_lam2=self.mix_lam2,
                mix_lam3=self.mix_lam3,
                mix_w1=self.mix_w1,
                mix_w2=self.mix_w2,
            ).astype(int)
        self._demand_idx = 0

        # IMPORTANT (PPO feature hygiene):
        # The observation vector includes (initial_green_demand, initial_yellow_demand).
        # In earlier drafts we set those using demand_stream[0] at reset(). That leaks
        # the first-period demand before the agent acts and also creates a constant
        # demand feature if not updated later.
        #
        # To avoid any leakage and keep the feature interpretable, we initialize to 0
        # and update them each period to the *realized* demand split of the previous
        # period (see step()).
        self.initial_green_demand = 0.0
        self.initial_yellow_demand = 0.0

        # Update cached inventory position
        self._update_cached_inventory_levels()

        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Any):
        # Convert action to integer order quantity
        if isinstance(action, (np.ndarray, list, tuple)):
            # SB3 sometimes returns array([a]) or scalar array
            action = np.asarray(action).item()
        q_t = int(action)
        q_t = int(np.clip(q_t, 0, self.max_order))

        # -----------------------------
        # Demand for this period
        # -----------------------------
        assert self.demand_stream is not None
        if self._demand_idx >= len(self.demand_stream):
            # Safety: extend stream if someone runs longer than episode_length
            extra = sample_demand_stream(
                rng=self.rng,
                mean=self.mean_demand,
                cv=self.coef_of_var,
                horizon=int(self.episode_length),
                demand_type=self.demand_type,
                n=int(self.demand_dmax),
                mix_pi_high=self.mix_pi_high,
                mix_pi_calm=self.mix_pi_calm,
                mix_pi_normal=self.mix_pi_normal,
                mix_pi_promo=self.mix_pi_promo,
                mix_lam1=self.mix_lam1,
                mix_lam2=self.mix_lam2,
                mix_lam3=self.mix_lam3,
                mix_w1=self.mix_w1,
                mix_w2=self.mix_w2,
            ).astype(int)
            self.demand_stream = np.concatenate([self.demand_stream, extra])

        total_demand = int(self.demand_stream[self._demand_idx])
        self._demand_idx += 1

        green_demand = int(np.rint(self.beta * total_demand))
        green_demand = max(0, min(total_demand, green_demand))
        yellow_demand = int(total_demand - green_demand)

        # Update the demand features included in the observation vector.
        # Interpretation: the NEXT observation (start of next period) contains the
        # realized demand split of the PREVIOUS period.
        self.initial_green_demand = float(green_demand)
        self.initial_yellow_demand = float(yellow_demand)

        # -----------------------------
        # Issue demand from age buckets (fifo / lifo / random)
        # -----------------------------
        served_green, rem_green = self._issue_from_buckets(self.green_stock[: self.m], int(green_demand), self.issuing_policy)
        served_yellow, rem_yellow = self._issue_from_buckets(self.yellow_stock, int(yellow_demand), self.issuing_policy)

        satisfied_green = float(served_green)
        satisfied_yellow = float(served_yellow)
        lost_green = float(rem_green)
        lost_yellow = float(rem_yellow)

        # -----------------------------
        # Costs / profit (raw units)
        # -----------------------------
        # -----------------------------
        revenue = self.p1 * satisfied_green + self.p2 * satisfied_yellow

        order_cost = (self.c + self.tti_unit_cost) * q_t
        if q_t > 0:
            order_cost += self.fixed_order_cost

        # Outdating is at end of period (bucket 0 leftovers)
        expired_green = float(self.green_stock[0])
        expired_yellow = float(self.yellow_stock[0]) if self.yellow_stock.size else 0.0
        outdating_cost = self.w * (expired_green + expired_yellow)

        # Holding charged on post-service leftovers (including those that will outdate at end)
        on_hand_green_after_service = float(np.sum(self.green_stock[: self.m]))
        on_hand_yellow_after_service = float(np.sum(self.yellow_stock))
        holding_cost = self.h * (on_hand_green_after_service + on_hand_yellow_after_service)

        lost_sales_cost = self.b1 * lost_green + self.b2 * lost_yellow

        # Total cost (raw units) excluding revenue. This is the object typically
        # reported in inventory papers (average cost per period).
        raw_cost = order_cost + holding_cost + lost_sales_cost + outdating_cost

        # Net profit (raw units)
        raw_profit = revenue - raw_cost
        reward = raw_profit / self.reward_scale

        # -----------------------------
        # Transition: aging + pipeline shift
        # -----------------------------
        # Green shift (includes pipeline positions). Remove expired bucket by clearing last bucket after roll.
        self.green_stock = np.roll(self.green_stock, -1)
        self.green_stock[-1] = 0

        # Yellow shift (remove expired bucket)
        if self.yellow_stock.size:
            self.yellow_stock = np.roll(self.yellow_stock, -1)
            self.yellow_stock[-1] = 0

        # Inject the new order into pipeline tail (or directly into freshest on-hand if L=1)
        if self.L <= 1:
            # No pipeline; order arrives immediately as freshest green
            self.green_stock[self.m - 1] += int(q_t)
        else:
            # Pipeline tail position is last index
            self.green_stock[-1] = int(q_t)

        # Deterioration green->yellow AFTER shift.
        # Discrete per-unit model: for each on-hand green bucket i,
        # deteriorated_i ~ Binomial(n_i, alpha).
        #
        # This keeps inventory integer-valued under deterioration.
        if self.yellow_stock.size:
            k = min(self.m - 1, self.green_stock.size, self.yellow_stock.size)
            if k > 0 and self.alpha > 0:
                n = self.green_stock[:k].astype(np.int64)
                deteriorated = self.rng.binomial(n=n, p=float(self.alpha)).astype(np.int64)
                self.green_stock[:k] -= deteriorated
                self.yellow_stock[:k] += deteriorated

        self.current_step += 1

        # Gymnasium semantics: reaching the fixed episode length is a TIME LIMIT.
        # It is not a "true" terminal state of the Markov chain. For PPO (and other
        # actor-critic methods) this matters: time-limit truncations should still
        # bootstrap the value function.
        terminated = False
        truncated = bool(self.current_step >= self.episode_length)

        # Update cached inventory levels for info/policies
        self._update_cached_inventory_levels()

        info: Dict[str, Any] = {
            # Raw (unscaled) economics
            "raw_profit": float(raw_profit),
            "raw_cost": float(raw_cost),
            "revenue": float(revenue),
            "order_cost": float(order_cost),
            "holding_cost": float(holding_cost),
            "lost_sales_cost": float(lost_sales_cost),
            "outdating_cost": float(outdating_cost),
            "reward_scale": float(self.reward_scale),

            # Demand split
            "demand_total": int(total_demand),
            "demand_green": int(green_demand),
            "demand_yellow": int(yellow_demand),
            "beta_excel": float(self.beta_excel),
            "beta_paper": float(self.beta),

            # Service / losses
            "satisfied_green": float(satisfied_green),
            "satisfied_yellow": float(satisfied_yellow),
            "lost_sales_green": float(lost_green),
            "lost_sales_yellow": float(lost_yellow),

            # Outdating
            "expired_green": float(expired_green),
            "expired_yellow": float(expired_yellow),

            # Inventory levels (explicit on-hand vs pipeline)
            "on_hand_green": float(self.on_hand_green),
            "pipeline_green": float(self.pipeline_green),
            "on_hand_yellow": float(self.on_hand_yellow),
            "inventory_position_total": float(self.inventory_position_total),

            # Extra costs (helpful for sensitivity tables)
            "fixed_order_cost": float(self.fixed_order_cost),
            "tti_unit_cost": float(self.tti_unit_cost),
            "sensor_error_eps": float(self.sensor_error_eps),
        }

        obs = self._get_observation()

        # Stable-Baselines3 uses this flag to detect time-limit cutoffs and bootstrap
        # correctly. Since we implement the horizon ourselves (not via gymnasium.wrappers.TimeLimit),
        # we set it explicitly.
        info["TimeLimit.truncated"] = bool(truncated and not terminated)
        if info["TimeLimit.truncated"]:
            info["terminal_observation"] = obs.copy()

        return obs, float(reward), terminated, truncated, info

    # -----------------------------
    # Helpers
    # -----------------------------
    
    def _issue_from_buckets(self, buckets: np.ndarray, demand: int, policy: str) -> Tuple[int, int]:
        """Issue inventory from age buckets.

        Parameters
        ----------
        buckets:
            1D array of nonnegative integers. Modified in place.
        demand:
            Nonnegative integer demand.
        policy:
            "fifo", "lifo", or "random".

        Semantics
        ---------
        - fifo: consume oldest first (index 0 upward)
        - lifo: consume youngest first (reverse)
        - random: discrete random issuing. We draw `served` units uniformly at random
          from the multiset of all on-hand units (no replacement). This is implemented
          via a multivariate hypergeometric allocation across buckets.

        Returns
        -------
        (served, remaining_demand)
        """
        demand_i = int(demand)
        if demand_i <= 0:
            return 0, 0
        if buckets is None or len(buckets) == 0:
            return 0, demand_i

        policy_s = str(policy).strip().lower()

        if policy_s == "fifo":
            indices = range(len(buckets))
            remaining = demand_i
            served = 0
            for i in indices:
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

        if policy_s == "lifo":
            indices = range(len(buckets) - 1, -1, -1)
            remaining = demand_i
            served = 0
            for i in indices:
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

        if policy_s == "random":
            total = int(np.sum(buckets))
            if total <= 0:
                return 0, demand_i

            served = int(demand_i if demand_i < total else total)
            remaining = int(demand_i - served)
            if served <= 0:
                return 0, demand_i

            # Sequential hypergeometric draws yield a multivariate hypergeometric allocation.
            remaining_sample = served
            remaining_total = total
            for i in range(len(buckets)):
                if remaining_sample <= 0:
                    break
                good = int(buckets[i])
                if good <= 0:
                    remaining_total -= good
                    continue
                bad = int(remaining_total - good)
                if bad < 0:
                    bad = 0
                draw = int(self.rng.hypergeometric(good, bad, remaining_sample))
                if draw > good:
                    draw = good
                buckets[i] = int(good - draw)
                remaining_sample -= draw
                remaining_total -= good

            return int(served), int(remaining)

        raise ValueError(f"Unknown issuing policy: {policy_s!r}")


    def _update_cached_inventory_levels(self) -> None:
        self.on_hand_green = float(np.sum(self.green_stock[: self.m]))
        self.pipeline_green = float(np.sum(self.green_stock[self.m :])) if self.green_stock.size > self.m else 0.0
        self.on_hand_yellow = float(np.sum(self.yellow_stock))
        self.inventory_position_total = self.on_hand_green + self.pipeline_green + self.on_hand_yellow


    def _apil_correction_deterministic_from_obs(
        self,
        *,
        green_obs: np.ndarray,
        yellow_obs: np.ndarray,
        return_components: bool = False,
    ) -> float | tuple[float, float, float]:
        """Deterministic APIL-style correction term based on mean demand.

        Returns
        -------
        float
            C_det(s) = sum_{t=0}^{L-1} (outdating_t - lost_t) under the configured issuing policy,
            with arrivals coming only from the *existing* pipeline (no new orders).

        Notes
        -----
        - Mirrors the event ordering of `step()`:
            demand -> issue (fifo/lifo/random) (green then yellow) -> outdating -> shift/age -> deterioration.
        - Uses continuous split of mean demand:
            dg = beta * mean_demand, dy = mean_demand - dg
          (no integer rounding), consistent with APIL in PIL_GY_policy.py.
        - Uses the provided (possibly sensor-noisy) observations, so this feature does not
          leak hidden state when sensor_error_eps > 0.
        """
        m = int(self.m)
        L = int(self.L)
        if L <= 0:
            return 0.0

        mu = float(self.mean_demand)
        beta = float(np.clip(self.beta, 0.0, 1.0))
        alpha = float(np.clip(self.Alpha, 0.0, 1.0))

        # Work on copies (float) so we can simulate fractional flows if sensor noise is active.
        G = np.asarray(green_obs, dtype=np.float64).reshape(-1).copy()
        Y = np.asarray(yellow_obs, dtype=np.float64).reshape(-1).copy()
        y_len = int(Y.size)

        lost_total = 0.0
        outd_total = 0.0

        for _ in range(L):
            dg = beta * mu
            dy = mu - dg

            # ----------------- serve GREEN and YELLOW demand from on-hand buckets
            policy = str(getattr(self, "issuing_policy", "fifo")).strip().lower()

            # GREEN demand: issue from first m buckets of G
            need_g = float(dg)
            g_len = int(min(m, int(G.size)))
            if policy == "random":
                total_g = float(np.sum(G[:g_len])) if g_len > 0 else 0.0
                if total_g > 0.0:
                    served_g = min(need_g, total_g)
                    frac = served_g / total_g
                    # Expected uniform random picking => proportional depletion
                    G[:g_len] = G[:g_len] * (1.0 - frac)
                    need_g -= served_g
                lost_g = max(0.0, need_g)
            else:
                if policy == "lifo":
                    idxs_g = range(g_len - 1, -1, -1)
                else:
                    idxs_g = range(g_len)
                for i in idxs_g:
                    if need_g <= 0.0:
                        break
                    take = min(float(G[i]), need_g)
                    G[i] -= take
                    need_g -= take
                lost_g = max(0.0, need_g)

            # YELLOW demand: issue from Y buckets
            need_y = float(dy)
            if policy == "random":
                total_y = float(np.sum(Y)) if y_len > 0 else 0.0
                if total_y > 0.0:
                    served_y = min(need_y, total_y)
                    frac_y = served_y / total_y
                    Y[:y_len] = Y[:y_len] * (1.0 - frac_y)
                    need_y -= served_y
                lost_y = max(0.0, need_y)
            else:
                if policy == "lifo":
                    idxs_y = range(y_len - 1, -1, -1)
                else:
                    idxs_y = range(y_len)
                for i in idxs_y:
                    if need_y <= 0.0:
                        break
                    take = min(float(Y[i]), need_y)
                    Y[i] -= take
                    need_y -= take
                lost_y = max(0.0, need_y)

            # ----------------- outdating (oldest leftovers)
            outd_g = float(G[0]) if G.size > 0 else 0.0
            outd_y = float(Y[0]) if y_len > 0 else 0.0
            outd_total += (outd_g + outd_y)
            lost_total += (lost_g + lost_y)

            # ----------------- expire oldest, then age/pipeline shift (no new order injected)
            if G.size > 0:
                G[0] = 0.0
                G = np.roll(G, -1)
                G[-1] = 0.0  # no new tail arrival during correction

            if y_len > 0:
                Y[0] = 0.0
                Y = np.roll(Y, -1)
                Y[-1] = 0.0

            # ----------------- deterioration after shift: move alpha fraction of
            # green buckets with remaining life 1..m-1 (indices 0..m-2) into yellow
            if alpha > 0.0 and y_len > 0 and m > 1:
                k = min(m - 1, y_len)
                if k > 0 and G.size >= k:
                    det = alpha * G[:k]
                    G[:k] -= det
                    Y[:k] += det

        corr = float(outd_total - lost_total)
        if return_components:
            return corr, float(outd_total), float(lost_total)
        return corr

    def _get_observation(self) -> np.ndarray:
        """Return observation vector, optionally corrupted by sensor_error_eps."""
        green_obs = self.green_stock.astype(np.float64).copy()
        yellow_obs = self.yellow_stock.astype(np.float64).copy() if self.yellow_stock.size else self.yellow_stock.astype(np.float64)

        eps = self.sensor_error_eps
        if eps > 0 and self.yellow_stock.size:
            # Deterministic misclassification mixing between green and yellow on-hand buckets
            # For buckets 0..m-2 (remaining life 1..m-1), swap an eps-fraction in expectation.
            k = min(self.m - 1, yellow_obs.size)
            if k > 0:
                g_part = green_obs[:k].copy()
                y_part = yellow_obs[:k].copy()
                green_obs[:k] = (1.0 - eps) * g_part + eps * y_part
                yellow_obs[:k] = (1.0 - eps) * y_part + eps * g_part

        # IMPORTANT: keep observation feature scales comparable.
        # In the original TTI code, episodes were short (e.g., 10 steps), so `current_step`
        # was O(10). In the revised long-run setting, `current_step` can be O(10^3) or more.
        # If we pass it raw, it dominates the neural network input and can harm PPO training.
        # Normalizing by episode length restores a stable [0,1] scale.
        #step_feature = float(self.current_step) / float(max(1, self.episode_length))
        step_feature = 0

        # APIL-style deterministic projection feature (no future random demand leakage):
        #   C_det(s_t) = sum_{tau=0}^{L-1} (outdating_tau - lost_tau)
        # APIL-style deterministic projection feature (no future random demand leakage):
        #   C_det(s_t) = sum_{tau=0}^{L-1} (outdating_tau - lost_tau)
        if self.obs_extra_features:
            corr_det, outd_det, lost_det = self._apil_correction_deterministic_from_obs(
                green_obs=green_obs,
                yellow_obs=yellow_obs,
                return_components=True,
            )
        else:
            corr_det = self._apil_correction_deterministic_from_obs(
                green_obs=green_obs,
                yellow_obs=yellow_obs,
            )
            outd_det = 0.0
            lost_det = 0.0

        # Observed inventory position (on-hand + pipeline + yellow).
        ip_obs = float(np.sum(green_obs)) + float(np.sum(yellow_obs))

        # Projected inventory after L periods under deterministic mean-demand rollout:
        #   proj_stock = IP_obs - L*mu - C_det(s)
        proj_stock = ip_obs - float(self.L) * float(self.mean_demand) - float(corr_det)
        if proj_stock < 0.0:
            # Numerical safety (should be >=0 by construction)
            proj_stock = 0.0
        base_feats = np.array(
            [
                self.initial_green_demand,
                self.initial_yellow_demand,
                step_feature,
                proj_stock,
                float(self.mean_demand),
                float(self.coef_of_var),
                float(self.Alpha),
                float(self.beta),
            ],
            dtype=float,
        )

        if self.obs_extra_features:
            extra_feats = np.array([ip_obs, float(outd_det), float(lost_det)], dtype=float)
            obs = np.concatenate([green_obs, yellow_obs, base_feats, extra_feats])
        else:
            obs = np.concatenate([green_obs, yellow_obs, base_feats])
        return obs.astype(np.float32)



