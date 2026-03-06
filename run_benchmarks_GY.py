# run_benchmarks_GY.py
"""
run_benchmarks_GY.py

Evaluate benchmark policies for the Green/Yellow perishable inventory model.

Evaluates (long-run average profit per period after burn-in):
- Base-stock (BS)
- PIL-GY (Bu et al. 2025 inspired)
- PPO (loads trained model from logs/...)

Reproducibility points:
- LONG-RUN objective: horizon_T with burn_in, average per period
- Fair benchmarking: identical demand_stream injection across policies
- Outputs mean, std, and 95% CI over replications

Typical command:
    python run_benchmarks_GY.py --config-ids 1,2,3,4,5 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Safety for some Windows/conda stacks (torch + MKL OpenMP runtime) ---
# If you have no issue, these are harmless. If you see:
#   "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# this prevents a hard crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from demand_models import sample_demand_stream
import pandas as pd

# Matplotlib only for saving figures (no interactive display needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Stable-Baselines3 imports for PPO and Normalization
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    pass

from InventoryEnvGY_Config import InventoryEnvGYConfig
from PIL_GY_policy import pil_order_quantity

# ---------------------------------------------------------------------
# USER PARAMETERS (interactive-friendly)
# ---------------------------------------------------------------------

CONFIG_EXCEL_PATH = "configurations.xlsx"

# Default: first 5 configurations (edit as needed)
CONFIG_IDS = None  # None -> all configuration IDs found in configurations.xlsx

# Quick mode = smaller horizons/replications for fast sanity checks.
# For paper-quality results, set QUICK_MODE = False.
QUICK_MODE = False

# Long-run evaluation parameters
T_HORIZON_FULL = 2000
BURN_IN_FULL = 400
TAIL_DROP_STEPS = 0  # set to 0 to use the full horizon (no tail trimming)
OBS_EXTRA_FEATURES = True  # keep PPO obs enriched (deterministic features only)
N_REPS_FULL = 30

T_HORIZON_QUICK = 2000
BURN_IN_QUICK = 300
N_REPS_QUICK = 3

# Tuning parameters (grid search)
TUNE_HORIZON_FULL = 2000
TUNE_BURN_IN_FULL = 200
TUNE_REPS_FULL = 10

TUNE_HORIZON_QUICK = 800
TUNE_BURN_IN_QUICK = 200
TUNE_REPS_QUICK = 2

# Base-stock / PIL grid
S_GRID = list(range(0, 60))

# PIL parameters
PIL_USE_CORRECTION = True
PIL_N_MC_TUNE = 0
PIL_N_MC_EVAL = 0

# Parallelization (BS/PIL only; PPO eval kept in main process to avoid torch+mp issues)
PARALLEL_REPS = True
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Seeds
TUNE_SEED0 = 10_000
EVAL_SEED0 = 20_000

# Where PPO models live
LOGS_DIR = "logs"

# Output
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _to_int_action(action: Any) -> int:
    """Robust conversion of SB3/gymnasium actions to a Python int.

    Handles:
      - Python int / numpy integer
      - floats
      - numpy arrays with shape () or (1,) etc.
    """
    if action is None:
        return 0
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, float):
        return int(np.rint(action))
    a = np.asarray(action)
    if a.shape == ():
        return int(a.item())
    return int(a.reshape(-1)[0])


def _ensure_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
    logs = base_dir / LOGS_DIR
    results = base_dir / RESULTS_DIR
    plots = base_dir / PLOTS_DIR
    logs.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    return logs, results, plots


def _gamma_params(mean: float, cv: float) -> Tuple[float, float]:
    """Gamma(shape, scale) from mean and coefficient of variation."""
    cv2 = float(cv) ** 2
    shape = 1.0 / cv2 if cv2 > 0 else 1e9
    scale = float(mean) / shape
    return shape, scale


def generate_demand_stream(
    mean: Optional[float] = None,
    cv: Optional[float] = None,
    horizon: int = 0,
    seed: int = 0,
    *,
    mean_demand: Optional[float] = None,
    coef_of_var: Optional[float] = None,
    demand_type: str = "beta_binomial",
    demand_dmax: Optional[int] = None,
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
) -> np.ndarray:
    """Integer IID demand stream generated from the configured demand model."""

    if mean is None:
        mean = mean_demand
    if cv is None:
        cv = coef_of_var
    if mean is None:
        raise TypeError("generate_demand_stream requires mean (or mean_demand)")

    dt = str(demand_type).strip().lower()

    # CV is required for all bounded calibrated families and for gamma.
    # For legacy mix_poisson_3regime, CV is ignored, so we can accept cv=None by setting a dummy value.
    if cv is None:
        if dt in ("mix_poisson_3regime", "mix_poisson_3", "mix3", "mix_poisson"):
            cv = 1.0
        else:
            raise TypeError("generate_demand_stream requires cv (or coef_of_var) for this demand_type")

    n = int(demand_dmax) if demand_dmax is not None else 20
    rng = np.random.default_rng(int(seed))

    return sample_demand_stream(
        rng=rng,
        mean=float(mean),
        cv=float(cv),
        horizon=int(horizon),
        demand_type=dt,
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


def load_config_row(excel_path: Path, config_id: int) -> pd.Series:
    """Load one configuration row by id.

    Priority:
    1) If the Excel file contains a 'configuration' column, use it (robust to re-ordering).
    2) Otherwise, fall back to 1-indexed row position (legacy).
    """
    df = pd.read_excel(excel_path, engine="openpyxl")

    if "configuration" in df.columns:
        sel = df.loc[df["configuration"] == int(config_id)]
        if sel.empty:
            raise ValueError(f"config_id={config_id} not found in {excel_path} (column 'configuration').")
        row = sel.iloc[0]
    else:
        idx = int(config_id) - 1
        if idx < 0 or idx >= len(df):
            raise ValueError(f"config_id={config_id} out of range 1..{len(df)} for {excel_path}.")
        row = df.iloc[idx]

    # Hard guard: this project does not use Gamma demand
    dt = str(row.get("demand_type", "")).strip().lower()
    if dt in ("", "nan", "none"):
        raise ValueError(
            f"Config id={int(config_id)} has no valid 'demand_type' in {excel_path}. "
            "Set demand_type explicitly (e.g., beta_binomial, mix_binom_3regime, mix_poisson_3regime)."
        )
    if dt.startswith("gamma"):
        raise ValueError(
            f"Config id={int(config_id)} uses demand_type='{dt}', but Gamma demand is not allowed. "
            "Update configurations.xlsx to a bounded demand family."
        )

    return row


def build_env_config(row: pd.Series, *, episode_length: int, issuing_policy: str = "fifo") -> Dict[str, Any]:
    """Map Excel row -> env config dict. Keeps configurations.xlsx unchanged."""

    env_cfg: Dict[str, Any] = {
        "m": int(row.get("m", 3)),
        "L": int(row.get("L", 2)),
        "Alpha": float(row.get("Alpha", 0.0)),
        "beta_excel": float(row.get("beta_excel", row.get("beta", 0.05))),
        "obs_extra_features": bool(OBS_EXTRA_FEATURES),
        "mean_demand": float(row.get("mean_demand", row.get("mean_dem", 3.0))),
        "coef_of_var": float(row.get("coef_of_var", row.get("cv", 0.1))),
        # demand model
        "demand_type": str(row.get("demand_type", "beta_binomial")),
        "demand_dmax": int(row.get("demand_dmax", row.get("dmax", 65))),
        # optional regime probabilities for binomial mixtures
        "mix_pi_high": row.get("mix_pi_high", row.get("pi_high", None)),
        "mix_pi_calm": row.get("mix_pi_calm", None),
        "mix_pi_normal": row.get("mix_pi_normal", None),
        "mix_pi_promo": row.get("mix_pi_promo", None),
        # legacy poisson-mixture knobs (optional)
        "mix_lam1": row.get("mix_lam1", row.get("lam1", None)),
        "mix_lam2": row.get("mix_lam2", row.get("lam2", None)),
        "mix_lam3": row.get("mix_lam3", row.get("lam3", None)),
        "mix_w1": row.get("mix_w1", row.get("w1", None)),
        "mix_w2": row.get("mix_w2", row.get("w2", None)),
        "p1": float(row.get("p1", 25.0)),
        "p2": float(row.get("p2", 15.0)),
        "c": float(row.get("c", 0.0)),
        "h": float(row.get("h", 1.0)),
        "b1": float(row.get("b1", 10.0)),
        "b2": float(row.get("b2", 5.0)),
        "w": float(row.get("w", 3.0)),
        # optional extensions (defaults keep original experiments unchanged)
        "fixed_order_cost": float(row.get("fixed_order_cost", 0.0)),
        "tti_unit_cost": float(row.get("tti_unit_cost", 0.0)),
        "sensor_error_eps": float(row.get("sensor_error_eps", 0.0)),
        "issuing_policy": str(issuing_policy).strip().lower(),
        # RL scaling / bounds
        "reward_scale": float(row.get("reward_scale", 100.0)),
        "max_order": int(row.get("max_order", 60)),
        # episodic interface for SB3; long-run is handled by evaluation horizon+burn-in
        "episode_length": int(episode_length),
    }
    return env_cfg


@dataclass
class ReplicationStats:
    # Core economics (raw units)
    profit_mean_per_period: float
    cost_mean_per_period: float
    revenue_mean_per_period: float
    lost_mean_per_period: float
    waste_mean_per_period: float
    satisfied_mean_per_period: float
    stock_green_mean: float
    stock_yellow_mean: float
    expired_green_mean: float
    expired_yellow_mean: float
    lost_green_mean: float
    lost_yellow_mean: float
    satisfied_green_mean: float
    satisfied_yellow_mean: float


def _run_one_replication(
    env_cfg: Dict[str, Any],
    demand_stream: np.ndarray,
    burn_in: int,
    policy_kind: str,
    *,
    S_bs: Optional[int] = None,
    S_pil: Optional[int] = None,
    pil_use_correction: bool = True,
    pil_n_mc: int = 100,
    pil_seed: int = 12345,
    pil_use_observed_state: bool = False,
    ppo_model_path: Optional[Path] = None,
    ppo_model: Optional[Any] = None,
    ppo_vec_normalize: Optional[Any] = None,  # Added: normalization object
) -> ReplicationStats:
    """Simulate ONE replication for a given policy (BS/PIL/PPO) on a fixed demand stream."""

    env_cfg_r = dict(env_cfg)
    env_cfg_r["demand_stream"] = np.asarray(demand_stream, dtype=int)

    env = InventoryEnvGYConfig(env_cfg_r)

    # Policy wiring
    rng_pil = np.random.default_rng(int(pil_seed))

    if policy_kind == "PPO":
        if ppo_model is None:
            if ppo_model_path is None or not ppo_model_path.exists():
                raise FileNotFoundError(f"PPO model not found: {ppo_model_path}")
            from stable_baselines3 import PPO  # local import to keep BS/PIL workers torch-free
            ppo_model = PPO.load(str(ppo_model_path))

    obs, _ = env.reset()

    T = len(demand_stream)
    if burn_in >= T:
        raise ValueError("burn_in must be < horizon")

    # accumulators (post burn-in)
    n_eval = 0
    sum_profit = 0.0
    sum_cost = 0.0
    sum_revenue = 0.0
    sum_lost = 0.0
    sum_waste = 0.0
    sum_satisfied = 0.0
    sum_stock_green = 0.0
    sum_stock_yellow = 0.0

    sum_exp_g = 0.0
    sum_exp_y = 0.0
    sum_lost_g = 0.0
    sum_lost_y = 0.0
    sum_sat_g = 0.0
    sum_sat_y = 0.0

    end_t = int(max(0, T - int(TAIL_DROP_STEPS)))

    for t in range(end_t):
        if policy_kind == "BS":
            if S_bs is None:
                raise ValueError("S_bs is required for BS")
            inv_pos = float(env.inventory_position_total)
            q = int(max(0, min(env.max_order, round(S_bs - inv_pos))))

        elif policy_kind == "PIL":
            # Project the current **age profile** forward by L periods (APIL when pil_use_correction=False).
            if S_pil is None:
                raise ValueError("S_pil is required for PIL")
            # If sensor_error_eps > 0 and pil_use_observed_state=True, use the *observed* (potentially misclassified)
            # G/Y profile from the environment observation.
            if bool(pil_use_observed_state):
                green_len = int(env.m + max(env.L - 1, 0))
                yellow_len = int(max(env.m - 1, 0))
                obs_vec = np.asarray(obs, dtype=np.float64).reshape(-1)
                if obs_vec.size < green_len + yellow_len:
                    raise RuntimeError("Observation vector too short to extract (green,yellow) stocks.")
                g_full = obs_vec[:green_len].copy()
                y_full = obs_vec[green_len:green_len + yellow_len].copy()
                green_on_hand = g_full[: env.m].copy()
                green_pipeline = g_full[env.m :].copy() if env.L > 1 else np.zeros((0,), dtype=np.float64)
                yellow_on_hand = y_full.copy()
            else:
                green_on_hand = env.green_stock[: env.m].copy()
                green_pipeline = env.green_stock[env.m :].copy() if env.L > 1 else np.zeros((0,), dtype=np.float64)
                yellow_on_hand = env.yellow_stock.copy()

            q = pil_order_quantity(
                S_pil=S_pil,
                green_on_hand=green_on_hand,
                green_pipeline=green_pipeline,
                yellow_on_hand=yellow_on_hand,
                m=env.m,
                L=env.L,
                alpha=env.alpha,
                beta=env.beta,  # paper-consistent beta
                mean_demand=env.mean_demand,
                coef_of_var=env.coef_of_var,
                demand_type=env.demand_type,
                demand_dmax=env.demand_dmax,
                mix_pi_high=env.mix_pi_high,
                mix_pi_calm=env.mix_pi_calm,
                mix_pi_normal=env.mix_pi_normal,
                mix_pi_promo=env.mix_pi_promo,
                mix_lam1=env.mix_lam1,
                mix_lam2=env.mix_lam2,
                mix_lam3=env.mix_lam3,
                mix_w1=env.mix_w1,
                mix_w2=env.mix_w2,
                max_order=env.max_order,
                n_mc=int(pil_n_mc) if pil_use_correction else 0,
                rng=rng_pil,
                issuing_policy=str(getattr(env, 'issuing_policy', 'fifo')),
            )

        elif policy_kind == "PPO":
            if ppo_model is None:
                raise ValueError("PPO model failed to load")
            
            # --- Normalize Observation if stats exist ---
            if ppo_vec_normalize is not None:
                # normalize_obs expects shape (n_envs, obs_dim)
                obs_norm = ppo_vec_normalize.normalize_obs(obs[np.newaxis, :])
                action, _ = ppo_model.predict(obs_norm, deterministic=True)
            else:
                action, _ = ppo_model.predict(obs, deterministic=True)
                
            q = _to_int_action(action)
            q = int(max(0, min(env.max_order, q)))

        else:
            raise ValueError(f"Unknown policy_kind={policy_kind}")

        obs, reward, terminated, truncated, info = env.step(q)

        if t >= burn_in:
            n_eval += 1
            sum_profit += float(info.get("raw_profit", 0.0))
            sum_cost += float(info.get("raw_cost", 0.0))
            sum_revenue += float(info.get("revenue", 0.0))

            lost_g = float(info.get("lost_sales_green", 0.0))
            lost_y = float(info.get("lost_sales_yellow", 0.0))
            exp_g = float(info.get("expired_green", 0.0))
            exp_y = float(info.get("expired_yellow", 0.0))
            sat_g = float(info.get("satisfied_green", 0.0))
            sat_y = float(info.get("satisfied_yellow", 0.0))

            sum_lost += (lost_g + lost_y)
            sum_waste += (exp_g + exp_y)
            sum_satisfied += (sat_g + sat_y)

            sum_stock_green += float(info.get("on_hand_green", 0.0))
            sum_stock_yellow += float(info.get("on_hand_yellow", 0.0))

            sum_exp_g += exp_g
            sum_exp_y += exp_y
            sum_lost_g += lost_g
            sum_lost_y += lost_y
            sum_sat_g += sat_g
            sum_sat_y += sat_y

        if terminated or truncated:
            # SB3 episodes end, but for long-run eval we just stop at horizon length.
            break

    if n_eval == 0:
        raise RuntimeError("No evaluation steps collected (check burn_in/horizon)")

    return ReplicationStats(
        profit_mean_per_period=sum_profit / n_eval,
        cost_mean_per_period=sum_cost / n_eval,
        revenue_mean_per_period=sum_revenue / n_eval,
        lost_mean_per_period=sum_lost / n_eval,
        waste_mean_per_period=sum_waste / n_eval,
        satisfied_mean_per_period=sum_satisfied / n_eval,
        stock_green_mean=sum_stock_green / n_eval,
        stock_yellow_mean=sum_stock_yellow / n_eval,
        expired_green_mean=sum_exp_g / n_eval,
        expired_yellow_mean=sum_exp_y / n_eval,
        lost_green_mean=sum_lost_g / n_eval,
        lost_yellow_mean=sum_lost_y / n_eval,
        satisfied_green_mean=sum_sat_g / n_eval,
        satisfied_yellow_mean=sum_sat_y / n_eval,
    )


def _worker_run_one_replication(task):
    """Picklable worker for ProcessPoolExecutor (Windows-safe).

    Backward-compatible: accepts either the legacy 9-tuple or the newer 10-tuple
    (with pil_use_observed_state).
    """
    if len(task) == 9:
        (
            cfg,
            stream,
            burn_in,
            policy_kind,
            S_bs,
            S_pil,
            pil_use_correction,
            pil_n_mc,
            pil_seed,
        ) = task
        pil_use_observed_state = False
    else:
        (
            cfg,
            stream,
            burn_in,
            policy_kind,
            S_bs,
            S_pil,
            pil_use_correction,
            pil_n_mc,
            pil_seed,
            pil_use_observed_state,
        ) = task

    return _run_one_replication(
        cfg,
        stream,
        burn_in,
        policy_kind,
        S_bs=S_bs,
        S_pil=S_pil,
        pil_use_correction=pil_use_correction,
        pil_n_mc=pil_n_mc,
        pil_seed=pil_seed,
        pil_use_observed_state=bool(pil_use_observed_state),
        ppo_model_path=None,
    )


def _mean_std_ci(x: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(x, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, ci95


def load_training_meta(logs_dir: Path, config_id: int) -> Dict[str, Any]:
    """Load PPO training metadata for a configuration.

    We prefer the canonical `training_meta.json`, but we also accept `metadata.json`
    as a backward-compatible fallback (older runs).
    """
    cand = [
        logs_dir / f"config_{config_id}" / "training_meta.json",
        logs_dir / f"config_{config_id}" / "metadata.json",
    ]
    for meta_path in cand:
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


def _resolve_meta_path(base_dir: Path, p: str) -> Path:
    """Resolve a path stored in JSON metadata.

    The JSON may contain Windows-style backslashes even when running on other OSes.
    We normalize separators and interpret relative paths as relative to `base_dir`.
    """
    s = str(p).strip().replace("\\", "/")
    pp = Path(s)
    if pp.is_absolute():
        return pp
    return base_dir / pp


def resolve_ppo_checkpoint(
    *,
    base_dir: Path,
    logs_dir: Path,
    config_id: int,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Resolve which PPO checkpoint should be evaluated for a configuration.

    Correct priority (important):
    1) training_meta.json -> meta['paths']['best_model'] (or 'final_model')
    2) training_meta.json -> meta['hyperparam_sweep']['best_run_dir']/best_model.zip
    3) Canonical copy at logs/config_<id>/best_model.zip
    4) Legacy file logs/config_<id>/best_model_evalcallback.zip (LAST, can be stale)
    5) Newest logs/config_<id>/**/best_model.zip (fallback)
    """
    cfg_dir = logs_dir / f"config_{int(config_id)}"

    if meta is None:
        meta = load_training_meta(logs_dir, int(config_id))

    # 1) Use meta paths first
    if isinstance(meta, dict):
        paths = meta.get("paths", {}) or {}
        for key in ("best_model", "final_model"):
            p = paths.get(key)
            if p:
                cand = _resolve_meta_path(base_dir, str(p))
                if cand.exists():
                    return cand

        # 2) Sweep best_run_dir
        sweep = meta.get("hyperparam_sweep", {}) or {}
        best_run_dir = sweep.get("best_run_dir")
        if best_run_dir:
            cand = _resolve_meta_path(base_dir, str(best_run_dir)) / "best_model.zip"
            if cand.exists():
                return cand

    # 3) Canonical copy
    canonical = cfg_dir / "best_model.zip"
    if canonical.exists():
        return canonical

    # 4) Legacy fallback (LAST)
    alt = cfg_dir / "best_model_evalcallback.zip"
    if alt.exists():
        return alt

    # 5) Fallback: newest nested best_model.zip
    cands = list(cfg_dir.rglob("best_model.zip"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]

    return None


def tune_level(
    env_cfg: Dict[str, Any],
    policy_kind: str,
    S_grid: Optional[List[int]],
    horizon: int,
    burn_in: int,
    reps: int,
    seed0: int,
    *,
    pil_use_correction: bool,
    pil_n_mc: int,
) -> int:
    """Grid search for S (BS or PIL) using fixed demand streams for fairness.

    Speed note
    ----------
    To keep the benchmark responsive, we apply a simple **early-stop** heuristic:
    after the best level S* has been found, we stop once we observe `patience`
    consecutive candidates above S* that do not improve the mean profit.
    """

    assert policy_kind in {"BS", "PIL"}

    if S_grid is None:
        S_grid = list(range(0, int(env_cfg["max_order"]) + 1))
    S_grid = sorted(int(s) for s in S_grid)

    mean_d = float(env_cfg["mean_demand"])
    cv = float(env_cfg["coef_of_var"])

    dt = str(env_cfg.get("demand_type", "beta_binomial"))
    streams = [
        generate_demand_stream(
            mean_d,
            cv,
            horizon,
            seed0 + r,
            demand_type=dt,
            demand_dmax=env_cfg.get("demand_dmax", None),
            mix_pi_high=env_cfg.get("mix_pi_high", None),
            mix_pi_calm=env_cfg.get("mix_pi_calm", None),
            mix_pi_normal=env_cfg.get("mix_pi_normal", None),
            mix_pi_promo=env_cfg.get("mix_pi_promo", None),
            mix_lam1=env_cfg.get("mix_lam1", None),
            mix_lam2=env_cfg.get("mix_lam2", None),
            mix_lam3=env_cfg.get("mix_lam3", None),
            mix_w1=env_cfg.get("mix_w1", None),
            mix_w2=env_cfg.get("mix_w2", None),
        )
        for r in range(reps)
    ]

    best_S = int(S_grid[0])
    best_mean = -1e18
    since_best = 0
    patience = 5  # evaluate a few more points after the current best, then stop

    for S in S_grid:
        vals = []
        for r, stream in enumerate(streams):
            rep = _run_one_replication(
                env_cfg,
                stream,
                burn_in,
                policy_kind,
                S_bs=S if policy_kind == "BS" else None,
                S_pil=S if policy_kind == "PIL" else None,
                pil_use_correction=pil_use_correction,
                pil_n_mc=pil_n_mc,
                pil_seed=seed0 + 999 + r,
                ppo_model_path=None,
            )
            vals.append(rep.profit_mean_per_period)
        m = float(np.mean(vals))

        if m > best_mean:
            best_mean = m
            best_S = int(S)
            since_best = 0
        else:
            if int(S) > best_S:
                since_best += 1
                if since_best >= patience:
                    break

    return int(best_S)


def _plot_policy_comparison(df: pd.DataFrame, plots_dir: Path) -> None:
    """Create a simple figure for reporting-focused convergence/robustness reporting."""

    if df.empty:
        return

    # If Alpha varies, plot profit vs Alpha; else bar plot
    if "Alpha" in df.columns and df["Alpha"].nunique() > 1:
        df2 = df.sort_values("Alpha")
        x = df2["Alpha"].values

        plt.figure()
        plt.plot(x, df2["profit_BS_mean"].values, marker="o", label="BS")
        plt.plot(x, df2["profit_PIL_mean"].values, marker="o", label="PIL")
        plt.plot(x, df2["profit_PPO_mean"].values, marker="o", label="PPO")
        plt.xlabel("Alpha")
        plt.ylabel("Mean profit per period (raw units)")
        plt.title("Policy comparison vs deterioration rate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "policy_comparison_profit_vs_alpha.png", dpi=150)
        plt.close()

        # Also plot costs (lower is better)
        if {"cost_BS_mean", "cost_PIL_mean", "cost_PPO_mean"}.issubset(df2.columns):
            plt.figure()
            plt.plot(x, df2["cost_BS_mean"].values, marker="o", label="BS")
            plt.plot(x, df2["cost_PIL_mean"].values, marker="o", label="PIL")
            plt.plot(x, df2["cost_PPO_mean"].values, marker="o", label="PPO")
            plt.xlabel("Alpha")
            plt.ylabel("Mean cost per period (raw units)")
            plt.title("Policy comparison vs deterioration rate (cost)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "policy_comparison_cost_vs_alpha.png", dpi=150)
            plt.close()
    else:
        # bar plot of mean profit for each config (x axis = config_id)
        plt.figure(figsize=(10, 4))
        x = np.arange(len(df))
        w = 0.25
        plt.bar(x - w, df["profit_BS_mean"].values, width=w, label="BS")
        plt.bar(x, df["profit_PIL_mean"].values, width=w, label="PIL")
        plt.bar(x + w, df["profit_PPO_mean"].values, width=w, label="PPO")
        plt.xticks(x, df["configuration"].values)
        plt.xlabel("Configuration")
        plt.ylabel("Mean profit per period (raw units)")
        plt.title("Policy comparison")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "policy_comparison_profit.png", dpi=150)
        plt.close()

        # Cost bar plot
        if {"cost_BS_mean", "cost_PIL_mean", "cost_PPO_mean"}.issubset(df.columns):
            plt.figure(figsize=(10, 4))
            x = np.arange(len(df))
            w = 0.25
            plt.bar(x - w, df["cost_BS_mean"].values, width=w, label="BS")
            plt.bar(x, df["cost_PIL_mean"].values, width=w, label="PIL")
            plt.bar(x + w, df["cost_PPO_mean"].values, width=w, label="PPO")
            plt.xticks(x, df["configuration"].values)
            plt.xlabel("Configuration")
            plt.ylabel("Mean cost per period (raw units)")
            plt.title("Policy comparison (cost)")
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "policy_comparison_cost.png", dpi=150)
            plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run policy benchmarks (BS, PIL, PPO).")
    p.add_argument("--config-ids", type=str, default="", help="Comma-separated list of configuration IDs.")
    p.add_argument("--excel", type=str, default=CONFIG_EXCEL_PATH, help="Excel file with configurations.")
    p.add_argument("--quick", action="store_true", help="Use fast settings (fewer reps, shorter horizons).")
    p.add_argument("--issuing-policy", type=str, default="fifo", choices=["fifo", "lifo", "random"])

    # Parallelization
    p.add_argument(
        "--parallel-configs",
        action="store_true",
        help="Parallelize across configuration IDs (recommended when running many configs).",
    )
    p.add_argument(
        "--max-config-workers",
        type=int,
        default=0,
        help="Max workers for --parallel-configs. 0 => cpu_count()-1.",
    )

    return p.parse_args()


def _set_single_thread_env() -> None:
    """Best-effort protection against oversubscription on MKL/OpenBLAS stacks."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _benchmark_one_config(
    *,
    cid: int,
    base_dir: Path,
    excel_path: Path,
    logs_dir: Path,
    quick: bool,
    parallel_reps: bool,
    issuing_policy: str = "fifo",
) -> Dict[str, Any]:
    """Evaluate one configuration and return the output row dict.

    Notes
    -----
    - If `parallel_reps` is False, replications are evaluated sequentially.
      This is the correct setting when parallelizing across configs.
    """
    issuing_policy = str(os.environ.get("INVENTORY_ISSUING_POLICY", issuing_policy)).strip().lower()
    if issuing_policy not in {"fifo", "lifo", "random"}:
        issuing_policy = "fifo"


    # Settings
    if quick:
        T_HORIZON = T_HORIZON_QUICK
        BURN_IN = BURN_IN_QUICK
        N_REPS = N_REPS_QUICK
        TUNE_HORIZON = TUNE_HORIZON_QUICK
        TUNE_BURN_IN = TUNE_BURN_IN_QUICK
        TUNE_REPS = TUNE_REPS_QUICK
    else:
        T_HORIZON = T_HORIZON_FULL
        BURN_IN = BURN_IN_FULL
        N_REPS = N_REPS_FULL
        TUNE_HORIZON = TUNE_HORIZON_FULL
        TUNE_BURN_IN = TUNE_BURN_IN_FULL
        TUNE_REPS = TUNE_REPS_FULL

    row = load_config_row(excel_path, cid)
    env_cfg = build_env_config(row, episode_length=T_HORIZON, issuing_policy=issuing_policy)

    # Tune levels
    t0 = time.time()
    S_BS = tune_level(
        env_cfg,
        policy_kind="BS",
        S_grid=S_GRID,
        horizon=TUNE_HORIZON,
        burn_in=TUNE_BURN_IN,
        reps=TUNE_REPS,
        seed0=TUNE_SEED0 + 1000 * cid,
        pil_use_correction=False,
        pil_n_mc=0,
    )
    S_PIL = tune_level(
        env_cfg,
        policy_kind="PIL",
        S_grid=S_GRID,
        horizon=TUNE_HORIZON,
        burn_in=TUNE_BURN_IN,
        reps=TUNE_REPS,
        seed0=TUNE_SEED0 + 1000 * cid + 100,
        pil_use_correction=PIL_USE_CORRECTION,
        pil_n_mc=PIL_N_MC_TUNE,
    )
    tune_seconds = time.time() - t0

    # Pre-generate demand streams for fair evaluation
    dt = str(env_cfg.get("demand_type", "beta_binomial"))
    streams = [
        generate_demand_stream(
            float(env_cfg["mean_demand"]),
            float(env_cfg["coef_of_var"]),
            int(T_HORIZON),
            int(EVAL_SEED0 + 1000 * cid + r),
            demand_type=dt,
            demand_dmax=env_cfg.get("demand_dmax", None),
            mix_pi_high=env_cfg.get("mix_pi_high", None),
            mix_pi_calm=env_cfg.get("mix_pi_calm", None),
            mix_pi_normal=env_cfg.get("mix_pi_normal", None),
            mix_pi_promo=env_cfg.get("mix_pi_promo", None),
            mix_lam1=env_cfg.get("mix_lam1", None),
            mix_lam2=env_cfg.get("mix_lam2", None),
            mix_lam3=env_cfg.get("mix_lam3", None),
            mix_w1=env_cfg.get("mix_w1", None),
            mix_w2=env_cfg.get("mix_w2", None),
        )
        for r in range(int(N_REPS))
    ]

    # Evaluate BS and PIL (torch-free)
    def eval_many(policy_kind: str) -> List[ReplicationStats]:
        if (not parallel_reps) or (int(N_REPS) <= 6):
            out: List[ReplicationStats] = []
            for r, s in enumerate(streams):
                out.append(
                    _run_one_replication(
                        env_cfg,
                        s,
                        int(BURN_IN),
                        policy_kind,
                        S_bs=int(S_BS) if policy_kind == "BS" else None,
                        S_pil=int(S_PIL) if policy_kind == "PIL" else None,
                        pil_use_correction=PIL_USE_CORRECTION,
                        pil_n_mc=PIL_N_MC_EVAL,
                        pil_seed=int(EVAL_SEED0 + 77 + r),
                        ppo_model_path=None,
                    )
                )
            return out

        tasks = []
        for r, s in enumerate(streams):
            tasks.append(
                (
                    env_cfg,
                    s,
                    int(BURN_IN),
                    policy_kind,
                    int(S_BS),
                    int(S_PIL),
                    bool(PIL_USE_CORRECTION),
                    int(PIL_N_MC_EVAL),
                    int(EVAL_SEED0 + 77 + r),
                )
            )

        # Use spawn explicitly for Windows reliability
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(processes=min(MAX_WORKERS, len(tasks))) as pool:
            return pool.map(_worker_run_one_replication, tasks)

    stats_bs = eval_many("BS")
    stats_pil = eval_many("PIL")

    def summarize(stats: List[ReplicationStats]) -> Tuple[float, float, float, float, float, float, float, float]:
        vals = np.array([s.profit_mean_per_period for s in stats], dtype=float)
        costs = np.array([s.cost_mean_per_period for s in stats], dtype=float)
        revs = np.array([s.revenue_mean_per_period for s in stats], dtype=float)
        lost = np.array([s.lost_mean_per_period for s in stats], dtype=float)
        waste = np.array([s.waste_mean_per_period for s in stats], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = float(1.96 * std / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

        c_mean = float(np.mean(costs))
        c_std = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
        c_ci = float(1.96 * c_std / np.sqrt(len(costs))) if len(costs) > 1 else 0.0

        return mean, ci, float(np.mean(revs)), c_mean, c_ci, float(np.mean(lost)), float(np.mean(waste)), std

    profit_BS_mean, profit_BS_ci, rev_BS_mean, cost_BS_mean, cost_BS_ci, lost_BS_mean, waste_BS_mean, std_BS = summarize(stats_bs)
    profit_PIL_mean, profit_PIL_ci, rev_PIL_mean, cost_PIL_mean, cost_PIL_ci, lost_PIL_mean, waste_PIL_mean, std_PIL = summarize(stats_pil)

    # PPO eval (kept sequential by default)
    profit_PPO_mean = float("nan")
    profit_PPO_ci = float("nan")
    rev_PPO_mean = float("nan")
    cost_PPO_mean = float("nan")
    cost_PPO_ci = float("nan")
    lost_PPO_mean = float("nan")
    waste_PPO_mean = float("nan")
    std_PPO = float("nan")

    cfg_dir = logs_dir / f"config_{cid}"

    # Load PPO meta early so we resolve the *exact* checkpoint produced by training.
    # This prevents evaluating a stale logs/config_<id>/best_model.zip when a hyperparameter
    # sweep stored the best checkpoint deeper under logs/config_<id>/hp_*/best_model.zip.
    meta = load_training_meta(logs_dir, cid)
    ppo_model_path = resolve_ppo_checkpoint(base_dir=base_dir, logs_dir=logs_dir, config_id=cid, meta=meta)
    ppo_model_path_str = str(ppo_model_path) if ppo_model_path is not None else ""
    ppo_model_mtime = (
        float(ppo_model_path.stat().st_mtime) if (ppo_model_path is not None and ppo_model_path.exists()) else float("nan")
    )

    if ppo_model_path is not None:
        print(f"[bench] config={cid}: PPO checkpoint -> {ppo_model_path_str}")

    ppo_stats: List[ReplicationStats] = []

    if (ppo_model_path is not None) and ppo_model_path.exists():
        try:
            # Load once (torch load is expensive) and reuse across replications
            from stable_baselines3 import PPO as SB3PPO
            ppo_model = SB3PPO.load(str(ppo_model_path))
            
            # --- Check and load normalization stats ---
            ppo_vec_normalize = None
            vec_path = ppo_model_path.parent / "vecnormalize.pkl"
            if vec_path.exists():
                # We need a dummy environment to load VecNormalize stats
                # Using lambda to defer env creation
                venv_dummy = DummyVecEnv([lambda: InventoryEnvGYConfig(env_cfg)])
                ppo_vec_normalize = VecNormalize.load(str(vec_path), venv_dummy)
                # Important: disable training and reward normalization during evaluation
                ppo_vec_normalize.training = False
                ppo_vec_normalize.norm_reward = False
                print(f"[bench] config={cid}: Loaded VecNormalize stats from {vec_path}")

            for r, s in enumerate(streams):
                ppo_stats.append(
                    _run_one_replication(
                        env_cfg,
                        s,
                        int(BURN_IN),
                        "PPO",
                        S_bs=None,
                        S_pil=None,
                        pil_use_correction=PIL_USE_CORRECTION,
                        pil_n_mc=PIL_N_MC_EVAL,
                        pil_seed=int(EVAL_SEED0 + 77 + r),
                        ppo_model_path=ppo_model_path,
                        ppo_model=ppo_model,
                        ppo_vec_normalize=ppo_vec_normalize,
                    )
                )

        except Exception as e:
            print(f"[bench] WARNING: PPO evaluation skipped for config={cid}: {e}")

    if ppo_stats:
        profit_PPO_mean, profit_PPO_ci, rev_PPO_mean, cost_PPO_mean, cost_PPO_ci, lost_PPO_mean, waste_PPO_mean, std_PPO = summarize(ppo_stats)

    # PPO meta already loaded above (used both for checkpoint resolution and for reporting)
    beta_excel = float(env_cfg.get("beta_excel", env_cfg.get("beta", 0.0)))

    row_out: Dict[str, Any] = {
        "configuration": int(cid),
        "issuing_policy": str(issuing_policy),
        "m": int(env_cfg["m"]),
        "L": int(env_cfg["L"]),
        "Alpha": float(env_cfg["Alpha"]),
        "beta_excel": float(beta_excel),
        "beta_paper": float(1.0 - beta_excel),
        "mean_demand": float(env_cfg["mean_demand"]),
        "coef_of_var": float(env_cfg["coef_of_var"]),
        "demand_type": str(env_cfg.get("demand_type", "")),
        "demand_dmax": int(env_cfg.get("demand_dmax", 0)),
        "max_order": int(env_cfg.get("max_order", 0)),
        "S_BS": int(S_BS),
        "S_PIL": int(S_PIL),
        "tune_seconds": float(tune_seconds),
        "profit_BS_mean": float(profit_BS_mean),
        "profit_BS_ci95": float(profit_BS_ci),
        "profit_PIL_mean": float(profit_PIL_mean),
        "profit_PIL_ci95": float(profit_PIL_ci),
        "profit_PPO_mean": float(profit_PPO_mean),
        "profit_PPO_ci95": float(profit_PPO_ci),
        "revenue_BS_mean": float(rev_BS_mean),
        "revenue_PIL_mean": float(rev_PIL_mean),
        "revenue_PPO_mean": float(rev_PPO_mean),
        "cost_BS_mean": float(cost_BS_mean),
        "cost_BS_ci95": float(cost_BS_ci),
        "cost_PIL_mean": float(cost_PIL_mean),
        "cost_PIL_ci95": float(cost_PIL_ci),
        "cost_PPO_mean": float(cost_PPO_mean),
        "cost_PPO_ci95": float(cost_PPO_ci),
        "lost_units_BS_mean": float(lost_BS_mean),
        "lost_units_PIL_mean": float(lost_PIL_mean),
        "lost_units_PPO_mean": float(lost_PPO_mean),
        "waste_units_BS_mean": float(waste_BS_mean),
        "waste_units_PIL_mean": float(waste_PIL_mean),
        "waste_units_PPO_mean": float(waste_PPO_mean),
        "profit_BS_std": float(std_BS),
        "profit_PIL_std": float(std_PIL),
        "profit_PPO_std": float(std_PPO),
        "ppo_model_path": ppo_model_path_str,
        "ppo_model_mtime": float(ppo_model_mtime),
        "ppo_train_seconds": meta.get("training_time_seconds", float("nan")),
        "ppo_total_timesteps": meta.get("total_timesteps", float("nan")),
        "ppo_net_arch": meta.get("ppo_hyperparams", {}).get("net_arch", ""),
        "ppo_learning_rate_start": meta.get("ppo_hyperparams", {}).get("learning_rate_start", ""),
        "ppo_learning_rate_end": meta.get("ppo_hyperparams", {}).get("learning_rate_end", ""),
        "ppo_n_steps": meta.get("ppo_hyperparams", {}).get("n_steps", ""),
        "ppo_batch_size": meta.get("ppo_hyperparams", {}).get("batch_size", ""),
        "ppo_n_epochs": meta.get("ppo_hyperparams", {}).get("n_epochs", ""),
        "ppo_gamma": meta.get("ppo_hyperparams", {}).get("gamma", ""),
        "ppo_clip_range": meta.get("ppo_hyperparams", {}).get("clip_range", ""),
        "ppo_ent_coef": meta.get("ppo_hyperparams", {}).get("ent_coef", ""),
        "ppo_final_eval_mean_profit_per_period": meta.get("final_eval", {}).get("mean_profit_per_period_raw", ""),
        "ppo_final_eval_timesteps": meta.get("final_eval", {}).get("timesteps", ""),
    }

    return row_out


def _worker_benchmark_one_config(task: Tuple[int, str, str, bool, str]) -> Dict[str, Any]:
    """Child-process wrapper for one configuration."""
    cid, base_dir_s, excel_s, quick, issuing_policy = task
    _set_single_thread_env()

    base_dir = Path(base_dir_s)
    logs_dir, _, _ = _ensure_dirs(base_dir)
    excel_path = Path(excel_s)

    return _benchmark_one_config(
        cid=int(cid),
        base_dir=base_dir,
        excel_path=excel_path,
        logs_dir=logs_dir,
        quick=bool(quick),
        parallel_reps=False,  # avoid nested parallelism
        issuing_policy=issuing_policy,
    )
def main() -> None:
    args = _parse_cli()

    base_dir = Path(__file__).resolve().parent
    logs_dir, results_dir, plots_dir = _ensure_dirs(base_dir)
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(exist_ok=True)


    # Config IDs selection: CLI overrides. Otherwise, use CONFIG_IDS if set, else all IDs from Excel.
    if args.config_ids:
        config_ids = [int(x.strip()) for x in args.config_ids.split(",") if x.strip()]
    elif CONFIG_IDS is not None:
        config_ids = list(CONFIG_IDS)
    else:
        try:
            df_all = pd.read_excel(str(args.excel) if args.excel else str(base_dir / "configurations.xlsx"))
            col = "configuration" if "configuration" in df_all.columns else df_all.columns[0]
            config_ids = sorted({int(x) for x in df_all[col].dropna().tolist()})
        except Exception:
            config_ids = []
    quick = bool(args.quick or QUICK_MODE)

    # Issuing policy: prefer environment variable set by orchestration runner
    issuing_policy = str(os.environ.get("INVENTORY_ISSUING_POLICY", args.issuing_policy)).strip().lower()
    if issuing_policy not in {"fifo", "lifo", "random"}:
        issuing_policy = "fifo"

    if quick:
        T_HORIZON = T_HORIZON_QUICK
        BURN_IN = BURN_IN_QUICK
        N_REPS = N_REPS_QUICK
        TUNE_HORIZON = TUNE_HORIZON_QUICK
        TUNE_BURN_IN = TUNE_BURN_IN_QUICK
        TUNE_REPS = TUNE_REPS_QUICK
    else:
        T_HORIZON = T_HORIZON_FULL
        BURN_IN = BURN_IN_FULL
        N_REPS = N_REPS_FULL
        TUNE_HORIZON = TUNE_HORIZON_FULL
        TUNE_BURN_IN = TUNE_BURN_IN_FULL
        TUNE_REPS = TUNE_REPS_FULL

    excel_path = base_dir / CONFIG_EXCEL_PATH
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing {excel_path}. Copy configurations.xlsx into the package folder.")

    rows_out: List[Dict[str, Any]] = []

    parallel_configs = bool(args.parallel_configs)
    max_config_workers = int(args.max_config_workers)
    if max_config_workers <= 0:
        max_config_workers = max(1, (os.cpu_count() or 2) - 1)

    # If we parallelize across configs, disable rep-level parallelization to avoid nested pools.
    parallel_reps = bool(PARALLEL_REPS) and (not parallel_configs)

    if parallel_configs and len(config_ids) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        ctx = get_context("spawn")
        tasks = [(int(cid), str(base_dir), str(excel_path), bool(quick), str(issuing_policy)) for cid in config_ids]
        n_workers = min(int(max_config_workers), len(tasks))

        print(f"[benchmarks] Parallel configs enabled: workers={n_workers}, rep_parallel={parallel_reps}")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            fut_map = {ex.submit(_worker_benchmark_one_config, t): t[0] for t in tasks}
            for fut in as_completed(fut_map):
                cid = fut_map[fut]
                row_out = fut.result()
                rows_out.append(row_out)

                # progress print (same fields as before)
                try:
                    print(
                        f"Config {cid}: S_BS={row_out.get('S_BS')} | S_PIL={row_out.get('S_PIL')} | "
                        f"profit_BS={float(row_out.get('profit_BS_mean', float('nan'))):.3f} | "
                        f"profit_PIL={float(row_out.get('profit_PIL_mean', float('nan'))):.3f} | "
                        f"profit_PPO={float(row_out.get('profit_PPO_mean', float('nan'))):.3f}"
                    )
                except Exception:
                    pass
    else:
        for cid in config_ids:
            row_out = _benchmark_one_config(
                cid=int(cid),
                base_dir=base_dir,
                excel_path=excel_path,
                logs_dir=logs_dir,
                quick=bool(quick),
                parallel_reps=parallel_reps,
                issuing_policy=issuing_policy,
            )
            rows_out.append(row_out)

            print(
                f"Config {cid}: S_BS={row_out.get('S_BS')} | S_PIL={row_out.get('S_PIL')} | "
                f"profit_BS={float(row_out.get('profit_BS_mean', float('nan'))):.3f} | "
                f"profit_PIL={float(row_out.get('profit_PIL_mean', float('nan'))):.3f} | "
                f"profit_PPO={float(row_out.get('profit_PPO_mean', float('nan'))):.3f}"
            )

    # Keep rows ordered by configuration for stable CSV diffs
    rows_out = sorted(rows_out, key=lambda d: int(d.get("configuration", 0)))
    out_df = pd.DataFrame(rows_out)
    out_path = results_dir / "policy_comparison.csv"
    out_path_policy = results_dir / f"policy_comparison_{issuing_policy}.csv"
    # On Windows, writing can fail with PermissionError if the CSV is currently
    # open in Excel or another process. Instead of crashing the whole run_all
    # pipeline, fall back to a timestamped filename.
    out_path_written = out_path
    try:
        out_df.to_csv(out_path, index=False)
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path_written = results_dir / f"policy_comparison_{ts}.csv"
        out_df.to_csv(out_path_written, index=False)
        print(
            f"[bench] WARNING: Could not write '{out_path}' (file locked). "
            f"Wrote '{out_path_written}' instead. Close the CSV and re-run to overwrite."
        )

    out_path_policy_written = out_path_policy
    try:
        out_df.to_csv(out_path_policy, index=False)
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path_policy_written = results_dir / f"policy_comparison_{issuing_policy}_{ts}.csv"
        out_df.to_csv(out_path_policy_written, index=False)
        print(
            f"[bench] WARNING: Could not write '{out_path_policy}' (file locked). "
            f"Wrote '{out_path_policy_written}' instead."
        )

    # Meta for run reproducibility (horizons, seeds, tuning settings)
    meta_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": bool(quick),
        "issuing_policy": str(issuing_policy),
        "parallel_configs": bool(parallel_configs),
        "max_config_workers": int(max_config_workers),
        "config_ids": [int(x) for x in config_ids],
        "outputs": {
            "policy_comparison_csv": str(out_path_written),
            "policy_comparison_policy_csv": str(out_path_policy_written),
        },
        "eval": {
            "horizon": int(T_HORIZON),
            "burn_in": int(BURN_IN),
            "n_reps": int(N_REPS),
            "seed0": int(EVAL_SEED0),
            "demand_model": "sample_demand_stream (configurations.xlsx demand_type, bounded by demand_dmax)",
        },
        "tuning": {
            "horizon": int(TUNE_HORIZON),
            "burn_in": int(TUNE_BURN_IN),
            "n_reps": int(TUNE_REPS),
            "seed0": int(TUNE_SEED0),
            "grid": "0..max_order (early-stop patience=5)",
        },
        "pil": {
            "use_correction": bool(PIL_USE_CORRECTION),
            "n_mc_tune": int(PIL_N_MC_TUNE),
            "n_mc_eval": int(PIL_N_MC_EVAL),
        },
    }
    meta_path = results_dir / "policy_comparison_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    _plot_policy_comparison(out_df, plots_dir)

    # Duplicate plots under ./figures as well (some users look for that folder name)
    try:
        import shutil
        for p in plots_dir.glob("*.png"):
            shutil.copyfile(p, figures_dir / p.name)
    except Exception:
        pass

    print(f"\nSaved: {out_path_written}")
    print(f"Saved: {out_path_policy_written}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()


