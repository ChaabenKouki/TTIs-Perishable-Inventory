# train_ppo_GY.py
"""
Train PPO policies for the Green/Yellow perishable inventory environment.

Inputs:
- configurations.xlsx (path passed with --excel)
- configuration id(s) passed by --config-id or --config-ids

Main outputs:
- logs/config_<id>/best_model.zip
- logs/config_<id>/vecnormalize.pkl
- logs/config_<id>/training_meta.json

Typical command:
    python train_ppo_GY.py --excel configurations.xlsx --config-id 1 --quick
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import gymnasium as gym  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from InventoryEnvGY_Config import InventoryEnvGYConfig


# ---------------------------------------------------------------------------
# Helper: Load hyperparameters from JSON file
# ---------------------------------------------------------------------------
def _load_hyperparam_from_file(hyper_file: str) -> List[Dict[str, Any]]:
    """Load hyperparameters from a JSON file (e.g., from bestparm.py output)."""
    path = Path(hyper_file)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {hyper_file}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"Expected list or dict in {hyper_file}, got {type(data)}")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
EXCEL_PATH = "configurations.xlsx"

EPISODE_LENGTH_TRAIN = 2000
EPISODE_LENGTH_EVAL = 2000

TOTAL_TIMESTEPS_QUICK = 200_000
TOTAL_TIMESTEPS_FULL = 300_000
TOTAL_TIMESTEPS_FULL_CAP = 400_000

REWARD_SCALE = 100.0
OBS_EXTRA_FEATURES = True
MAX_ORDER = 60

# Eval settings
N_EVAL_EPISODES_CB = 5
EVAL_FREQ_DIVISOR = 20
EVAL_FREQ_MIN_QUICK = 1_000
EVAL_FREQ_MIN_FULL = 2_000

FINAL_EVAL_N_EPISODES = 40
FINAL_EVAL_BURN_IN_FRACTION = 0.20
TAIL_DROP_STEPS = 0

# Early stopping
NO_IMPROVEMENT_EVALS = 15
MIN_EVALS_BEFORE_STOP = 20

EARLY_STOP_FULL_MIN_STEPS = 150_000
EARLY_STOP_FULL_PATIENCE_STEPS = 40_000
EARLY_STOP_FULL_MIN_FRACTION = 0.60
EARLY_STOP_FULL_PATIENCE_FRACTION = 0.10

# --- MISSING CONSTANTS RESTORED HERE ---
TRAIN_LOG_EVERY_N_STEPS = 1
TRAIN_ROLLING_WINDOW = 200
TRAIN_PLOT_MAX_POINTS = 200_000
TRAIN_ZOOM_FRACTION = 0.20


def _compute_total_timesteps(*, quick: bool, cv: float) -> int:
    if bool(quick):
        return int(TOTAL_TIMESTEPS_QUICK)

    base = int(TOTAL_TIMESTEPS_FULL)
    cap = int(TOTAL_TIMESTEPS_FULL_CAP)

    try:
        cv_f = float(cv)
    except Exception:
        cv_f = float("nan")

    if not (cv_f == cv_f):
        return base

    if cv_f < 0.25:
        mult = 1.50
    elif cv_f < 0.35:
        mult = 1.25
    elif cv_f <= 0.65:
        mult = 1.00
    elif cv_f < 0.90:
        mult = 1.25
    else:
        mult = 1.50

    return int(min(cap, round(base * mult)))


# ---------------------------------------------------------------------------
# HYPERPARAMETERS (STABLE SET)
# ---------------------------------------------------------------------------
HYPERPARAM_CANDIDATES: List[Dict[str, Any]] = [

    # 1) YOUR ORIGINAL (unchanged)
    {
        "name": "stable_long_run",
        "net_arch": {"pi": [64, 64], "vf": [128, 128]},
        "lr_start": 3e-4,
        "lr_end": 1e-5,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_start": 0.2,
        "clip_end": 0.05,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "clip_range_vf": None,
    },

    # 2) YOUR ORIGINAL (unchanged)
    {
        "name": "fast_learner",
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "lr_start": 4e-4,
        "lr_end": 4e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_start": 0.2,
        "clip_end": 0.1,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "clip_range_vf": None,
    },

    # 3) stable_long_run + slightly lower entropy (still high)
    {
        "name": "stable_long_run_ent007",
        "net_arch": {"pi": [64, 64], "vf": [128, 128]},
        "lr_start": 3e-4,
        "lr_end": 1e-5,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_start": 0.2,
        "clip_end": 0.05,
        "ent_coef": 0.07,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "clip_range_vf": None,
    },

    # 4) stable_long_run + tighter KL (more conservative updates)
    {
        "name": "stable_long_run_kl005",
        "net_arch": {"pi": [64, 64], "vf": [128, 128]},
        "lr_start": 3e-4,
        "lr_end": 1e-5,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_start": 0.2,
        "clip_end": 0.05,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.005,
        "clip_range_vf": None,
    },

    # 5) stable_long_run + smaller LR start (reduce oscillation, keep everything else)
    {
        "name": "stable_long_run_lr2e4",
        "net_arch": {"pi": [64, 64], "vf": [128, 128]},
        "lr_start": 2e-4,
        "lr_end": 1e-5,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_start": 0.2,
        "clip_end": 0.05,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "clip_range_vf": None,
    },

    # 6) fast_learner + slightly higher entropy (still in your style)
    {
        "name": "fast_learner_ent007",
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "lr_start": 4e-4,
        "lr_end": 4e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_start": 0.2,
        "clip_end": 0.1,
        "ent_coef": 0.07,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "clip_range_vf": None,
    },

    # 7) fast_learner + tighter clip end (less late noise)
    {
        "name": "fast_learner_clip07",
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "lr_start": 4e-4,
        "lr_end": 4e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_start": 0.2,
        "clip_end": 0.07,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "clip_range_vf": None,
    },

    # 8) fast_learner + slightly larger critic (helps value stability)
    {
        "name": "fast_learner_bigV",
        "net_arch": {"pi": [64, 64], "vf": [128, 128]},
        "lr_start": 4e-4,
        "lr_end": 4e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_start": 0.2,
        "clip_end": 0.1,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "clip_range_vf": None,
    },
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_config_ids(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _read_config_row(excel_path: str, config_id: int) -> Dict[str, Any]:
    df = pd.read_excel(excel_path, engine="openpyxl").reset_index(drop=True)
    if "configuration" in df.columns:
        sub = df.loc[df["configuration"].astype(int) == int(config_id)]
        if sub.empty:
            raise ValueError(f"Config id={int(config_id)} not found in {excel_path}")
        return sub.iloc[0].to_dict()

    idx = int(config_id) - 1
    if idx < 0 or idx >= len(df):
        raise ValueError(f"Config id={int(config_id)} out of range for {excel_path}")
    return df.iloc[idx].to_dict()


def _assert_non_gamma_demand(row: Dict[str, Any], *, config_id: int, excel_path: str) -> None:
    dt = str(row.get("demand_type", "")).strip().lower()
    if dt in ("", "nan", "none") or dt.startswith("gamma"):
        raise ValueError(f"Config id={int(config_id)}: invalid demand_type='{dt}' in {excel_path}")


def _make_env_config(base: Dict[str, Any], *, episode_length: int, issuing_policy: str = "fifo") -> Dict[str, Any]:
    m = int(base.get("m", 3))
    L = int(base.get("L", 1))

    beta_excel = float(base.get("beta_excel", base.get("beta", 0.0)))
    beta_excel = float(np.clip(beta_excel, 0.0, 1.0))

    cfg: Dict[str, Any] = {
        "m": m,
        "L": L,
        "Alpha": float(base.get("Alpha", 0.0)),
        "beta_excel": beta_excel,
        "obs_extra_features": bool(OBS_EXTRA_FEATURES),
        "mean_demand": float(base.get("mean_demand", 3.0)),
        "coef_of_var": float(base.get("coef_of_var", 0.5)),
        "demand_type": str(base.get("demand_type", "beta_binomial")),
        "demand_dmax": int(base.get("demand_dmax", base.get("dmax", 65))),
        "mix_pi_high": base.get("mix_pi_high", base.get("pi_high", None)),
        "mix_pi_calm": base.get("mix_pi_calm", None),
        "mix_pi_normal": base.get("mix_pi_normal", None),
        "mix_pi_promo": base.get("mix_pi_promo", None),
        "mix_lam1": float(base.get("mix_lam1", base.get("lam1", 0.5))),
        "mix_lam2": base.get("mix_lam2", base.get("lam2", None)),
        "mix_lam3": float(base.get("mix_lam3", base.get("lam3", 15.0))),
        "mix_w1": float(base.get("mix_w1", base.get("w1", 0.01))),
        "mix_w2": float(base.get("mix_w2", base.get("w2", 1.0))),
        "p1": float(base.get("p1", 20.0)),
        "p2": float(base.get("p2", 0.0)),
        "c": float(base.get("c", 0.0)),
        "h": float(base.get("h", 1.0)),
        "b1": float(base.get("b1", 10.0)),
        "b2": float(base.get("b2", 0.0)),
        "w": float(base.get("w", 3.0)),
        "fixed_order_cost": float(base.get("fixed_order_cost", 0.0)),
        "tti_unit_cost": float(base.get("tti_unit_cost", 0.0)),
        "reward_scale": float(REWARD_SCALE),
        "max_order": int(base.get("max_order", MAX_ORDER)),
        "episode_length": int(episode_length),
        "warm_up_fraction": 0.0,
        "sensor_error_eps": 0.0,
        "issuing_policy": str(issuing_policy).strip().lower(),
        "seed": None,
    }
    return cfg


def _make_lr_schedule(lr_start: float, lr_end: float):
    lr_start = float(lr_start)
    lr_end = float(lr_end)

    def _schedule(progress_remaining: float) -> float:
        return float(lr_end + (lr_start - lr_end) * float(progress_remaining))

    return _schedule


def _make_clip_schedule(clip_start: float, clip_end: float):
    clip_start = float(clip_start)
    clip_end = float(clip_end)

    def _schedule(progress_remaining: float) -> float:
        return float(clip_end + (clip_start - clip_end) * float(progress_remaining))

    return _schedule


def _to_int_action(action: Any) -> int:
    if action is None:
        return 0
    if np.isscalar(action):
        return int(action)
    arr = np.asarray(action)
    if arr.ndim == 0:
        return int(arr.item())
    return int(arr.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class TrainStepProfitCallback(BaseCallback):
    """Logs raw profit per period during PPO training."""
    def __init__(self, *, out_dir: Path, total_timesteps_budget: int, reward_scale: float, log_every_n_steps: int = 1, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.out_dir = Path(out_dir)
        self.total_timesteps_budget = int(max(1, total_timesteps_budget))
        self.reward_scale = float(reward_scale)
        self.log_every_n_steps = int(max(1, log_every_n_steps))
        self._i = 0
        self._timesteps: Optional[np.ndarray] = None
        self._raw_profit: Optional[np.ndarray] = None

    def _on_training_start(self) -> None:
        n = int(self.total_timesteps_budget // self.log_every_n_steps) + 2
        self._timesteps = np.empty(n, dtype=np.int64)
        self._raw_profit = np.empty(n, dtype=np.float64)
        self._i = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps % self.log_every_n_steps) != 0:
            return True

        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)

        raw = None
        if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
            raw = infos[0].get("raw_profit", None)

        if raw is None and rewards is not None:
            try:
                r0 = float(np.asarray(rewards).reshape(-1)[0])
                raw = r0 * self.reward_scale
            except Exception:
                raw = float("nan")

        if self._timesteps is None or self._raw_profit is None:
            return True

        if self._i >= self._timesteps.size:
            self._timesteps = np.pad(self._timesteps, (0, 10000), constant_values=0)
            self._raw_profit = np.pad(self._raw_profit, (0, 10000), constant_values=np.nan)

        self._timesteps[self._i] = int(self.num_timesteps)
        self._raw_profit[self._i] = float(raw)
        self._i += 1
        return True

    def _on_training_end(self) -> None:
        self.save()

    def save(self) -> Path:
        _ensure_dir(self.out_dir)
        out_path = self.out_dir / "train_step_profit.npz"
        if self._timesteps is None or self._raw_profit is None or self._i <= 0:
            np.savez_compressed(str(out_path), timesteps=np.array([], dtype=np.int64), raw_profit=np.array([], dtype=np.float64))
            return out_path
        np.savez_compressed(str(out_path), timesteps=self._timesteps[: self._i], raw_profit=self._raw_profit[: self._i])
        return out_path


def _plot_training_convergence(train_npz_path: Path, out_png_path: Path, *, rolling_window: int, max_points: int, zoom_fraction: float) -> Dict[str, float]:
    data = np.load(str(train_npz_path), allow_pickle=True)
    t = data.get("timesteps", np.array([], dtype=np.int64))
    x = data.get("raw_profit", np.array([], dtype=np.float64))

    if t.size == 0 or x.size == 0:
        return {}

    w = int(max(1, rolling_window))
    if x.size < w:
        mean, std, tt = x.copy(), np.zeros_like(x), t.copy()
    else:
        c1 = np.cumsum(np.insert(x, 0, 0.0))
        c2 = np.cumsum(np.insert(x * x, 0, 0.0))
        mean = (c1[w:] - c1[:-w]) / float(w)
        var = (c2[w:] - c2[:-w]) / float(w) - mean * mean
        std = np.sqrt(np.maximum(var, 0.0))
        tt = t[w - 1 :]

    def _downsample(arrs, max_n):
        if arrs[0].size <= max_n: return arrs
        step = int(math.ceil(arrs[0].size / float(max_n)))
        return [a[::step] for a in arrs]

    import matplotlib.pyplot as plt
    tt_p, mean_p, std_p = _downsample([tt, mean, std], max_points)

    plt.figure(figsize=(10, 5))
    plt.title("PPO convergence (training)")
    plt.xlabel("Timesteps")
    plt.ylabel("Profit")
    plt.fill_between(tt_p, mean_p - std_p, mean_p + std_p, alpha=0.25)
    plt.plot(tt_p, mean_p, label=f"Mean (w={w})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_png_path), dpi=100)
    plt.close()

    return {"final_train_mean": float(mean[-1]) if mean.size else 0.0}


class DeterministicProfitEvalCallback(BaseCallback):
    """
    Saves best_model.zip AND writes vecnormalize_best.pkl whenever improved.
    """
    def __init__(
        self,
        *,
        eval_cfg: Dict[str, Any],
        eval_freq: int,
        n_eval_episodes: int,
        burn_in_fraction: float,
        seed0: int,
        out_dir: Path,
        reward_scale: float,
        min_steps: int,
        patience_steps: int,
        vecnorm_train: VecNormalize,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_cfg = dict(eval_cfg)
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.burn_in_fraction = float(burn_in_fraction)
        self.seed0 = int(seed0)
        self.out_dir = Path(out_dir)
        self.reward_scale = float(reward_scale)
        self.min_steps = int(max(0, min_steps))
        self.patience_steps = int(max(0, patience_steps))
        self.vecnorm_train = vecnorm_train

        self.best_mean_profit = float("-inf")
        self.best_step = 0
        self._next_eval = self.eval_freq

    def _eval_mean_profit(self) -> float:
        env = InventoryEnvGYConfig(dict(self.eval_cfg, seed=self.seed0 + 777))
        T = int(self.eval_cfg.get("episode_length", EPISODE_LENGTH_EVAL))
        burn_in = int(round(self.burn_in_fraction * T))
        end_t = int(max(0, T - int(TAIL_DROP_STEPS)))

        ep_vals: List[float] = []

        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=int(self.seed0 + 10_000 + ep))
            done = False
            t = 0
            s = 0.0
            n = 0
            while not done:
                # normalize obs using TRAIN stats (in-memory)
                obs_norm = self.vecnorm_train.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
                action, _ = self.model.predict(obs_norm, deterministic=True)  # type: ignore[attr-defined]
                q = _to_int_action(action)
                obs, r, term, trunc, info = env.step(q)

                if (t >= burn_in) and (t < end_t):
                    raw = info.get("raw_profit", float(r) * self.reward_scale)
                    s += float(raw)
                    n += 1
                t += 1
                done = bool(term or trunc)
            ep_vals.append(s / max(1, n))

        return float(np.mean(ep_vals))

    def _on_step(self) -> bool:
        if int(self.num_timesteps) < int(self._next_eval):
            return True

        mean_profit = self._eval_mean_profit()
        improved = bool(mean_profit > self.best_mean_profit)
        if improved:
            self.best_mean_profit = float(mean_profit)
            self.best_step = int(self.num_timesteps)
            _ensure_dir(self.out_dir)
            # save best_model
            self.model.save(str(self.out_dir / "best_model.zip"))  # type: ignore[attr-defined]
            # save matching stats snapshot for that best_model
            self.vecnorm_train.save(str(self.out_dir / "vecnormalize_best.pkl"))

        # early stop (FULL/QUICK behavior preserved by your parameters)
        if int(self.num_timesteps) >= int(self.min_steps) and self.patience_steps > 0:
            if (int(self.num_timesteps) - int(self.best_step)) >= int(self.patience_steps):
                return False

        self._next_eval += int(self.eval_freq)
        return True


def _final_deterministic_eval(
    *,
    model: PPO,
    eval_cfg: Dict[str, Any],
    seed0: int,
    n_episodes: int,
    burn_in_fraction: float,
    vecnorm: Optional[VecNormalize],
) -> float:
    env = InventoryEnvGYConfig(dict(eval_cfg, seed=seed0 + 777))
    T = int(eval_cfg.get("episode_length", EPISODE_LENGTH_EVAL))
    burn_in = int(round(float(burn_in_fraction) * T))
    end_t = int(max(0, T - int(TAIL_DROP_STEPS)))

    vals: List[float] = []
    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(seed0 + 10_000 + ep))
        done = False
        t = 0
        s = 0.0
        n = 0
        while not done:
            if vecnorm is not None:
                obs_norm = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
                action, _ = model.predict(obs_norm, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            q = _to_int_action(action)
            obs, r, term, trunc, info = env.step(q)
            if (t >= burn_in) and (t < end_t):
                s += float(info.get("raw_profit", float(r) * REWARD_SCALE))
                n += 1
            t += 1
            done = bool(term or trunc)
        vals.append(s / max(1, n))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Training one candidate (ONE run_dir)
# ---------------------------------------------------------------------------
def _train_candidate(
    *,
    excel_path: str,
    row: Dict[str, Any],
    config_id: int,
    issuing_policy: str,
    out_dir: Path,
    quick: bool,
    train_seed: int,
    eval_seed_base: int,
    hyper: Dict[str, Any],
) -> Dict[str, Any]:
    _ensure_dir(out_dir)

    cv = float(row.get("coef_of_var", row.get("cv", float("nan"))))
    total_timesteps = _compute_total_timesteps(quick=bool(quick), cv=float(cv))

    min_eval = int(EVAL_FREQ_MIN_QUICK if quick else EVAL_FREQ_MIN_FULL)
    eval_freq = int(max(min_eval, int(total_timesteps // int(EVAL_FREQ_DIVISOR))))

    train_cfg = _make_env_config(row, episode_length=EPISODE_LENGTH_TRAIN, issuing_policy=issuing_policy)
    eval_cfg = _make_env_config(row, episode_length=EPISODE_LENGTH_EVAL, issuing_policy=issuing_policy)

    set_random_seed(int(train_seed))
    np.random.seed(int(train_seed))

    def make_train_env():
        c = dict(train_cfg)
        c["seed"] = int(train_seed)
        return InventoryEnvGYConfig(c)

    # VecNormalize training env
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    # schedules
    lr_schedule = _make_lr_schedule(hyper["lr_start"], hyper["lr_end"])
    clip_schedule = _make_clip_schedule(hyper["clip_start"], hyper["clip_end"])

    # early stop thresholds
    if quick:
        min_steps_es = int(MIN_EVALS_BEFORE_STOP) * int(eval_freq)
        patience_steps_es = int(NO_IMPROVEMENT_EVALS) * int(eval_freq)
    else:
        min_steps_es = int(max(EARLY_STOP_FULL_MIN_STEPS, EARLY_STOP_FULL_MIN_FRACTION * float(total_timesteps)))
        patience_steps_es = int(max(EARLY_STOP_FULL_PATIENCE_STEPS, EARLY_STOP_FULL_PATIENCE_FRACTION * float(total_timesteps)))

    cb_profit = DeterministicProfitEvalCallback(
        eval_cfg=eval_cfg,
        eval_freq=int(eval_freq),
        n_eval_episodes=int(N_EVAL_EPISODES_CB),
        burn_in_fraction=float(FINAL_EVAL_BURN_IN_FRACTION),
        seed0=int(eval_seed_base),
        out_dir=out_dir,
        reward_scale=float(REWARD_SCALE),
        min_steps=int(min_steps_es),
        patience_steps=int(patience_steps_es),
        vecnorm_train=train_env,
        verbose=0,
    )

    cb_step = TrainStepProfitCallback(
        out_dir=out_dir,
        total_timesteps_budget=int(total_timesteps),
        reward_scale=float(REWARD_SCALE),
        log_every_n_steps=TRAIN_LOG_EVERY_N_STEPS,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr_schedule,
        n_steps=int(hyper["n_steps"]),
        batch_size=int(hyper["batch_size"]),
        n_epochs=int(hyper["n_epochs"]),
        gamma=float(hyper["gamma"]),
        gae_lambda=float(hyper["gae_lambda"]),
        clip_range=clip_schedule,
        ent_coef=float(hyper["ent_coef"]),
        vf_coef=float(hyper["vf_coef"]),
        max_grad_norm=float(hyper["max_grad_norm"]),
        target_kl=float(hyper["target_kl"]) if hyper.get("target_kl", None) is not None else None,
        clip_range_vf=None,
        policy_kwargs={"net_arch": hyper["net_arch"]},
        normalize_advantage=True,
        verbose=0,
        seed=int(train_seed),
        device="cpu",
    )

    t0 = time.time()
    model.learn(total_timesteps=int(total_timesteps), callback=CallbackList([cb_profit, cb_step]))
    train_seconds = float(time.time() - t0)

    # Save final model + final stats snapshot
    final_model_path = out_dir / "ppo_model.zip"
    model.save(str(final_model_path))
    train_env.save(str(out_dir / "vecnormalize_final.pkl"))

    # Robust selection: best_model vs final_model WITH THEIR MATCHED STATS
    best_model_path = out_dir / "best_model.zip"
    best_stats_path = out_dir / "vecnormalize_best.pkl"
    final_stats_path = out_dir / "vecnormalize_final.pkl"

    # Load stats objects for scoring
    stats_best = None
    if best_stats_path.exists():
        dummy = DummyVecEnv([lambda: InventoryEnvGYConfig(dict(eval_cfg, seed=eval_seed_base + 777))])
        stats_best = VecNormalize.load(str(best_stats_path), dummy)
        stats_best.training = False
        stats_best.norm_reward = False

    dummy2 = DummyVecEnv([lambda: InventoryEnvGYConfig(dict(eval_cfg, seed=eval_seed_base + 777))])
    stats_final = VecNormalize.load(str(final_stats_path), dummy2)
    stats_final.training = False
    stats_final.norm_reward = False

    score_best = float("-inf")
    if best_model_path.exists() and stats_best is not None:
        try:
            m_best = PPO.load(str(best_model_path), device="cpu")
            score_best = _final_deterministic_eval(
                model=m_best,
                eval_cfg=eval_cfg,
                seed0=int(eval_seed_base),
                n_episodes=int(FINAL_EVAL_N_EPISODES),
                burn_in_fraction=float(FINAL_EVAL_BURN_IN_FRACTION),
                vecnorm=stats_best,
            )
        except Exception:
            score_best = float("-inf")

    # score final
    score_final = _final_deterministic_eval(
        model=model,
        eval_cfg=eval_cfg,
        seed0=int(eval_seed_base),
        n_episodes=int(FINAL_EVAL_N_EPISODES),
        burn_in_fraction=float(FINAL_EVAL_BURN_IN_FRACTION),
        vecnorm=stats_final,
    )

    # Winner -> write best_model.zip and vecnormalize.pkl
    winner = "final"
    best_score = score_final

    if score_best > score_final:
        winner = "best"
        best_score = score_best

    # Ensure benchmark-friendly files exist:
    #   best_model.zip + vecnormalize.pkl (matching)
    if winner == "final":
        shutil.copy2(str(final_model_path), str(best_model_path))
        shutil.copy2(str(final_stats_path), str(out_dir / "vecnormalize.pkl"))
    else:
        # best_model already exists; ensure its matching stats is copied to vecnormalize.pkl
        shutil.copy2(str(best_stats_path), str(out_dir / "vecnormalize.pkl"))

    meta = {
        "config_id": int(config_id),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": float(train_seconds),
        "total_timesteps_budget": int(total_timesteps),
        "final_scores": {"best_score": float(score_best), "final_score": float(score_final), "winner": winner, "winner_score": float(best_score)},
        "hyper": dict(hyper),
        "paths": {
            "best_model": str(best_model_path),
            "vecnormalize": str(out_dir / "vecnormalize.pkl"),
            "final_model": str(final_model_path),
            "vecnormalize_final": str(final_stats_path),
            "vecnormalize_best": str(best_stats_path),
        },
    }
    with open(out_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    # Optional plots
    if (out_dir / "train_step_profit.npz").exists():
        _plot_training_convergence(out_dir / "train_step_profit.npz", out_dir / "ppo_convergence.png", rolling_window=TRAIN_ROLLING_WINDOW, max_points=TRAIN_PLOT_MAX_POINTS, zoom_fraction=TRAIN_ZOOM_FRACTION)

    try:
        train_env.close()
    except Exception:
        pass

    return {"winner_score": float(best_score), "winner": winner, "meta": meta}


# ---------------------------------------------------------------------------
# Train per config
# ---------------------------------------------------------------------------
def train_one_config(
    *,
    excel_path: str,
    config_id: int,
    issuing_policy: str,
    out_dir: Path,
    quick: bool,
    seed0: int,
    sweep: bool,
    max_candidates: int = 0,
    hyper_list: Optional[List[Dict[str, Any]]] = None,
) -> None:
    _ensure_dir(out_dir)
    row = _read_config_row(excel_path, int(config_id))
    _assert_non_gamma_demand(row, config_id=int(config_id), excel_path=str(excel_path))

    # Use provided hyper_list if available, otherwise use HYPERPARAM_CANDIDATES
    if hyper_list is not None:
        cands = list(hyper_list)
    else:
        cands = list(HYPERPARAM_CANDIDATES)
    
    if int(max_candidates) > 0:
        cands = cands[: int(max_candidates)]
    if not sweep:
        cands = [cands[0]]

    best_profit = float("-inf")
    best_dir: Optional[Path] = None

    for i, hyper in enumerate(cands, start=1):
        run_dir = out_dir / f"hp_{i:02d}_{hyper['name']}"
        print(f"[train_ppo] config={config_id} | candidate {i}/{len(cands)} -> {run_dir.name}")

        res = _train_candidate(
            excel_path=str(excel_path),
            row=row,
            config_id=int(config_id),
            issuing_policy=str(issuing_policy),
            out_dir=run_dir,
            quick=bool(quick),
            train_seed=int(seed0 + 1000 * i),
            eval_seed_base=int(seed0),
            hyper=hyper,
        )

        score = float(res["winner_score"])
        if score > best_profit:
            best_profit = score
            best_dir = run_dir

    if best_dir is None:
        raise RuntimeError("No candidate produced output")

    print(f"[train_ppo] config={config_id} | BEST = {best_dir.name} score={best_profit:.3f}")

    # Promote best candidate to config root (logs/config_<id>/...)
    _copy = shutil.copy2
    _copy(str(best_dir / "best_model.zip"), str(out_dir / "best_model.zip"))
    _copy(str(best_dir / "vecnormalize.pkl"), str(out_dir / "vecnormalize.pkl"))
    _copy(str(best_dir / "training_meta.json"), str(out_dir / "training_meta.json"))
    _copy(str(best_dir / "metadata.json"), str(out_dir / "metadata.json"))

    # Optional: copy diagnostics if exist
    for fname in ["train_step_profit.npz", "ppo_convergence.png", "ppo_model.zip", "vecnormalize_final.pkl", "vecnormalize_best.pkl"]:
        src = best_dir / fname
        if src.exists():
            _copy(str(src), str(out_dir / fname))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=str, default=EXCEL_PATH)
    p.add_argument("--config-id", type=int, default=1)
    p.add_argument("--config-ids", type=str, default="")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--issuing-policy", type=str, default="fifo", choices=["fifo", "lifo", "random"])
    p.add_argument("--sweep", action="store_true", default=False)
    p.add_argument("--no-sweep", action="store_true", default=False)
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--hyper-file", type=str, default=None, 
                   help="Path to JSON file with hyperparameters (e.g., from bestparm.py output)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    issuing_policy = str(os.environ.get("INVENTORY_ISSUING_POLICY", args.issuing_policy)).strip().lower()
    if issuing_policy not in {"fifo", "lifo", "random"}:
        issuing_policy = "fifo"

    if str(args.config_ids).strip():
        config_ids = _parse_config_ids(args.config_ids)
    else:
        config_ids = [int(args.config_id)]

    base_out = Path(args.out_dir) if args.out_dir else Path("logs")

    sweep = bool(args.sweep)
    if bool(args.no_sweep):
        sweep = False
    if bool(args.quick) and (not bool(args.sweep)):
        # Keep legacy behavior for quick runs unless user explicitly requests --sweep.
        sweep = False

    # Load hyperparameters from file if provided
    hyper_file = args.hyper_file
    hyper_list = None
    if hyper_file:
        print(f"[main] Loading hyperparameters from: {hyper_file}")
        hyper_list = _load_hyperparam_from_file(hyper_file)
        print(f"[main] Loaded {len(hyper_list)} hyperparameter(s)")

    for cid in config_ids:
        out_dir = base_out / f"config_{int(cid)}"
        train_one_config(
            excel_path=str(args.excel),
            config_id=int(cid),
            issuing_policy=str(issuing_policy),
            out_dir=out_dir,
            quick=bool(args.quick),
            seed0=int(args.seed),
            sweep=bool(sweep),
            max_candidates=int(args.max_candidates),
            hyper_list=hyper_list,
        )


if __name__ == "__main__":
    main()


