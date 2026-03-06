# run_sensitivity.py
"""
Sensitivity driver for the TTI Green/Yellow perishables model.

Main sensitivity dimensions:
1) Non-zero ordering and TTI costs:
   - fixed_order_cost: per-order setup cost, incurred if q_t > 0
   - tti_unit_cost: per-unit additive ordering cost (added to unit ordering cost c)

2) Robustness to perishability and lead time:
   - (m, L) grid experiments, with m = shelf life (green periods) and L = lead time.

3) Sensor imperfection:
   - sensor_error_eps in [0, 0.5], implemented inside InventoryEnvGY_Config._get_observation()
     as deterministic misclassification mixing between green/yellow age buckets.
   - This affects PPO directly via the observation.
   - For PIL, you can choose to compute orders from the observed (noisy) state in sensor scenarios.

Design goal:
- Provide a clean, reproducible sensitivity pipeline that reuses:
  - baseline S_BS and S_PIL from policy_comparison.csv (no re-tuning needed), and
  - baseline PPO hyperparameters from logs/config_{id}/training_meta.json (no re-screening needed).

Output:
- results/sensitivity_<tag>.csv
- results/sensitivity_<tag>_meta.json
- plots/*.png (optional, lightweight)

Notes on comparability:
- By default, BS/PIL use the baseline tuned levels (fixed_baseline_levels=True).
  This is a robustness test of the policy parameterization.
- You can enable per-scenario re-tuning of S_BS/S_PIL via --no-fixed-baseline-levels.
- PPO retraining per scenario is enabled by default (recommended when reward/observation changes).

Typical command:
    python run_sensitivity.py --excel configurations.xlsx --policy-csv results/policy_comparison.csv --config-ids 1 --quick --issuing-policy fifo --ppo-hyper-source training_meta
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import run_benchmarks_GY as bench
from InventoryEnvGY_Config import InventoryEnvGYConfig

# PPO is optional unless you ask for it (keeps script usable if SB3 isn't installed)
try:  # pragma: no cover
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
except Exception:  # pragma: no cover
    PPO = None
    CallbackList = None
    DummyVecEnv = None
    VecNormalize = None
    set_random_seed = None

try:  # pragma: no cover
    import train_ppo_GY as ppo_train_ref
except Exception:  # pragma: no cover
    ppo_train_ref = None


# ======================================================================================
# Parsing helpers
# ======================================================================================
def _parse_config_ids(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _resolve_dir(base_dir: Path, path_arg: str, default_name: str) -> Path:
    s = str(path_arg).strip()
    if not s:
        return base_dir / default_name
    p = Path(s)
    if p.is_absolute():
        return p
    return base_dir / p


def _ensure_dirs(base_dir: Path, *, results_dir_arg: str = "results", plots_dir_arg: str = "plots") -> Tuple[Path, Path]:
    results_dir = _resolve_dir(base_dir, results_dir_arg, "results")
    plots_dir = _resolve_dir(base_dir, plots_dir_arg, "plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def _resolve_logs_dir(base_dir: Path, logs_dir_arg: str) -> Path:
    s = str(logs_dir_arg).strip()
    if not s:
        return base_dir / "logs"
    p = Path(s)
    if p.is_absolute():
        return p
    return base_dir / p


def _set_single_thread_env() -> None:
    # Avoid CPU oversubscription and common Anaconda OpenMP issues.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:  # pragma: no cover
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


# ======================================================================================
# Policy-comparison loading and baseline selection
# ======================================================================================
def _load_policy_comparison_df(policy_csv: Path) -> pd.DataFrame:
    """Load policy_comparison.csv with normalized columns for robust filtering."""
    if not Path(policy_csv).exists():
        raise FileNotFoundError(f"Cannot find policy_comparison.csv at: {policy_csv}")
    df = pd.read_csv(policy_csv)

    required = {"configuration", "demand_type", "coef_of_var", "m", "L", "Alpha", "beta_excel"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"policy_comparison.csv missing required column(s) {missing}: {policy_csv}")

    df = df.copy()
    df["configuration"] = pd.to_numeric(df["configuration"], errors="coerce").astype("Int64")
    df["coef_of_var"] = pd.to_numeric(df["coef_of_var"], errors="coerce")
    df["m"] = pd.to_numeric(df["m"], errors="coerce").astype("Int64")
    df["L"] = pd.to_numeric(df["L"], errors="coerce").astype("Int64")
    df["Alpha"] = pd.to_numeric(df["Alpha"], errors="coerce")
    df["beta_excel"] = pd.to_numeric(df["beta_excel"], errors="coerce")
    df["demand_type_norm"] = df["demand_type"].astype(str).str.strip().str.lower()
    if "issuing_policy" in df.columns:
        df["issuing_policy_norm"] = df["issuing_policy"].astype(str).str.strip().str.lower()

    # Optional (for levels)
    if "S_BS" in df.columns:
        df["S_BS"] = pd.to_numeric(df["S_BS"], errors="coerce")
    if "S_PIL" in df.columns:
        df["S_PIL"] = pd.to_numeric(df["S_PIL"], errors="coerce")

    return df


def _load_policy_comparison_meta(policy_csv: Path) -> Dict[str, Any]:
    """Load sibling policy_comparison_meta.json if available."""
    meta_path = Path(policy_csv).with_name("policy_comparison_meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _filter_policy_df_issuing(policy_df: pd.DataFrame, issuing_policy: str) -> Tuple[pd.DataFrame, bool]:
    """Filter by issuing policy if present; return (filtered_df, policy_verified)."""
    if policy_df is None or policy_df.empty:
        return policy_df, False
    if "issuing_policy_norm" not in policy_df.columns:
        return policy_df, False
    ip = str(issuing_policy).strip().lower()
    sub = policy_df.loc[policy_df["issuing_policy_norm"] == ip].copy()
    if sub.empty:
        raise ValueError(
            f"policy_comparison contains issuing_policy column, but no rows match issuing_policy='{ip}'."
        )
    return sub, True


def _policy_csv_name_matches_issuing(policy_csv: Path, issuing_policy: str) -> bool:
    stem = str(policy_csv.stem).strip().lower()
    return stem.endswith(f"_{str(issuing_policy).strip().lower()}")


def _parse_demand_type_list(arg: str, policy_df: pd.DataFrame) -> List[str]:
    s = str(arg).strip().lower()
    if s in {"all", "*"}:
        return sorted([str(x).strip().lower() for x in policy_df["demand_type_norm"].dropna().unique().tolist()])
    return [p.strip().lower() for p in s.split(",") if p.strip()]


def _select_baseline_triples_fixed_params(
    policy_df: pd.DataFrame,
    *,
    cv_list: List[float],
    demand_types: List[str],
    Alpha: float,
    beta_excel: float,
    m: Optional[int] = None,
    L: Optional[int] = None,
) -> List[Tuple[int, float, str]]:
    """Select baseline (config_id, cv, demand_type) triples with fixed (Alpha, beta_excel, m, L).

    This enforces the user's requirement:
    - same Alpha/beta across all demand types and CV scenarios,
    - same (m,L) across all demand types (unless overridden later by the (m,L) sensitivity grid).

    Deterministic tie-break:
    - pick smallest configuration id (there should typically be exactly one match).
    """
    if policy_df is None or policy_df.empty:
        raise ValueError("policy_df is empty; cannot auto-select baselines")

    tol = 1e-9
    out: List[Tuple[int, float, str]] = []

    for dt in demand_types:
        dt_norm = str(dt).strip().lower()
        for cv in cv_list:
            cv_f = float(cv)
            sub = policy_df.loc[
                (policy_df["demand_type_norm"] == dt_norm)
                & (policy_df["coef_of_var"].sub(cv_f).abs() <= tol)
                & (policy_df["Alpha"].sub(float(Alpha)).abs() <= tol)
                & (policy_df["beta_excel"].sub(float(beta_excel)).abs() <= tol)
            ].copy()

            if m is not None:
                sub = sub.loc[sub["m"].astype("Int64") == int(m)].copy()
            if L is not None:
                sub = sub.loc[sub["L"].astype("Int64") == int(L)].copy()

            if sub.empty:
                raise ValueError(
                    f"No baseline match in policy_comparison.csv for demand_type='{dt_norm}', cv={cv_f}, "
                    f"Alpha={float(Alpha)}, beta_excel={float(beta_excel)}, m={m}, L={L}."
                )

            sub = sub.sort_values(by=["configuration"], ascending=[True], kind="mergesort")
            cid = int(sub.iloc[0]["configuration"])
            out.append((cid, cv_f, dt_norm))

    return out


def _levels_from_policy_df(policy_df: pd.DataFrame, cid: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (S_BS, S_PIL) for this configuration from policy_df."""
    if policy_df is None or policy_df.empty:
        return None, None
    sub = policy_df.loc[policy_df["configuration"].astype("Int64") == int(cid)]
    if sub.empty:
        return None, None
    r = sub.iloc[0]
    s_bs = int(r["S_BS"]) if ("S_BS" in r and np.isfinite(r["S_BS"])) else None
    s_pil = int(r["S_PIL"]) if ("S_PIL" in r and np.isfinite(r["S_PIL"])) else None
    return s_bs, s_pil




# ======================================================================================
# PPO hyperparameter loading (top_candidates.json) and baseline-row injection
# ======================================================================================
def _load_top_candidates_json(path: Path) -> List[Dict[str, Any]]:
    """Load a top_candidates.json produced by hyperparam_screen_spyder_*.py.

    Expected format: a JSON list of dicts. Each dict is a PPO hyperparameter set
    with keys like: name, net_arch, lr_start, lr_end, n_steps, batch_size, ...
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            out: List[Dict[str, Any]] = []
            for it in obj:
                if isinstance(it, dict):
                    out.append(dict(it))
            return out
    except Exception:
        pass
    return []


def _select_top_candidate(cands: List[Dict[str, Any]], *, idx: int = 0, name: str = "") -> Dict[str, Any]:
    """Select a candidate by name (preferred) or by index. Returns {} if not found."""
    if not cands:
        return {}
    nm = str(name).strip().lower()
    if nm:
        for c in cands:
            if str(c.get("name", "")).strip().lower() == nm:
                return dict(c)
    i = int(idx)
    if 0 <= i < len(cands):
        return dict(cands[i])
    return {}


def _baseline_row_from_policy_df(
    *,
    policy_df: pd.DataFrame,
    cid: int,
    cv: float,
    issuing_policy: str,
    env_cfg: Dict[str, Any],
    S_bs: int,
    S_pil: int,
    tune_seconds_fallback: float = 0.0,
) -> Dict[str, Any]:
    """Create the baseline output row by reading metrics from policy_comparison.csv.

    This avoids re-running the baseline simulation when baseline results already exist.
    """
    if policy_df is None or policy_df.empty:
        raise ValueError("policy_df is empty; cannot build baseline row from policy_comparison.csv")

    sub = policy_df.loc[policy_df["configuration"].astype("Int64") == int(cid)].copy()
    if "demand_type_norm" in sub.columns:
        dt_norm = str(env_cfg.get("demand_type", "")).strip().lower()
        if dt_norm:
            sub_dt = sub.loc[sub["demand_type_norm"] == dt_norm].copy()
            if not sub_dt.empty:
                sub = sub_dt
    if "coef_of_var" in sub.columns:
        sub_cv = sub.loc[sub["coef_of_var"].sub(float(cv)).abs() <= 1e-9].copy()
        if not sub_cv.empty:
            sub = sub_cv
    if "issuing_policy_norm" in sub.columns:
        sub_ip = sub.loc[sub["issuing_policy_norm"] == str(issuing_policy).strip().lower()].copy()
        if not sub_ip.empty:
            sub = sub_ip
    if sub.empty:
        raise ValueError(f"Baseline config_id={cid} not found in policy_comparison.csv; cannot reuse baseline metrics.")
    r = sub.iloc[0]

    def _gf(col: str) -> float:
        if col not in r:
            return float("nan")
        try:
            v = float(r[col])
        except Exception:
            return float("nan")
        return float(v) if np.isfinite(v) else float("nan")

    beta_excel = float(env_cfg.get("beta_excel", env_cfg.get("beta", 0.0)))
    # Baseline reused from policy CSV is not tuned in this run.
    tune_seconds = 0.0

    row_out: Dict[str, Any] = {
        "config_id": int(cid),
        "demand_type": str(env_cfg.get("demand_type", "")),
        "issuing_policy": str(issuing_policy),
        "cv_scenario": float(cv),
        "scenario_group": "baseline",
        "scenario_name": "baseline",
        "m": int(env_cfg["m"]),
        "L": int(env_cfg["L"]),
        "Alpha": float(env_cfg["Alpha"]),
        "beta_excel": float(beta_excel),
        "beta_paper": float(1.0 - beta_excel),
        "mean_demand": float(env_cfg["mean_demand"]),
        "coef_of_var": float(env_cfg["coef_of_var"]),
        "c": float(env_cfg.get("c", 0.0)),
        "tti_unit_cost": float(env_cfg.get("tti_unit_cost", 0.0)),
        "fixed_order_cost": float(env_cfg.get("fixed_order_cost", 0.0)),
        "sensor_error_eps": float(env_cfg.get("sensor_error_eps", 0.0)),
        "S_BS": int(S_bs),
        "S_PIL": int(S_pil),
        "tune_seconds": float(tune_seconds),
        "pil_uses_observed_state": False,
    }

    # Map policy_comparison columns -> sensitivity columns
    def _attach_metrics(pol: str, prefix: str) -> None:
        row_out[f"{pol}_profit_mean"] = _gf(f"profit_{prefix}_mean")
        row_out[f"{pol}_profit_std"] = _gf(f"profit_{prefix}_std")
        row_out[f"{pol}_profit_ci95"] = _gf(f"profit_{prefix}_ci95")

        row_out[f"{pol}_cost_mean"] = _gf(f"cost_{prefix}_mean")
        row_out[f"{pol}_cost_std"] = float("nan")  # not stored in policy_comparison
        row_out[f"{pol}_cost_ci95"] = _gf(f"cost_{prefix}_ci95")

        row_out[f"{pol}_revenue_mean"] = _gf(f"revenue_{prefix}_mean")
        row_out[f"{pol}_revenue_std"] = float("nan")
        row_out[f"{pol}_revenue_ci95"] = float("nan")

        row_out[f"{pol}_lost_mean"] = _gf(f"lost_units_{prefix}_mean")
        row_out[f"{pol}_waste_mean"] = _gf(f"waste_units_{prefix}_mean")

    _attach_metrics("BS", "BS")
    _attach_metrics("PIL", "PIL")
    _attach_metrics("PPO", "PPO")

    # PPO checkpoint path if present
    row_out["PPO_model_path"] = str(r.get("ppo_model_path", "")) if ("ppo_model_path" in r) else ""

    # Derived comparisons (profit and cost deltas)
    try:
        row_out["PPO_minus_BS_profit"] = float(row_out.get("PPO_profit_mean", np.nan) - row_out.get("BS_profit_mean", np.nan))
        row_out["PPO_minus_PIL_profit"] = float(row_out.get("PPO_profit_mean", np.nan) - row_out.get("PIL_profit_mean", np.nan))
        row_out["PIL_minus_BS_profit"] = float(row_out.get("PIL_profit_mean", np.nan) - row_out.get("BS_profit_mean", np.nan))

        row_out["PPO_minus_BS_cost"] = float(row_out.get("PPO_cost_mean", np.nan) - row_out.get("BS_cost_mean", np.nan))
        row_out["PPO_minus_PIL_cost"] = float(row_out.get("PPO_cost_mean", np.nan) - row_out.get("PIL_cost_mean", np.nan))
        row_out["PIL_minus_BS_cost"] = float(row_out.get("PIL_cost_mean", np.nan) - row_out.get("BS_cost_mean", np.nan))
    except Exception:
        pass

    return row_out


def _expand_template_pairs_to_all_demand_types(
    policy_df: pd.DataFrame,
    *,
    template_pairs: List[Tuple[int, float]],
    demand_types: List[str],
) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
    """Expand a small set of template (config_id, cv) pairs to all demand types.

    Use-case: run_all_spyder_with_bestparm.py may pass only 1 or 2 config IDs.
    If you want the expanded experiment set (all demand types, same Alpha/beta/m/L),
    this function builds the corresponding config IDs for each demand type and each CV.

    Returns:
      - expanded_pairs: list of (config_id, cv) across all demand_types and CVs
      - params_used: dict with Alpha, beta_excel, m, L inferred from templates
    """
    if policy_df is None or policy_df.empty:
        raise ValueError("policy_df is empty; cannot expand template pairs")

    tol = 1e-9

    # Infer baseline (Alpha, beta_excel, m, L) from template configuration(s)
    params = None
    for cid, _cv in template_pairs:
        sub = policy_df.loc[policy_df["configuration"].astype("Int64") == int(cid)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        cur = (float(r["Alpha"]), float(r["beta_excel"]), int(r["m"]), int(r["L"]))
        if params is None:
            params = cur
        else:
            if (abs(cur[0] - params[0]) > tol) or (abs(cur[1] - params[1]) > tol) or (cur[2] != params[2]) or (cur[3] != params[3]):
                raise ValueError(
                    "Template config IDs do not share the same (Alpha, beta_excel, m, L). "
                    f"Got {params} vs {cur}. Disable expansion with --no-expand-demand-types or pass consistent templates."
                )

    if params is None:
        raise ValueError("Could not infer baseline parameters from template_pairs (templates not found in policy_df).")

    Alpha, beta_excel, m, L = params

    cvs = sorted(set([float(cv) for (_cid, cv) in template_pairs]))

    expanded: List[Tuple[int, float]] = []
    for cv in cvs:
        triples = _select_baseline_triples_fixed_params(
            policy_df,
            cv_list=[float(cv)],
            demand_types=[str(dt).strip().lower() for dt in demand_types],
            Alpha=float(Alpha),
            beta_excel=float(beta_excel),
            m=int(m),
            L=int(L),
        )
        expanded.extend([(cid2, float(cv2)) for (cid2, cv2, _dt) in triples])

    # Deduplicate while preserving order
    seen = set()
    expanded_unique: List[Tuple[int, float]] = []
    for cid2, cv2 in expanded:
        key = (int(cid2), float(cv2))
        if key in seen:
            continue
        seen.add(key)
        expanded_unique.append((int(cid2), float(cv2)))

    return expanded_unique, {"Alpha": float(Alpha), "beta_excel": float(beta_excel), "m": int(m), "L": int(L)}

# ======================================================================================
# Tuning and evaluation utilities
# ======================================================================================
def _tune_levels_for_cfg(
    env_cfg: Dict[str, Any],
    quick: bool,
    seed: int,
    *,
    seed_bs: Optional[int] = None,
    seed_pil: Optional[int] = None,
) -> Tuple[int, int, float]:
    """Tune S_BS and S_PIL (fast grid search). Returns (S_BS, S_PIL, tune_seconds)."""
    t0 = time.time()

    max_order = int(env_cfg.get("max_order", 30))
    S_grid = list(range(0, max_order + 1))

    horizon = bench.TUNE_HORIZON_QUICK if quick else bench.TUNE_HORIZON_FULL
    burn_in = bench.TUNE_BURN_IN_QUICK if quick else bench.TUNE_BURN_IN_FULL
    reps = bench.TUNE_REPS_QUICK if quick else bench.TUNE_REPS_FULL

    seed_bs_eff = int(seed if seed_bs is None else seed_bs)
    seed_pil_eff = int(seed if seed_pil is None else seed_pil)

    S_bs = bench.tune_level(
        env_cfg,
        "BS",
        S_grid,
        horizon,
        burn_in,
        reps,
        seed0=seed_bs_eff,
        pil_use_correction=False,
        pil_n_mc=bench.PIL_N_MC_TUNE,
    )
    S_pil = bench.tune_level(
        env_cfg,
        "PIL",
        S_grid,
        horizon,
        burn_in,
        reps,
        seed0=seed_pil_eff,
        pil_use_correction=bench.PIL_USE_CORRECTION,
        pil_n_mc=bench.PIL_N_MC_TUNE,
    )

    return int(S_bs), int(S_pil), float(time.time() - t0)


def _evaluate_bs_pil(
    env_cfg: Dict[str, Any],
    demand_streams: List[np.ndarray],
    burn_in: int,
    policy_kind: str,
    *,
    S_bs: int,
    S_pil: int,
    pil_use_correction: bool,
    pil_n_mc: int,
    pil_seed_base: int,
    pil_use_observed_state: bool,
    max_workers: int,
) -> List[bench.ReplicationStats]:
    """Evaluate BS/PIL on fixed demand streams.

    Multiprocessing is used only when it is likely beneficial and stable.
    """
    assert policy_kind in {"BS", "PIL"}

    if max_workers <= 1 or len(demand_streams) <= 6:
        out: List[bench.ReplicationStats] = []
        for r, stream in enumerate(demand_streams):
            out.append(
                bench._run_one_replication(
                    env_cfg,
                    stream,
                    burn_in,
                    policy_kind,
                    S_bs=int(S_bs) if policy_kind == "BS" else None,
                    S_pil=int(S_pil) if policy_kind == "PIL" else None,
                    pil_use_correction=bool(pil_use_correction),
                    pil_n_mc=int(pil_n_mc),
                    pil_seed=int(pil_seed_base + r),
                    pil_use_observed_state=bool(pil_use_observed_state),
                    ppo_model_path=None,
                )
            )
        return out

    tasks = []
    for r, stream in enumerate(demand_streams):
        tasks.append(
            (
                env_cfg,
                stream,
                burn_in,
                policy_kind,
                int(S_bs),
                int(S_pil),
                bool(pil_use_correction),
                int(pil_n_mc),
                int(pil_seed_base + r),
                bool(pil_use_observed_state),
            )
        )

    try:  # pragma: no cover
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        stats: List[bench.ReplicationStats] = []
        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks)), mp_context=ctx) as ex:
            for st in ex.map(bench._worker_run_one_replication, tasks):
                stats.append(st)
        return stats
    except Exception as e:  # pragma: no cover
        print(f"[sensitivity] WARNING: replication multiprocessing failed ({type(e).__name__}: {e}); using sequential.")
        out: List[bench.ReplicationStats] = []
        for r, stream in enumerate(demand_streams):
            out.append(
                bench._run_one_replication(
                    env_cfg,
                    stream,
                    burn_in,
                    policy_kind,
                    S_bs=int(S_bs) if policy_kind == "BS" else None,
                    S_pil=int(S_pil) if policy_kind == "PIL" else None,
                    pil_use_correction=bool(pil_use_correction),
                    pil_n_mc=int(pil_n_mc),
                    pil_seed=int(pil_seed_base + r),
                    pil_use_observed_state=bool(pil_use_observed_state),
                    ppo_model_path=None,
                )
            )
        return out


def _evaluate_ppo(
    env_cfg: Dict[str, Any],
    demand_streams: List[np.ndarray],
    ppo_model_path: Path,
    burn_in: int,
) -> List[bench.ReplicationStats]:
    """Sequential PPO evaluation (stable across platforms)."""
    ppo_model = None
    ppo_vec_normalize = None
    if PPO is not None:
        try:  # pragma: no cover
            ppo_model = PPO.load(str(ppo_model_path))
            vec_path = ppo_model_path.parent / "vecnormalize.pkl"
            if vec_path.exists() and (DummyVecEnv is not None) and (VecNormalize is not None):
                venv_dummy = DummyVecEnv([lambda: InventoryEnvGYConfig(dict(env_cfg))])
                ppo_vec_normalize = VecNormalize.load(str(vec_path), venv_dummy)
                ppo_vec_normalize.training = False
                ppo_vec_normalize.norm_reward = False
        except Exception:
            ppo_model = None
            ppo_vec_normalize = None

    stats: List[bench.ReplicationStats] = []
    for stream in demand_streams:
        stats.append(
            bench._run_one_replication(
                env_cfg,
                stream,
                burn_in,
                "PPO",
                S_bs=None,
                S_pil=None,
                pil_use_correction=False,
                pil_n_mc=0,
                pil_seed=0,
                pil_use_observed_state=False,
                ppo_model_path=ppo_model_path,
                ppo_model=ppo_model,
                ppo_vec_normalize=ppo_vec_normalize,
            )
        )
    return stats


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


def _compute_train_like_timesteps(*, quick: bool, cv: float) -> Optional[int]:
    """Match train_ppo_GY.py budget logic when available."""
    if ppo_train_ref is None:
        return None
    fn = getattr(ppo_train_ref, "_compute_total_timesteps", None)
    if not callable(fn):
        return None
    try:
        out = int(fn(quick=bool(quick), cv=float(cv)))
        return out if out > 0 else None
    except Exception:
        return None


def _train_ppo_for_env_cfg(
    *,
    env_cfg: Dict[str, Any],
    out_dir: Path,
    hyper: Dict[str, Any],
    total_timesteps: int,
    train_seed: int,
    eval_seed_base: int,
    quick: bool,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train one PPO candidate for this exact scenario env config.

    Selection logic mirrors train_ppo_GY.py:
    - deterministic eval callback saves best_model.zip + vecnormalize_best.pkl
    - final model is also scored
    - winner is max(best_score, final_score)
    - benchmark-facing artifacts are best_model.zip + vecnormalize.pkl
    """
    if PPO is None:
        raise RuntimeError("stable-baselines3 is not available in this Python environment.")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_env_cfg = dict(env_cfg)
    # Keep PPO reward scaling aligned with train_ppo_GY.py.
    if ppo_train_ref is not None:
        try:
            rs = getattr(ppo_train_ref, "REWARD_SCALE", None)
            if rs is not None:
                train_env_cfg["reward_scale"] = float(rs)
        except Exception:
            pass
    train_env_cfg.setdefault("warm_up_fraction", 0.0)

    # Build training env for this exact scenario.
    def _make_env():
        c = dict(train_env_cfg)
        c["seed"] = int(train_seed)
        return InventoryEnvGYConfig(c)

    if set_random_seed is not None:
        set_random_seed(int(train_seed))
    np.random.seed(int(train_seed))

    venv = DummyVecEnv([_make_env])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    # Schedules.
    lr_start = float(hyper.get("lr_start", 3e-4))
    lr_end = float(hyper.get("lr_end", lr_start))
    clip_start = float(hyper.get("clip_start", 0.2))
    clip_end = float(hyper.get("clip_end", clip_start))

    lr_schedule = _make_lr_schedule(lr_start, lr_end) if lr_start != lr_end else float(lr_start)
    clip_schedule = _make_clip_schedule(clip_start, clip_end) if clip_start != clip_end else float(clip_start)

    # Network architecture (SB3 expects a list or dict with pi/vf)
    net_arch = hyper.get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
    policy_kwargs = {"net_arch": net_arch}

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=lr_schedule,
        n_steps=int(hyper.get("n_steps", 1024)),
        batch_size=int(hyper.get("batch_size", 64)),
        n_epochs=int(hyper.get("n_epochs", 10)),
        gamma=float(hyper.get("gamma", 0.99)),
        gae_lambda=float(hyper.get("gae_lambda", 0.95)),
        clip_range=clip_schedule,
        ent_coef=float(hyper.get("ent_coef", 0.0)),
        vf_coef=float(hyper.get("vf_coef", 0.5)),
        max_grad_norm=float(hyper.get("max_grad_norm", 0.5)),
        target_kl=float(hyper["target_kl"]) if hyper.get("target_kl", None) is not None else None,
        clip_range_vf=None,
        policy_kwargs=policy_kwargs,
        normalize_advantage=True,
        verbose=0,
        seed=int(train_seed),
        device=str(device),
    )

    callbacks: List[Any] = []
    eval_freq = int(max(1, int(total_timesteps // 20)))
    if (ppo_train_ref is not None) and (CallbackList is not None):
        min_eval = int(getattr(ppo_train_ref, "EVAL_FREQ_MIN_QUICK", 1000) if bool(quick) else getattr(ppo_train_ref, "EVAL_FREQ_MIN_FULL", 2000))
        eval_div = int(max(1, getattr(ppo_train_ref, "EVAL_FREQ_DIVISOR", 20)))
        eval_freq = int(max(min_eval, int(total_timesteps // eval_div)))

        if bool(quick):
            min_steps_es = int(getattr(ppo_train_ref, "MIN_EVALS_BEFORE_STOP", 20)) * int(eval_freq)
            patience_steps_es = int(getattr(ppo_train_ref, "NO_IMPROVEMENT_EVALS", 15)) * int(eval_freq)
        else:
            min_steps_es = int(
                max(
                    int(getattr(ppo_train_ref, "EARLY_STOP_FULL_MIN_STEPS", 150_000)),
                    float(getattr(ppo_train_ref, "EARLY_STOP_FULL_MIN_FRACTION", 0.60)) * float(total_timesteps),
                )
            )
            patience_steps_es = int(
                max(
                    int(getattr(ppo_train_ref, "EARLY_STOP_FULL_PATIENCE_STEPS", 40_000)),
                    float(getattr(ppo_train_ref, "EARLY_STOP_FULL_PATIENCE_FRACTION", 0.10)) * float(total_timesteps),
                )
            )

        eval_cfg = dict(train_env_cfg)

        cb_profit = ppo_train_ref.DeterministicProfitEvalCallback(
            eval_cfg=eval_cfg,
            eval_freq=int(eval_freq),
            n_eval_episodes=int(getattr(ppo_train_ref, "N_EVAL_EPISODES_CB", 5)),
            burn_in_fraction=float(getattr(ppo_train_ref, "FINAL_EVAL_BURN_IN_FRACTION", 0.20)),
            seed0=int(eval_seed_base),
            out_dir=out_dir,
            reward_scale=float(train_env_cfg.get("reward_scale", getattr(ppo_train_ref, "REWARD_SCALE", 100.0))),
            min_steps=int(min_steps_es),
            patience_steps=int(patience_steps_es),
            vecnorm_train=venv,
            verbose=0,
        )
        cb_step = ppo_train_ref.TrainStepProfitCallback(
            out_dir=out_dir,
            total_timesteps_budget=int(total_timesteps),
            reward_scale=float(train_env_cfg.get("reward_scale", getattr(ppo_train_ref, "REWARD_SCALE", 100.0))),
            log_every_n_steps=int(getattr(ppo_train_ref, "TRAIN_LOG_EVERY_N_STEPS", 1)),
            verbose=0,
        )
        callbacks = [cb_profit, cb_step]

    t0 = time.time()
    if callbacks and (CallbackList is not None):
        model.learn(total_timesteps=int(total_timesteps), callback=CallbackList(callbacks))
    else:
        model.learn(total_timesteps=int(total_timesteps))
    train_seconds = float(time.time() - t0)

    # Save final model + final stats snapshot.
    final_model_path = out_dir / "ppo_model.zip"
    model.save(str(final_model_path))
    final_stats_path = out_dir / "vecnormalize_final.pkl"
    try:
        venv.save(str(final_stats_path))
    except Exception:
        pass

    # Winner selection (best callback checkpoint vs final checkpoint), like train_ppo_GY.py.
    best_model_path = out_dir / "best_model.zip"
    best_stats_path = out_dir / "vecnormalize_best.pkl"
    out_vecnorm_path = out_dir / "vecnormalize.pkl"

    winner = "final"
    score_best = float("-inf")
    score_final = float("-inf")

    if (ppo_train_ref is not None) and (DummyVecEnv is not None) and (VecNormalize is not None):
        eval_cfg = dict(train_env_cfg)

        stats_best = None
        if best_stats_path.exists():
            try:
                dummy = DummyVecEnv([lambda: InventoryEnvGYConfig(dict(eval_cfg, seed=int(eval_seed_base) + 777))])
                stats_best = VecNormalize.load(str(best_stats_path), dummy)
                stats_best.training = False
                stats_best.norm_reward = False
            except Exception:
                stats_best = None

        stats_final = None
        if final_stats_path.exists():
            try:
                dummy2 = DummyVecEnv([lambda: InventoryEnvGYConfig(dict(eval_cfg, seed=int(eval_seed_base) + 777))])
                stats_final = VecNormalize.load(str(final_stats_path), dummy2)
                stats_final.training = False
                stats_final.norm_reward = False
            except Exception:
                stats_final = None

        if best_model_path.exists() and (stats_best is not None):
            try:
                m_best = PPO.load(str(best_model_path), device="cpu")
                score_best = float(
                    ppo_train_ref._final_deterministic_eval(
                        model=m_best,
                        eval_cfg=eval_cfg,
                        seed0=int(eval_seed_base),
                        n_episodes=int(getattr(ppo_train_ref, "FINAL_EVAL_N_EPISODES", 40)),
                        burn_in_fraction=float(getattr(ppo_train_ref, "FINAL_EVAL_BURN_IN_FRACTION", 0.20)),
                        vecnorm=stats_best,
                    )
                )
            except Exception:
                score_best = float("-inf")

        try:
            score_final = float(
                ppo_train_ref._final_deterministic_eval(
                    model=model,
                    eval_cfg=eval_cfg,
                    seed0=int(eval_seed_base),
                    n_episodes=int(getattr(ppo_train_ref, "FINAL_EVAL_N_EPISODES", 40)),
                    burn_in_fraction=float(getattr(ppo_train_ref, "FINAL_EVAL_BURN_IN_FRACTION", 0.20)),
                    vecnorm=stats_final,
                )
            )
        except Exception:
            score_final = float("-inf")

        if score_best > score_final:
            winner = "best"

    # Ensure benchmark-facing files always exist and are matched.
    if winner == "best" and best_model_path.exists() and best_stats_path.exists():
        try:
            shutil.copy2(str(best_stats_path), str(out_vecnorm_path))
        except Exception:
            winner = "final"
    if winner == "final":
        try:
            shutil.copy2(str(final_model_path), str(best_model_path))
        except Exception:
            pass
        try:
            shutil.copy2(str(final_stats_path), str(out_vecnorm_path))
        except Exception:
            pass

    if winner == "best":
        winner_score = float(score_best)
    else:
        winner_score = float(score_final)

    training_meta: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": float(train_seconds),
        "total_timesteps_budget": int(total_timesteps),
        "final_scores": {
            "best_score": float(score_best),
            "final_score": float(score_final),
            "winner": str(winner),
            "winner_score": float(winner_score),
        },
        "hyper": dict(hyper),
        "paths": {
            "best_model": str(best_model_path),
            "vecnormalize": str(out_vecnorm_path),
            "final_model": str(final_model_path),
            "vecnormalize_final": str(final_stats_path),
            "vecnormalize_best": str(best_stats_path),
        },
    }

    try:
        with open(out_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2)
    except Exception:
        pass

    try:
        venv.close()
    except Exception:
        pass

    return {
        "ppo_total_timesteps": int(total_timesteps),
        "ppo_train_seconds": float(train_seconds),
        "ppo_hyper_name": str(hyper.get("name", "")),
        "ppo_winner": str(winner),
        "ppo_winner_score": float(winner_score),
        "ppo_eval_seed_base": int(eval_seed_base),
        "ppo_eval_freq": int(eval_freq),
        "ppo_lr_start": float(lr_start),
        "ppo_lr_end": float(lr_end),
        "ppo_clip_start": float(clip_start),
        "ppo_clip_end": float(clip_end),
        "ppo_n_steps": int(hyper.get("n_steps", 1024)),
        "ppo_batch_size": int(hyper.get("batch_size", 64)),
        "ppo_n_epochs": int(hyper.get("n_epochs", 10)),
        "ppo_gamma": float(hyper.get("gamma", 0.99)),
        "ppo_gae_lambda": float(hyper.get("gae_lambda", 0.95)),
        "ppo_ent_coef": float(hyper.get("ent_coef", 0.0)),
        "ppo_vf_coef": float(hyper.get("vf_coef", 0.5)),
        "ppo_max_grad_norm": float(hyper.get("max_grad_norm", 0.5)),
        "ppo_target_kl": float(hyper.get("target_kl")) if hyper.get("target_kl", None) is not None else float("nan"),
        "ppo_net_arch": json.dumps(net_arch),
        "ppo_model_path": str(best_model_path if best_model_path.exists() else final_model_path),
        "ppo_training_meta_path": str(out_dir / "training_meta.json"),
    }


def _summarize(stats: List[bench.ReplicationStats]) -> Dict[str, float]:
    profits = np.asarray([s.profit_mean_per_period for s in stats], dtype=float)
    costs = np.asarray([s.cost_mean_per_period for s in stats], dtype=float)
    revenues = np.asarray([s.revenue_mean_per_period for s in stats], dtype=float)
    lost = np.asarray([s.lost_mean_per_period for s in stats], dtype=float)
    waste = np.asarray([s.waste_mean_per_period for s in stats], dtype=float)

    def _mean_std_ci(x: np.ndarray) -> tuple[float, float, float]:
        if x.size == 0:
            return float("nan"), float("nan"), float("nan")
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        ci = 1.96 * s / np.sqrt(float(x.size)) if x.size > 1 else 0.0
        return m, s, ci

    p_mean, p_std, p_ci95 = _mean_std_ci(profits)
    c_mean, c_std, c_ci95 = _mean_std_ci(costs)
    r_mean, r_std, r_ci95 = _mean_std_ci(revenues)

    return {
        "profit_mean": float(p_mean),
        "profit_std": float(p_std),
        "profit_ci95": float(p_ci95),
        "cost_mean": float(c_mean),
        "cost_std": float(c_std),
        "cost_ci95": float(c_ci95),
        "revenue_mean": float(r_mean),
        "revenue_std": float(r_std),
        "revenue_ci95": float(r_ci95),
        "lost_mean": float(np.mean(lost)) if lost.size else float("nan"),
        "waste_mean": float(np.mean(waste)) if waste.size else float("nan"),
    }


# ======================================================================================
# Scenario generation
# ======================================================================================
def _scenario_list(
    *,
    mode: str,
    base_cfg: Dict[str, Any],
    sensor_eps_list: List[float],
    fixed_order_cost_list: List[float],
    tti_unit_cost_list: List[float],
    m_list: List[int],
    L_list: List[int],
) -> List[Dict[str, Any]]:
    """Return perturbation scenarios only (baseline is added elsewhere)."""
    mode = str(mode).lower().strip()
    m0, L0 = int(base_cfg["m"]), int(base_cfg["L"])
    base_cost_ov = {"sensor_error_eps": 0.0, "fixed_order_cost": 0.0, "tti_unit_cost": 0.0}

    def _sensor_rows() -> List[Dict[str, Any]]:
        out = []
        for eps in sensor_eps_list:
            eps_f = float(eps)
            if eps_f <= 0.0:
                continue
            out.append({"group": "sensor", "name": f"eps_{eps_f:g}", "overrides": dict(base_cost_ov, sensor_error_eps=eps_f)})
        return out

    def _tti_rows() -> List[Dict[str, Any]]:
        out = []
        for tti in tti_unit_cost_list:
            tti_f = float(tti)
            if tti_f <= 0.0:
                continue
            out.append(
                {
                    "group": "tti_cost",
                    "name": f"tti_{tti_f:g}",
                    "overrides": dict(base_cost_ov, tti_unit_cost=tti_f),
                }
            )
        return out

    def _order_rows() -> List[Dict[str, Any]]:
        out = []
        for k in fixed_order_cost_list:
            k_f = float(k)
            if k_f <= 0.0:
                continue
            out.append(
                {
                    "group": "order_cost",
                    "name": f"K_{k_f:g}",
                    "overrides": dict(base_cost_ov, fixed_order_cost=k_f),
                }
            )
        return out

    def _ml_rows() -> List[Dict[str, Any]]:
        out = []
        for m in m_list:
            for L in L_list:
                if int(m) == m0 and int(L) == L0:
                    continue
                out.append(
                    {
                        "group": "ml",
                        "name": f"m{int(m)}_L{int(L)}",
                        "overrides": dict(base_cost_ov, m=int(m), L=int(L)),
                    }
                )
        return out

    if mode == "sensor":
        return _sensor_rows()
    if mode == "tti_cost":
        return _tti_rows()
    if mode == "order_cost":
        return _order_rows()
    if mode == "costs":
        # Backward-compatible alias.
        return _tti_rows() + _order_rows()
    if mode == "ml":
        return _ml_rows()

    if mode == "all":
        out: List[Dict[str, Any]] = []
        out += _sensor_rows()
        out += _tti_rows()
        out += _order_rows()
        out += _ml_rows()
        return out

    raise ValueError(f"Unknown mode: {mode}")


# ======================================================================================
# Plot helpers (lightweight)
# ======================================================================================
def _maybe_plot_sensor(df: pd.DataFrame, plots_dir: Path, *, tag: str, demand_type: str, cv: float) -> None:
    try:  # pragma: no cover
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        sub = df.loc[
            (df["demand_type"].astype(str).str.lower() == str(demand_type).lower())
            & (df["cv_scenario"].astype(float) == float(cv))
            & (df["scenario_group"].isin(["baseline", "sensor"]))
        ].copy()
        if sub.empty:
            return

        # Ensure baseline has eps=0 for plotting
        sub["sensor_error_eps"] = pd.to_numeric(sub["sensor_error_eps"], errors="coerce").fillna(0.0)
        sub = sub.sort_values("sensor_error_eps")
        x = sub["sensor_error_eps"].to_numpy(dtype=float)

        plt.figure(figsize=(8, 4))
        plt.title(f"Sensor error sensitivity | demand={demand_type} | cv={cv:g}")
        plt.xlabel("Sensor error eps")
        plt.ylabel("Mean cost per period")

        for pol in ["BS", "PIL", "PPO"]:
            ycol = f"{pol}_cost_mean"
            ecol = f"{pol}_cost_ci95"
            if ycol not in sub.columns:
                continue
            y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)
            e = pd.to_numeric(sub.get(ecol, np.nan), errors="coerce").to_numpy(dtype=float)
            if np.isfinite(e).any():
                plt.errorbar(x, y, yerr=e, marker="o", linewidth=1.2, label=pol)
            else:
                plt.plot(x, y, marker="o", linewidth=1.2, label=pol)

        plt.legend(loc="best")
        plt.tight_layout()
        out = plots_dir / f"sensitivity_{tag}_{demand_type}_cv{cv:g}_sensor.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
    except Exception:
        return


def _maybe_plot_cost_slices(df: pd.DataFrame, plots_dir: Path, *, tag: str, demand_type: str, cv: float) -> None:
    try:  # pragma: no cover
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        base = df.loc[
            (df["demand_type"].astype(str).str.lower() == str(demand_type).lower())
            & (df["cv_scenario"].astype(float) == float(cv))
        ].copy()
        if base.empty:
            return

        # Slice 1: vary fixed_order_cost with tti_unit_cost=0 (ordering setup cost only)
        sub1 = base.loc[(base["scenario_group"].isin(["baseline", "order_cost", "costs"]))].copy()
        sub1["fixed_order_cost"] = pd.to_numeric(sub1["fixed_order_cost"], errors="coerce").fillna(0.0)
        sub1["tti_unit_cost"] = pd.to_numeric(sub1["tti_unit_cost"], errors="coerce").fillna(0.0)
        sub1 = sub1.loc[sub1["tti_unit_cost"].abs() <= 1e-12].copy()
        if not sub1.empty:
            sub1 = sub1.sort_values("fixed_order_cost")
            x = sub1["fixed_order_cost"].to_numpy(dtype=float)

            plt.figure(figsize=(8, 4))
            plt.title(f"Ordering setup cost sensitivity | demand={demand_type} | cv={cv:g} | tti=0")
            plt.xlabel("fixed_order_cost")
            plt.ylabel("Mean cost per period")
            for pol in ["BS", "PIL", "PPO"]:
                ycol = f"{pol}_cost_mean"
                ecol = f"{pol}_cost_ci95"
                if ycol not in sub1.columns:
                    continue
                y = pd.to_numeric(sub1[ycol], errors="coerce").to_numpy(dtype=float)
                e = pd.to_numeric(sub1.get(ecol, np.nan), errors="coerce").to_numpy(dtype=float)
                if np.isfinite(e).any():
                    plt.errorbar(x, y, yerr=e, marker="o", linewidth=1.2, label=pol)
                else:
                    plt.plot(x, y, marker="o", linewidth=1.2, label=pol)
            plt.legend(loc="best")
            plt.tight_layout()
            out = plots_dir / f"sensitivity_{tag}_{demand_type}_cv{cv:g}_Kslice.png"
            plt.savefig(str(out), dpi=150)
            plt.close()

        # Slice 2: vary tti_unit_cost with fixed_order_cost=0 (unit TTI cost only)
        sub2 = base.loc[(base["scenario_group"].isin(["baseline", "tti_cost", "costs"]))].copy()
        sub2["fixed_order_cost"] = pd.to_numeric(sub2["fixed_order_cost"], errors="coerce").fillna(0.0)
        sub2["tti_unit_cost"] = pd.to_numeric(sub2["tti_unit_cost"], errors="coerce").fillna(0.0)
        sub2 = sub2.loc[sub2["fixed_order_cost"].abs() <= 1e-12].copy()
        if not sub2.empty:
            sub2 = sub2.sort_values("tti_unit_cost")
            x = sub2["tti_unit_cost"].to_numpy(dtype=float)

            plt.figure(figsize=(8, 4))
            plt.title(f"Unit TTI cost sensitivity | demand={demand_type} | cv={cv:g} | K=0")
            plt.xlabel("tti_unit_cost")
            plt.ylabel("Mean cost per period")
            for pol in ["BS", "PIL", "PPO"]:
                ycol = f"{pol}_cost_mean"
                ecol = f"{pol}_cost_ci95"
                if ycol not in sub2.columns:
                    continue
                y = pd.to_numeric(sub2[ycol], errors="coerce").to_numpy(dtype=float)
                e = pd.to_numeric(sub2.get(ecol, np.nan), errors="coerce").to_numpy(dtype=float)
                if np.isfinite(e).any():
                    plt.errorbar(x, y, yerr=e, marker="o", linewidth=1.2, label=pol)
                else:
                    plt.plot(x, y, marker="o", linewidth=1.2, label=pol)
            plt.legend(loc="best")
            plt.tight_layout()
            out = plots_dir / f"sensitivity_{tag}_{demand_type}_cv{cv:g}_ttislice.png"
            plt.savefig(str(out), dpi=150)
            plt.close()
    except Exception:
        return


def _maybe_plot_ml_heatmap(df: pd.DataFrame, plots_dir: Path, *, tag: str, demand_type: str, cv: float) -> None:
    try:  # pragma: no cover
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        sub = df.loc[
            (df["demand_type"].astype(str).str.lower() == str(demand_type).lower())
            & (df["cv_scenario"].astype(float) == float(cv))
            & (df["scenario_group"].isin(["baseline", "ml"]))
        ].copy()
        if sub.empty:
            return

        # Keep only (m,L) rows (baseline included)
        sub["m"] = pd.to_numeric(sub["m"], errors="coerce")
        sub["L"] = pd.to_numeric(sub["L"], errors="coerce")
        sub = sub.dropna(subset=["m", "L"]).copy()
        if sub.empty:
            return

        # Heatmap for each policy
        for pol in ["BS", "PIL", "PPO"]:
            val_col = f"{pol}_cost_mean"
            if val_col not in sub.columns:
                continue
            piv = sub.pivot_table(index="m", columns="L", values=val_col, aggfunc="mean")
            if piv.empty:
                continue

            plt.figure(figsize=(7, 5))
            plt.title(f"(m,L) sensitivity heatmap | {pol} | demand={demand_type} | cv={cv:g}")
            plt.xlabel("L")
            plt.ylabel("m")
            arr = piv.to_numpy(dtype=float)
            plt.imshow(arr, aspect="auto", origin="lower")
            plt.colorbar()
            plt.xticks(ticks=np.arange(piv.shape[1]), labels=[str(int(x)) for x in piv.columns.tolist()])
            plt.yticks(ticks=np.arange(piv.shape[0]), labels=[str(int(x)) for x in piv.index.tolist()])
            plt.tight_layout()
            out = plots_dir / f"sensitivity_{tag}_{demand_type}_cv{cv:g}_ml_{pol}.png"
            plt.savefig(str(out), dpi=150)
            plt.close()
    except Exception:
        return


# ======================================================================================
# Core scenario evaluation
# ======================================================================================
def _evaluate_one_scenario(
    *,
    cid: int,
    cv: float,
    cv_tag: int,
    scen_idx: int,
    scen_group: str,
    scen_name: str,
    overrides: Dict[str, Any],
    base_cfg: Dict[str, Any],
    base_cfg0: Dict[str, Any],
    logs_dir: Path,
    quick: bool,
    eval_horizon: int,
    eval_reps: int,
    burn_in: int,
    S_bs_base: int,
    S_pil_base: int,
    tune_seconds_base: float,
    sensor_affects_pil: bool,
    fixed_baseline_levels: bool,
    freeze_levels_on_sensor: bool,
    retrain_ppo: bool,
    skip_ppo: bool,
    rep_workers: int,
    demand_streams_base: Optional[List[np.ndarray]],
    ppo_timesteps: int,
    ppo_device: str,
    ppo_hyper_source: str,
    ppo_hyper_fallback: Optional[Dict[str, Any]],
    ppo_top_candidates: Optional[List[Dict[str, Any]]],
    eval_seed0: int,
) -> Dict[str, Any]:
    group = str(scen_group)
    name = str(scen_name)

    env_cfg = dict(base_cfg)
    env_cfg.update(dict(overrides))

    # Demand streams are independent of (m,L) and costs, so reuse whenever mean/cv unchanged.
    if demand_streams_base is not None and ("mean_demand" not in overrides) and ("coef_of_var" not in overrides):
        ds = demand_streams_base
    else:
        ds = [
            bench.generate_demand_stream(
                mean_demand=float(env_cfg["mean_demand"]),
                coef_of_var=float(env_cfg["coef_of_var"]),
                demand_type=str(env_cfg.get("demand_type", "beta_binomial")),
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
                horizon=int(eval_horizon),
                seed=int(int(eval_seed0) + 1000 * int(cid) + int(_r)),
            )
            for _r in range(int(eval_reps))
        ]

    # BS/PIL level selection
    if bool(fixed_baseline_levels):
        S_bs, S_pil, tune_seconds = int(S_bs_base), int(S_pil_base), 0.0
    elif group == "sensor" and bool(freeze_levels_on_sensor):
        S_bs, S_pil, tune_seconds = int(S_bs_base), int(S_pil_base), 0.0
    elif group == "baseline":
        S_bs, S_pil, tune_seconds = int(S_bs_base), int(S_pil_base), float(tune_seconds_base)
    else:
        S_bs, S_pil, tune_seconds = _tune_levels_for_cfg(
            env_cfg,
            quick=bool(quick),
            seed=int(int(cid) * 1000 + int(cv_tag) + 7 + 13 * (int(scen_idx) + 1)),
        )

    # Sensor noise affects PIL only when requested, and only for sensor scenarios.
    pil_use_obs = bool(sensor_affects_pil) and (group == "sensor") and (float(env_cfg.get("sensor_error_eps", 0.0)) > 0.0)

    bs_stats = _evaluate_bs_pil(
        env_cfg,
        ds,
        int(burn_in),
        "BS",
        S_bs=int(S_bs),
        S_pil=int(S_pil),
        pil_use_correction=bench.PIL_USE_CORRECTION,
        pil_n_mc=bench.PIL_N_MC_EVAL,
        pil_seed_base=int(int(eval_seed0) + 77),
        pil_use_observed_state=False,
        max_workers=int(rep_workers),
    )
    pil_stats = _evaluate_bs_pil(
        env_cfg,
        ds,
        int(burn_in),
        "PIL",
        S_bs=int(S_bs),
        S_pil=int(S_pil),
        pil_use_correction=bench.PIL_USE_CORRECTION,
        pil_n_mc=bench.PIL_N_MC_EVAL,
        pil_seed_base=int(int(eval_seed0) + 77),
        pil_use_observed_state=bool(pil_use_obs),
        max_workers=int(rep_workers),
    )

    # PPO: baseline or retrain
    ppo_stats: List[bench.ReplicationStats] = []
    ppo_meta: Dict[str, Any] = {}
    ppo_model_path: Optional[Path] = None

    if (not bool(skip_ppo)) and (PPO is not None):
        baseline_meta = bench.load_training_meta(logs_dir, int(cid))
        baseline_hyper = dict(baseline_meta.get("hyper", {}) or {})
        top_hyper_list = [dict(c) for c in (ppo_top_candidates or []) if isinstance(c, dict)]

        # Hyperparameters for sensitivity retraining:
        # - training_meta: per-config winner (preferred if available)
        # - top_candidates: sweep all candidates and keep best per scenario
        # - auto: training_meta if present else sweep all top_candidates
        hyper_src = str(ppo_hyper_source).strip().lower()
        fallback_hyper = dict(ppo_hyper_fallback or {})
        if hyper_src == "top_candidates":
            hyper_for_train_list = [dict(h) for h in top_hyper_list]
        elif hyper_src == "training_meta":
            hyper_for_train_list = [dict(baseline_hyper)] if baseline_hyper else []
        else:
            if baseline_hyper:
                hyper_for_train_list = [dict(baseline_hyper)]
            elif top_hyper_list:
                hyper_for_train_list = [dict(h) for h in top_hyper_list]
            else:
                hyper_for_train_list = []

        if (not hyper_for_train_list) and fallback_hyper:
            hyper_for_train_list = [dict(fallback_hyper)]

        baseline_model = bench.resolve_ppo_checkpoint(base_dir=Path(__file__).resolve().parent, logs_dir=logs_dir, config_id=int(cid), meta=baseline_meta)

        dim_changed = (int(env_cfg["m"]) != int(base_cfg0["m"])) or (int(env_cfg["L"]) != int(base_cfg0["L"]))
        must_retrain_for_dim = (group == "ml") and dim_changed

        retrain_this = (group != "baseline") and bool(retrain_ppo)

        # PPO model shape changes when (m,L) changes, so retraining is mandatory for the ml group.
        if bool(must_retrain_for_dim):
            retrain_this = True

        # If the baseline checkpoint is missing, train a checkpoint once so PPO can still be evaluated.
        if (group == "baseline") and (baseline_model is None or (not Path(baseline_model).exists())):
            retrain_this = True

        if retrain_this:
            # Scenario tag (stable filesystem-safe)
            def _fmt(x: float) -> str:
                s = f"{float(x):g}"
                return s.replace(".", "p")

            scen_tag = f"{group}_{name}"
            if group == "sensor":
                scen_tag = f"{group}_eps{_fmt(env_cfg.get('sensor_error_eps', 0.0))}"
            if group in {"costs", "tti_cost", "order_cost"}:
                scen_tag = f"{group}_K{_fmt(env_cfg.get('fixed_order_cost', 0.0))}_tti{_fmt(env_cfg.get('tti_unit_cost', 0.0))}"
            if group == "ml":
                scen_tag = f"{group}_m{int(env_cfg.get('m', 0))}_L{int(env_cfg.get('L', 0))}"

            # Store sensitivity models under logs_dir/sensitivity/
            ppo_out_dir = logs_dir / "sensitivity" / str(env_cfg.get("issuing_policy", "fifo")) / f"config_{int(cid)}" / f"cv{int(cv_tag)}" / scen_tag
            train_candidates = [dict(h) for h in hyper_for_train_list] if hyper_for_train_list else [dict()]

            total_train_seconds = 0.0
            best_score = float("-inf")
            best_meta: Dict[str, Any] = {}
            best_model: Optional[Path] = None
            scenario_seed0 = int(int(cid) * 10_000 + int(cv_tag) * 10 + 123 + int(scen_idx))

            for cand_idx, cand_hyper in enumerate(train_candidates):
                cand_h = dict(cand_hyper or {})
                cand_rank = int(cand_idx) + 1
                cand_name_raw = str(cand_h.get("name", f"cand_{int(cand_rank):03d}")).strip()
                cand_name_safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in cand_name_raw).strip("_")
                if not cand_name_safe:
                    cand_name_safe = f"cand_{int(cand_rank):03d}"

                cand_out_dir = ppo_out_dir / f"hp_{int(cand_rank):02d}_{cand_name_safe}"
                cand_train_seed = int(scenario_seed0 + 1000 * int(cand_rank))

                try:
                    ppo_meta_i = _train_ppo_for_env_cfg(
                        env_cfg=env_cfg,
                        out_dir=cand_out_dir,
                        hyper=cand_h,
                        total_timesteps=int(ppo_timesteps),
                        train_seed=int(cand_train_seed),
                        eval_seed_base=int(scenario_seed0),
                        quick=bool(quick),
                        device=str(ppo_device),
                    )
                    total_train_seconds += float(ppo_meta_i.get("ppo_train_seconds", 0.0))

                    model_i = Path(str(ppo_meta_i.get("ppo_model_path", cand_out_dir / "best_model.zip")))
                    if not model_i.exists():
                        continue
                    cand_score = float(ppo_meta_i.get("ppo_winner_score", float("-inf")))
                    if (best_model is None) or (np.isfinite(cand_score) and (not np.isfinite(best_score) or cand_score > best_score)):
                        best_score = float(cand_score)
                        best_meta = dict(ppo_meta_i)
                        best_model = Path(model_i)
                        best_meta["ppo_hyper_name"] = str(cand_h.get("name", best_meta.get("ppo_hyper_name", "")))
                        best_meta["ppo_hyper_index"] = int(cand_idx)
                except Exception:
                    continue

            if best_model is None:
                raise RuntimeError(
                    f"PPO retraining failed for all candidates (config={int(cid)}, cv={float(cv):g}, scenario={group}:{name})."
                )

            ppo_meta = dict(best_meta)
            ppo_meta["ppo_train_seconds"] = float(total_train_seconds)
            ppo_meta["ppo_candidates_tried"] = int(len(train_candidates))
            ppo_model_path = Path(best_model)
        else:
            ppo_model_path = baseline_model

        if (not ppo_stats) and ppo_model_path is not None and Path(ppo_model_path).exists():
            ppo_stats = _evaluate_ppo(env_cfg, ds, Path(ppo_model_path), burn_in=int(burn_in))

    beta_excel = float(env_cfg.get("beta_excel", env_cfg.get("beta", 0.0)))
    row_out: Dict[str, Any] = {
        "config_id": int(cid),
        "demand_type": str(env_cfg.get("demand_type", "")),
        "issuing_policy": str(env_cfg.get("issuing_policy", "")),
        "cv_scenario": float(cv),
        "scenario_group": group,
        "scenario_name": name,
        "m": int(env_cfg["m"]),
        "L": int(env_cfg["L"]),
        "Alpha": float(env_cfg["Alpha"]),
        "beta_excel": float(beta_excel),
        "beta_paper": float(1.0 - beta_excel),
        "mean_demand": float(env_cfg["mean_demand"]),
        "coef_of_var": float(env_cfg["coef_of_var"]),
        "c": float(env_cfg.get("c", 0.0)),
        "tti_unit_cost": float(env_cfg.get("tti_unit_cost", 0.0)),
        "fixed_order_cost": float(env_cfg.get("fixed_order_cost", 0.0)),
        "sensor_error_eps": float(env_cfg.get("sensor_error_eps", 0.0)),
        "S_BS": int(S_bs),
        "S_PIL": int(S_pil),
        "tune_seconds": float(tune_seconds),
        "pil_uses_observed_state": bool(pil_use_obs),
    }

    row_out.update({f"BS_{k}": v for k, v in _summarize(bs_stats).items()})
    row_out.update({f"PIL_{k}": v for k, v in _summarize(pil_stats).items()})

    if ppo_stats:
        row_out.update({f"PPO_{k}": v for k, v in _summarize(ppo_stats).items()})
        row_out["PPO_model_path"] = str(ppo_model_path) if ppo_model_path is not None else ""
        # Keep output compact: include only PPO training time/budget descriptors.
        for k in ("ppo_total_timesteps", "ppo_train_seconds", "ppo_hyper_name"):
            if k in ppo_meta:
                row_out[f"PPO_{k}"] = ppo_meta[k]
    else:
        row_out["PPO_profit_mean"] = float("nan")
        row_out["PPO_cost_mean"] = float("nan")
        row_out["PPO_model_path"] = str(ppo_model_path) if ppo_model_path is not None else ""

    # Derived comparisons (profit and cost deltas)
    try:
        row_out["PPO_minus_BS_profit"] = float(row_out.get("PPO_profit_mean", np.nan) - row_out.get("BS_profit_mean", np.nan))
        row_out["PPO_minus_PIL_profit"] = float(row_out.get("PPO_profit_mean", np.nan) - row_out.get("PIL_profit_mean", np.nan))
        row_out["PIL_minus_BS_profit"] = float(row_out.get("PIL_profit_mean", np.nan) - row_out.get("BS_profit_mean", np.nan))

        row_out["PPO_minus_BS_cost"] = float(row_out.get("PPO_cost_mean", np.nan) - row_out.get("BS_cost_mean", np.nan))
        row_out["PPO_minus_PIL_cost"] = float(row_out.get("PPO_cost_mean", np.nan) - row_out.get("PIL_cost_mean", np.nan))
        row_out["PIL_minus_BS_cost"] = float(row_out.get("PIL_cost_mean", np.nan) - row_out.get("BS_cost_mean", np.nan))
    except Exception:
        pass

    return row_out


def _worker_eval_one_scenario(kwargs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Process-pool worker for scenario-level parallelism."""
    _set_single_thread_env()
    k = dict(kwargs or {})
    k["logs_dir"] = Path(str(k.get("logs_dir", "")))
    scen_idx = int(k.get("scen_idx", 0))
    row_out = _evaluate_one_scenario(**k)
    return scen_idx, row_out


def _add_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-(demand_type,cv,issuing_policy) deltas vs baseline row."""
    if df is None or df.empty:
        return df

    df = df.copy()
    key_cols = ["demand_type", "cv_scenario", "issuing_policy"]

    # Identify baseline rows
    base = df.loc[df["scenario_group"] == "baseline"].copy()
    if base.empty:
        return df

    # Use first baseline per key (should be unique).
    base = base.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="first")

    # Merge baseline metrics
    keep_cols = key_cols + [
        "BS_cost_mean",
        "PIL_cost_mean",
        "PPO_cost_mean",
        "BS_profit_mean",
        "PIL_profit_mean",
        "PPO_profit_mean",
    ]
    base_small = base[keep_cols].copy()
    base_small = base_small.rename(
        columns={
            "BS_cost_mean": "BS_cost_mean_baseline",
            "PIL_cost_mean": "PIL_cost_mean_baseline",
            "PPO_cost_mean": "PPO_cost_mean_baseline",
            "BS_profit_mean": "BS_profit_mean_baseline",
            "PIL_profit_mean": "PIL_profit_mean_baseline",
            "PPO_profit_mean": "PPO_profit_mean_baseline",
        }
    )

    df = df.merge(base_small, on=key_cols, how="left")

    for pol in ["BS", "PIL", "PPO"]:
        df[f"{pol}_cost_delta_vs_baseline"] = pd.to_numeric(df[f"{pol}_cost_mean"], errors="coerce") - pd.to_numeric(
            df[f"{pol}_cost_mean_baseline"], errors="coerce"
        )
        df[f"{pol}_profit_delta_vs_baseline"] = pd.to_numeric(df[f"{pol}_profit_mean"], errors="coerce") - pd.to_numeric(
            df[f"{pol}_profit_mean_baseline"], errors="coerce"
        )

    return df


def _compact_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the exact reporting-focused compact schema."""
    if df is None or df.empty:
        return df

    out = df.copy()
    if "baseline_config_id" not in out.columns and "config_id" in out.columns:
        out["baseline_config_id"] = pd.to_numeric(out["config_id"], errors="coerce")
    if "tune_seconds_used" not in out.columns and "tune_seconds" in out.columns:
        out["tune_seconds_used"] = pd.to_numeric(out["tune_seconds"], errors="coerce")
    if "ppo_train_seconds" not in out.columns:
        if "PPO_ppo_train_seconds" in out.columns:
            out["ppo_train_seconds"] = pd.to_numeric(out["PPO_ppo_train_seconds"], errors="coerce").fillna(0.0)
        else:
            out["ppo_train_seconds"] = 0.0

    compact_cols = [
        "issuing_policy",
        "demand_type",
        "coef_of_var",
        "baseline_config_id",
        "scenario_group",
        "scenario_name",
        "Alpha",
        "beta_excel",
        "m",
        "L",
        "tti_unit_cost",
        "fixed_order_cost",
        "sensor_error_eps",
        "BS_profit_mean",
        "PIL_profit_mean",
        "PPO_profit_mean",
        "BS_profit_ci95",
        "PIL_profit_ci95",
        "PPO_profit_ci95",
        "tune_seconds_used",
        "ppo_train_seconds",
    ]
    for c in compact_cols:
        if c in out.columns:
            continue
        if c in {"tune_seconds_used", "ppo_train_seconds"}:
            out[c] = 0.0
        else:
            out[c] = np.nan
    return out.loc[:, compact_cols].copy()


# ======================================================================================
# Main
# ======================================================================================
def main() -> None:
    _set_single_thread_env()

    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=str, default="configurations.xlsx")
    p.add_argument("--results-dir", type=str, default="results", help="Directory containing policy_comparison CSV and output CSV/meta.")
    p.add_argument("--policy-csv", type=str, default="", help="Path to policy_comparison.csv (baseline levels, baseline selection).")
    p.add_argument("--baseline-from-policy", action="store_true", default=True, help="Reuse baseline metrics from policy_comparison.csv (skip baseline simulation).")
    p.add_argument("--no-baseline-from-policy", action="store_false", dest="baseline_from_policy", help="Simulate baseline instead of reusing policy_comparison.csv.")
    p.add_argument(
        "--strict-policy-match",
        action="store_true",
        default=True,
        help="If baseline is reused from policy_comparison.csv, require issuing-policy match (column or filename suffix).",
    )
    p.add_argument(
        "--no-strict-policy-match",
        action="store_false",
        dest="strict_policy_match",
        help="Allow baseline reuse even when policy_comparison.csv does not explicitly encode issuing policy.",
    )

    # PPO hyperparameter reuse (no rescreening)
    p.add_argument("--top-candidates-json", type=str, default="hp_screen_logs_best_420_50k_80k/top_candidates.json", help="Path to top_candidates.json (PPO screening output).")
    p.add_argument("--ppo-hyper-source", type=str, default="top_candidates", choices=["auto", "training_meta", "top_candidates"], help="Source of PPO hyperparameters for sensitivity retraining.")
    p.add_argument("--ppo-hyper-idx", type=int, default=0, help="Fallback candidate index (used only when not sweeping top_candidates).")
    p.add_argument("--ppo-hyper-name", type=str, default="", help="Fallback candidate name (used only when not sweeping top_candidates).")
    p.add_argument("--config-ids", type=str, default="", help="Optional: explicit config IDs. If empty, baselines auto-selected.")
    p.add_argument("--logs-dir", type=str, default="logs")
    p.add_argument("--issuing-policy", type=str, default="fifo", choices=["fifo", "lifo", "random"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mode", type=str, default="all", choices=["all", "sensor", "costs", "tti_cost", "order_cost", "ml"])

    # Demand variability scenarios
    p.add_argument("--cv-list", type=str, default="0.8,1.2", help="Comma-separated CV values (default: 0.8,1.2).")
    p.add_argument("--pair-config-cv", action="store_true", default=True, help="Pair config_ids with cv_list by position.")
    p.add_argument("--no-pair-config-cv", action="store_false", dest="pair_config_cv", help="Use cartesian product of config_ids x cv_list.")

    # Auto-baseline selection (when --config-ids is empty)
    p.add_argument("--auto-select-baseline", action="store_true", default=True)
    p.add_argument("--no-auto-select-baseline", action="store_false", dest="auto_select_baseline")

    p.add_argument("--demand-types", type=str, default="all", help="Comma-separated demand types, or 'all' (default).")
    p.add_argument("--expand-demand-types", action="store_true", default=True, help="If config-ids are provided, expand to all demand types using the template (Alpha,beta,m,L).")
    p.add_argument("--no-expand-demand-types", action="store_false", dest="expand_demand_types", help="Do not expand config-ids to all demand types.")
    p.add_argument("--expansion-param-source", type=str, default="args", choices=["args", "template"], help="When expanding to all demand types, use params from CLI (args) or infer from template config IDs.")

    # Fixed baseline (Alpha, beta_excel) shared across all demand types and CV scenarios
    p.add_argument("--baseline-alpha", type=float, default=0.1)
    p.add_argument("--baseline-beta-excel", type=float, default=0.1)
    p.add_argument("--baseline-m", type=int, default=3)
    p.add_argument("--baseline-L", type=int, default=1)

    # Sensitivity grids (wide reporting-focused defaults)
    p.add_argument("--sensor-eps-list", type=str, default="0,0.05,0.10,0.20,0.30,0.40,0.50", help="Sensor error eps values (env clips to [0,0.5]).")
    p.add_argument("--fixed-order-cost-list", type=str, default="0,5,10,50,100", help="Fixed per-order costs K (>=0).")
    p.add_argument("--tti-unit-cost-list", type=str, default="0,0.05,0.10,0.20,0.50,1.0,2.0", help="Unit TTI costs (>=0).")
    p.add_argument("--m-list", type=str, default="2,3,4,5", help="m list for (m,L) robustness.")
    p.add_argument("--L-list", type=str, default="1,2,3,4", help="L list for (m,L) robustness.")

    # Policy parameters handling
    p.add_argument("--levels-source", type=str, default="policy_comparison", choices=["policy_comparison", "tune"])
    p.add_argument("--fixed-baseline-levels", action="store_true", default=True, help="Use baseline S_BS/S_PIL for all scenarios.")
    p.add_argument("--no-fixed-baseline-levels", action="store_false", dest="fixed_baseline_levels")

    # Sensor error handling for PIL
    p.add_argument("--sensor-affects-pil", action="store_true", default=True)
    p.add_argument("--no-sensor-affects-pil", action="store_false", dest="sensor_affects_pil")
    p.add_argument("--freeze-levels-on-sensor", action="store_true", default=True)
    p.add_argument("--no-freeze-levels-on-sensor", action="store_false", dest="freeze_levels_on_sensor")

    # PPO retraining control
    p.add_argument("--retrain-ppo", action="store_true", default=True, help="Retrain PPO for sensitivity scenarios (default: enabled).")
    p.add_argument("--no-retrain-ppo", action="store_false", dest="retrain_ppo", help="Do not retrain PPO except when (m,L) changes.")
    p.add_argument("--skip-ppo", action="store_true", help="Skip PPO evaluation/training (BS/PIL only).")

    p.add_argument("--ppo-train-timesteps", type=int, default=0, help="Override PPO training timesteps (0 => auto).")
    p.add_argument("--ppo-timesteps-scale", type=float, default=1.00, help="Scale PPO budget when auto; 1.0 matches train_ppo_GY.py budget.")
    p.add_argument("--ppo-device", type=str, default="cpu", help="PPO device (cpu or cuda).")

    # Parallelism (replications only; scenario-level parallelism is intentionally not default for stability)
    p.add_argument(
        "--parallel-scenarios",
        action="store_true",
        default=False,
        help="Parallelize across scenarios (uses spawn workers; replication workers are forced to 1 inside scenario workers).",
    )
    p.add_argument(
        "--no-parallel-scenarios",
        action="store_false",
        dest="parallel_scenarios",
        help="Disable scenario-level parallelism (evaluate scenarios sequentially).",
    )

    p.add_argument("--max-workers", type=int, default=0, help="0 => use cpu_count()-1 for scenario/replication workers.")

    args = p.parse_args()

    base_dir = Path(__file__).resolve().parent
    results_dir, plots_dir = _ensure_dirs(base_dir, results_dir_arg=str(args.results_dir), plots_dir_arg="plots")
    logs_dir = _resolve_logs_dir(base_dir, args.logs_dir)

    # Use CLI issuing policy as the single source of truth for this run.
    issuing_policy = str(args.issuing_policy).strip().lower()
    if issuing_policy not in {"fifo", "lifo", "random"}:
        issuing_policy = "fifo"

    quick = bool(args.quick)

    # Evaluation settings consistent with run_benchmarks_GY.py
    eval_reps = int(bench.N_REPS_QUICK if quick else bench.N_REPS_FULL)
    eval_horizon = int(bench.T_HORIZON_QUICK if quick else bench.T_HORIZON_FULL)
    burn_in = int(bench.BURN_IN_QUICK if quick else bench.BURN_IN_FULL)

    cv_list = [float(max(1e-6, v)) for v in _parse_float_list(args.cv_list)]

    sensor_eps_list = [float(max(0.0, v)) for v in _parse_float_list(args.sensor_eps_list)]
    fixed_order_cost_list = [float(max(0.0, v)) for v in _parse_float_list(args.fixed_order_cost_list)]
    tti_unit_cost_list = [float(max(0.0, v)) for v in _parse_float_list(args.tti_unit_cost_list)]

    m_list = sorted(set([int(x) for x in _parse_int_list(args.m_list)]))
    L_list = sorted(set([int(x) for x in _parse_int_list(args.L_list)]))
    if not m_list:
        m_list = [max(1, int(args.baseline_m))]
    if not L_list:
        L_list = [max(1, int(args.baseline_L))]

    max_workers = int(args.max_workers)
    if max_workers <= 0:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    # Resolve policy_comparison.csv path (for baseline selection and levels)
    if str(args.policy_csv).strip():
        pc_path = Path(str(args.policy_csv))
        if not pc_path.is_absolute():
            pc_path = base_dir / pc_path
    else:
        # If per-issuing-policy policy_comparison files exist, prefer them.
        pc_candidate = results_dir / f"policy_comparison_{issuing_policy}.csv"
        pc_path = pc_candidate if pc_candidate.exists() else (results_dir / "policy_comparison.csv")

    policy_df: Optional[pd.DataFrame] = None
    policy_match_verified = False
    if str(args.levels_source).strip().lower() == "policy_comparison" or bool(args.auto_select_baseline):
        policy_df = _load_policy_comparison_df(pc_path)
        policy_df, has_policy_col = _filter_policy_df_issuing(policy_df, issuing_policy)
        policy_match_verified = bool(has_policy_col or _policy_csv_name_matches_issuing(pc_path, issuing_policy))

    baseline_from_policy_effective = bool(getattr(args, "baseline_from_policy", True))
    if baseline_from_policy_effective and bool(getattr(args, "strict_policy_match", True)) and (not policy_match_verified):
        print(
            "[sensitivity] WARNING: baseline_from_policy disabled because policy_comparison issuing policy "
            "cannot be verified (no issuing_policy column and filename has no policy suffix)."
        )
        baseline_from_policy_effective = False

    # Seed alignment with benchmark run used to produce policy_comparison.csv
    eval_seed0 = int(getattr(bench, "EVAL_SEED0", 20_000))
    eval_seed0_source = "benchmark_default"
    policy_meta = _load_policy_comparison_meta(pc_path)
    try:
        meta_seed = policy_meta.get("eval", {}).get("seed0", None) if isinstance(policy_meta, dict) else None
        if meta_seed is not None:
            eval_seed0 = int(meta_seed)
            eval_seed0_source = "policy_comparison_meta"
    except Exception:
        pass

    # Load PPO screening candidates (optional) for hyperparameter reuse
    top_candidates_path = Path(str(getattr(args, "top_candidates_json", "")).strip())
    if str(top_candidates_path) and (not top_candidates_path.is_absolute()):
        top_candidates_path = base_dir / top_candidates_path
    top_candidates = _load_top_candidates_json(top_candidates_path) if str(top_candidates_path) else []
    ppo_hyper_source = str(getattr(args, "ppo_hyper_source", "top_candidates")).strip().lower()
    ppo_hyper_fallback = _select_top_candidate(
        top_candidates,
        idx=int(getattr(args, "ppo_hyper_idx", 0)),
        name=str(getattr(args, "ppo_hyper_name", "")),
    )
    if ppo_hyper_source == "top_candidates":
        if not top_candidates:
            raise RuntimeError(
                "PPO hyperparameters source is top_candidates, but top_candidates.json is empty or unreadable. "
                f"Check --top-candidates-json path: {top_candidates_path}"
            )


    config_ids = _parse_config_ids(args.config_ids) if str(args.config_ids).strip() else []

    run_pairs: List[Tuple[int, float]] = []
    baseline_triples: Optional[List[Tuple[int, float, str]]] = None

    if (not config_ids) and bool(args.auto_select_baseline):
        if policy_df is None:
            raise RuntimeError("Auto baseline selection requires policy_df")
        demand_types = _parse_demand_type_list(str(args.demand_types), policy_df)
        baseline_triples = _select_baseline_triples_fixed_params(
            policy_df,
            cv_list=cv_list,
            demand_types=demand_types,
            Alpha=float(args.baseline_alpha),
            beta_excel=float(args.baseline_beta_excel),
            m=int(args.baseline_m) if args.baseline_m is not None else None,
            L=int(args.baseline_L) if args.baseline_L is not None else None,
        )
        run_pairs = [(cid, cv) for (cid, cv, _dt) in baseline_triples]

        print("[baseline] Auto-selected fixed-parameter baselines from policy_comparison.csv:")
        for cid, cv, dt in baseline_triples:
            print(f"  demand_type={dt} | cv={cv:g} | config_id={cid} | Alpha={float(args.baseline_alpha):g} | beta_excel={float(args.baseline_beta_excel):g}")
    else:
        if not config_ids:
            raise ValueError("No config IDs provided and auto baseline selection disabled.")
        if bool(args.pair_config_cv):
            if len(config_ids) == len(cv_list):
                run_pairs = [(int(cid), float(cv_list[i])) for i, cid in enumerate(config_ids)]
            elif len(config_ids) == 1:
                run_pairs = [(int(config_ids[0]), float(cv)) for cv in cv_list]
            else:
                raise ValueError(
                    f"--pair-config-cv requires len(config_ids)==len(cv_list) or exactly one config_id; got {len(config_ids)} and {len(cv_list)}."
                )
        else:
            run_pairs = [(int(cid), float(cv)) for cid in config_ids for cv in cv_list]


    # Baseline parameters used for tagging and reporting.
    baseline_params_used = {
        "Alpha": float(args.baseline_alpha),
        "beta_excel": float(args.baseline_beta_excel),
        "m": int(args.baseline_m),
        "L": int(args.baseline_L),
    }

    # Optional: if config IDs are provided but you still want to run the expanded set
    # (all demand types, same Alpha/beta/m/L), expand using the template config(s).
    if config_ids and bool(getattr(args, "expand_demand_types", True)):
        if policy_df is None:
            policy_df = _load_policy_comparison_df(pc_path)
            policy_df, has_policy_col = _filter_policy_df_issuing(policy_df, issuing_policy)
            policy_match_verified = bool(has_policy_col or _policy_csv_name_matches_issuing(pc_path, issuing_policy))

        demand_types = _parse_demand_type_list(str(args.demand_types), policy_df)
        cv_values = sorted(set([float(cv) for (_cid, cv) in run_pairs]))

        if str(getattr(args, "expansion_param_source", "args")).strip().lower() == "template":
            expanded_pairs, params = _expand_template_pairs_to_all_demand_types(
                policy_df,
                template_pairs=run_pairs,
                demand_types=demand_types,
            )
            run_pairs = expanded_pairs
            baseline_params_used = dict(params)
        else:
            # Use the user-selected fixed baseline parameters from CLI.
            expanded: List[Tuple[int, float]] = []
            for cvv in cv_values:
                triples = _select_baseline_triples_fixed_params(
                    policy_df,
                    cv_list=[float(cvv)],
                    demand_types=demand_types,
                    Alpha=float(args.baseline_alpha),
                    beta_excel=float(args.baseline_beta_excel),
                    m=int(args.baseline_m) if args.baseline_m is not None else None,
                    L=int(args.baseline_L) if args.baseline_L is not None else None,
                )
                expanded.extend([(cid2, float(cv2)) for (cid2, cv2, _dt) in triples])

            # Deduplicate while preserving order
            seen = set()
            expanded_unique: List[Tuple[int, float]] = []
            for cid2, cv2 in expanded:
                key = (int(cid2), float(cv2))
                if key in seen:
                    continue
                seen.add(key)
                expanded_unique.append((int(cid2), float(cv2)))

            run_pairs = expanded_unique
            baseline_params_used = {
                "Alpha": float(args.baseline_alpha),
                "beta_excel": float(args.baseline_beta_excel),
                "m": int(args.baseline_m),
                "L": int(args.baseline_L),
            }

        # If the user did not explicitly pass m_list/L_list, re-derive them around the baseline used for expansion.
        if not str(args.m_list).strip():
            m0 = int(baseline_params_used["m"])
            m_list = sorted(set([max(1, m0 - 1), m0, m0 + 1, m0 + 2]))
        if not str(args.L_list).strip():
            L0 = int(baseline_params_used["L"])
            L_list = sorted(set([max(1, L0 - 1), L0, L0 + 1, L0 + 2]))

        print("[baseline] Expanded config IDs to all demand types using:")
        print(f"  Alpha={baseline_params_used['Alpha']:g} | beta_excel={baseline_params_used['beta_excel']:g} | m={baseline_params_used['m']} | L={baseline_params_used['L']}")
        print(f"  demand_types={demand_types} | cv_values={cv_values}")
        print(f"  total pairs={len(run_pairs)}")

    # Metadata
    tag_bits = [
        str(args.mode).lower(),
        "quick" if quick else "full",
        issuing_policy,
        f"a{float(baseline_params_used['Alpha']):g}".replace(".", "p"),
        f"b{float(baseline_params_used['beta_excel']):g}".replace(".", "p"),
    ]
    tag = "_".join(tag_bits)

    meta_out: Dict[str, Any] = {
        "mode": str(args.mode).lower(),
        "quick": bool(quick),
        "issuing_policy": str(issuing_policy),
        "cv_list": [float(x) for x in cv_list],
        "eval_horizon": int(eval_horizon),
        "burn_in": int(burn_in),
        "eval_reps": int(eval_reps),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_alpha": float(baseline_params_used["Alpha"]),
        "baseline_beta_excel": float(baseline_params_used["beta_excel"]),
        "baseline_m": int(baseline_params_used["m"]),
        "baseline_L": int(baseline_params_used["L"]),
        "sensor_eps_list": [float(x) for x in sensor_eps_list],
        "fixed_order_cost_list": [float(x) for x in fixed_order_cost_list],
        "tti_unit_cost_list": [float(x) for x in tti_unit_cost_list],
        "m_list": [int(x) for x in m_list],
        "L_list": [int(x) for x in L_list],
        "levels_source": str(args.levels_source),
        "fixed_baseline_levels": bool(args.fixed_baseline_levels),
        "sensor_affects_pil": bool(args.sensor_affects_pil),
        "freeze_levels_on_sensor": bool(args.freeze_levels_on_sensor),
        "retrain_ppo": bool(args.retrain_ppo),
        "skip_ppo": bool(args.skip_ppo),
        "parallel_scenarios_requested": bool(args.parallel_scenarios),
        "ppo_train_timesteps_override": int(args.ppo_train_timesteps),
        "ppo_timesteps_scale": float(args.ppo_timesteps_scale),
        "ppo_device": str(args.ppo_device),
        "max_workers": int(max_workers),
        "logs_dir": str(logs_dir),
        "policy_csv": str(pc_path),
        "baseline_from_policy_requested": bool(getattr(args, "baseline_from_policy", True)),
        "baseline_from_policy_used": bool(baseline_from_policy_effective),
        "strict_policy_match": bool(getattr(args, "strict_policy_match", True)),
        "policy_match_verified": bool(policy_match_verified),
        "eval_seed0_used": int(eval_seed0),
        "eval_seed0_source": str(eval_seed0_source),
        "top_candidates_json": str(top_candidates_path) if "top_candidates_path" in locals() else "",
        "ppo_hyper_source": str(ppo_hyper_source),
        "ppo_top_candidates_count": int(len(top_candidates)),
        "ppo_hyper_idx": int(getattr(args, "ppo_hyper_idx", 0)),
        "ppo_hyper_name": str(getattr(args, "ppo_hyper_name", "")),
        "ppo_hyper_selected_name": (
            "__all_top_candidates__"
            if str(ppo_hyper_source) == "top_candidates"
            else (str(ppo_hyper_fallback.get("name", "")) if isinstance(ppo_hyper_fallback, dict) else "")
        ),
        "config_cv_pairs": [{"config_id": int(cid), "cv": float(cv)} for cid, cv in run_pairs],
    }

    # Cache excel rows by config id
    row_cache: Dict[int, Any] = {}

    rows_out: List[Dict[str, Any]] = []
    scenario_parallel_requested = bool(args.parallel_scenarios)
    scenario_parallel_used_any = False
    scenario_parallel_fallback_any = False
    scenario_parallel_workers_used = 0

    for cid, cv in run_pairs:
        if int(cid) not in row_cache:
            row_cache[int(cid)] = bench.load_config_row(args.excel, int(cid))
        row = row_cache[int(cid)]

        # Base env config, then override CV and set issuing policy.
        base_cfg0 = bench.build_env_config(row, episode_length=int(eval_horizon), issuing_policy=issuing_policy)
        base_cfg = dict(base_cfg0)
        base_cfg["coef_of_var"] = float(cv)
        base_cfg["issuing_policy"] = str(issuing_policy)

        cv_tag = int(round(1000 * float(cv)))

        # Demand streams reused across all scenarios for this (config,cv).
        demand_streams_base = [
            bench.generate_demand_stream(
                mean_demand=float(base_cfg["mean_demand"]),
                coef_of_var=float(base_cfg["coef_of_var"]),
                demand_type=str(base_cfg.get("demand_type", "beta_binomial")),
                demand_dmax=base_cfg.get("demand_dmax", None),
                mix_pi_high=base_cfg.get("mix_pi_high", None),
                mix_pi_calm=base_cfg.get("mix_pi_calm", None),
                mix_pi_normal=base_cfg.get("mix_pi_normal", None),
                mix_pi_promo=base_cfg.get("mix_pi_promo", None),
                mix_lam1=base_cfg.get("mix_lam1", None),
                mix_lam2=base_cfg.get("mix_lam2", None),
                mix_lam3=base_cfg.get("mix_lam3", None),
                mix_w1=base_cfg.get("mix_w1", None),
                mix_w2=base_cfg.get("mix_w2", None),
                horizon=int(eval_horizon),
                seed=int(int(eval_seed0) + 1000 * int(cid) + int(r)),
            )
            for r in range(int(eval_reps))
        ]

        # Scenario list (baseline row always included)
        scenarios = _scenario_list(
            mode=str(args.mode).lower(),
            base_cfg=base_cfg,
            sensor_eps_list=sensor_eps_list,
            fixed_order_cost_list=fixed_order_cost_list,
            tti_unit_cost_list=tti_unit_cost_list,
            m_list=m_list,
            L_list=L_list,
        )
        scenarios = [
            {
                "group": "baseline",
                "name": "baseline",
                "overrides": {"sensor_error_eps": 0.0, "fixed_order_cost": 0.0, "tti_unit_cost": 0.0},
            }
        ] + scenarios

        # Baseline levels (S_BS, S_PIL)
        tune_seconds_base = 0.0
        if str(args.levels_source).strip().lower() == "policy_comparison":
            s_bs, s_pil = _levels_from_policy_df(policy_df, int(cid)) if policy_df is not None else (None, None)
            if (s_bs is None) or (s_pil is None):
                raise RuntimeError(
                    f"Missing S_BS/S_PIL for config={cid} in policy_comparison.csv ({pc_path}). "
                    "Run run_benchmarks_GY.py first, or use --levels-source tune."
                )
            S_bs_base, S_pil_base = int(s_bs), int(s_pil)
        else:
            # Benchmark-compatible baseline tuning seeds:
            # BS: TUNE_SEED0 + 1000*cid, PIL: BS seed + 100
            tune_seed_bs = int(getattr(bench, "TUNE_SEED0", 10_000) + 1000 * int(cid))
            tune_seed_pil = int(tune_seed_bs + 100)
            S_bs_base, S_pil_base, tune_seconds_base = _tune_levels_for_cfg(
                base_cfg,
                quick=quick,
                seed=int(tune_seed_bs),
                seed_bs=int(tune_seed_bs),
                seed_pil=int(tune_seed_pil),
            )

        # PPO timesteps for this (config,cv) (auto: same budget logic as train_ppo_GY.py, then optional scale)
        ppo_timesteps_override = int(args.ppo_train_timesteps)
        if ppo_timesteps_override > 0:
            ppo_timesteps = int(ppo_timesteps_override)
        else:
            cv_for_budget = float(base_cfg.get("coef_of_var", cv))
            base_budget_train_like = _compute_train_like_timesteps(quick=bool(quick), cv=float(cv_for_budget))
            if base_budget_train_like is not None:
                base_budget = int(base_budget_train_like)
            else:
                baseline_meta = bench.load_training_meta(logs_dir, int(cid))
                base_budget = int(baseline_meta.get("total_timesteps_budget", 200_000 if quick else 300_000))
            ppo_timesteps = int(max(10_000, round(float(args.ppo_timesteps_scale) * float(base_budget))))

        # Scenario-level parallelism (with deterministic per-scenario seeds).
        parallel_scenarios_this = bool(scenario_parallel_requested) and (int(max_workers) > 1) and (len(scenarios) > 1)

        scenario_tasks: List[Dict[str, Any]] = []
        for scen_idx, scen in enumerate(scenarios):
            scen_group = str(scen.get("group", "")).strip().lower()

            # Baseline: if already available in policy_comparison.csv, reuse it (skip simulation).
            if (scen_group == "baseline") and bool(baseline_from_policy_effective):
                if policy_df is None:
                    raise RuntimeError("baseline_from_policy=True requires --policy-csv (policy_df) to be loadable.")
                row_out = _baseline_row_from_policy_df(
                    policy_df=policy_df,
                    cid=int(cid),
                    cv=float(cv),
                    issuing_policy=str(issuing_policy),
                    env_cfg=dict(base_cfg),
                    S_bs=int(S_bs_base),
                    S_pil=int(S_pil_base),
                    tune_seconds_fallback=float(tune_seconds_base),
                )
                rows_out.append(row_out)
                print(f"[sensitivity] config={cid} cv={cv:g} group=baseline name=baseline_from_policy done.")
                continue

            task_kwargs: Dict[str, Any] = {
                "cid": int(cid),
                "cv": float(cv),
                "cv_tag": int(cv_tag),
                "scen_idx": int(scen_idx),
                "scen_group": str(scen["group"]),
                "scen_name": str(scen["name"]),
                "overrides": dict(scen.get("overrides", {})),
                "base_cfg": dict(base_cfg),
                "base_cfg0": dict(base_cfg0),
                "logs_dir": str(logs_dir),
                "quick": bool(quick),
                "eval_horizon": int(eval_horizon),
                "eval_reps": int(eval_reps),
                "burn_in": int(burn_in),
                "S_bs_base": int(S_bs_base),
                "S_pil_base": int(S_pil_base),
                "tune_seconds_base": float(tune_seconds_base),
                "sensor_affects_pil": bool(args.sensor_affects_pil),
                "fixed_baseline_levels": bool(args.fixed_baseline_levels),
                "freeze_levels_on_sensor": bool(args.freeze_levels_on_sensor),
                "retrain_ppo": bool(args.retrain_ppo),
                "skip_ppo": bool(args.skip_ppo),
                "rep_workers": 1 if parallel_scenarios_this else int(max_workers),
                # Avoid large IPC payload in scenario workers; demand streams are deterministically regenerated.
                "demand_streams_base": None if parallel_scenarios_this else demand_streams_base,
                "ppo_timesteps": int(ppo_timesteps),
                "ppo_device": str(args.ppo_device),
                "ppo_hyper_source": str(ppo_hyper_source),
                "ppo_hyper_fallback": ppo_hyper_fallback,
                "ppo_top_candidates": top_candidates,
                "eval_seed0": int(eval_seed0),
            }
            scenario_tasks.append(task_kwargs)

        if parallel_scenarios_this and scenario_tasks:
            try:  # pragma: no cover
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor, as_completed

                ctx = mp.get_context("spawn")
                n_workers = int(min(int(max_workers), len(scenario_tasks)))
                scenario_parallel_workers_used = max(int(scenario_parallel_workers_used), int(n_workers))

                out_rows_ordered: List[Tuple[int, Dict[str, Any]]] = []
                with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=ctx) as ex:
                    fut_map = {ex.submit(_worker_eval_one_scenario, kw): kw for kw in scenario_tasks}
                    for fut in as_completed(fut_map):
                        kw = fut_map[fut]
                        idx_done, row_out = fut.result()
                        out_rows_ordered.append((int(idx_done), row_out))
                        print(
                            f"[sensitivity] config={cid} cv={cv:g} "
                            f"group={row_out.get('scenario_group', '')} name={row_out.get('scenario_name', '')} done."
                        )

                out_rows_ordered.sort(key=lambda x: int(x[0]))
                rows_out.extend([r for _i, r in out_rows_ordered])
                scenario_parallel_used_any = True
            except Exception as e:  # pragma: no cover
                print(
                    f"[sensitivity] WARNING: scenario multiprocessing failed "
                    f"({type(e).__name__}: {e}); falling back to sequential for this config/cv."
                )
                scenario_parallel_fallback_any = True
                parallel_scenarios_this = False

        if (not parallel_scenarios_this) and scenario_tasks:
            for kw in scenario_tasks:
                row_out = _evaluate_one_scenario(
                    cid=int(kw["cid"]),
                    cv=float(kw["cv"]),
                    cv_tag=int(kw["cv_tag"]),
                    scen_idx=int(kw["scen_idx"]),
                    scen_group=str(kw["scen_group"]),
                    scen_name=str(kw["scen_name"]),
                    overrides=dict(kw["overrides"]),
                    base_cfg=dict(kw["base_cfg"]),
                    base_cfg0=dict(kw["base_cfg0"]),
                    logs_dir=Path(str(kw["logs_dir"])),
                    quick=bool(kw["quick"]),
                    eval_horizon=int(kw["eval_horizon"]),
                    eval_reps=int(kw["eval_reps"]),
                    burn_in=int(kw["burn_in"]),
                    S_bs_base=int(kw["S_bs_base"]),
                    S_pil_base=int(kw["S_pil_base"]),
                    tune_seconds_base=float(kw["tune_seconds_base"]),
                    sensor_affects_pil=bool(kw["sensor_affects_pil"]),
                    fixed_baseline_levels=bool(kw["fixed_baseline_levels"]),
                    freeze_levels_on_sensor=bool(kw["freeze_levels_on_sensor"]),
                    retrain_ppo=bool(kw["retrain_ppo"]),
                    skip_ppo=bool(kw["skip_ppo"]),
                    rep_workers=int(kw["rep_workers"]),
                    demand_streams_base=kw["demand_streams_base"],
                    ppo_timesteps=int(kw["ppo_timesteps"]),
                    ppo_device=str(kw["ppo_device"]),
                    ppo_hyper_source=str(kw["ppo_hyper_source"]),
                    ppo_hyper_fallback=kw["ppo_hyper_fallback"],
                    ppo_top_candidates=kw["ppo_top_candidates"],
                    eval_seed0=int(kw["eval_seed0"]),
                )
                rows_out.append(row_out)
                print(
                    f"[sensitivity] config={cid} cv={cv:g} "
                    f"group={row_out.get('scenario_group', '')} name={row_out.get('scenario_name', '')} done."
                )

    meta_out["parallel_scenarios_used_any"] = bool(scenario_parallel_used_any)
    meta_out["parallel_scenarios_fallback_any"] = bool(scenario_parallel_fallback_any)
    meta_out["parallel_scenarios_workers_used"] = int(scenario_parallel_workers_used)

    df = pd.DataFrame(rows_out)
    df = _add_baseline_deltas(df)
    df = _compact_output_columns(df)

    # Output files
    out_dir = results_dir / f"sensitivity_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sensitivity_results.csv"
    out_csv_written = out_csv
    try:
        df.to_csv(out_csv, index=False)
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv_written = out_dir / f"sensitivity_results_{ts}.csv"
        df.to_csv(out_csv_written, index=False)
        print(f"[sensitivity] WARNING: could not overwrite '{out_csv}' (file locked). Wrote '{out_csv_written}' instead.")

    meta_out["n_rows"] = int(len(df))
    meta_out_path = out_dir / "sensitivity_meta.json"
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    # Plots per demand_type and CV
    try:  # pragma: no cover
        if "demand_type" in df.columns and "cv_scenario" in df.columns:
            for dt in sorted(df["demand_type"].dropna().astype(str).unique().tolist()):
                for cv in sorted(df["cv_scenario"].dropna().astype(float).unique().tolist()):
                    _maybe_plot_sensor(df, plots_dir, tag=tag, demand_type=dt, cv=float(cv))
                    _maybe_plot_cost_slices(df, plots_dir, tag=tag, demand_type=dt, cv=float(cv))
                    _maybe_plot_ml_heatmap(df, plots_dir, tag=tag, demand_type=dt, cv=float(cv))
    except Exception:
        pass

    print(f"[sensitivity] Wrote: {out_csv_written}")
    print(f"[sensitivity] Wrote: {meta_out_path}")


if __name__ == "__main__":
    main()


