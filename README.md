# Reproducibility Guide

This folder contains the code used in our paper:
*Towards Optimized Perishable Inventory Systems: Integrating Time-Temperature Indicators with Deep Reinforcement Learning*
(Sirine Taleb, Chaaben Kouki, Lama Moussawi-Haidar).

This guide explains how to reproduce the results in csv.

## 1) Start Here

Run commands from the `reproducibility` folder.

```powershell
cd reproducibility
```

## 2) Environment Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import numpy, pandas, gymnasium, torch, stable_baselines3; print('ok')"
```

## 3) Paper Setting (Issuing Policy)

The paper reports FIFO issuing. Set:

```powershell
$env:INVENTORY_ISSUING_POLICY = "fifo"
```

(`lifo` and `random` are available for additional tests, but not used for reported paper results.)

## 4) Choose Which Configurations to Run

Use one of the following options before running Quick Mode or Full Mode.

### Option A: One configuration

```powershell
$IDS = "1"
```

### Option B: First N configurations (example: first 3)

This is the equivalent of running a prefix slice like `[1:3]` on the ordered configuration list.

```powershell
$N = 3
$IDS = python -c "import pandas as pd; df=pd.read_excel('configurations.xlsx'); c='configuration' if 'configuration' in df.columns else df.columns[0]; ids=df[c].dropna().astype(int).tolist(); print(','.join(map(str, ids[:$N])))"
```

### Option C: All configurations

```powershell
$IDS = python -c "import pandas as pd; df=pd.read_excel('configurations.xlsx'); c='configuration' if 'configuration' in df.columns else df.columns[0]; ids=df[c].dropna().astype(int).tolist(); print(','.join(map(str, ids)))"
```

Note:
- In `run_benchmarks_GY.py`, internal `CONFIG_IDS = None` means all configurations.
- In this README, we use explicit `--config-ids $IDS` for clarity and reproducibility.

## 5) Quick Mode

Quick Mode is for fast verification.

```powershell
python train_ppo_GY.py --excel configurations.xlsx --config-ids $IDS --quick
python run_benchmarks_GY.py --excel configurations.xlsx --config-ids $IDS --quick
python run_sensitivity.py --excel configurations.xlsx --policy-csv results/policy_comparison.csv --config-ids $IDS --quick --issuing-policy fifo --ppo-hyper-source training_meta
```

If Quick Mode succeeds, you should see:
- `logs/config_<id>/...`
- `results/policy_comparison.csv`
- `results/sensitivity_*.csv`

## 6) Full Mode

Full Mode is the same pipeline without `--quick`.

```powershell
python train_ppo_GY.py --excel configurations.xlsx --config-ids $IDS
python run_benchmarks_GY.py --excel configurations.xlsx --config-ids $IDS
python run_sensitivity.py --excel configurations.xlsx --policy-csv results/policy_comparison.csv --config-ids $IDS --issuing-policy fifo --ppo-hyper-source training_meta
```

This supports:
- full run on one config
- full run on first `N` configs
- full run on all configs

## 7) Main Outputs

- `results/policy_comparison.csv`
- `results/sensitivity_*.csv`

Additional files:
- `logs/config_<id>/...` (trained PPO models used by benchmark and sensitivity runs)
- `results/*.json` (run metadata)
- `results_R2/ppo_hyperparameter_profiles_R2.csv` and `results_R2/ppo_hyperparameter_selection_by_configuration_R2.csv` (PPO settings used in the revised experiments)

## 8) File Roles

- `configurations.xlsx`: experiment definitions
- `demand_models.py`: demand generation
- `InventoryEnvGY_Config.py`: environment dynamics
- `PIL_GY_policy.py`: PIL policy
- `train_ppo_GY.py`: PPO training
- `run_benchmarks_GY.py`: BS/PIL/PPO evaluation
- `run_sensitivity.py`: sensitivity experiments

## 9) Practical Notes

- Keep `configurations.xlsx` unchanged for direct comparability.
- Pipeline order must be:
  `train_ppo_GY.py` -> `run_benchmarks_GY.py` -> `run_sensitivity.py`.
- If a CSV is open in Excel during write, close it and rerun the command.
