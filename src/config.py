from __future__ import annotations

from pathlib import Path

# Root of the repository (two levels up from this file: src/config.py)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Default data path — override via environment variable DATA_PATH if needed
import os as _os

DATA_PATH: Path = Path(
    _os.environ.get("DATA_PATH", str(REPO_ROOT / "logistics_dataset.csv"))
)

# RL training defaults — centralised so notebooks and tests stay in sync
RL_DEFAULTS = {
    "horizon": 30,
    "episodes": 2500,
    "alpha": 0.15,
    "gamma": 0.92,
    "epsilon_start": 0.9,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "eval_runs": 20,
    "eval_top_n": 500,
    "random_seed": 42,
}

# Expedite simulation
EXPEDITE_PREMIUM = 0.35       # 35 % cost uplift over standard handling
STOCKOUT_PENALTY_MULT = 1.5   # penalty = multiplier × unit_price × daily_demand

# Feature engineering thresholds
HIGH_VALUE_PERCENTILE = 0.90  # top 10 % by value_score → high_value_flag
ZONE_CONGESTION_THRESHOLD = 0.75  # zone_load_ratio above this = high-pressure
