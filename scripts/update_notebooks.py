"""Update notebooks to use src package instead of inline duplicated code.

Strategy: work by absolute cell index (all cells, not just code cells) so
removals and replacements are unambiguous.  Every target index was verified
against the live notebooks before writing this script.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def load(name: str) -> dict:
    with open(name, encoding="utf-8") as f:
        return json.load(f)


def save(nb: dict, name: str) -> None:
    with open(name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {name}")


def set_code(nb: dict, all_idx: int, text: str) -> None:
    """Replace source of cell at *all_idx* (across all cell types)."""
    cell = nb["cells"][all_idx]
    assert cell["cell_type"] == "code", f"Cell {all_idx} is not a code cell"
    lines = text.splitlines(keepends=True)
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None


# ---------------------------------------------------------------------------
# Shared templates
# ---------------------------------------------------------------------------

SLOTTING_IMPORTS = """\
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")

from src.environments.zone_slotting import (
    ZoneGeometry,
    SlottingMetrics,
    predict_picking_time,
    ZoneSlottingEnv,
    SlottingQLearner,
    SLOT_VALUE_PER_SECOND,
    OVERLOAD_WEIGHT,
    UTILIZATION_SOFT_LIMIT,
    NEGATIVE_MOVE_PENALTY,
    MIN_TIME_GAIN_SECONDS,
)
from src.config import DATA_PATH
"""

DQN_IMPORTS = """\
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_context("talk")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this notebook. "
        "Example: `pip install torch`."
    ) from exc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

from src.environments.zone_slotting import (
    ZoneGeometry,
    SlottingMetrics,
    predict_picking_time,
    ZoneSlottingEnvDQN,
    SLOT_VALUE_PER_SECOND,
    OVERLOAD_WEIGHT,
    UTILIZATION_SOFT_LIMIT,
    NEGATIVE_MOVE_PENALTY,
    MIN_TIME_GAIN_SECONDS,
    REWARD_SCALE,
    REWARD_CLIP,
)
from src.config import DATA_PATH
"""

QLEARNING_IMPORTS = """\
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")

from src.config import DATA_PATH
"""

DATA_LOADING = """\
df = pd.read_csv(DATA_PATH)
df.head()
"""

ZONE_GEOMETRY_SETUP = """\
geom = ZoneGeometry(df)
df = geom.enrich_dataframe(df)

zones = geom.zones
zone_capacity = geom.zone_capacity
zone_totals = geom.zone_totals
zone_positions = geom.zone_positions
staging_point = geom.staging_point
grid_h = geom.grid_h
grid_w = geom.grid_w
travel_distance = geom.travel_distance

df[["item_id", "zone", "daily_demand", "picking_time_seconds", "travel_distance", "zone_load_ratio"]].head()
"""

IMPORT_PLACEHOLDER = "# Imported from src.environments.zone_slotting\n"


def _patch_env_call(text: str, old: str, new: str) -> str:
    if old in text:
        return text.replace(old, new)
    print(f"  WARNING: pattern not found: {old!r}")
    return text


# ---------------------------------------------------------------------------
# Dynamic slotting (zone assignment).ipynb
# Verified cell map (all-cell indices):
#   2  = code[0]  imports
#   3  = code[1]  data loading
#   4  = code[2]  zone geometry
#   5  = code[3]  baseline zone summary (keep)
#   7  = code[4]  SlottingMetrics + env + constants  → import marker
#   8  = code[5]  SlottingQLearner                   → import marker
#   10 = code[6]  training                           → patch env call
#   12 = code[8]  policies (keep)
#   13 = code[9]  apply_slotting_policy              → patch env call
# ---------------------------------------------------------------------------

def update_main_slotting() -> None:
    name = "Dynamic slotting (zone assignment).ipynb"
    print(f"\nUpdating {name}")
    nb = load(name)

    set_code(nb, 2, SLOTTING_IMPORTS)
    set_code(nb, 3, DATA_LOADING)
    set_code(nb, 4, ZONE_GEOMETRY_SETUP)
    # code[3] (baseline zone summary) at all[5] — keep unchanged
    set_code(nb, 7, IMPORT_PLACEHOLDER)   # SlottingMetrics / ZoneSlottingEnv
    set_code(nb, 8, IMPORT_PLACEHOLDER)   # SlottingQLearner

    # Training cell: patch ZoneSlottingEnv instantiation
    src = "".join(nb["cells"][10]["source"])
    src = _patch_env_call(
        src,
        "env = ZoneSlottingEnv(slotting_subset, zones, zone_capacity)",
        "env = ZoneSlottingEnv(slotting_subset, zones, zone_capacity, geom)",
    )
    set_code(nb, 10, src)

    # apply_slotting_policy: patch env instantiation
    src = "".join(nb["cells"][13]["source"])
    src = _patch_env_call(
        src,
        "env = ZoneSlottingEnv(frame, zones, zone_capacity)\n",
        "env = ZoneSlottingEnv(frame, zones, zone_capacity, geom)\n",
    )
    set_code(nb, 13, src)

    save(nb, name)


# ---------------------------------------------------------------------------
# DQN implementation.ipynb
# Verified cell map (all-cell indices):
#   2  = code[0]  imports
#   4  = code[1]  data loading
#   5  = code[2]  zone geometry
#   6  = code[3]  baseline zone summary (keep)
#   7  = code[4]  SlottingMetrics + ZoneSlottingEnvDQN → import marker
#   (code[5] = EpisodeReplayBuffer — keep)
#   (code[6] = training cell)
# ---------------------------------------------------------------------------

def update_dqn() -> None:
    name = "DQN implementation.ipynb"
    print(f"\nUpdating {name}")
    nb = load(name)

    # Map all-cell indices
    all_cells = nb["cells"]
    code_positions = [i for i, c in enumerate(all_cells) if c["cell_type"] == "code"]
    # code_positions[0]=imports, [1]=data, [2]=zone_geo, [3]=baseline, [4]=SlottingMetrics+envDQN, [6]=training
    set_code(nb, code_positions[0], DQN_IMPORTS)
    set_code(nb, code_positions[1], DATA_LOADING)
    set_code(nb, code_positions[2], ZONE_GEOMETRY_SETUP)
    set_code(nb, code_positions[4], IMPORT_PLACEHOLDER)

    # Training cell: patch ZoneSlottingEnvDQN instantiation
    src = "".join(all_cells[code_positions[6]]["source"])
    src = _patch_env_call(
        src,
        "env = ZoneSlottingEnvDQN(slotting_subset, zones, zone_capacity, feature_stats)",
        "env = ZoneSlottingEnvDQN(slotting_subset, zones, zone_capacity, geom, feature_stats)",
    )
    set_code(nb, code_positions[6], src)

    save(nb, name)


# ---------------------------------------------------------------------------
# warhouse-logistics-rl.ipynb
# Verified cell map (all-cell indices):
#   code[0] = imports
#   code[1] = data loading (has hardcoded DATA_PATH = Path(...))
# ---------------------------------------------------------------------------

def update_qlearning() -> None:
    name = "warhouse-logistics-rl.ipynb"
    print(f"\nUpdating {name}")
    nb = load(name)

    code_positions = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

    set_code(nb, code_positions[0], QLEARNING_IMPORTS)

    # Remove the hardcoded DATA_PATH assignment line from the data-loading cell
    src = "".join(nb["cells"][code_positions[1]]["source"])
    src = src.replace('DATA_PATH = Path("logistics_dataset.csv")\n\n', "")
    src = src.replace('DATA_PATH = Path("logistics_dataset.csv")\n', "")
    set_code(nb, code_positions[1], src)

    save(nb, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    update_main_slotting()
    update_dqn()
    update_qlearning()
    print("\nAll notebooks updated.")
