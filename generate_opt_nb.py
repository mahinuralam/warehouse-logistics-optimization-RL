"""Generate Optimized_Dynamic_Slotting.ipynb from the src package.

Run from the repository root:
    python generate_opt_nb.py
"""
import nbformat as nbf


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()

    code_imports = """\
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

    code_data = """\
df = pd.read_csv(DATA_PATH)

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

print("Data loaded and preprocessed.")
df.head()
"""

    code_train = """\
np.random.seed(42)
random.seed(42)

high_pressure = df[df["zone_load_ratio"] >= 0.75]
if len(high_pressure) < 400:
    slotting_subset = (
        df.sort_values(["zone_load_ratio", "daily_demand"], ascending=[False, False])
        .head(600)
        .reset_index(drop=True)
    )
else:
    slotting_subset = (
        high_pressure.sort_values("daily_demand", ascending=False)
        .head(700)
        .reset_index(drop=True)
    )

env = ZoneSlottingEnv(slotting_subset, zones, zone_capacity, geom)
agent = SlottingQLearner(env.state_space, env.action_space)

episodes = 1000
epsilon = 0.9
epsilon_decay = 0.995
min_epsilon = 0.05
training_log = []

print("Starting training...")
for episode in range(episodes):
    state_idx = env._state_index(env.reset())
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state_idx, epsilon)
        next_state_idx, reward, done, _ = env.step(action)
        agent.update(state_idx, action, reward, next_state_idx, done)
        state_idx = next_state_idx
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes} — reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        training_log.append({"episode": episode + 1, "reward": total_reward})

print("Training complete.")

log_df = pd.DataFrame(training_log)
plt.figure(figsize=(10, 5))
plt.plot(log_df["episode"], log_df["reward"], marker="o")
plt.title("Training reward per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()
"""

    code_eval = """\
def baseline_policy(state_idx: int, env: ZoneSlottingEnv) -> int:
    return env.zones.index(env.current_row["zone"])


def rl_policy(state_idx: int, env: ZoneSlottingEnv) -> int:
    return agent.act(state_idx, 0.0)


def apply_slotting_policy(frame: pd.DataFrame, policy_fn):
    env = ZoneSlottingEnv(frame, zones, zone_capacity, geom)
    state_idx = env._state_index(env.reset())
    records, done = [], False

    while not done:
        action = policy_fn(state_idx, env)
        next_state_idx, reward, done, metrics = env.step(action)
        records.append({
            "item_id": metrics.item_id,
            "old_zone": metrics.old_zone,
            "new_zone": metrics.new_zone,
            "baseline_time": metrics.baseline_time,
            "new_time": metrics.new_time,
            "travel_old": travel_distance(metrics.old_zone),
            "travel_new": metrics.travel,
            "demand": metrics.demand,
            "move_cost": metrics.move_cost,
            "time_gain_seconds": metrics.time_gain_seconds,
            "net_value": metrics.net_value,
        })
        state_idx = next_state_idx

    return pd.DataFrame(records)


def summarize_assignments(assignments: pd.DataFrame) -> dict:
    if assignments.empty:
        return {}
    w = assignments["demand"]
    return {
        "avg_picking_time_new": float(np.average(assignments["new_time"], weights=w)),
        "avg_picking_time_baseline": float(np.average(assignments["baseline_time"], weights=w)),
        "avg_travel_new": float(np.average(assignments["travel_new"], weights=w)),
        "avg_travel_old": float(np.average(assignments["travel_old"], weights=w)),
        "total_time_gain_seconds": float(assignments["time_gain_seconds"].sum()),
        "total_net_value": float(assignments["net_value"].sum()),
        "total_moves": int((assignments["old_zone"] != assignments["new_zone"]).sum()),
    }
"""

    code_compare = """\
print("Running evaluation...")
baseline_df = apply_slotting_policy(df, baseline_policy)
rl_df = apply_slotting_policy(df, rl_policy)

comparison = pd.DataFrame(
    [summarize_assignments(baseline_df), summarize_assignments(rl_df)],
    index=["Baseline", "RL Policy"],
)
print(comparison.T)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

import seaborn as sns
sns.histplot(baseline_df["baseline_time"], color="blue", label="Baseline", kde=True, alpha=0.4, ax=axes[0])
sns.histplot(rl_df["new_time"], color="green", label="RL Policy", kde=True, alpha=0.4, ax=axes[0])
axes[0].set_title("Picking time distribution")
axes[0].set_xlabel("Picking time (seconds)")
axes[0].legend()

sns.histplot(baseline_df["travel_old"], color="blue", label="Baseline", kde=True, alpha=0.4, ax=axes[1])
sns.histplot(rl_df["travel_new"], color="green", label="RL Policy", kde=True, alpha=0.4, ax=axes[1])
axes[1].set_title("Travel distance distribution")
axes[1].set_xlabel("Travel distance (grid units)")
axes[1].legend()

plt.tight_layout()
plt.show()
"""

    cells = [
        nbf.v4.new_markdown_cell("# Optimized Dynamic Slotting with Forecast & Turnover"),
        nbf.v4.new_markdown_cell("## 1. Imports & Setup"),
        nbf.v4.new_code_cell(code_imports),
        nbf.v4.new_markdown_cell("## 2. Data Loading & Zone Geometry"),
        nbf.v4.new_code_cell(code_data),
        nbf.v4.new_markdown_cell("## 3. Training"),
        nbf.v4.new_code_cell(code_train),
        nbf.v4.new_markdown_cell("## 4. Evaluation"),
        nbf.v4.new_code_cell(code_eval),
        nbf.v4.new_markdown_cell("## 5. Results & Visualization"),
        nbf.v4.new_code_cell(code_compare),
    ]
    nb.cells = cells

    with open("Optimized_Dynamic_Slotting.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Optimized_Dynamic_Slotting.ipynb created.")


if __name__ == "__main__":
    create_notebook()
