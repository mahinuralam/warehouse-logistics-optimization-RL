from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants shared by both tabular and DQN environments
# ---------------------------------------------------------------------------

SLOT_VALUE_PER_SECOND: float = 0.22
OVERLOAD_WEIGHT: float = 55.0
UTILIZATION_SOFT_LIMIT: float = 0.92
NEGATIVE_MOVE_PENALTY: float = 25.0
MIN_TIME_GAIN_SECONDS: float = 45.0
REWARD_SCALE: float = 100.0
REWARD_CLIP: float = 50.0


# ---------------------------------------------------------------------------
# Zone geometry
# ---------------------------------------------------------------------------

class ZoneGeometry:
    """Encapsulates the spatial layout of warehouse zones.

    Call :meth:`enrich_dataframe` once to add ``travel_distance``,
    ``zone_demand_total``, ``zone_capacity``, and ``zone_load_ratio``
    columns to a raw inventory DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.zones: List[str] = sorted(df["zone"].unique())
        self.grid_w: int = math.ceil(math.sqrt(len(self.zones)))
        self.grid_h: int = math.ceil(len(self.zones) / self.grid_w)
        self.zone_positions: Dict[str, Tuple[int, int]] = {
            zone: (idx // self.grid_w, idx % self.grid_w)
            for idx, zone in enumerate(self.zones)
        }
        self.staging_point: np.ndarray = np.array(
            [self.grid_h / 2, self.grid_w / 2], dtype=float
        )
        self.zone_totals: Dict[str, float] = (
            df.groupby("zone")["daily_demand"].sum().to_dict()
        )
        self.zone_capacity: Dict[str, float] = {
            z: max(total * 1.15, total + 25)
            for z, total in self.zone_totals.items()
        }

    def travel_distance(self, zone: str) -> float:
        coord = np.array(self.zone_positions.get(zone, (0, 0)), dtype=float)
        return float(np.linalg.norm(coord - self.staging_point))

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial columns to *df* and return the enriched copy."""
        df = df.copy()
        df["travel_distance"] = df["zone"].map(self.travel_distance)
        df["zone_demand_total"] = df["zone"].map(self.zone_totals)
        df["zone_capacity"] = df["zone"].map(self.zone_capacity)
        df["zone_load_ratio"] = df["zone_demand_total"] / df["zone_capacity"]
        return df


# ---------------------------------------------------------------------------
# Shared dataclass + helpers
# ---------------------------------------------------------------------------

@dataclass
class SlottingMetrics:
    baseline_time: float
    new_time: float
    travel: float
    move_cost: float
    reward: float
    demand: float
    layout_score: float
    item_id: str
    old_zone: str
    new_zone: str
    time_gain_seconds: float
    value_gain: float
    net_value: float


def _safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row[col]
        return float(default) if pd.isna(val) else float(val)
    except Exception:
        return float(default)


def predict_picking_time(
    row: pd.Series,
    target_zone: str,
    zone_loads: Dict[str, float],
    capacities: Dict[str, float],
    zone_geometry: ZoneGeometry,
) -> float:
    """Estimate picking time for *row* relocated to *target_zone*."""
    base_time = _safe_get(row, "picking_time_seconds", 60.0)
    load_ratio = zone_loads.get(target_zone, 0.0) / max(capacities.get(target_zone, 1.0), 1.0)
    current_zone = str(row.get("zone", target_zone))
    if target_zone == current_zone:
        travel_factor = _safe_get(row, "travel_distance", 0.0)
    else:
        travel_factor = zone_geometry.travel_distance(target_zone)
    layout_bonus = max(0.0, _safe_get(row, "layout_efficiency_score", 0.5) - 0.5)
    congestion_penalty = 1 + 0.45 * min(load_ratio, 2.0)
    travel_penalty = 8 + 6 * travel_factor
    time = base_time * congestion_penalty + travel_penalty - layout_bonus * 6
    return max(time, base_time * 0.5)


# ---------------------------------------------------------------------------
# Tabular Q-learning environment
# ---------------------------------------------------------------------------

class ZoneSlottingEnv:
    """Discrete-state zone slotting environment for tabular Q-learning."""

    def __init__(
        self,
        frame: pd.DataFrame,
        zones: List[str],
        capacities: Dict[str, float],
        zone_geometry: ZoneGeometry,
        move_cost_base: float = 8.0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.zones = zones
        self.capacities = capacities
        self.geom = zone_geometry
        self.action_space = len(zones)
        self.move_cost_base = move_cost_base
        self.reset()

    def reset(self):
        self.items = self.frame.sample(frac=1.0, random_state=None).reset_index(drop=True)
        self.ptr = 0
        self.zone_loads = {z: self.geom.zone_totals.get(z, 0.0) for z in self.zones}
        self.current_row = self.items.iloc[self.ptr]
        return self._observe(self.current_row)

    def _observe(self, row: pd.Series):
        forecast_bin = int(np.digitize([_safe_get(row, "forecasted_demand_next_7d")], [50, 150, 300])[0])
        turnover_bin = int(np.digitize([_safe_get(row, "turnover_ratio")], [4, 8, 12])[0])
        load_ratio = self.zone_loads.get(str(row.get("zone", "")), 0.0) / max(
            self.capacities.get(str(row.get("zone", "")), 1.0), 1.0
        )
        load_bin = int(np.digitize([load_ratio], [0.6, 0.9, 1.1])[0])
        travel_bin = int(np.digitize([_safe_get(row, "travel_distance")], [1.0, 2.5, 4.0])[0])
        return (forecast_bin, turnover_bin, load_bin, travel_bin)

    def _state_index(self, state_tuple) -> int:
        f, tu, l, tr = state_tuple
        return f + 4 * tu + 16 * l + 64 * tr

    @property
    def state_space(self) -> int:
        return 4 * 4 * 4 * 4

    def step(self, action: int):
        row = self.current_row
        target_zone = self.zones[action]
        old_zone = str(row.get("zone", target_zone))
        demand = _safe_get(row, "daily_demand")

        baseline_time = predict_picking_time(row, old_zone, self.zone_loads, self.capacities, self.geom)

        original_loads = dict(self.zone_loads)
        updated_loads = dict(original_loads)
        updated_loads[old_zone] = max(0.0, updated_loads.get(old_zone, 0.0) - demand)
        updated_loads[target_zone] = updated_loads.get(target_zone, 0.0) + demand

        new_time = predict_picking_time(row, target_zone, updated_loads, self.capacities, self.geom)
        move_cost = 0.0 if target_zone == old_zone else self.move_cost_base * (
            1 + _safe_get(row, "stock_level") / 200
        )

        time_gain_seconds = (baseline_time - new_time) * demand
        value_gain = time_gain_seconds * SLOT_VALUE_PER_SECOND

        capacity_new = max(self.capacities.get(target_zone, updated_loads[target_zone]), 1.0)
        utilization_new = updated_loads[target_zone] / capacity_new
        overload_penalty = OVERLOAD_WEIGHT * max(utilization_new - 1.0, 0.0) * demand
        soft_penalty = 12.0 * max(utilization_new - UTILIZATION_SOFT_LIMIT, 0.0) * demand

        raw_net_value = value_gain - move_cost - overload_penalty - soft_penalty
        move_accepted = raw_net_value > 0 and time_gain_seconds > MIN_TIME_GAIN_SECONDS

        if move_accepted:
            self.zone_loads = updated_loads
            net_value = raw_net_value
            reward = net_value
        else:
            self.zone_loads = original_loads
            target_zone = old_zone
            new_time = baseline_time
            move_cost = 0.0
            time_gain_seconds = 0.0
            value_gain = 0.0
            net_value = 0.0
            reward = raw_net_value - NEGATIVE_MOVE_PENALTY

        metrics = SlottingMetrics(
            baseline_time=baseline_time,
            new_time=new_time,
            travel=self.geom.travel_distance(target_zone),
            move_cost=move_cost,
            reward=reward,
            demand=demand,
            layout_score=_safe_get(row, "layout_efficiency_score"),
            item_id=str(row.get("item_id", "")),
            old_zone=old_zone,
            new_zone=target_zone,
            time_gain_seconds=time_gain_seconds,
            value_gain=value_gain,
            net_value=net_value,
        )

        self.ptr += 1
        done = self.ptr >= len(self.items)
        if not done:
            self.current_row = self.items.iloc[self.ptr]
            next_state = self._observe(self.current_row)
        else:
            next_state = (0, 0, 0, 0)

        return self._state_index(next_state), reward, bool(done), metrics


# ---------------------------------------------------------------------------
# Tabular Q-learning agent
# ---------------------------------------------------------------------------

class SlottingQLearner:
    """Tabular Q-learning agent for zone slotting."""

    def __init__(
        self,
        state_space: int,
        action_space: int,
        alpha: float = 0.12,
        gamma: float = 0.9,
    ) -> None:
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((state_space, action_space))

    def act(self, state_idx: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_space)
        return int(np.argmax(self.q_table[state_idx]))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state_idx]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error


# ---------------------------------------------------------------------------
# DQN environment (continuous feature vector + one-hot zone encoding)
# ---------------------------------------------------------------------------

class ZoneSlottingEnvDQN:
    """Continuous-state zone slotting environment for deep Q-learning."""

    def __init__(
        self,
        frame: pd.DataFrame,
        zones: List[str],
        capacities: Dict[str, float],
        zone_geometry: ZoneGeometry,
        feature_stats: Dict[str, float],
        move_cost_base: float = 8.0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.zones = zones
        self.capacities = capacities
        self.geom = zone_geometry
        self.action_space = len(zones)
        self.move_cost_base = float(move_cost_base)
        self.feature_stats = feature_stats
        self.zone_to_idx = {z: i for i, z in enumerate(self.zones)}
        self.reset()

    def reset(self) -> List[float]:
        self.items = self.frame.sample(frac=1.0, random_state=None).reset_index(drop=True)
        self.ptr = 0
        self.zone_loads = {z: float(self.geom.zone_totals.get(z, 0.0)) for z in self.zones}
        self.current_row = self.items.iloc[self.ptr]
        return self._observe(self.current_row)

    def _observe(self, row: pd.Series) -> List[float]:
        max_demand = max(self.feature_stats.get("max_demand", 1.0), 1.0)
        max_forecast = max(self.feature_stats.get("max_forecast", 1.0), 1.0)
        max_turnover = max(self.feature_stats.get("max_turnover", 1.0), 1.0)
        max_stock = max(self.feature_stats.get("max_stock", 1.0), 1.0)
        max_travel = max(self.feature_stats.get("max_travel", 1.0), 1.0)

        demand = _safe_get(row, "daily_demand")
        forecast = _safe_get(row, "forecasted_demand_next_7d", demand)
        turnover = _safe_get(row, "turnover_ratio")
        stock = _safe_get(row, "stock_level")
        layout = _safe_get(row, "layout_efficiency_score", 0.5)
        travel = _safe_get(row, "travel_distance")
        zone = str(row.get("zone", self.zones[0]))
        util = self.zone_loads.get(zone, 0.0) / max(self.capacities.get(zone, 1.0), 1.0)

        demand_n = float(np.log1p(demand) / np.log1p(max_demand))
        forecast_n = float(np.log1p(forecast) / np.log1p(max_forecast))
        turnover_n = float(np.clip(turnover / max_turnover, 0.0, 2.0) / 2.0)
        stock_n = float(np.clip(stock / max_stock, 0.0, 2.0) / 2.0)
        travel_n = float(np.clip(travel / max_travel, 0.0, 1.5) / 1.5)
        layout_n = float(np.clip(layout, 0.0, 1.2) / 1.2)
        util_n = float(np.clip(util, 0.0, 2.0) / 2.0)

        one_hot = [0.0] * self.action_space
        one_hot[self.zone_to_idx.get(zone, 0)] = 1.0
        return [forecast_n, turnover_n, util_n, travel_n, stock_n, layout_n, demand_n] + one_hot

    @property
    def state_dim(self) -> int:
        return 7 + self.action_space

    def _scale_reward(self, raw: float) -> float:
        return float(np.clip(raw / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))

    def step(self, action: int):
        row = self.current_row
        action = int(action)
        target_zone = self.zones[action]
        old_zone = str(row.get("zone", target_zone))
        demand = _safe_get(row, "daily_demand")
        baseline_time = predict_picking_time(row, old_zone, self.zone_loads, self.capacities, self.geom)

        if target_zone == old_zone:
            metrics = SlottingMetrics(
                baseline_time=baseline_time,
                new_time=baseline_time,
                travel=self.geom.travel_distance(old_zone),
                move_cost=0.0,
                reward=0.0,
                demand=demand,
                layout_score=_safe_get(row, "layout_efficiency_score", 0.5),
                item_id=str(row.get("item_id", "")),
                old_zone=old_zone,
                new_zone=old_zone,
                time_gain_seconds=0.0,
                value_gain=0.0,
                net_value=0.0,
            )
            self.ptr += 1
            done = self.ptr >= len(self.items)
            if not done:
                self.current_row = self.items.iloc[self.ptr]
                next_state = self._observe(self.current_row)
            else:
                next_state = [0.0] * self.state_dim
            return next_state, 0.0, bool(done), metrics

        original_loads = dict(self.zone_loads)
        updated_loads = dict(original_loads)
        updated_loads[old_zone] = max(0.0, updated_loads.get(old_zone, 0.0) - demand)
        updated_loads[target_zone] = updated_loads.get(target_zone, 0.0) + demand

        new_time = predict_picking_time(row, target_zone, updated_loads, self.capacities, self.geom)
        stock_level = _safe_get(row, "stock_level")
        move_cost = self.move_cost_base * (1 + stock_level / 200.0)

        time_gain_seconds = (baseline_time - new_time) * demand
        value_gain = time_gain_seconds * SLOT_VALUE_PER_SECOND

        capacity_new = max(self.capacities.get(target_zone, updated_loads[target_zone]), 1.0)
        utilization_new = updated_loads[target_zone] / capacity_new
        overload_penalty = OVERLOAD_WEIGHT * max(utilization_new - 1.0, 0.0) * demand
        soft_penalty = 12.0 * max(utilization_new - UTILIZATION_SOFT_LIMIT, 0.0) * demand

        raw_net_value = value_gain - move_cost - overload_penalty - soft_penalty
        move_accepted = raw_net_value > 0 and time_gain_seconds > MIN_TIME_GAIN_SECONDS

        if move_accepted:
            self.zone_loads = updated_loads
            net_value = raw_net_value
        else:
            self.zone_loads = original_loads
            target_zone = old_zone
            new_time = baseline_time
            move_cost = 0.0
            time_gain_seconds = 0.0
            value_gain = 0.0
            net_value = 0.0
            raw_net_value = -NEGATIVE_MOVE_PENALTY

        reward = self._scale_reward(raw_net_value if move_accepted else raw_net_value)

        metrics = SlottingMetrics(
            baseline_time=baseline_time,
            new_time=new_time,
            travel=self.geom.travel_distance(target_zone),
            move_cost=move_cost,
            reward=reward,
            demand=demand,
            layout_score=_safe_get(row, "layout_efficiency_score", 0.5),
            item_id=str(row.get("item_id", "")),
            old_zone=old_zone,
            new_zone=target_zone,
            time_gain_seconds=time_gain_seconds,
            value_gain=value_gain,
            net_value=net_value,
        )

        self.ptr += 1
        done = self.ptr >= len(self.items)
        if not done:
            self.current_row = self.items.iloc[self.ptr]
            next_state = self._observe(self.current_row)
        else:
            next_state = [0.0] * self.state_dim

        return next_state, float(reward), bool(done), metrics
