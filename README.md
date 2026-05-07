# Warehouse Logistics RL

Reinforcement learning approaches for warehouse inventory optimization and dynamic zone slotting.

## Overview

This project applies RL techniques to two warehouse logistics problems:

1. **Stockout Risk Prioritization** — tabular Q-learning agent that decides when to expedite reorders, trading premium shipping cost against stockout penalties.
2. **Dynamic Zone Slotting** — both a tabular Q-learner and a BiLSTM DQN that relocate SKUs across warehouse zones to minimize congestion-adjusted picking time.

## Project Structure

```
├── src/                                     # Importable Python package
│   ├── config.py                            # Paths, RL hyperparameters, thresholds
│   ├── models/
│   │   └── inventory.py                     # InventoryItem dataclass (23 fields + derived properties)
│   ├── repositories/
│   │   ├── base.py                          # IInventoryRepository interface
│   │   └── csv_repository.py               # CSV-backed implementation
│   ├── services/
│   │   └── inventory_service.py            # Business logic (risk queries, zone summaries)
│   └── environments/
│       └── zone_slotting.py                # ZoneGeometry, ZoneSlottingEnv, ZoneSlottingEnvDQN,
│                                           #   SlottingQLearner, SlottingMetrics, predict_picking_time
│
├── warhouse-logistics-rl.ipynb             # Tabular Q-learning for stockout prevention
├── DQN implementation.ipynb               # BiLSTM DQN for zone assignment
├── Dynamic slotting (zone assignment).ipynb  # Tabular RL zone slotting
├── generate_opt_nb.py                      # Generates Optimized_Dynamic_Slotting.ipynb
│
├── tests/                                  # pytest suite (52 tests)
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_csv_repository.py
│   └── test_inventory_service.py
│
├── scripts/
│   └── update_notebooks.py                # Utility: patches notebook cells to use src
│
├── logistics_dataset.csv                   # 3,204 warehouse inventory records (2024)
├── documentation                           # Dataset feature descriptions
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Architecture

### Repository Pattern (`src/`)

All data access flows through a typed interface so notebooks, services, and tests are decoupled from the CSV file:

```python
from src.repositories import CSVInventoryRepository
from src.services import InventoryService
from src.config import DATA_PATH

repo    = CSVInventoryRepository(DATA_PATH)
service = InventoryService(repo)

# High-level queries
candidates = service.get_expedite_candidates(days_of_cover_threshold=2.0)
summary    = service.zone_summary()
stats      = service.summary_stats()
```

### Domain Model (`src/models/inventory.py`)

`InventoryItem` is a plain dataclass covering all 23 dataset columns. Derived metrics are computed on-demand as `@property` methods — no duplication across notebooks:

| Property | Formula |
|---|---|
| `days_of_cover` | `stock_level / daily_demand` |
| `demand_volatility` | `demand_std_dev / daily_demand` |
| `value_score` | `unit_price × daily_demand` |
| `criticality_score` | weighted combination of cover, volatility, stockout history |
| `forecast_ratio` | `forecasted_demand_next_7d / (daily_demand × 7)` |

### Shared RL Environments (`src/environments/zone_slotting.py`)

Eliminates the copy-pasted `SlottingMetrics`, `ZoneSlottingEnv`, `SlottingQLearner`, `ZoneSlottingEnvDQN`, zone-geometry helpers, and constants that previously appeared verbatim in every notebook:

```python
from src.environments.zone_slotting import (
    ZoneGeometry, ZoneSlottingEnv, SlottingQLearner,
    ZoneSlottingEnvDQN, SlottingMetrics, predict_picking_time,
)

geom = ZoneGeometry(df)          # computes grid, travel distances, capacities
df   = geom.enrich_dataframe(df) # adds travel_distance, zone_load_ratio, …
env  = ZoneSlottingEnv(subset, geom.zones, geom.zone_capacity, geom)
```

## Dataset

3,204 warehouse inventory records from 2024 with 23 features:

| Category | Columns |
|---|---|
| Identifiers | `item_id`, `storage_location_id`, `zone`, `category` |
| Inventory | `stock_level`, `reorder_point`, `reorder_frequency_days`, `lead_time_days` |
| Demand | `daily_demand`, `demand_std_dev`, `forecasted_demand_next_7d`, `total_orders_last_month` |
| Costs | `unit_price`, `handling_cost_per_unit`, `holding_cost_per_unit_day` |
| Operations | `picking_time_seconds`, `item_popularity_score` |
| Performance | `stockout_count_last_month`, `order_fulfillment_rate`, `turnover_ratio`, `layout_efficiency_score`, `KPI_score` |
| Temporal | `last_restock_date` |

## Setup

```bash
pip install -r requirements-dev.txt   # includes pytest
```

Override the default data path via environment variable:

```bash
DATA_PATH=/path/to/custom.csv jupyter notebook warhouse-logistics-rl.ipynb
```

## Running Notebooks

```bash
jupyter notebook warhouse-logistics-rl.ipynb         # Q-learning stockout agent
jupyter notebook "DQN implementation.ipynb"          # BiLSTM DQN zone slotter
jupyter notebook "Dynamic slotting (zone assignment).ipynb"  # tabular zone slotter
python generate_opt_nb.py && jupyter notebook Optimized_Dynamic_Slotting.ipynb
```

## Tests

```bash
pytest tests/ -v   # 52 tests — models, repository, service
```

## Key Results

| Metric | Value |
|---|---|
| Stockout reduction (Q-learning) | ~73 % |
| Avoided penalty vs expedite cost ratio | ~6× |
| Expedite precision (prevented / triggered) | ~2.4 |
