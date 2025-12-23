# 🏭 Warehouse Logistics RL

Reinforcement learning approaches for warehouse inventory optimization and dynamic slotting.

## 📋 Overview

This project applies RL techniques to two warehouse logistics problems:

1. **Stockout Risk Prioritization** - Q-learning agent for expedite decisions to prevent stockouts
2. **Dynamic Zone Slotting** - DQN agent for SKU relocation to minimize picking time

## 📁 Project Structure

```
├── warhouse-logistics-rl.ipynb          # Tabular Q-learning for stockout prevention
├── DQN implementation.ipynb             # Deep Q-Network for zone assignment
├── Dynamic slotting (zone assignment).ipynb  # Zone slotting baseline & analysis
├── logistics_dataset.csv                # 3,204 warehouse inventory records (2024)
└── documentation                        # Dataset feature descriptions
```

## 🎯 Tasks

### Task 1: Stockout Risk Prioritization
- Simulates stock dynamics with daily demand variability
- Agent decides when to expedite orders (1-day lead time vs standard)
- Balances expedite costs against stockout penalties
- Targets high-value SKUs for protection

### Task 2: Dynamic Zone Slotting (DQN)
- Neural network Q-function with replay buffer & target network
- Relocates SKUs across zones to reduce congestion-adjusted picking time
- Accounts for movement costs vs picking time savings

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `stock_level` | Current inventory quantity |
| `daily_demand` | Average daily demand |
| `lead_time_days` | Delivery time after reorder |
| `zone` | Warehouse storage zone |
| `picking_time_seconds` | Retrieval time |
| `KPI_score` | Target performance metric |

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn torch
```

## 🚀 Usage

```bash
jupyter notebook warhouse-logistics-rl.ipynb  # Q-learning
jupyter notebook "DQN implementation.ipynb"   # Deep RL
```

## 📈 Key Metrics

- Stockout prevention rate
- Expedite cost vs penalty trade-off
- Picking time reduction
- Zone congestion levels
