from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class InventoryItem:
    """Domain model for a single warehouse inventory record."""

    # --- Identifiers ---
    item_id: str
    storage_location_id: str
    zone: str
    category: str

    # --- Inventory levels ---
    stock_level: float
    reorder_point: float
    reorder_frequency_days: int
    lead_time_days: int

    # --- Demand ---
    daily_demand: float
    demand_std_dev: float
    forecasted_demand_next_7d: float
    total_orders_last_month: int

    # --- Costs & pricing ---
    unit_price: float
    handling_cost_per_unit: float
    holding_cost_per_unit_day: float

    # --- Warehouse operations ---
    picking_time_seconds: float
    item_popularity_score: float

    # --- Performance ---
    stockout_count_last_month: int
    order_fulfillment_rate: float
    turnover_ratio: float
    layout_efficiency_score: float
    kpi_score: float

    # --- Temporal ---
    last_restock_date: Optional[date] = field(default=None)

    # ------------------------------------------------------------------
    # Derived properties — computed on demand, no storage overhead
    # ------------------------------------------------------------------

    @property
    def days_of_cover(self) -> float:
        """Current stock divided by average daily demand."""
        return self.stock_level / (self.daily_demand + 1e-6)

    @property
    def demand_volatility(self) -> float:
        """Coefficient of variation for daily demand."""
        return self.demand_std_dev / (self.daily_demand + 1e-6)

    @property
    def value_score(self) -> float:
        """Unit price × daily demand; proxy for revenue risk."""
        return self.unit_price * self.daily_demand

    @property
    def criticality_score(self) -> float:
        """Composite score weighting cover, volatility, and recent stockouts.

        Mirrors the feature engineering used during RL training.
        """
        recent_flag = int(self.stockout_count_last_month > 0)
        return (
            0.4 * (1.0 / (self.days_of_cover + 1.0))
            + 0.3 * self.demand_volatility
            + 0.3 * recent_flag
        )

    @property
    def forecast_ratio(self) -> float:
        """Forecasted demand for next 7 days vs expected average weekly demand."""
        avg_weekly = self.daily_demand * 7.0
        return self.forecasted_demand_next_7d / (avg_weekly + 1e-6)

    def __repr__(self) -> str:
        return (
            f"InventoryItem(item_id={self.item_id!r}, zone={self.zone!r}, "
            f"category={self.category!r}, stock_level={self.stock_level}, "
            f"days_of_cover={self.days_of_cover:.2f})"
        )
