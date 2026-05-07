"""Shared fixtures for the test suite."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.models import InventoryItem
from src.repositories import CSVInventoryRepository
from src.services import InventoryService

# ---------------------------------------------------------------------------
# Minimal in-memory items used across all test modules
# ---------------------------------------------------------------------------

ITEM_A = InventoryItem(
    item_id="ITM00001",
    storage_location_id="L01",
    zone="A",
    category="Pharma",
    stock_level=200.0,
    reorder_point=50.0,
    reorder_frequency_days=7,
    lead_time_days=3,
    daily_demand=20.0,
    demand_std_dev=2.0,
    forecasted_demand_next_7d=140.0,
    total_orders_last_month=600,
    unit_price=100.0,
    handling_cost_per_unit=2.0,
    holding_cost_per_unit_day=0.5,
    picking_time_seconds=30.0,
    item_popularity_score=0.8,
    stockout_count_last_month=0,
    order_fulfillment_rate=0.95,
    turnover_ratio=5.0,
    layout_efficiency_score=0.9,
    kpi_score=0.75,
)

ITEM_B = InventoryItem(
    item_id="ITM00002",
    storage_location_id="L02",
    zone="B",
    category="Automotive",
    stock_level=10.0,
    reorder_point=50.0,   # below reorder point
    reorder_frequency_days=14,
    lead_time_days=5,
    daily_demand=40.0,
    demand_std_dev=8.0,
    forecasted_demand_next_7d=300.0,
    total_orders_last_month=900,
    unit_price=200.0,
    handling_cost_per_unit=5.0,
    holding_cost_per_unit_day=1.5,
    picking_time_seconds=60.0,
    item_popularity_score=0.4,
    stockout_count_last_month=5,
    order_fulfillment_rate=0.60,
    turnover_ratio=12.0,
    layout_efficiency_score=0.5,
    kpi_score=0.45,
)

ITEM_C = InventoryItem(
    item_id="ITM00003",
    storage_location_id="L03",
    zone="A",
    category="Groceries",
    stock_level=500.0,
    reorder_point=30.0,
    reorder_frequency_days=3,
    lead_time_days=2,
    daily_demand=5.0,
    demand_std_dev=0.5,
    forecasted_demand_next_7d=35.0,
    total_orders_last_month=150,
    unit_price=10.0,
    handling_cost_per_unit=0.5,
    holding_cost_per_unit_day=0.1,
    picking_time_seconds=15.0,
    item_popularity_score=0.9,
    stockout_count_last_month=0,
    order_fulfillment_rate=0.99,
    turnover_ratio=2.0,
    layout_efficiency_score=0.95,
    kpi_score=0.88,
)

ALL_ITEMS = [ITEM_A, ITEM_B, ITEM_C]


# ---------------------------------------------------------------------------
# Temporary CSV fixture
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "item_id", "storage_location_id", "zone", "category",
    "stock_level", "reorder_point", "reorder_frequency_days", "lead_time_days",
    "daily_demand", "demand_std_dev", "forecasted_demand_next_7d",
    "total_orders_last_month", "unit_price", "handling_cost_per_unit",
    "holding_cost_per_unit_day", "picking_time_seconds", "item_popularity_score",
    "stockout_count_last_month", "order_fulfillment_rate", "turnover_ratio",
    "layout_efficiency_score", "KPI_score", "last_restock_date",
]


def _item_to_row(item: InventoryItem) -> dict:
    return {
        "item_id": item.item_id,
        "storage_location_id": item.storage_location_id,
        "zone": item.zone,
        "category": item.category,
        "stock_level": item.stock_level,
        "reorder_point": item.reorder_point,
        "reorder_frequency_days": item.reorder_frequency_days,
        "lead_time_days": item.lead_time_days,
        "daily_demand": item.daily_demand,
        "demand_std_dev": item.demand_std_dev,
        "forecasted_demand_next_7d": item.forecasted_demand_next_7d,
        "total_orders_last_month": item.total_orders_last_month,
        "unit_price": item.unit_price,
        "handling_cost_per_unit": item.handling_cost_per_unit,
        "holding_cost_per_unit_day": item.holding_cost_per_unit_day,
        "picking_time_seconds": item.picking_time_seconds,
        "item_popularity_score": item.item_popularity_score,
        "stockout_count_last_month": item.stockout_count_last_month,
        "order_fulfillment_rate": item.order_fulfillment_rate,
        "turnover_ratio": item.turnover_ratio,
        "layout_efficiency_score": item.layout_efficiency_score,
        "KPI_score": item.kpi_score,
        "last_restock_date": "2024-01-15" if item.last_restock_date is None else str(item.last_restock_date),
    }


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Write the three test items to a temporary CSV and return its path."""
    csv_file = tmp_path / "test_inventory.csv"
    with csv_file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in ALL_ITEMS:
            writer.writerow(_item_to_row(item))
    return csv_file


@pytest.fixture
def csv_repo(tmp_csv: Path) -> CSVInventoryRepository:
    return CSVInventoryRepository(tmp_csv)


@pytest.fixture
def service(csv_repo: CSVInventoryRepository) -> InventoryService:
    return InventoryService(csv_repo)
