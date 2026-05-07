from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..models import InventoryItem
from .base import IInventoryRepository


def _parse_date(value: object) -> Optional[date]:
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _row_to_item(row: pd.Series) -> InventoryItem:
    return InventoryItem(
        item_id=str(row["item_id"]),
        storage_location_id=str(row["storage_location_id"]),
        zone=str(row["zone"]),
        category=str(row["category"]),
        stock_level=float(row["stock_level"]),
        reorder_point=float(row["reorder_point"]),
        reorder_frequency_days=int(row["reorder_frequency_days"]),
        lead_time_days=int(row["lead_time_days"]),
        daily_demand=float(row["daily_demand"]),
        demand_std_dev=float(row["demand_std_dev"]),
        forecasted_demand_next_7d=float(row["forecasted_demand_next_7d"]),
        total_orders_last_month=int(row["total_orders_last_month"]),
        unit_price=float(row["unit_price"]),
        handling_cost_per_unit=float(row["handling_cost_per_unit"]),
        holding_cost_per_unit_day=float(row["holding_cost_per_unit_day"]),
        picking_time_seconds=float(row["picking_time_seconds"]),
        item_popularity_score=float(row["item_popularity_score"]),
        stockout_count_last_month=int(row["stockout_count_last_month"]),
        order_fulfillment_rate=float(row["order_fulfillment_rate"]),
        turnover_ratio=float(row["turnover_ratio"]),
        layout_efficiency_score=float(row["layout_efficiency_score"]),
        kpi_score=float(row["KPI_score"]),
        last_restock_date=_parse_date(row.get("last_restock_date")),
    )


class CSVInventoryRepository(IInventoryRepository):
    """Reads inventory data from a CSV file and exposes it through the
    :class:`IInventoryRepository` interface.

    The file is loaded once on construction and cached in memory.  Pass
    *eager=False* to defer loading until the first query.
    """

    def __init__(self, csv_path: Path | str, *, eager: bool = True) -> None:
        self._path = Path(csv_path)
        self._cache: Optional[List[InventoryItem]] = None
        self._index: Optional[Dict[str, InventoryItem]] = None
        if eager:
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        df = pd.read_csv(self._path)
        self._cache = [_row_to_item(row) for _, row in df.iterrows()]
        self._index = {item.item_id: item for item in self._cache}

    def _items(self) -> List[InventoryItem]:
        if self._cache is None:
            self._load()
        return self._cache  # type: ignore[return-value]

    def _idx(self) -> Dict[str, InventoryItem]:
        if self._index is None:
            self._load()
        return self._index  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # IInventoryRepository implementation
    # ------------------------------------------------------------------

    def get_all(self) -> List[InventoryItem]:
        return list(self._items())

    def get_by_id(self, item_id: str) -> Optional[InventoryItem]:
        return self._idx().get(item_id)

    def get_by_zone(self, zone: str) -> List[InventoryItem]:
        return [i for i in self._items() if i.zone == zone]

    def get_by_category(self, category: str) -> List[InventoryItem]:
        return [i for i in self._items() if i.category == category]

    def get_below_reorder_point(self) -> List[InventoryItem]:
        return [i for i in self._items() if i.stock_level <= i.reorder_point]

    def get_top_by_criticality(self, n: int) -> List[InventoryItem]:
        return sorted(self._items(), key=lambda i: i.criticality_score, reverse=True)[:n]

    def get_top_by_value(self, n: int) -> List[InventoryItem]:
        return sorted(self._items(), key=lambda i: i.value_score, reverse=True)[:n]

    def count(self) -> int:
        return len(self._items())

    def distinct_zones(self) -> List[str]:
        return sorted({i.zone for i in self._items()})

    def distinct_categories(self) -> List[str]:
        return sorted({i.category for i in self._items()})

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read the CSV from disk, replacing the in-memory cache."""
        self._cache = None
        self._index = None
        self._load()

    def __repr__(self) -> str:
        loaded = self._cache is not None
        return f"CSVInventoryRepository(path={self._path!r}, loaded={loaded})"
