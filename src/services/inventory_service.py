from __future__ import annotations

from typing import Dict, List, Optional

from ..models import InventoryItem
from ..repositories import IInventoryRepository


class InventoryService:
    """Orchestrates inventory queries and business logic.

    All domain operations that span multiple repository calls, require
    derived metrics, or enforce business rules live here — keeping both
    the repository (pure data access) and the notebooks (presentation)
    free of cross-cutting concerns.
    """

    def __init__(self, repository: IInventoryRepository) -> None:
        self._repo = repository

    # ------------------------------------------------------------------
    # Pass-through queries
    # ------------------------------------------------------------------

    def get_all(self) -> List[InventoryItem]:
        return self._repo.get_all()

    def get_by_id(self, item_id: str) -> Optional[InventoryItem]:
        return self._repo.get_by_id(item_id)

    def get_by_zone(self, zone: str) -> List[InventoryItem]:
        return self._repo.get_by_zone(zone)

    def get_by_category(self, category: str) -> List[InventoryItem]:
        return self._repo.get_by_category(category)

    # ------------------------------------------------------------------
    # Reorder & stockout risk
    # ------------------------------------------------------------------

    def get_reorder_candidates(self) -> List[InventoryItem]:
        """Items at or below their reorder point, sorted by criticality."""
        items = self._repo.get_below_reorder_point()
        return sorted(items, key=lambda i: i.criticality_score, reverse=True)

    def get_stockout_risk_items(self, days_of_cover_threshold: float = 3.0) -> List[InventoryItem]:
        """Items with fewer than *days_of_cover_threshold* days of remaining stock."""
        return [
            i for i in self._repo.get_all()
            if i.days_of_cover < days_of_cover_threshold
        ]

    def get_high_volatility_items(self, volatility_threshold: float = 0.3) -> List[InventoryItem]:
        """Items whose demand coefficient of variation exceeds the threshold."""
        return [
            i for i in self._repo.get_all()
            if i.demand_volatility > volatility_threshold
        ]

    # ------------------------------------------------------------------
    # Prioritisation for RL / expedite decisions
    # ------------------------------------------------------------------

    def get_top_critical_items(self, n: int = 100) -> List[InventoryItem]:
        """Top *n* items ranked by criticality score (used to seed RL training)."""
        return self._repo.get_top_by_criticality(n)

    def get_high_value_items(self, percentile: float = 0.9) -> List[InventoryItem]:
        """Items in the top (1 - *percentile*) of the value-score distribution."""
        all_items = self._repo.get_all()
        if not all_items:
            return []
        scores = sorted(i.value_score for i in all_items)
        cutoff_idx = int(len(scores) * percentile)
        threshold = scores[cutoff_idx]
        return [i for i in all_items if i.value_score >= threshold]

    def get_expedite_candidates(
        self,
        days_of_cover_threshold: float = 2.0,
        criticality_threshold: float = 0.2,
    ) -> List[InventoryItem]:
        """Items that are both low on stock and highly critical — prime candidates
        for an expedited reorder in the RL simulation."""
        return [
            i for i in self._repo.get_all()
            if i.days_of_cover < days_of_cover_threshold
            and i.criticality_score > criticality_threshold
        ]

    # ------------------------------------------------------------------
    # Zone-level analytics
    # ------------------------------------------------------------------

    def zone_summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate KPIs grouped by warehouse zone."""
        result: Dict[str, Dict[str, float]] = {}
        for zone in self._repo.distinct_zones():
            items = self._repo.get_by_zone(zone)
            if not items:
                continue
            n = len(items)
            result[zone] = {
                "item_count": n,
                "avg_stock_level": sum(i.stock_level for i in items) / n,
                "avg_daily_demand": sum(i.daily_demand for i in items) / n,
                "avg_days_of_cover": sum(i.days_of_cover for i in items) / n,
                "total_value_score": sum(i.value_score for i in items),
                "avg_kpi_score": sum(i.kpi_score for i in items) / n,
                "reorder_count": sum(1 for i in items if i.stock_level <= i.reorder_point),
                "stockout_count": sum(i.stockout_count_last_month for i in items),
            }
        return result

    def category_summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate KPIs grouped by item category."""
        result: Dict[str, Dict[str, float]] = {}
        for cat in self._repo.distinct_categories():
            items = self._repo.get_by_category(cat)
            if not items:
                continue
            n = len(items)
            result[cat] = {
                "item_count": n,
                "avg_criticality_score": sum(i.criticality_score for i in items) / n,
                "avg_value_score": sum(i.value_score for i in items) / n,
                "avg_kpi_score": sum(i.kpi_score for i in items) / n,
                "total_stockouts_last_month": sum(i.stockout_count_last_month for i in items),
                "avg_order_fulfillment_rate": sum(i.order_fulfillment_rate for i in items) / n,
            }
        return result

    # ------------------------------------------------------------------
    # Dataset stats
    # ------------------------------------------------------------------

    def summary_stats(self) -> Dict[str, object]:
        """High-level statistics about the loaded dataset."""
        items = self._repo.get_all()
        n = len(items)
        if n == 0:
            return {"total_items": 0}
        return {
            "total_items": n,
            "zones": self._repo.distinct_zones(),
            "categories": self._repo.distinct_categories(),
            "reorder_candidates": len(self._repo.get_below_reorder_point()),
            "avg_kpi_score": round(sum(i.kpi_score for i in items) / n, 4),
            "avg_days_of_cover": round(sum(i.days_of_cover for i in items) / n, 2),
            "avg_criticality_score": round(sum(i.criticality_score for i in items) / n, 4),
            "total_stockouts_last_month": sum(i.stockout_count_last_month for i in items),
        }
