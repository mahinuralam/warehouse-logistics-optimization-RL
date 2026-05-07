"""Unit tests for InventoryService."""
from __future__ import annotations

import pytest


class TestPassThroughQueries:
    def test_get_all(self, service):
        assert len(service.get_all()) == 3

    def test_get_by_id_found(self, service):
        item = service.get_by_id("ITM00001")
        assert item is not None
        assert item.item_id == "ITM00001"

    def test_get_by_id_not_found(self, service):
        assert service.get_by_id("NOPE") is None

    def test_get_by_zone(self, service):
        assert len(service.get_by_zone("A")) == 2

    def test_get_by_category(self, service):
        assert len(service.get_by_category("Automotive")) == 1


class TestReorderAndRisk:
    def test_reorder_candidates_sorted_by_criticality(self, service):
        candidates = service.get_reorder_candidates()
        # Only ITEM_B is below its reorder point
        assert len(candidates) == 1
        assert candidates[0].item_id == "ITM00002"

    def test_stockout_risk_low_threshold(self, service):
        # ITEM_B: days_of_cover ≈ 0.25, so it should appear at threshold=1
        risky = service.get_stockout_risk_items(days_of_cover_threshold=1.0)
        ids = [i.item_id for i in risky]
        assert "ITM00002" in ids
        assert "ITM00001" not in ids

    def test_stockout_risk_high_threshold_returns_all(self, service):
        risky = service.get_stockout_risk_items(days_of_cover_threshold=1000.0)
        assert len(risky) == 3

    def test_high_volatility_items(self, service):
        # ITEM_B: demand_std_dev=8, daily_demand=40 → volatility=0.2 < 0.3
        # ITEM_A: volatility=0.1, ITEM_C: 0.5/5=0.1 — none exceed 0.3 by default
        low_vol = service.get_high_volatility_items(volatility_threshold=0.3)
        # ITEM_B volatility = 8/40 = 0.2, below threshold
        assert "ITM00002" not in [i.item_id for i in low_vol]

    def test_high_volatility_items_low_threshold(self, service):
        all_volatile = service.get_high_volatility_items(volatility_threshold=0.0)
        assert len(all_volatile) == 3


class TestPrioritisation:
    def test_get_top_critical_items(self, service):
        top = service.get_top_critical_items(n=2)
        assert len(top) == 2
        # Should be sorted descending by criticality
        assert top[0].criticality_score >= top[1].criticality_score

    def test_get_high_value_items(self, service):
        # ITEM_B: value_score = 200*40 = 8000 — highest
        high_val = service.get_high_value_items(percentile=0.9)
        ids = [i.item_id for i in high_val]
        assert "ITM00002" in ids

    def test_get_expedite_candidates(self, service):
        # ITEM_B: days_of_cover ≈ 0.25, criticality > 0
        candidates = service.get_expedite_candidates(
            days_of_cover_threshold=1.0,
            criticality_threshold=0.0,
        )
        assert any(i.item_id == "ITM00002" for i in candidates)


class TestZoneSummary:
    def test_zone_keys_match_distinct_zones(self, service):
        summary = service.zone_summary()
        assert set(summary.keys()) == {"A", "B"}

    def test_zone_a_item_count(self, service):
        assert service.zone_summary()["A"]["item_count"] == 2

    def test_zone_b_has_reorder(self, service):
        assert service.zone_summary()["B"]["reorder_count"] == 1


class TestCategorySummary:
    def test_category_keys_present(self, service):
        summary = service.category_summary()
        assert "Pharma" in summary
        assert "Automotive" in summary
        assert "Groceries" in summary

    def test_automotive_stockout_count(self, service):
        assert service.category_summary()["Automotive"]["total_stockouts_last_month"] == 5


class TestSummaryStats:
    def test_total_items(self, service):
        stats = service.summary_stats()
        assert stats["total_items"] == 3

    def test_reorder_candidates_count(self, service):
        assert service.summary_stats()["reorder_candidates"] == 1

    def test_avg_kpi_score_is_float(self, service):
        assert isinstance(service.summary_stats()["avg_kpi_score"], float)

    def test_empty_repo(self):
        from unittest.mock import MagicMock
        from src.services import InventoryService

        empty_repo = MagicMock()
        empty_repo.get_all.return_value = []
        svc = InventoryService(empty_repo)
        assert svc.summary_stats() == {"total_items": 0}
