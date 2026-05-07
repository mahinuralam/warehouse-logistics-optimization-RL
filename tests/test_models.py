"""Unit tests for the InventoryItem domain model."""
from __future__ import annotations

import pytest

from tests.conftest import ITEM_A, ITEM_B, ITEM_C


class TestDerivedProperties:
    def test_days_of_cover(self):
        # 200 stock / 20 demand ≈ 10
        assert abs(ITEM_A.days_of_cover - 10.0) < 0.01

    def test_days_of_cover_low_stock(self):
        # 10 stock / 40 demand ≈ 0.25
        assert abs(ITEM_B.days_of_cover - 0.25) < 0.01

    def test_demand_volatility(self):
        # std/mean = 2/20 = 0.1
        assert abs(ITEM_A.demand_volatility - 0.1) < 0.001

    def test_value_score(self):
        # 100 * 20 = 2000
        assert abs(ITEM_A.value_score - 2000.0) < 0.01

    def test_criticality_score_high_stockout(self):
        # ITEM_B has stockout_count > 0, so criticality should be higher
        assert ITEM_B.criticality_score > ITEM_A.criticality_score

    def test_criticality_score_range(self):
        for item in [ITEM_A, ITEM_B, ITEM_C]:
            assert 0.0 <= item.criticality_score <= 1.0

    def test_forecast_ratio(self):
        # ITEM_A: forecasted=140, avg_weekly=140 → ratio ≈ 1.0
        assert abs(ITEM_A.forecast_ratio - 1.0) < 0.01

    def test_repr_contains_item_id(self):
        assert "ITM00001" in repr(ITEM_A)


class TestEdgeCases:
    def test_zero_demand_no_division_error(self):
        from src.models import InventoryItem

        zero_demand_item = InventoryItem(
            item_id="ITM99999",
            storage_location_id="L99",
            zone="Z",
            category="Test",
            stock_level=100.0,
            reorder_point=10.0,
            reorder_frequency_days=1,
            lead_time_days=1,
            daily_demand=0.0,
            demand_std_dev=0.0,
            forecasted_demand_next_7d=0.0,
            total_orders_last_month=0,
            unit_price=1.0,
            handling_cost_per_unit=0.1,
            holding_cost_per_unit_day=0.01,
            picking_time_seconds=1.0,
            item_popularity_score=0.0,
            stockout_count_last_month=0,
            order_fulfillment_rate=1.0,
            turnover_ratio=0.0,
            layout_efficiency_score=1.0,
            kpi_score=0.5,
        )
        # No ZeroDivisionError — epsilon guard in place
        _ = zero_demand_item.days_of_cover
        _ = zero_demand_item.demand_volatility
        _ = zero_demand_item.forecast_ratio
