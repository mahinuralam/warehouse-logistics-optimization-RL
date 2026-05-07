"""Unit tests for CSVInventoryRepository."""
from __future__ import annotations

import pytest

from src.repositories import CSVInventoryRepository
from tests.conftest import ITEM_A, ITEM_B, ITEM_C


class TestLoading:
    def test_eager_load_count(self, csv_repo):
        assert csv_repo.count() == 3

    def test_lazy_load(self, tmp_csv):
        repo = CSVInventoryRepository(tmp_csv, eager=False)
        # Cache not populated yet
        assert repo._cache is None
        # Triggers load
        assert repo.count() == 3
        assert repo._cache is not None

    def test_reload(self, csv_repo):
        csv_repo.reload()
        assert csv_repo.count() == 3

    def test_repr(self, csv_repo):
        assert "CSVInventoryRepository" in repr(csv_repo)
        assert "loaded=True" in repr(csv_repo)


class TestGetAll:
    def test_returns_all_items(self, csv_repo):
        items = csv_repo.get_all()
        assert len(items) == 3

    def test_returns_inventory_items(self, csv_repo):
        from src.models import InventoryItem
        for item in csv_repo.get_all():
            assert isinstance(item, InventoryItem)


class TestGetById:
    def test_existing_id(self, csv_repo):
        item = csv_repo.get_by_id("ITM00001")
        assert item is not None
        assert item.item_id == "ITM00001"

    def test_missing_id_returns_none(self, csv_repo):
        assert csv_repo.get_by_id("DOES_NOT_EXIST") is None


class TestGetByZone:
    def test_zone_a_has_two_items(self, csv_repo):
        items = csv_repo.get_by_zone("A")
        assert len(items) == 2

    def test_zone_b_has_one_item(self, csv_repo):
        items = csv_repo.get_by_zone("B")
        assert len(items) == 1
        assert items[0].item_id == "ITM00002"

    def test_unknown_zone_returns_empty(self, csv_repo):
        assert csv_repo.get_by_zone("Z") == []


class TestGetByCategory:
    def test_pharma_category(self, csv_repo):
        items = csv_repo.get_by_category("Pharma")
        assert len(items) == 1
        assert items[0].item_id == "ITM00001"

    def test_unknown_category_returns_empty(self, csv_repo):
        assert csv_repo.get_by_category("Unknown") == []


class TestBelowReorderPoint:
    def test_item_b_is_below_reorder_point(self, csv_repo):
        # ITEM_B: stock_level=10, reorder_point=50
        items = csv_repo.get_below_reorder_point()
        ids = [i.item_id for i in items]
        assert "ITM00002" in ids

    def test_items_a_and_c_not_below_reorder_point(self, csv_repo):
        items = csv_repo.get_below_reorder_point()
        ids = [i.item_id for i in items]
        assert "ITM00001" not in ids
        assert "ITM00003" not in ids


class TestTopQueries:
    def test_top_by_criticality_respects_n(self, csv_repo):
        top2 = csv_repo.get_top_by_criticality(2)
        assert len(top2) == 2

    def test_top_by_criticality_ordering(self, csv_repo):
        items = csv_repo.get_top_by_criticality(3)
        scores = [i.criticality_score for i in items]
        assert scores == sorted(scores, reverse=True)

    def test_top_by_value_ordering(self, csv_repo):
        items = csv_repo.get_top_by_value(3)
        scores = [i.value_score for i in items]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_exceeds_total_returns_all(self, csv_repo):
        assert len(csv_repo.get_top_by_criticality(999)) == 3


class TestAggregates:
    def test_distinct_zones(self, csv_repo):
        zones = csv_repo.distinct_zones()
        assert zones == ["A", "B"]

    def test_distinct_categories(self, csv_repo):
        cats = csv_repo.distinct_categories()
        assert "Pharma" in cats
        assert "Automotive" in cats
        assert "Groceries" in cats
