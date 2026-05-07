from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import InventoryItem


class IInventoryRepository(ABC):
    """Abstract contract for all inventory data sources.

    Concrete implementations (CSV, SQLite, REST, …) must satisfy this
    interface so that higher layers stay decoupled from storage details.
    """

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def get_all(self) -> List[InventoryItem]:
        """Return every inventory record."""

    @abstractmethod
    def get_by_id(self, item_id: str) -> Optional[InventoryItem]:
        """Return a single item by its unique ID, or None if not found."""

    @abstractmethod
    def get_by_zone(self, zone: str) -> List[InventoryItem]:
        """Return all items assigned to *zone*."""

    @abstractmethod
    def get_by_category(self, category: str) -> List[InventoryItem]:
        """Return all items belonging to *category*."""

    @abstractmethod
    def get_below_reorder_point(self) -> List[InventoryItem]:
        """Return items whose stock level has fallen to or below their reorder point."""

    @abstractmethod
    def get_top_by_criticality(self, n: int) -> List[InventoryItem]:
        """Return the *n* items with the highest criticality score."""

    @abstractmethod
    def get_top_by_value(self, n: int) -> List[InventoryItem]:
        """Return the *n* items with the highest value score (unit_price × daily_demand)."""

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    @abstractmethod
    def count(self) -> int:
        """Total number of records in the data source."""

    @abstractmethod
    def distinct_zones(self) -> List[str]:
        """Sorted list of distinct warehouse zones."""

    @abstractmethod
    def distinct_categories(self) -> List[str]:
        """Sorted list of distinct item categories."""
