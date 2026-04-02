from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class DayCountConvention:
    days_in_year: float = 365.0

@dataclass(frozen=True)
class MarketConventions:
    contract_multiplier: int = 100  # US equity options: 1 contract = 100 shares
    day_count: DayCountConvention = field(default_factory=DayCountConvention)