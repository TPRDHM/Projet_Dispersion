from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

from .config import MarketConventions
from .instruments import OptionContract, Stock
from .pricing import BlackScholesInputs, BlackScholesPricer


@dataclass(frozen=True)
class Position:
    """
    Quantity convention:
    - For options: quantity = number of contracts (can be negative for short).
    - For stock: quantity = number of shares (can be negative for short).
    """
    instrument: Stock | OptionContract
    quantity: float


@dataclass
class Portfolio:
    positions: list[Position] = field(default_factory=list)
    conventions: MarketConventions = MarketConventions()

    def add(self, position: Position) -> None:
        self.positions.append(position)

    def theta_total_per_day(
        self,
        asof: date,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
        pricer: BlackScholesPricer,
    ) -> float:
        """
        Total portfolio theta per day, in currency units (not per share).
        For equity options, multiply per-share theta by contract multiplier and quantity.
        """
        total = 0.0
        for p in self.positions:
            if isinstance(p.instrument, Stock):
                continue # theta ~ 0 for the stock leg

            opt = p.instrument
            S = spot_by_symbol[opt.underlying.symbol]
            q = dividend_yield_cc_by_symbol.get(opt.underlying.symbol, 0.0)
            T = pricer.year_fraction(asof, opt.expiry, day_count=pricer.conventions.day_count.days_in_year)
            vol = vol_by_option[opt]

            x = BlackScholesInputs(
                spot=S,
                strike=opt.strike,
                time_to_expiry_years=T,
                rate_cc=rate_cc,
                dividend_yield_cc=q,
                vol=vol,
            )
            theta_share_day = pricer.theta_per_day(opt, x)
            theta_contract_day = theta_share_day * self.conventions.contract_multiplier
            total += theta_contract_day * p.quantity

        return total