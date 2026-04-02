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
    - Options: number of contracts
    - Stock: number of shares
    """
    instrument: Stock | OptionContract
    quantity: float


@dataclass
class Portfolio:
    positions: list[Position] = field(default_factory=list)
    conventions: MarketConventions = MarketConventions()

    def add(self, position: Position) -> None:
        self.positions.append(position)

    # ===============================
    # ===== THETA (UNCHANGED) =======
    # ===============================
    def theta_total_per_day(
        self,
        asof: date,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
        pricer: BlackScholesPricer,
    ) -> float:

        total = 0.0

        for p in self.positions:

            if isinstance(p.instrument, Stock):
                continue  # stock has no theta

            opt = p.instrument
            S = spot_by_symbol[opt.underlying.symbol]
            q = dividend_yield_cc_by_symbol.get(opt.underlying.symbol, 0.0)

            T = pricer.year_fraction(asof, opt.expiry)
            vol = vol_by_option[opt]

            x = BlackScholesInputs(
                spot=S,
                strike=opt.strike,
                time_to_expiry_years=T,
                rate_cc=rate_cc,
                dividend_yield_cc=q,
                vol=vol,
            )

            theta = pricer.theta_per_day(opt, x)
            theta_contract = theta * self.conventions.contract_multiplier

            total += theta_contract * p.quantity

        return total

    # =========================================
    # >>> MODIF DISPERSION : DELTA PAR SYMBOLE
    # =========================================
    def delta_by_symbol(
        self,
        asof: date,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
        pricer: BlackScholesPricer,
    ) -> dict[str, float]:
        """
        Return delta exposure in number of shares per underlying.
        Used for delta hedging each leg separately (SPY / AAPL).
        """

        delta_map: dict[str, float] = {}

        for p in self.positions:

            # ===== STOCK =====
            if isinstance(p.instrument, Stock):
                symbol = p.instrument.symbol
                delta_map[symbol] = delta_map.get(symbol, 0.0) + p.quantity
                continue

            # ===== OPTION =====
            opt = p.instrument
            symbol = opt.underlying.symbol

            S = spot_by_symbol[symbol]
            q = dividend_yield_cc_by_symbol.get(symbol, 0.0)

            T = pricer.year_fraction(asof, opt.expiry)
            vol = vol_by_option[opt]

            x = BlackScholesInputs(
                spot=S,
                strike=opt.strike,
                time_to_expiry_years=T,
                rate_cc=rate_cc,
                dividend_yield_cc=q,
                vol=vol,
            )

            # >>> NEED delta from pricing.py
            delta = pricer.delta(opt, x)

            delta_contract = delta * self.conventions.contract_multiplier

            delta_map[symbol] = delta_map.get(symbol, 0.0) + delta_contract * p.quantity

        return delta_map