from __future__ import annotations
from dataclasses import dataclass
from datetime import date

from .instruments import OptionContract
from .portfolio import Portfolio, Position
from .pricing import BlackScholesPricer


@dataclass
class ThetaHedgeResult:
    hedge_contract: OptionContract
    hedge_quantity: float
    portfolio_theta_before: float
    portfolio_theta_after: float


@dataclass
class ThetaHedger:
    """
    Build a theta hedge by trading ONE hedge option contract (simplest 
    replicable approach).
    """
    pricer: BlackScholesPricer

    def hedge_to_target_theta(
        self,
        portfolio: Portfolio,
        asof: date,
        hedge_contract: OptionContract,
        target_theta_per_day: float,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
    ) -> ThetaHedgeResult:

        theta_before = portfolio.theta_total_per_day(
            asof=asof,
            spot_by_symbol=spot_by_symbol,
            rate_cc=rate_cc,
            dividend_yield_cc_by_symbol=dividend_yield_cc_by_symbol,
            vol_by_option=vol_by_option,
            pricer=self.pricer,
        )

        # Compute hedge option theta per contract/day by building a one￾contract "portfolio"
        tmp = Portfolio(conventions=portfolio.conventions)
        tmp.add(Position(instrument=hedge_contract, quantity=1.0))

        hedge_theta_1 = tmp.theta_total_per_day(
            asof=asof,
            spot_by_symbol=spot_by_symbol,
            rate_cc=rate_cc,
            dividend_yield_cc_by_symbol=dividend_yield_cc_by_symbol,
            vol_by_option=vol_by_option,
            pricer=self.pricer,
        )

        if abs(hedge_theta_1) < 1e-12:
            raise ValueError("Hedge option theta is ~0; choose a different hedge contract.")

        # Solve: theta_before + n * hedge_theta_1 = target
        n = (target_theta_per_day - theta_before) / hedge_theta_1

        # Apply hedge
        portfolio.add(Position(instrument=hedge_contract, quantity=n))

        theta_after = portfolio.theta_total_per_day(
            asof=asof,
            spot_by_symbol=spot_by_symbol,
            rate_cc=rate_cc,
            dividend_yield_cc_by_symbol=dividend_yield_cc_by_symbol,
            vol_by_option=vol_by_option,
            pricer=self.pricer,
        )

        return ThetaHedgeResult(
            hedge_contract=hedge_contract,
            hedge_quantity=n,
            portfolio_theta_before=theta_before,
            portfolio_theta_after=theta_after,
        )