from __future__ import annotations
from dataclasses import dataclass
from datetime import date

from src.theta_hedging.instruments import Stock, OptionContract, OptionType
from src.theta_hedging.market_data import YFinanceMarketData, FredRateProvider
from src.theta_hedging.pricing import BlackScholesPricer
from src.theta_hedging.portfolio import Portfolio, Position
from src.theta_hedging.strategies import ThetaHedger


@dataclass(frozen=True)
class Selection:
    portfolio_contract: OptionContract
    hedge_contract: OptionContract


def pick_two_calls_same_expiry(symbol: str, expiry: date, md: YFinanceMarketData, spot: float) -> Selection:
    """
    Simple deterministic selection:
    - portfolio option: call strike closest to spot (ATM-ish)
    - hedge option: call other nearby strike (2nd closest) to avoid identical instrument
    """
    stk = Stock(symbol)
    tmp = OptionContract(underlying=stk, option_type=OptionType.CALL, strike=0.0, expiry=expiry)
    chain = md.get_option_chain(tmp)

    strikes = sorted([float(k) for k in chain["strike"].tolist()])
    strikes_sorted = sorted(strikes, key=lambda k: abs(k - spot))

    if len(strikes_sorted) < 2:
        raise ValueError("Not enough strikes to pick two distinct options for this expiry.")

    k1, k2 = strikes_sorted[0], strikes_sorted[1]
    c1 = OptionContract(underlying=stk, option_type=OptionType.CALL, strike=k1, expiry=expiry)
    c2 = OptionContract(underlying=stk, option_type=OptionType.CALL, strike=k2, expiry=expiry)
    return Selection(portfolio_contract=c1, hedge_contract=c2)


def main() -> None:
    # === Parameters you may change ===
    underlying = "AAPL" # Example; replace by "SPY" etc.
    portfolio_contracts_qty = 10.0 # long 10 calls
    target_theta_per_day = 0.0 # theta-neutral
    # =================================

    asof = date.today()

    md = YFinanceMarketData()
    rates = FredRateProvider(series_id="DGS3MO")
    pricer = BlackScholesPricer()
    hedger = ThetaHedger(pricer=pricer)

    # Spot / dividend
    spot = md.get_spot(underlying)
    q_simple = md.get_dividend_yield(underlying) # already a decimal; treat as continuous approx
    q_cc = pricer.to_continuous_rate(q_simple)

    # Risk-free
    r_simple = rates.get_annual_rate(asof)
    r_cc = pricer.to_continuous_rate(r_simple)

    # Expiries
    expiries = md.list_expirations(underlying)
    if not expiries:
        raise ValueError("No option expirations returned by the data source.")
    expiry = expiries[0]

    # nearest expiry; for robustness pick based on your project rules
    sel = pick_two_calls_same_expiry(underlying, expiry, md, spot)

    # Quotes + implied vols
    q1 = md.get_option_quote(sel.portfolio_contract)
    q2 = md.get_option_quote(sel.hedge_contract)

    if q1.implied_vol is None or q2.implied_vol is None:
        raise ValueError("Implied volatility missing in the chain; choose another expiry/strike or data source.")

    vol_by_option = {
        sel.portfolio_contract: float(q1.implied_vol),
        sel.hedge_contract: float(q2.implied_vol),
    }

    spot_by_symbol = {underlying: float(spot)}
    div_by_symbol = {underlying: float(q_cc)}

    # Portfolio build
    ptf = Portfolio()
    ptf.add(Position(instrument=sel.portfolio_contract, quantity=portfolio_contracts_qty))

    theta_before = ptf.theta_total_per_day(
        asof=asof,
        spot_by_symbol=spot_by_symbol,
        rate_cc=r_cc,
        dividend_yield_cc_by_symbol=div_by_symbol,
        vol_by_option=vol_by_option,
        pricer=pricer,
    )

    print("=== Inputs ===")
    print(f"As-of date: {asof.isoformat()}")
    print(f"Underlying: {underlying} Spot: {spot:.4f}")
    print(f"Risk-free (DGS3MO) simple: {r_simple:.4%} cc: {r_cc:.4%}")
    print(f"Dividend yield simple: {q_simple:.4%} cc: {q_cc:.4%}")
    print(f"Expiry selected: {expiry.isoformat()}")

    print("\n=== Instruments ===")
    print(f"Portfolio option: {sel.portfolio_contract.pretty_symbol()} IV={q1.implied_vol:.4f}")
    print(f"Hedge option: {sel.hedge_contract.pretty_symbol()} IV={q2.implied_vol:.4f}")

    print("\n=== Theta before hedge ===")
    print(f"Portfolio theta/day (currency): {theta_before:.6f}")

    # Hedge
    res = hedger.hedge_to_target_theta(
        portfolio=ptf,
        asof=asof,
        hedge_contract=sel.hedge_contract,
        target_theta_per_day=target_theta_per_day,
        spot_by_symbol=spot_by_symbol,
        rate_cc=r_cc,
        dividend_yield_cc_by_symbol=div_by_symbol,
        vol_by_option=vol_by_option,
    )

    print("\n=== Hedge result ===")
    print(f"Hedge quantity (contracts): {res.hedge_quantity:.6f}")
    print(f"Theta/day before: {res.portfolio_theta_before:.6f}")
    print(f"Theta/day after: {res.portfolio_theta_after:.6f}")


if __name__ == "__main__":
    main()