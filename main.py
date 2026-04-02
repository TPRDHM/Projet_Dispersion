from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

from src.Strategies.instruments import Stock, OptionContract, OptionType
from src.Strategies.market_data import YFinanceMarketData, ConstantRateProvider
from src.Strategies.pricing import BlackScholesInputs, BlackScholesPricer
from src.Strategies.portfolio import Portfolio, Position
from src.Strategies.strategies import DispersionSizer, DeltaHedger


MatchMetric = Literal["theta_notional", "vega_notional"]


@dataclass(frozen=True)
class Straddle:
    call: OptionContract
    put: OptionContract


def pick_common_expiry(
    exp1: list[date],
    exp2: list[date],
    asof: date,
    min_days: int,
    max_days: int,
) -> date:
    common = sorted(set(exp1).intersection(set(exp2)))

    for d in common:
        dt = (d - asof).days
        if min_days <= dt <= max_days:
            return d

    if not common:
        raise ValueError("No common expiry found between SPY and AAPL.")

    return common[0]


def pick_atm_strike(symbol: str, expiry: date, md: YFinanceMarketData, spot: float) -> float:
    stk = Stock(symbol)
    tmp = OptionContract(
        underlying=stk,
        option_type=OptionType.CALL,
        strike=0.0,
        expiry=expiry,
    )

    chain = md.get_option_chain(tmp)
    strikes = sorted([float(k) for k in chain["strike"].tolist()])

    return min(strikes, key=lambda k: abs(k - spot))


def build_straddle(symbol: str, expiry: date, strike: float) -> Straddle:
    stk = Stock(symbol)

    return Straddle(
        call=OptionContract(
            underlying=stk,
            option_type=OptionType.CALL,
            strike=strike,
            expiry=expiry,
        ),
        put=OptionContract(
            underlying=stk,
            option_type=OptionType.PUT,
            strike=strike,
            expiry=expiry,
        ),
    )


def straddle_metric_per_contract(
    metric: MatchMetric,
    asof: date,
    straddle: Straddle,
    spot: float,
    r_cc: float,
    q_cc: float,
    iv_call: float,
    iv_put: float,
    pricer: BlackScholesPricer,
    multiplier: int = 100,
) -> float:
    out = 0.0

    for opt, iv in [(straddle.call, iv_call), (straddle.put, iv_put)]:
        T = pricer.year_fraction(asof, opt.expiry, day_count=pricer.conventions.day_count.days_in_year)

        x = BlackScholesInputs(
            spot=spot,
            strike=opt.strike,
            time_to_expiry_years=T,
            rate_cc=r_cc,
            dividend_yield_cc=q_cc,
            vol=iv,
        )

        g = pricer.greeks_per_share(opt, x)

        if metric == "theta_notional":
            out += g.theta_per_day * multiplier
        elif metric == "vega_notional":
            out += g.vega * multiplier
        else:
            raise ValueError("Unknown metric.")

    return out


def main() -> None:
    asof = date.today()

    metric: MatchMetric = "theta_notional"
    base_qty_spy_straddle = -1.0
    min_days = 20
    max_days = 45
    rate_simple = 0.0

    md = YFinanceMarketData()
    rates = ConstantRateProvider(annual_rate_simple=rate_simple)
    pricer = BlackScholesPricer()
    sizer = DispersionSizer(pricer=pricer)
    hedger = DeltaHedger(pricer=pricer)

    spy_symbol = "SPY"
    aapl_symbol = "AAPL"

    spot_spy = md.get_spot(spy_symbol)
    spot_aapl = md.get_spot(aapl_symbol)

    q_spy_cc = pricer.to_continuous_rate(md.get_dividend_yield(spy_symbol))
    q_aapl_cc = pricer.to_continuous_rate(md.get_dividend_yield(aapl_symbol))
    r_cc = pricer.to_continuous_rate(rates.get_annual_rate(asof))

    exp_spy = md.list_expirations(spy_symbol)
    exp_aapl = md.list_expirations(aapl_symbol)

    expiry = pick_common_expiry(exp_spy, exp_aapl, asof, min_days, max_days)

    k_spy = pick_atm_strike(spy_symbol, expiry, md, spot_spy)
    k_aapl = pick_atm_strike(aapl_symbol, expiry, md, spot_aapl)

    spy_straddle = build_straddle(spy_symbol, expiry, k_spy)
    aapl_straddle = build_straddle(aapl_symbol, expiry, k_aapl)

    q_spy_call = md.get_option_quote(spy_straddle.call)
    q_spy_put = md.get_option_quote(spy_straddle.put)
    q_aapl_call = md.get_option_quote(aapl_straddle.call)
    q_aapl_put = md.get_option_quote(aapl_straddle.put)

    for q in [q_spy_call, q_spy_put, q_aapl_call, q_aapl_put]:
        if q.implied_vol is None:
            raise ValueError("Missing implied volatility in option chain.")

    spy_metric_1 = straddle_metric_per_contract(
        metric=metric,
        asof=asof,
        straddle=spy_straddle,
        spot=spot_spy,
        r_cc=r_cc,
        q_cc=q_spy_cc,
        iv_call=float(q_spy_call.implied_vol),
        iv_put=float(q_spy_put.implied_vol),
        pricer=pricer,
    )

    aapl_metric_1 = straddle_metric_per_contract(
        metric=metric,
        asof=asof,
        straddle=aapl_straddle,
        spot=spot_aapl,
        r_cc=r_cc,
        q_cc=q_aapl_cc,
        iv_call=float(q_aapl_call.implied_vol),
        iv_put=float(q_aapl_put.implied_vol),
        pricer=pricer,
    )

    sz = sizer.match(
        metric=metric,
        qty_spy_straddle=base_qty_spy_straddle,
        spy_metric_per_straddle=spy_metric_1,
        aapl_metric_per_straddle=aapl_metric_1,
    )

    ptf = Portfolio()

    ptf.add(Position(spy_straddle.call, sz.qty_spy_straddle))
    ptf.add(Position(spy_straddle.put, sz.qty_spy_straddle))

    ptf.add(Position(aapl_straddle.call, sz.qty_aapl_straddle))
    ptf.add(Position(aapl_straddle.put, sz.qty_aapl_straddle))

    spot_by_symbol = {
        spy_symbol: float(spot_spy),
        aapl_symbol: float(spot_aapl),
    }

    div_by_symbol = {
        spy_symbol: float(q_spy_cc),
        aapl_symbol: float(q_aapl_cc),
    }

    vol_by_option = {
        spy_straddle.call: float(q_spy_call.implied_vol),
        spy_straddle.put: float(q_spy_put.implied_vol),
        aapl_straddle.call: float(q_aapl_call.implied_vol),
        aapl_straddle.put: float(q_aapl_put.implied_vol),
    }

    trades = hedger.hedge_to_zero_delta(
        portfolio=ptf,
        asof=asof,
        spot_by_symbol=spot_by_symbol,
        rate_cc=r_cc,
        dividend_yield_cc_by_symbol=div_by_symbol,
        vol_by_option=vol_by_option,
    )

    theta_total = ptf.theta_total_per_day(
        asof=asof,
        spot_by_symbol=spot_by_symbol,
        rate_cc=r_cc,
        dividend_yield_cc_by_symbol=div_by_symbol,
        vol_by_option=vol_by_option,
        pricer=pricer,
    )

    delta_map = ptf.delta_by_symbol(
        asof=asof,
        spot_by_symbol=spot_by_symbol,
        rate_cc=r_cc,
        dividend_yield_cc_by_symbol=div_by_symbol,
        vol_by_option=vol_by_option,
        pricer=pricer,
    )

    print("=== Dispersion Trade Snapshot ===")
    print(f"As-of: {asof.isoformat()}")
    print(f"Expiry: {expiry.isoformat()}")
    print()

    print("=== Straddles ===")
    print(f"SPY spot={spot_spy:.4f} strike={k_spy:.2f} qty={sz.qty_spy_straddle:.6f}")
    print(f"AAPL spot={spot_aapl:.4f} strike={k_aapl:.2f} qty={sz.qty_aapl_straddle:.6f}")
    print()

    print("=== Metric Matching ===")
    print(f"Metric used: {metric}")
    print(f"SPY total metric: {sz.spy_metric_value:.6f}")
    print(f"AAPL total metric: {sz.aapl_metric_value:.6f}")
    print()

    print("=== Portfolio Theta ===")
    print(f"Portfolio theta/day: {theta_total:.6f}")
    print()

    print("=== Delta by Symbol Before Hedge Execution ===")
    for symbol, delta_val in delta_map.items():
        print(f"{symbol}: {delta_val:.6f} shares")
    print()

    print("=== Hedge Trades to Execute ===")
    for t in trades:
        print(f"{t.symbol}: trade {t.trade_shares:.6f} shares -> new stock position {t.new_total_shares:.6f}")


if __name__ == "__main__":
    main()