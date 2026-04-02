from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import date

from .config import MarketConventions
from .instruments import OptionContract, OptionType
from .math_utils import norm_cdf, norm_pdf, clamp


@dataclass(frozen=True)
class BlackScholesInputs:
    spot: float
    strike: float
    time_to_expiry_years: float
    rate_cc: float # continuously compounded
    dividend_yield_cc: float # continuously compounded
    vol: float # annualized (decimal)


@dataclass
class BlackScholesPricer:
    conventions: MarketConventions = MarketConventions()

    def _d1_d2(self, x: BlackScholesInputs) -> tuple[float, float]:
        T = max(x.time_to_expiry_years, 1e-10)
        vol = max(x.vol, 1e-10)
        num = math.log(x.spot / x.strike) + (x.rate_cc - x.dividend_yield_cc + 0.5 * vol * vol) * T
        den = vol * math.sqrt(T)
        d1 = num / den
        d2 = d1 - vol * math.sqrt(T)
        return d1, d2

    def price(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        d1, d2 = self._d1_d2(x)
        S, K, T = x.spot, x.strike, x.time_to_expiry_years
        r, q = x.rate_cc, x.dividend_yield_cc

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if contract.option_type == OptionType.CALL:
            return disc_q * S * norm_cdf(d1) - disc_r * K * norm_cdf(d2)
        else:
            return disc_r * K * norm_cdf(-d2) - disc_q * S * norm_cdf(-d1)

    def theta_per_year(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        """
        Theta in price units per *year* (per share), using BS closed forms.
        Convention: theta = -∂V/∂T with T in years (as in many textbooks/notes).
        """
        d1, d2 = self._d1_d2(x)
        S, K, T = x.spot, x.strike, x.time_to_expiry_years
        r, q, vol = x.rate_cc, x.dividend_yield_cc, x.vol

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        first_term = -(disc_q * S * norm_pdf(d1) * vol) / (2.0 * math.sqrt(max(T, 1e-10)))

        if contract.option_type == OptionType.CALL:
            return first_term + q * disc_q * S * norm_cdf(d1) - r * disc_r * K * norm_cdf(d2)

        # Put theta via put-call parity:
        # P = C + K e^{-rT} - S e^{-qT}
        # theta_put = theta_call + r K e^{-rT} - q S e^{-qT}
        call_theta = first_term + q * disc_q * S * norm_cdf(d1) - r * disc_r * K * norm_cdf(d2)
        return call_theta + r * disc_r * K - q * disc_q * S

    def theta_per_day(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        """
        Theta per calendar day (per share). Many platforms quote theta per day.
        """
        theta_y = self.theta_per_year(contract, x)
        return theta_y / self.conventions.day_count.days_in_year

    @staticmethod
    def year_fraction(asof: date, expiry: date, day_count: float = 365.0) -> float:
        days = (expiry - asof).days
        return max(days, 0) / day_count

    @staticmethod
    def to_continuous_rate(simple_annual_rate: float) -> float:
        """
        Convert a simple annual yield into a continuously-compounded approximation.
        r_cc = ln(1 + r_simple). For small rates, r_cc ~ r_simple.
        """
        r = clamp(simple_annual_rate, -0.99, 10.0)
        return math.log(1.0 + r)