from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr

from .instruments import OptionContract, OptionQuote, OptionType, Stock

class SpotProvider(Protocol):
    def get_spot(self, symbol: str) -> float:
        ...

class RateProvider(Protocol):
    def get_annual_rate(self, asof: date) -> float:
        """
        Return an annual risk-free rate as a decimal (e.g. 0.05 for 5%).
        The strategy/pricer decides whether/how to convert to continuous 
        compounding.
        """
        ...

@dataclass
class YFinanceMarketData(SpotProvider):
    """
    Market data provider using yfinance (Yahoo Finance endpoints).
    Best suited for education & prototyping (snapshots).
    """
    def get_spot(self, symbol: str) -> float:
        t = yf.Ticker(symbol)
        px = t.fast_info.get("last_price", None)
        if px is None:
            hist = t.history(period="5d")
            if hist.empty:
                raise ValueError(f"Cannot fetch spot price for {symbol}")
            px = float(hist["Close"].iloc[-1])
        return float(px)

    def get_dividend_yield(self, symbol: str) -> float:
        """
        Attempt to fetch dividend yield as decimal.
        If unavailable, return 0.
        """
        t = yf.Ticker(symbol)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        dy = info.get("dividendYield", None)
        if dy is None:
            return 0.0
        return float(dy)

    def list_expirations(self, symbol: str) -> list[date]:
        t = yf.Ticker(symbol)
        expirations = getattr(t, "options", None)
        if not expirations:
            return []
        out: list[date] = []
        for s in expirations:
            out.append(date.fromisoformat(s))
        return out

    def get_option_chain(self, contract: OptionContract) -> pd.DataFrame:
        """
        Return the full option chain (calls/puts) for the expiry.
        You will filter for a specific strike/type in a helper function.
        """
        t = yf.Ticker(contract.underlying.symbol)
        chain = t.option_chain(contract.expiry.isoformat())
        df = chain.calls if contract.option_type == OptionType.CALL else chain.puts
        df = df.copy()
        df["optionType"] = contract.option_type.value
        df["expiry"] = contract.expiry.isoformat()
        return df

    def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        """
        Extract a single strike quote row from the yfinance chain for the 
        given expiry.
        """
        df = self.get_option_chain(contract)
        row = df.loc[df["strike"] == float(contract.strike)]
        if row.empty:
            # Sometimes strike is float with minor rounding.
            row = df.iloc[(df["strike"] - float(contract.strike)).abs().argsort()[:1]]
        r0 = row.iloc[0].to_dict()

        def _f(x):
            try:
                if x is None:
                    return None
                if pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None

        # yfinance provides lastTradeDate as pandas Timestamp (often tz-aware)
        ts = r0.get("lastTradeDate", None)
        ts_str = None
        if ts is not None and not pd.isna(ts):
            try:
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                if isinstance(ts, datetime):
                    ts_str = ts.astimezone().isoformat()
                else:
                    ts_str = str(ts)
            except Exception:
                ts_str = str(ts)

        return OptionQuote(
            last=_f(r0.get("lastPrice", None)),
            bid=_f(r0.get("bid", None)),
            ask=_f(r0.get("ask", None)),
            implied_vol=_f(r0.get("impliedVolatility", None)),
            timestamp_utc=ts_str,
        )

@dataclass
class FredRateProvider(RateProvider):
    """
    Risk-free proxy via FRED constant maturity yields.
    Default: DGS3MO (3-month Treasury constant maturity).
    """
    series_id: str = "DGS3MO"

    def get_annual_rate(self, asof: date) -> float:
        start = asof.replace(year=asof.year - 1)
        df = pdr.get_data_fred(self.series_id, start=start, end=asof)
        if df.empty:
            raise ValueError(f"FRED returned empty series {self.series_id}")
        # FRED yields in percent; take the last available observation <= asof
        s = df[self.series_id].dropna()
        if s.empty:
            raise ValueError(f"FRED series {self.series_id} contains only NaNs")
        y_percent = float(s.iloc[-1])
        return y_percent / 100.0