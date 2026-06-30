"""DEMA Cross strategy."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "dema_cross"


def _dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average = 2*EMA(n) - EMA(EMA(n))."""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2


class DEMACrossStrategy(BaseStrategy):
    """DEMA crossover strategy.

    Entry: Fast DEMA crosses above Slow DEMA.
    Exit: Fast DEMA crosses below Slow DEMA.
    """

    @property
    def name(self) -> str:
        return f"DEMA {self.params.get('dema_fast', 12)}/{self.params.get('dema_slow', 26)} Cross ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "dema_fast": 12,
            "dema_slow": 26,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buys when {p['dema_fast']}-bar DEMA crosses above {p['dema_slow']}-bar DEMA. "
            f"Exits on the reverse crossover."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        close = df["close"]

        min_bars = p["dema_slow"] * 2 + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        dema_fast = _dema(close, p["dema_fast"])
        dema_slow = _dema(close, p["dema_slow"])
        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_fast = float(dema_fast.iloc[-1])
        prev_fast = float(dema_fast.iloc[-2])
        curr_slow = float(dema_slow.iloc[-1])
        prev_slow = float(dema_slow.iloc[-2])
        curr_atr = float(atr_14.iloc[-1])

        cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        entry = cross_up
        exit_ = cross_down

        spread = abs(curr_fast - curr_slow)
        confidence = min(1.0, spread / (curr_atr * 2)) if curr_atr > 0 else 0.5

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            indicators={
                "dema_fast": round(curr_fast, 4),
                "dema_slow": round(curr_slow, 4),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "dema_fast": (8, 16, 2),
            "dema_slow": (20, 34, 2),
        }


STRATEGY_CLASS = DEMACrossStrategy

STRATEGIES = [
    ("PREBUILT-DEMA-CROSS", DEMACrossStrategy, {"_asset": "BTC"}),
]
