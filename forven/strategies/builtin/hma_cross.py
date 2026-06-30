"""HMA Cross strategy."""

import math

import numpy as np
import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "hma_cross"


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))
    wma_half = _wma(series, half_period)
    wma_full = _wma(series, period)
    diff = 2 * wma_half - wma_full
    return _wma(diff, sqrt_period)


class HMACrossStrategy(BaseStrategy):
    """Hull Moving Average crossover strategy.

    Entry: Fast HMA crosses above Slow HMA.
    Exit: Fast HMA crosses below Slow HMA.
    """

    @property
    def name(self) -> str:
        return f"HMA {self.params.get('hma_fast', 9)}/{self.params.get('hma_slow', 21)} Cross ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "hma_fast": 9,
            "hma_slow": 21,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buys when {p['hma_fast']}-bar HMA crosses above {p['hma_slow']}-bar HMA. "
            f"Exits on the reverse crossover."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        close = df["close"]

        slow = p["hma_slow"]
        min_bars = slow + int(math.sqrt(slow)) + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        hma_fast = _hma(close, p["hma_fast"])
        hma_slow = _hma(close, slow)
        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_fast = float(hma_fast.iloc[-1])
        prev_fast = float(hma_fast.iloc[-2])
        curr_slow = float(hma_slow.iloc[-1])
        prev_slow = float(hma_slow.iloc[-2])
        curr_atr = float(atr_14.iloc[-1])

        cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        entry = cross_up
        exit_ = cross_down

        # Confidence from separation magnitude
        spread = abs(curr_fast - curr_slow)
        confidence = min(1.0, spread / (curr_atr * 2)) if curr_atr > 0 else 0.5

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            indicators={
                "hma_fast": round(curr_fast, 4),
                "hma_slow": round(curr_slow, 4),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "hma_fast": (5, 15, 2),
            "hma_slow": (15, 30, 3),
        }


STRATEGY_CLASS = HMACrossStrategy

STRATEGIES = [
    ("PREBUILT-HMA-CROSS", HMACrossStrategy, {"_asset": "BTC"}),
]
