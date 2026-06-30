"""TEMA Cross strategy."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "tema_cross"


def _tema(series: pd.Series, period: int) -> pd.Series:
    """Triple Exponential Moving Average = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


class TEMACrossStrategy(BaseStrategy):
    """TEMA crossover strategy.

    Entry: Fast TEMA crosses above Slow TEMA.
    Exit: Fast TEMA crosses below Slow TEMA.
    """

    @property
    def name(self) -> str:
        return f"TEMA {self.params.get('tema_fast', 12)}/{self.params.get('tema_slow', 26)} Cross ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "tema_fast": 12,
            "tema_slow": 26,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buys when {p['tema_fast']}-bar TEMA crosses above {p['tema_slow']}-bar TEMA. "
            f"Exits on the reverse crossover."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        close = df["close"]

        min_bars = p["tema_slow"] * 3 + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        tema_fast = _tema(close, p["tema_fast"])
        tema_slow = _tema(close, p["tema_slow"])
        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_fast = float(tema_fast.iloc[-1])
        prev_fast = float(tema_fast.iloc[-2])
        curr_slow = float(tema_slow.iloc[-1])
        prev_slow = float(tema_slow.iloc[-2])
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
                "tema_fast": round(curr_fast, 4),
                "tema_slow": round(curr_slow, 4),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "tema_fast": (8, 16, 2),
            "tema_slow": (20, 34, 2),
        }


STRATEGY_CLASS = TEMACrossStrategy

STRATEGIES = [
    ("PREBUILT-TEMA-CROSS", TEMACrossStrategy, {"_asset": "BTC"}),
]
