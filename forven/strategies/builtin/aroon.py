"""Aroon strategy."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "aroon"


class AroonStrategy(BaseStrategy):
    """Aroon indicator trend-following strategy.

    Entry: Aroon Up crosses above Aroon Down AND Aroon Up > upper threshold.
    Exit: Aroon Down crosses above Aroon Up.
    """

    @property
    def name(self) -> str:
        return f"Aroon {self.params.get('aroon_period', 25)} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "aroon_period": 25,
            "upper_threshold": 70,
            "lower_threshold": 30,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Enters when Aroon Up ({p['aroon_period']}) crosses above Aroon Down "
            f"and Aroon Up > {p['upper_threshold']}. "
            f"Exits when Aroon Down crosses above Aroon Up."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        high = df["high"]
        low = df["low"]
        close = df["close"]
        period = p["aroon_period"]

        min_bars = period + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        # Aroon Up = ((period - bars since highest high) / period) * 100
        # Aroon Down = ((period - bars since lowest low) / period) * 100
        aroon_up = high.rolling(period + 1).apply(
            lambda x: ((period - (period - x.values.argmax())) / period) * 100,
            raw=False,
        )
        aroon_down = low.rolling(period + 1).apply(
            lambda x: ((period - (period - x.values.argmin())) / period) * 100,
            raw=False,
        )

        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_aroon_up = float(aroon_up.iloc[-1])
        prev_aroon_up = float(aroon_up.iloc[-2])
        curr_aroon_down = float(aroon_down.iloc[-1])
        prev_aroon_down = float(aroon_down.iloc[-2])
        curr_atr = float(atr_14.iloc[-1])

        up_cross = prev_aroon_up <= prev_aroon_down and curr_aroon_up > curr_aroon_down
        down_cross = prev_aroon_down <= prev_aroon_up and curr_aroon_down > curr_aroon_up

        entry = up_cross and curr_aroon_up > p["upper_threshold"]
        exit_ = down_cross

        confidence = min(1.0, curr_aroon_up / 100) if entry else 0.0

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(confidence, 4),
            indicators={
                "aroon_up": round(curr_aroon_up, 2),
                "aroon_down": round(curr_aroon_down, 2),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "aroon_period": (15, 35, 5),
            "upper_threshold": (60, 80, 10),
            "lower_threshold": (20, 40, 10),
        }


STRATEGY_CLASS = AroonStrategy

STRATEGIES = [
    ("PREBUILT-AROON", AroonStrategy, {"_asset": "BTC"}),
]
