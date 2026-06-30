"""TRIX strategy."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "trix"


class TrixStrategy(BaseStrategy):
    """TRIX indicator strategy.

    TRIX = 1-bar rate-of-change of a triple-smoothed EMA.
    Signal line = EMA of TRIX.
    Entry: TRIX crosses above signal line AND TRIX > 0.
    Exit: TRIX crosses below signal line.
    """

    @property
    def name(self) -> str:
        return f"TRIX {self.params.get('trix_period', 15)}/{self.params.get('signal_period', 9)} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "trix_period": 15,
            "signal_period": 9,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Enters when TRIX({p['trix_period']}) crosses above its {p['signal_period']}-bar "
            f"signal line and TRIX > 0. Exits on TRIX crossing below signal."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        close = df["close"]
        trix_period = p["trix_period"]
        signal_period = p["signal_period"]

        min_bars = trix_period * 3 + signal_period + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        # Triple-smoothed EMA
        ema1 = close.ewm(span=trix_period, adjust=False).mean()
        ema2 = ema1.ewm(span=trix_period, adjust=False).mean()
        ema3 = ema2.ewm(span=trix_period, adjust=False).mean()

        # TRIX = 1-bar ROC of triple-smoothed EMA (percentage)
        trix = ema3.pct_change() * 100

        # Signal line = EMA of TRIX
        signal_line = trix.ewm(span=signal_period, adjust=False).mean()

        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_trix = float(trix.iloc[-1])
        prev_trix = float(trix.iloc[-2])
        curr_signal = float(signal_line.iloc[-1])
        prev_signal = float(signal_line.iloc[-2])
        curr_atr = float(atr_14.iloc[-1])

        cross_up = prev_trix <= prev_signal and curr_trix > curr_signal
        cross_down = prev_trix >= prev_signal and curr_trix < curr_signal

        entry = cross_up and curr_trix > 0
        exit_ = cross_down

        confidence = min(1.0, abs(curr_trix) * 10) if entry else 0.0

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            indicators={
                "trix": round(curr_trix, 6),
                "signal_line": round(curr_signal, 6),
                "trix_positive": bool(curr_trix > 0),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "trix_period": (10, 20, 5),
            "signal_period": (5, 12, 1),
        }


STRATEGY_CLASS = TrixStrategy

STRATEGIES = [
    ("PREBUILT-TRIX", TrixStrategy, {"_asset": "BTC"}),
]
