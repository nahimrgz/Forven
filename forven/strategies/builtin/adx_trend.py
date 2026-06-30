"""ADX Trend strategy."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "adx_trend"


class ADXTrendStrategy(BaseStrategy):
    """ADX-based trend-following strategy using Directional Indicators.

    Entry: ADX > threshold AND +DI crosses above -DI.
    Exit: ADX drops below threshold OR -DI crosses above +DI.
    """

    @property
    def name(self) -> str:
        return f"ADX Trend {self.params.get('adx_period', 14)} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "adx_period": 14,
            "adx_threshold": 25,
            "di_period": 14,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Enters when ADX({p['adx_period']}) > {p['adx_threshold']} and +DI crosses above -DI. "
            f"Exits when ADX falls below threshold or -DI crosses above +DI."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import adx, atr

        p = self.params
        high = df["high"]
        low = df["low"]
        close = df["close"]

        di_period = p["di_period"]
        adx_period = p["adx_period"]
        min_bars = max(di_period, adx_period) * 2 + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        # Compute +DM / -DM
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

        atr_vals = atr(df, di_period)

        # Smoothed DI
        plus_di = 100 * (plus_dm.ewm(span=di_period, adjust=False).mean() / atr_vals)
        minus_di = 100 * (minus_dm.ewm(span=di_period, adjust=False).mean() / atr_vals)

        adx_vals = adx(df, adx_period)
        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_adx = float(adx_vals.iloc[-1])
        curr_plus_di = float(plus_di.iloc[-1])
        prev_plus_di = float(plus_di.iloc[-2])
        curr_minus_di = float(minus_di.iloc[-1])
        prev_minus_di = float(minus_di.iloc[-2])
        curr_atr = float(atr_14.iloc[-1])

        adx_strong = curr_adx > p["adx_threshold"]
        plus_cross_up = prev_plus_di <= prev_minus_di and curr_plus_di > curr_minus_di
        minus_cross_up = prev_minus_di <= prev_plus_di and curr_minus_di > curr_plus_di

        entry = adx_strong and plus_cross_up
        exit_ = (curr_adx < p["adx_threshold"]) or minus_cross_up

        confidence = min(1.0, curr_adx / 50) if adx_strong else 0.0

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(confidence, 4),
            indicators={
                "adx": round(curr_adx, 1),
                "plus_di": round(curr_plus_di, 2),
                "minus_di": round(curr_minus_di, 2),
                "adx_strong": bool(adx_strong),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "adx_period": (10, 20, 2),
            "adx_threshold": (20, 35, 5),
            "di_period": (10, 20, 2),
        }


STRATEGY_CLASS = ADXTrendStrategy

STRATEGIES = [
    ("PREBUILT-ADX-TREND", ADXTrendStrategy, {"_asset": "BTC"}),
]
