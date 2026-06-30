"""Linear Regression Channel strategy."""

import numpy as np
import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "linear_regression"


class LinearRegressionStrategy(BaseStrategy):
    """Linear regression channel breakout strategy.

    Entry: Price breaks above the upper channel (regression line + num_std * std).
    Exit: Price drops below the regression line.
    """

    @property
    def name(self) -> str:
        return f"LinReg {self.params.get('period', 50)}/{self.params.get('num_std', 2.0)} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "period": 50,
            "num_std": 2.0,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Enters when price breaks above the {p['period']}-bar linear regression channel "
            f"(+{p['num_std']} std). Exits when price drops below the regression line."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        close = df["close"]
        period = p["period"]

        min_bars = period + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        # Compute linear regression over the last `period` bars
        window = close.iloc[-period:].values.astype(float)
        x = np.arange(period, dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        reg_line = slope * x + intercept
        residuals = window - reg_line
        std_dev = float(np.std(residuals))

        # Current regression value is at x = period - 1
        reg_value = float(reg_line[-1])
        upper_channel = reg_value + p["num_std"] * std_dev
        lower_channel = reg_value - p["num_std"] * std_dev

        # Previous bar regression (recompute for previous window)
        prev_window = close.iloc[-(period + 1):-1].values.astype(float)
        prev_slope, prev_intercept = np.polyfit(x, prev_window, 1)
        prev_reg_line = prev_slope * x + prev_intercept
        prev_residuals = prev_window - prev_reg_line
        prev_std = float(np.std(prev_residuals))
        prev_reg_value = float(prev_reg_line[-1])
        prev_upper = prev_reg_value + p["num_std"] * prev_std

        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        curr_atr = float(atr_14.iloc[-1])

        # Entry: price breaks above upper channel
        entry = prev_close <= prev_upper and curr_close > upper_channel
        # Exit: price drops below regression line
        exit_ = curr_close < reg_value

        # Confidence based on how far above the regression line
        if curr_close > reg_value and std_dev > 0:
            confidence = min(1.0, (curr_close - reg_value) / (p["num_std"] * std_dev))
        else:
            confidence = 0.0

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            indicators={
                "reg_value": round(reg_value, 4),
                "upper_channel": round(upper_channel, 4),
                "lower_channel": round(lower_channel, 4),
                "slope": round(slope, 6),
                "std_dev": round(std_dev, 6),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "period": (30, 80, 10),
            "num_std": (1.5, 3.0, 0.5),
        }


STRATEGY_CLASS = LinearRegressionStrategy

STRATEGIES = [
    ("PREBUILT-LINREG", LinearRegressionStrategy, {"_asset": "BTC"}),
]
