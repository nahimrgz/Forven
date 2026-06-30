"""EMA trend pullback strategy confirmed by RSI recovery."""

from __future__ import annotations

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "ema_rsi_pullback"


class EMARSIPullbackStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return f"EMA RSI Pullback ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "SOL")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "ema_fast": 12,
            "ema_slow": 34,
            "rsi_period": 14,
            "rsi_entry": 45,
            "rsi_exit": 62,
            "leverage": 2.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "RANGE_BOUND"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buy a pullback inside an EMA uptrend when RSI({p['rsi_period']}) reclaims "
            f"{p['rsi_entry']}; exit when RSI loses {p['rsi_exit']} or price loses the slow EMA."
        )

    def _indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        from forven.strategies.indicators import rsi

        p = self.params
        close = df["close"]
        ema_fast = close.ewm(span=int(p["ema_fast"]), adjust=False).mean()
        ema_slow = close.ewm(span=int(p["ema_slow"]), adjust=False).mean()
        rsi_series = rsi(close, int(p["rsi_period"]))

        return pd.DataFrame(
            {"close": close, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi_series},
            index=df.index,
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        indicators = self._indicator_frame(df)
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2] if len(indicators) > 1 else current

        trend_ok = bool(current["ema_fast"] > current["ema_slow"] and current["close"] >= current["ema_fast"])
        rsi_reclaim = bool(previous["rsi"] < float(self.params["rsi_entry"]) and current["rsi"] >= float(self.params["rsi_entry"]))
        entry = trend_ok and rsi_reclaim
        exit_ = bool(
            (previous["rsi"] >= float(self.params["rsi_exit"]) and current["rsi"] < float(self.params["rsi_exit"]))
            or current["close"] < current["ema_slow"]
        )

        return Signal(
            entry_signal=entry,
            exit_signal=exit_,
            price=round(float(current["close"]), 4),
            direction="long",
            confidence=round(min(1.0, float(current["rsi"]) / 100.0), 4),
            indicators={
                "ema_fast": round(float(current["ema_fast"]), 4),
                "ema_slow": round(float(current["ema_slow"]), 4),
                "rsi": round(float(current["rsi"]), 2),
                "trend_ok": trend_ok,
            },
        )

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        indicators = self._indicator_frame(df)
        trend_ok = (indicators["ema_fast"] > indicators["ema_slow"]) & (indicators["close"] >= indicators["ema_fast"])
        rsi_reclaim = (indicators["rsi"].shift(1) < float(self.params["rsi_entry"])) & (
            indicators["rsi"] >= float(self.params["rsi_entry"])
        )
        rsi_rollover = (indicators["rsi"].shift(1) >= float(self.params["rsi_exit"])) & (
            indicators["rsi"] < float(self.params["rsi_exit"])
        )

        entry_signals = trend_ok & rsi_reclaim
        exit_signals = rsi_rollover | (indicators["close"] < indicators["ema_slow"])
        return entry_signals.fillna(False), exit_signals.fillna(False)

    def parameter_space(self) -> dict:
        return {
            "ema_fast": (8, 20, 2),
            "ema_slow": (21, 55, 5),
            "rsi_entry": (40, 55, 5),
            "rsi_exit": (55, 70, 5),
        }


STRATEGY_CLASS = EMARSIPullbackStrategy

STRATEGIES = [
    ("PREBUILT-EMA-RSI-PULLBACK-SOL", EMARSIPullbackStrategy, {"_asset": "SOL"}),
]
