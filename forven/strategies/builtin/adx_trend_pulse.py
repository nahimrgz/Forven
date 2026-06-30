"""ADX trend continuation strategy with EMA pullback reclaim."""

from __future__ import annotations

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "trend_pulse_adx"


class ADXTrendPulseStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return f"ADX Trend Pulse ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "ema_fast": 21,
            "ema_slow": 55,
            "adx_period": 14,
            "adx_threshold": 18.0,
            "pullback_bars": 4,
            "leverage": 2.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Trend continuation on {self.asset}: reclaim the {p['ema_fast']}-bar EMA after a "
            f"{p['pullback_bars']}-bar pullback while ADX({p['adx_period']}) stays above "
            f"{p['adx_threshold']}."
        )

    def _indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        from forven.strategies.indicators import adx

        p = self.params
        close = df["close"]
        ema_fast = close.ewm(span=int(p["ema_fast"]), adjust=False).mean()
        ema_slow = close.ewm(span=int(p["ema_slow"]), adjust=False).mean()
        adx_series = adx(df, int(p["adx_period"]))
        lookback = max(int(p["pullback_bars"]), 2)
        recent_pullback = close.shift(1).rolling(lookback).min() <= ema_fast.shift(1)

        return pd.DataFrame(
            {
                "close": close,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "adx": adx_series,
                "recent_pullback": recent_pullback.fillna(False),
            },
            index=df.index,
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        indicators = self._indicator_frame(df)
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2] if len(indicators) > 1 else current

        trend_ok = bool(current["close"] > current["ema_slow"])
        adx_ok = bool(current["adx"] >= float(self.params["adx_threshold"]))
        reclaim = bool(previous["close"] <= previous["ema_fast"] and current["close"] > current["ema_fast"])
        entry = trend_ok and adx_ok and bool(current["recent_pullback"]) and reclaim
        exit_ = bool(current["close"] < current["ema_fast"] or current["adx"] < float(self.params["adx_threshold"]))

        return Signal(
            entry_signal=entry,
            exit_signal=exit_,
            price=round(float(current["close"]), 4),
            direction="long",
            confidence=round(min(1.0, float(current["adx"]) / 40.0) if adx_ok else 0.0, 4),
            indicators={
                "ema_fast": round(float(current["ema_fast"]), 4),
                "ema_slow": round(float(current["ema_slow"]), 4),
                "adx": round(float(current["adx"]), 2),
                "trend_ok": trend_ok,
                "recent_pullback": bool(current["recent_pullback"]),
            },
        )

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        indicators = self._indicator_frame(df)
        close = indicators["close"]
        ema_fast = indicators["ema_fast"]
        ema_slow = indicators["ema_slow"]
        adx_series = indicators["adx"]

        reclaim = (close.shift(1) <= ema_fast.shift(1)) & (close > ema_fast)
        trend_ok = close > ema_slow
        adx_ok = adx_series >= float(self.params["adx_threshold"])

        entry_signals = reclaim & trend_ok & adx_ok & indicators["recent_pullback"]
        exit_signals = (close < ema_fast) | (adx_series < float(self.params["adx_threshold"]))

        return entry_signals.fillna(False), exit_signals.fillna(False)

    def parameter_space(self) -> dict:
        return {
            "ema_fast": (10, 30, 5),
            "ema_slow": (40, 80, 5),
            "adx_threshold": (12, 28, 4),
            "pullback_bars": (2, 6, 1),
        }


STRATEGY_CLASS = ADXTrendPulseStrategy

STRATEGIES = [
    ("PREBUILT-ADX-TREND-PULSE-BTC", ADXTrendPulseStrategy, {"_asset": "BTC"}),
]
