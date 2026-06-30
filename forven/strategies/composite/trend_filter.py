"""TrendFilterStrategy — pre-wires MA trend direction + ADX gate.

Subclass fills in _entry_signal(df) returning a bool Series.
The base handles trend direction, ADX gate, ATR stops, and position sizing.

Example usage:
    class TrendKeltnerStrategy(TrendFilterStrategy):
        strategy_type = "trend_keltner"

        def _entry_signal(self, df):
            from forven.strategies.indicators import keltner_channel
            kc = keltner_channel(df, period=self.p("kc_period", 20), mult=self.p("kc_mult", 2.0))
            return df["close"] > kc["upper"]  # breakout above upper band
"""

from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal


class TrendFilterStrategy(BaseStrategy):
    """Base: MA trend direction + ADX gate + subclass entry signal."""

    strategy_type: str = "trend_filter_base"

    @property
    def default_params(self) -> dict:
        return {
            "ma_period": 100,
            "ma_type": "ema",   # "ema" or "sma"
            "adx_period": 14,
            "adx_min": 20,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "risk_pct": 0.01,
            "leverage": 3.0,
        }

    def parameter_space(self) -> dict:
        return {
            "ma_period": (50, 200, 25),
            "adx_min": (15, 30, 5),
            "atr_stop_mult": (1.0, 2.5, 0.5),
        }

    def p(self, key: str, default=None):
        return self.params.get(key, self.default_params.get(key, default))

    @abstractmethod
    def _entry_signal(self, df: pd.DataFrame) -> "pd.Series[bool]":
        """Return a boolean Series: True on bars where entry condition is met."""

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 10:
            return Signal()
        try:
            from forven.strategies.indicators import adx as calc_adx, atr as calc_atr

            ma_period = max(2, int(self.p("ma_period", 100)))
            ma_type = str(self.p("ma_type", "ema")).lower()
            adx_min = float(self.p("adx_min", 20))

            if ma_type == "sma":
                ma = df["close"].rolling(ma_period).mean()
            else:
                ma = df["close"].ewm(span=ma_period, adjust=False).mean()

            adx_vals = calc_adx(df, period=int(self.p("adx_period", 14)))
            atr_vals = calc_atr(df, period=int(self.p("atr_period", 14)))

            trend_up = df["close"].iloc[-1] > ma.iloc[-1]
            adx_ok = adx_vals.iloc[-1] >= adx_min if not pd.isna(adx_vals.iloc[-1]) else False

            entry_series = self._entry_signal(df)
            entry_now = bool(entry_series.iloc[-1]) if len(entry_series) else False

            entry = trend_up and adx_ok and entry_now
            stop_dist = float(atr_vals.iloc[-1]) * float(self.p("atr_stop_mult", 1.5))
            price = float(df["close"].iloc[-1])

            return Signal(
                entry_signal=entry,
                exit_signal=not trend_up,
                price=price,
                direction="long",
                confidence=min(1.0, adx_vals.iloc[-1] / 50) if adx_ok else 0.0,
                indicators={
                    "ma": float(ma.iloc[-1]),
                    "adx": float(adx_vals.iloc[-1]),
                    "atr_stop": stop_dist,
                    "trend_up": trend_up,
                },
            )
        except Exception:
            return Signal()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
        try:
            from forven.strategies.indicators import adx as calc_adx

            ma_period = max(2, int(self.p("ma_period", 100)))
            ma_type = str(self.p("ma_type", "ema")).lower()
            adx_min = float(self.p("adx_min", 20))

            if ma_type == "sma":
                ma = df["close"].rolling(ma_period).mean()
            else:
                ma = df["close"].ewm(span=ma_period, adjust=False).mean()

            adx_vals = calc_adx(df, period=int(self.p("adx_period", 14)))
            trend_up = df["close"] > ma
            adx_ok = adx_vals >= adx_min

            entry_series = self._entry_signal(df).astype(bool)
            entry = trend_up & adx_ok & entry_series

            return pd.DataFrame({
                "entry_signal": entry,
                "exit_signal": ~trend_up,
            }, index=df.index).fillna(False)
        except Exception:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
