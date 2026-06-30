"""MomentumConfirmationStrategy — pre-wires dual-signal momentum + confirmation gate.

Subclass fills in _primary_signal(df) and _confirmation(df), both returning bool Series.
The base handles RSI momentum gate, volume confirmation, ATR stops, and position sizing.

Example usage:
    class MACDVolumeStrategy(MomentumConfirmationStrategy):
        strategy_type = "macd_volume"

        def _primary_signal(self, df):
            from forven.strategies.indicators import macd
            m = macd(df)
            return (m["macd"] > m["signal"]) & (m["macd"] > 0)

        def _confirmation(self, df):
            vol_ma = df["volume"].rolling(20).mean()
            return df["volume"] > vol_ma * 1.5  # volume surge confirms
"""

from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal


class MomentumConfirmationStrategy(BaseStrategy):
    """Base: RSI momentum gate + volume confirmation + subclass dual signal."""

    strategy_type: str = "momentum_confirmation_base"

    @property
    def default_params(self) -> dict:
        return {
            "rsi_period": 14,
            "rsi_min": 45,       # min RSI for entry (avoid oversold chasing)
            "rsi_max": 75,       # max RSI (avoid overbought entries)
            "vol_mult": 1.2,     # volume must be N× its MA
            "vol_period": 20,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "risk_pct": 0.01,
            "leverage": 3.0,
        }

    def parameter_space(self) -> dict:
        return {
            "rsi_period": (10, 21, 7),
            "rsi_min": (40, 55, 5),
            "rsi_max": (65, 80, 5),
            "vol_mult": (1.0, 2.0, 0.25),
        }

    def p(self, key: str, default=None):
        return self.params.get(key, self.default_params.get(key, default))

    @abstractmethod
    def _primary_signal(self, df: pd.DataFrame) -> "pd.Series[bool]":
        """Return bool Series: True on bars where primary momentum condition is met."""

    @abstractmethod
    def _confirmation(self, df: pd.DataFrame) -> "pd.Series[bool]":
        """Return bool Series: True on bars where confirmation condition is met."""

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 10:
            return Signal()
        try:
            from forven.strategies.indicators import rsi as calc_rsi, atr as calc_atr

            rsi_period = int(self.p("rsi_period", 14))
            rsi_min = float(self.p("rsi_min", 45))
            rsi_max = float(self.p("rsi_max", 75))
            vol_mult = float(self.p("vol_mult", 1.2))
            vol_period = int(self.p("vol_period", 20))

            rsi_vals = calc_rsi(df, period=rsi_period)
            atr_vals = calc_atr(df, period=int(self.p("atr_period", 14)))

            rsi_now = float(rsi_vals.iloc[-1]) if not pd.isna(rsi_vals.iloc[-1]) else 50.0
            rsi_ok = rsi_min <= rsi_now <= rsi_max

            # Volume gate
            if "volume" in df.columns and len(df) >= vol_period:
                vol_ma = df["volume"].rolling(vol_period).mean().iloc[-1]
                vol_ok = df["volume"].iloc[-1] > vol_ma * vol_mult if vol_ma > 0 else True
            else:
                vol_ok = True

            primary_series = self._primary_signal(df)
            confirm_series = self._confirmation(df)

            primary_now = bool(primary_series.iloc[-1]) if len(primary_series) else False
            confirm_now = bool(confirm_series.iloc[-1]) if len(confirm_series) else False

            entry = rsi_ok and vol_ok and primary_now and confirm_now
            stop_dist = float(atr_vals.iloc[-1]) * float(self.p("atr_stop_mult", 1.5))
            price = float(df["close"].iloc[-1])

            return Signal(
                entry_signal=entry,
                exit_signal=(rsi_now > rsi_max + 5),
                price=price,
                direction="long",
                confidence=min(1.0, (rsi_now - rsi_min) / (rsi_max - rsi_min)) if rsi_ok else 0.0,
                indicators={
                    "rsi": rsi_now,
                    "vol_ok": vol_ok,
                    "atr_stop": stop_dist,
                    "primary": primary_now,
                    "confirmation": confirm_now,
                },
            )
        except Exception:
            return Signal()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
        try:
            from forven.strategies.indicators import rsi as calc_rsi

            rsi_period = int(self.p("rsi_period", 14))
            rsi_min = float(self.p("rsi_min", 45))
            rsi_max = float(self.p("rsi_max", 75))
            vol_mult = float(self.p("vol_mult", 1.2))
            vol_period = int(self.p("vol_period", 20))

            rsi_vals = calc_rsi(df, period=rsi_period)
            rsi_ok = (rsi_vals >= rsi_min) & (rsi_vals <= rsi_max)

            if "volume" in df.columns:
                vol_ma = df["volume"].rolling(vol_period).mean()
                vol_ok = df["volume"] > vol_ma * vol_mult
            else:
                vol_ok = pd.Series(True, index=df.index)

            primary_series = self._primary_signal(df).astype(bool)
            confirm_series = self._confirmation(df).astype(bool)

            entry = rsi_ok & vol_ok & primary_series & confirm_series

            return pd.DataFrame({
                "entry_signal": entry,
                "exit_signal": rsi_vals > (rsi_max + 5),
            }, index=df.index).fillna(False)
        except Exception:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
