"""MeanReversionGateStrategy — pre-wires Bollinger Band mean reversion + volatility gate.

Subclass fills in _volatility_ok(df) returning a bool Series to add a custom
volatility or regime filter (e.g. ATR percentile, ADX below threshold, funding rate).
The base handles BB deviation detection, RSI oversold gate, and ATR stops.

Example usage:
    class BBRSIReversionStrategy(MeanReversionGateStrategy):
        strategy_type = "bb_rsi_reversion"

        def _volatility_ok(self, df):
            from forven.strategies.indicators import adx
            adx_vals = adx(df, period=14)
            return adx_vals < 25  # only trade when market is ranging, not trending
"""

from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal


class MeanReversionGateStrategy(BaseStrategy):
    """Base: Bollinger Band extension + RSI oversold/overbought + subclass volatility gate."""

    strategy_type: str = "mean_reversion_gate_base"

    @property
    def default_params(self) -> dict:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 35,    # long entry when RSI this low
            "rsi_overbought": 65,  # exit signal
            "atr_period": 14,
            "atr_stop_mult": 2.0,  # wider stop for mean reversion
            "risk_pct": 0.01,
            "leverage": 2.0,       # lower leverage for counter-trend
        }

    def parameter_space(self) -> dict:
        return {
            "bb_period": (14, 30, 4),
            "bb_std": (1.5, 2.5, 0.5),
            "rsi_oversold": (25, 40, 5),
            "rsi_overbought": (60, 75, 5),
        }

    def p(self, key: str, default=None):
        return self.params.get(key, self.default_params.get(key, default))

    @abstractmethod
    def _volatility_ok(self, df: pd.DataFrame) -> "pd.Series[bool]":
        """Return bool Series: True on bars where volatility/regime is suitable for reversion."""

    def _bb(self, df: pd.DataFrame):
        """Compute Bollinger Bands; returns (upper, lower, mid)."""
        period = int(self.p("bb_period", 20))
        std_mult = float(self.p("bb_std", 2.0))
        mid = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        return mid + std_mult * std, mid - std_mult * std, mid

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 10:
            return Signal()
        try:
            from forven.strategies.indicators import rsi as calc_rsi, atr as calc_atr

            rsi_period = int(self.p("rsi_period", 14))
            rsi_oversold = float(self.p("rsi_oversold", 35))
            rsi_overbought = float(self.p("rsi_overbought", 65))

            rsi_vals = calc_rsi(df, period=rsi_period)
            atr_vals = calc_atr(df, period=int(self.p("atr_period", 14)))

            upper, lower, mid = self._bb(df)

            price = float(df["close"].iloc[-1])
            rsi_now = float(rsi_vals.iloc[-1]) if not pd.isna(rsi_vals.iloc[-1]) else 50.0

            below_lower = price < float(lower.iloc[-1])
            rsi_low = rsi_now <= rsi_oversold

            vol_series = self._volatility_ok(df)
            vol_ok = bool(vol_series.iloc[-1]) if len(vol_series) else True

            entry = below_lower and rsi_low and vol_ok
            exit_sig = price > float(mid.iloc[-1]) or rsi_now >= rsi_overbought

            stop_dist = float(atr_vals.iloc[-1]) * float(self.p("atr_stop_mult", 2.0))

            return Signal(
                entry_signal=entry,
                exit_signal=exit_sig,
                price=price,
                direction="long",
                confidence=max(0.0, min(1.0, (rsi_oversold - rsi_now) / rsi_oversold)) if rsi_low else 0.0,
                indicators={
                    "bb_upper": float(upper.iloc[-1]),
                    "bb_lower": float(lower.iloc[-1]),
                    "bb_mid": float(mid.iloc[-1]),
                    "rsi": rsi_now,
                    "atr_stop": stop_dist,
                    "vol_ok": vol_ok,
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
            rsi_oversold = float(self.p("rsi_oversold", 35))
            rsi_overbought = float(self.p("rsi_overbought", 65))

            rsi_vals = calc_rsi(df, period=rsi_period)
            upper, lower, mid = self._bb(df)

            below_lower = df["close"] < lower
            rsi_low = rsi_vals <= rsi_oversold

            vol_series = self._volatility_ok(df).astype(bool)

            entry = below_lower & rsi_low & vol_series
            exit_sig = (df["close"] > mid) | (rsi_vals >= rsi_overbought)

            return pd.DataFrame({
                "entry_signal": entry,
                "exit_signal": exit_sig,
            }, index=df.index).fillna(False)
        except Exception:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
