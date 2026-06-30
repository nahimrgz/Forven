"""FundingRegimeStrategy — pre-wires funding rate extreme detection + technical filter.

Subclass fills in _technical_filter(df) returning a bool Series for the entry trigger.
The base handles funding rate z-score gate (extreme funding = fade opportunity),
open interest divergence check, and ATR stops.

Example usage:
    class FundingFadeStrategy(FundingRegimeStrategy):
        strategy_type = "funding_fade_rsi"

        def _technical_filter(self, df):
            from forven.strategies.indicators import rsi
            rsi_vals = rsi(df, period=14)
            # Fade: enter long when funding is extremely negative (longs squeezed)
            # and RSI confirms oversold
            return rsi_vals < 35
"""

from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal


class FundingRegimeStrategy(BaseStrategy):
    """Base: funding rate extreme + OI divergence gate + subclass technical filter."""

    strategy_type: str = "funding_regime_base"

    @property
    def default_params(self) -> dict:
        return {
            "funding_period": 48,   # lookback for z-score (number of bars)
            "funding_threshold": 1.5,  # |z-score| above this = extreme funding
            "funding_direction": "negative",  # "negative" = fade shorts, "positive" = fade longs
            "oi_period": 14,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "risk_pct": 0.01,
            "leverage": 2.0,
        }

    def parameter_space(self) -> dict:
        return {
            "funding_threshold": (1.0, 2.5, 0.5),
            "funding_period": (24, 96, 24),
            "atr_stop_mult": (1.0, 2.5, 0.5),
        }

    def p(self, key: str, default=None):
        return self.params.get(key, self.default_params.get(key, default))

    @abstractmethod
    def _technical_filter(self, df: pd.DataFrame) -> "pd.Series[bool]":
        """Return bool Series: True on bars where technical entry condition is met."""

    def _funding_extreme(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Returns (is_extreme, z_score). Graceful if no funding_rate column."""
        if "funding_rate" not in df.columns:
            return False, 0.0
        from forven.strategies.indicators import funding_rate_zscore
        period = int(self.p("funding_period", 48))
        threshold = float(self.p("funding_threshold", 1.5))
        direction = str(self.p("funding_direction", "negative")).lower()
        z = funding_rate_zscore(df, period=period)
        z_now = float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else 0.0
        if direction == "negative":
            is_extreme = z_now <= -threshold
        elif direction == "positive":
            is_extreme = z_now >= threshold
        else:
            is_extreme = abs(z_now) >= threshold
        return is_extreme, z_now

    def _funding_extreme_series(self, df: pd.DataFrame) -> "pd.Series[bool]":
        """Vectorized funding extreme check."""
        if "funding_rate" not in df.columns:
            return pd.Series(False, index=df.index)
        from forven.strategies.indicators import funding_rate_zscore
        period = int(self.p("funding_period", 48))
        threshold = float(self.p("funding_threshold", 1.5))
        direction = str(self.p("funding_direction", "negative")).lower()
        z = funding_rate_zscore(df, period=period)
        if direction == "negative":
            return z <= -threshold
        elif direction == "positive":
            return z >= threshold
        else:
            return z.abs() >= threshold

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 10:
            return Signal()
        try:
            from forven.strategies.indicators import atr as calc_atr, oi_price_divergence

            atr_vals = calc_atr(df, period=int(self.p("atr_period", 14)))
            funding_extreme, z_now = self._funding_extreme(df)

            # OI divergence (price up + OI down = weak move, bearish for longs)
            oi_div = oi_price_divergence(df, period=int(self.p("oi_period", 14)))
            oi_ok = not bool(oi_div.iloc[-1])  # no bearish OI divergence

            tech_series = self._technical_filter(df)
            tech_ok = bool(tech_series.iloc[-1]) if len(tech_series) else False

            entry = funding_extreme and oi_ok and tech_ok
            stop_dist = float(atr_vals.iloc[-1]) * float(self.p("atr_stop_mult", 1.5))
            price = float(df["close"].iloc[-1])

            return Signal(
                entry_signal=entry,
                exit_signal=False,  # subclass or time-based exit
                price=price,
                direction="long",
                confidence=min(1.0, abs(z_now) / 3.0) if funding_extreme else 0.0,
                indicators={
                    "funding_z": z_now,
                    "funding_extreme": funding_extreme,
                    "oi_ok": oi_ok,
                    "atr_stop": stop_dist,
                    "technical_ok": tech_ok,
                },
            )
        except Exception:
            return Signal()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
        try:
            from forven.strategies.indicators import oi_price_divergence

            funding_series = self._funding_extreme_series(df)
            oi_div = oi_price_divergence(df, period=int(self.p("oi_period", 14)))
            oi_ok = ~oi_div.astype(bool)
            tech_series = self._technical_filter(df).astype(bool)

            entry = funding_series & oi_ok & tech_series

            return pd.DataFrame({
                "entry_signal": entry,
                "exit_signal": pd.Series(False, index=df.index),
            }, index=df.index).fillna(False)
        except Exception:
            return pd.DataFrame({"entry_signal": False, "exit_signal": False}, index=df.index)
