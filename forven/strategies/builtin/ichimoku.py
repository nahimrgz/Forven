"""Ichimoku Cloud strategy."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "ichimoku"


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud trend-following strategy.

    Entry: Tenkan-sen crosses above Kijun-sen AND price is above the cloud (Senkou Span A & B).
    Exit: Tenkan-sen crosses below Kijun-sen.
    """

    @property
    def name(self) -> str:
        return f"Ichimoku {self.params.get('tenkan_period', 9)}/{self.params.get('kijun_period', 26)} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "leverage": 1.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buys when Tenkan-sen ({p['tenkan_period']}) crosses above Kijun-sen ({p['kijun_period']}) "
            f"and price is above the Ichimoku cloud. "
            f"Senkou Span B period: {p['senkou_b_period']}."
        )

    def generate_signals(self, df: pd.DataFrame):
        """Vectorized twin of generate_signal — the SINGLE source of entry/exit logic."""
        p = self.params
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tenkan_p = p["tenkan_period"]
        kijun_p = p["kijun_period"]
        senkou_b_p = p["senkou_b_period"]

        # Tenkan-sen: midpoint of highest high and lowest low over tenkan_period
        tenkan = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
        # Kijun-sen: midpoint over kijun_period
        kijun = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2
        # Senkou Span A: midpoint of Tenkan and Kijun, shifted forward kijun_period bars
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
        # Senkou Span B: midpoint of highest high and lowest low over senkou_b_period, shifted
        senkou_b = ((high.rolling(senkou_b_p).max() + low.rolling(senkou_b_p).min()) / 2).shift(kijun_p)

        # Per-bar method falls back to close when a Senkou span is NaN (warmup).
        senkou_a_eff = senkou_a.where(senkou_a.notna(), close)
        senkou_b_eff = senkou_b.where(senkou_b.notna(), close)
        cloud_top = pd.concat([senkou_a_eff, senkou_b_eff], axis=1).max(axis=1)
        above_cloud = close > cloud_top

        prev_tenkan = tenkan.shift(1)
        prev_kijun = kijun.shift(1)
        cross_up = (prev_tenkan <= prev_kijun) & (tenkan > kijun)
        cross_down = (prev_tenkan >= prev_kijun) & (tenkan < kijun)

        entry = cross_up & above_cloud
        exit_ = cross_down
        return entry.fillna(False), exit_.fillna(False)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr

        p = self.params
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tenkan_p = p["tenkan_period"]
        kijun_p = p["kijun_period"]
        senkou_b_p = p["senkou_b_period"]

        min_bars = senkou_b_p + kijun_p + 2
        if len(df) < min_bars:
            return Signal(entry_signal=False, exit_signal=False, price=0.0,
                          direction="long", confidence=0.0, indicators={})

        # Tenkan-sen: midpoint of highest high and lowest low over tenkan_period
        tenkan = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
        # Kijun-sen: midpoint over kijun_period
        kijun = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2
        # Senkou Span A: midpoint of Tenkan and Kijun, shifted forward kijun_period bars
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
        # Senkou Span B: midpoint of highest high and lowest low over senkou_b_period, shifted
        senkou_b = ((high.rolling(senkou_b_p).max() + low.rolling(senkou_b_p).min()) / 2).shift(kijun_p)

        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_tenkan = float(tenkan.iloc[-1])
        curr_kijun = float(kijun.iloc[-1])
        curr_senkou_a = float(senkou_a.iloc[-1]) if pd.notna(senkou_a.iloc[-1]) else curr_close
        curr_senkou_b = float(senkou_b.iloc[-1]) if pd.notna(senkou_b.iloc[-1]) else curr_close
        curr_atr = float(atr_14.iloc[-1])

        cloud_top = max(curr_senkou_a, curr_senkou_b)
        above_cloud = curr_close > cloud_top

        entries, exits = self.generate_signals(df)

        # Confidence based on distance above cloud
        if above_cloud and cloud_top > 0:
            confidence = min(1.0, (curr_close - cloud_top) / (curr_atr * 3)) if curr_atr > 0 else 0.5
        else:
            confidence = 0.0

        return Signal(
            entry_signal=bool(entries.iloc[-1]),
            exit_signal=bool(exits.iloc[-1]),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            indicators={
                "tenkan": round(curr_tenkan, 4),
                "kijun": round(curr_kijun, 4),
                "senkou_a": round(curr_senkou_a, 4),
                "senkou_b": round(curr_senkou_b, 4),
                "above_cloud": bool(above_cloud),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "tenkan_period": (7, 12, 1),
            "kijun_period": (20, 30, 2),
            "senkou_b_period": (44, 60, 4),
        }


STRATEGY_CLASS = IchimokuStrategy

STRATEGIES = [
    ("PREBUILT-ICHIMOKU", IchimokuStrategy, {"_asset": "BTC"}),
]
