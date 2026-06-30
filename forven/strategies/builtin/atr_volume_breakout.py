"""ATR breakout strategy confirmed by expanding volume."""

from __future__ import annotations

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "atr_volume_breakout"


class ATRVolumeBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return f"ATR Volume Breakout ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "ETH")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "atr_period": 14,
            "atr_multiplier": 0.7,
            "volume_period": 20,
            "volume_multiplier": 1.2,
            "breakout_lookback": 20,
            "exit_ema": 12,
            "leverage": 2.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "HIGH_VOL"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Breakout continuation using ATR({p['atr_period']}) and volume expansion above "
            f"{p['volume_multiplier']}x the {p['volume_period']}-bar average."
        )

    def _indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        from forven.strategies.indicators import atr

        p = self.params
        close = df["close"]
        high = df["high"]
        volume = df.get("volume", pd.Series(1.0, index=df.index, dtype=float))
        rolling_high = high.shift(1).rolling(int(p["breakout_lookback"])).max()
        atr_series = atr(df, int(p["atr_period"]))
        volume_ma = volume.rolling(int(p["volume_period"])).mean()
        exit_ema = close.ewm(span=int(p["exit_ema"]), adjust=False).mean()

        return pd.DataFrame(
            {
                "close": close,
                "rolling_high": rolling_high,
                "atr": atr_series,
                "volume": volume,
                "volume_ma": volume_ma,
                "exit_ema": exit_ema,
            },
            index=df.index,
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        indicators = self._indicator_frame(df)
        current = indicators.iloc[-1]

        breakout_level = float(current["rolling_high"]) + (float(current["atr"]) * float(self.params["atr_multiplier"]))
        volume_ok = bool(current["volume"] >= current["volume_ma"] * float(self.params["volume_multiplier"]))
        entry = bool(current["close"] > breakout_level and volume_ok)
        exit_ = bool(current["close"] < current["exit_ema"])

        return Signal(
            entry_signal=entry,
            exit_signal=exit_,
            price=round(float(current["close"]), 4),
            direction="long",
            confidence=round(min(1.0, float(current["volume"]) / max(float(current["volume_ma"]), 1.0) / 2.0), 4),
            indicators={
                "rolling_high": round(float(current["rolling_high"]), 4),
                "atr": round(float(current["atr"]), 4),
                "breakout_level": round(breakout_level, 4),
                "volume_ma": round(float(current["volume_ma"]), 2),
                "volume_ok": volume_ok,
                "exit_ema": round(float(current["exit_ema"]), 4),
            },
        )

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        indicators = self._indicator_frame(df)
        breakout_level = indicators["rolling_high"] + (indicators["atr"] * float(self.params["atr_multiplier"]))
        volume_ok = indicators["volume"] >= indicators["volume_ma"] * float(self.params["volume_multiplier"])

        entry_signals = (indicators["close"] > breakout_level) & volume_ok
        exit_signals = indicators["close"] < indicators["exit_ema"]
        return entry_signals.fillna(False), exit_signals.fillna(False)

    def parameter_space(self) -> dict:
        return {
            "atr_multiplier": (0.5, 1.5, 0.2),
            "volume_multiplier": (1.0, 2.0, 0.2),
            "breakout_lookback": (10, 30, 5),
        }


STRATEGY_CLASS = ATRVolumeBreakoutStrategy

STRATEGIES = [
    ("PREBUILT-ATR-VOLUME-BREAKOUT-ETH", ATRVolumeBreakoutStrategy, {"_asset": "ETH"}),
]
