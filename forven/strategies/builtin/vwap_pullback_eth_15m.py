"""ETH 15m VWAP pullback template with trend and RSI confirmation."""

from __future__ import annotations

import pandas as pd

from forven.strategies.indicators import rsi as compute_rsi
from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "vwap_pullback_eth_15m"


def _rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    volume_sum = volume.rolling(period).sum().replace(0.0, pd.NA)
    return (typical_price * volume).rolling(period).sum() / volume_sum


class Eth15mVWAPPullbackStrategy(BaseStrategy):
    """Preset ETH 15m pullback strategy for pipeline experimentation."""

    @property
    def name(self) -> str:
        return "ETH 15m VWAP Pullback"

    @property
    def asset(self) -> str:
        return str(self.params.get("_asset") or "ETH")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "_asset": "ETH",
            "timeframe": "15m",
            "vwap_period": 32,
            "distance_pct": 0.015,
            "ema_regime": 96,
            "slope_bars": 8,
            "rsi_period": 14,
            "rsi_entry": 35,
            "rsi_exit": 60,
            "leverage": 1.0,
        }

    def data_requirements(self) -> list[dict]:
        warmup = max(
            int(self.params.get("vwap_period", 32)),
            int(self.params.get("ema_regime", 96)),
            int(self.params.get("rsi_period", 14)),
            int(self.params.get("slope_bars", 8)),
        )
        return [
            {
                "asset": self.asset,
                "exchange": "any",
                "timeframe": str(self.params.get("timeframe") or "15m"),
                "min_bars": max(warmup + 100, 500),
            }
        ]

    def describe(self) -> str:
        p = self.params
        return (
            f"ETH 15m pullback template. Buys when price trades at least {p['distance_pct']:.2%} below "
            f"rolling VWAP({p['vwap_period']}) while price stays above a rising EMA({p['ema_regime']}) "
            f"and RSI({p['rsi_period']}) is below {p['rsi_entry']}. Exits back at VWAP, on RSI recovery "
            f"above {p['rsi_exit']}, or when price loses the regime EMA."
        )

    def _indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        indicators = pd.DataFrame(index=df.index)
        indicators["close"] = df["close"].astype(float)
        indicators["vwap"] = _rolling_vwap(df, int(p["vwap_period"]))
        indicators["ema_regime"] = indicators["close"].ewm(
            span=int(p["ema_regime"]),
            adjust=False,
        ).mean()
        indicators["rsi"] = compute_rsi(indicators["close"], int(p["rsi_period"]))
        indicators["distance_to_vwap"] = (
            (indicators["vwap"] - indicators["close"]) / indicators["vwap"]
        )
        indicators["trend_ok"] = indicators["close"] > indicators["ema_regime"]
        indicators["ema_rising"] = indicators["ema_regime"] > indicators["ema_regime"].shift(
            int(p["slope_bars"])
        )
        return indicators

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        warmup = max(
            int(self.params.get("vwap_period", 32)),
            int(self.params.get("ema_regime", 96)),
            int(self.params.get("rsi_period", 14)),
            int(self.params.get("slope_bars", 8)),
        ) + 2
        curr_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
        if len(df) < warmup:
            return Signal(
                entry_signal=False,
                exit_signal=False,
                price=curr_close,
                direction="long",
                confidence=0.0,
                indicators={},
            )

        indicators = self._indicator_frame(df)
        latest = indicators.iloc[-1]
        latest_vwap = latest["vwap"]
        latest_ema = latest["ema_regime"]
        latest_rsi = latest["rsi"]
        latest_distance = latest["distance_to_vwap"]

        if pd.isna(latest_vwap) or pd.isna(latest_ema) or pd.isna(latest_rsi):
            return Signal(
                entry_signal=False,
                exit_signal=False,
                price=curr_close,
                direction="long",
                confidence=0.0,
                indicators={},
            )

        entry = bool(
            latest_distance >= float(self.params["distance_pct"])
            and bool(latest["trend_ok"])
            and bool(latest["ema_rising"])
            and float(latest_rsi) < float(self.params["rsi_entry"])
        )
        exit_ = bool(
            curr_close >= float(latest_vwap)
            or float(latest_rsi) > float(self.params["rsi_exit"])
            or curr_close < float(latest_ema)
        )
        confidence = 0.0
        threshold = float(self.params["distance_pct"])
        if entry and threshold > 0:
            confidence = min(1.0, float(latest_distance) / threshold)

        return Signal(
            entry_signal=entry,
            exit_signal=exit_,
            price=round(curr_close, 4),
            direction="long",
            confidence=round(confidence, 4),
            indicators={
                "vwap": round(float(latest_vwap), 4),
                "ema_regime": round(float(latest_ema), 4),
                "rsi": round(float(latest_rsi), 2),
                "distance_to_vwap": round(float(latest_distance), 5),
                "trend_ok": bool(latest["trend_ok"]),
                "ema_rising": bool(latest["ema_rising"]),
            },
        )

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        indicators = self._indicator_frame(df)
        entry_signals = (
            (indicators["distance_to_vwap"] >= float(self.params["distance_pct"]))
            & indicators["trend_ok"]
            & indicators["ema_rising"]
            & (indicators["rsi"] < float(self.params["rsi_entry"]))
        )
        exit_signals = (
            (indicators["close"] >= indicators["vwap"])
            | (indicators["rsi"] > float(self.params["rsi_exit"]))
            | (indicators["close"] < indicators["ema_regime"])
        )
        return entry_signals.fillna(False), exit_signals.fillna(False)

    def parameter_space(self) -> dict:
        return {
            "vwap_period": (24, 40, 4),
            "distance_pct": (0.0125, 0.0175, 0.0025),
            "ema_regime": (72, 144, 24),
            "slope_bars": (4, 8, 4),
            "rsi_entry": (35, 45, 5),
            "rsi_exit": (50, 60, 5),
        }


STRATEGY_CLASS = Eth15mVWAPPullbackStrategy
STRATEGIES: list[tuple[str, type[Eth15mVWAPPullbackStrategy], dict]] = []
