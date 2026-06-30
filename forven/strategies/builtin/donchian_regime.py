"""Regime-aware Donchian breakout strategy tuned for durable trend capture."""

from __future__ import annotations

import pandas as pd

from forven.strategies.indicators import adx as compute_adx
from forven.strategies.base import BaseStrategy, Signal
from forven.strategies.builtin.donchian import donchian_bands, resolve_donchian_period

TYPE_NAME = "donchian_regime"


def _resolve_exit_period(params: dict | None) -> int:
    payload = params or {}
    raw_value = payload.get("exit_period", payload.get("donchian_exit_period", 20))
    try:
        period = int(raw_value)
    except (TypeError, ValueError):
        period = 20
    return max(period, 2)


def _resolve_ema_period(params: dict | None) -> int:
    payload = params or {}
    raw_value = payload.get(
        "ema_period",
        payload.get("ema_regime", payload.get("trend_ema", payload.get("regime_ema200", 200))),
    )
    try:
        period = int(raw_value)
    except (TypeError, ValueError):
        period = 200
    return max(period, 2)


class DonchianRegimeStrategy(BaseStrategy):
    """Long-only Donchian breakout that trades only when the trend is healthy."""

    @property
    def name(self) -> str:
        return f"Donchian Regime Breakout ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "period": 55,
            "exit_period": 20,
            "ema_period": 200,
            "adx_period": 14,
            "adx_min": 25,
            "_asset": "BTC",
        }

    def data_requirements(self) -> list[dict]:
        warmup = max(
            resolve_donchian_period(self.params),
            _resolve_exit_period(self.params),
            _resolve_ema_period(self.params),
            int(self.params.get("adx_period", 14)),
        )
        timeframe = str(self.params.get("timeframe") or "1d").strip() or "1d"
        return [
            {
                "asset": self.asset,
                "exchange": "any",
                "timeframe": timeframe,
                "min_bars": max(warmup + 50, 260),
            }
        ]

    def describe(self) -> str:
        entry_period = resolve_donchian_period(self.params)
        exit_period = _resolve_exit_period(self.params)
        ema_period = _resolve_ema_period(self.params)
        adx_period = int(self.params.get("adx_period", 14))
        adx_min = float(self.params.get("adx_min", 25))
        return (
            f"Long-only breakout that buys a {entry_period}-bar Donchian high when price is "
            f"above the {ema_period}-bar EMA and ADX({adx_period}) is at least {adx_min:g}. "
            f"It exits on a {exit_period}-bar Donchian low or a close back under the EMA trend filter."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        entry_period = resolve_donchian_period(self.params)
        exit_period = _resolve_exit_period(self.params)
        ema_period = _resolve_ema_period(self.params)
        adx_period = int(self.params.get("adx_period", 14))
        adx_min = float(self.params.get("adx_min", 25))

        curr_close = float(df["close"].iloc[-1])
        min_bars = max(entry_period, exit_period, ema_period, adx_period) + 2
        if len(df) < min_bars:
            return Signal(
                entry_signal=False,
                exit_signal=False,
                price=round(curr_close, 4),
                direction="long",
                confidence=0.0,
            )

        close = df["close"]
        ema = close.ewm(span=ema_period, adjust=False).mean()
        adx_val = compute_adx(df, adx_period)
        upper_prev, _, _ = donchian_bands(df, entry_period)
        _, _, exit_lower = donchian_bands(df, exit_period)

        prev_close = float(close.iloc[-2])
        curr_upper = upper_prev.iloc[-1]
        curr_exit_lower = exit_lower.iloc[-1]
        curr_ema = ema.iloc[-1]
        curr_adx = adx_val.iloc[-1]

        if (
            pd.isna(curr_upper)
            or pd.isna(curr_exit_lower)
            or pd.isna(curr_ema)
            or pd.isna(curr_adx)
        ):
            return Signal(
                entry_signal=False,
                exit_signal=False,
                price=round(curr_close, 4),
                direction="long",
                confidence=0.0,
            )

        upper_value = float(curr_upper)
        exit_lower_value = float(curr_exit_lower)
        ema_value = float(curr_ema)
        adx_value = float(curr_adx)

        trend_ok = curr_close > ema_value and adx_value >= adx_min
        entry = prev_close <= upper_value and curr_close > upper_value and trend_ok
        exit_ = (prev_close >= exit_lower_value and curr_close < exit_lower_value) or curr_close < ema_value

        confidence = 0.0
        if entry:
            confidence = min(1.0, max(0.0, (adx_value - adx_min + 10.0) / 25.0))

        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(confidence, 3),
            indicators={
                "upper": round(upper_value, 4),
                "exit_lower": round(exit_lower_value, 4),
                "ema": round(ema_value, 4),
                "adx": round(adx_value, 2),
                "trend_ok": bool(trend_ok),
            },
        )

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        entry_period = resolve_donchian_period(self.params)
        exit_period = _resolve_exit_period(self.params)
        ema_period = _resolve_ema_period(self.params)
        adx_period = int(self.params.get("adx_period", 14))
        adx_min = float(self.params.get("adx_min", 25))

        close = df["close"]
        close_prev = close.shift(1)
        ema = close.ewm(span=ema_period, adjust=False).mean()
        adx_val = compute_adx(df, adx_period)
        upper_prev, _, _ = donchian_bands(df, entry_period)
        _, _, exit_lower = donchian_bands(df, exit_period)

        trend_ok = (close > ema) & (adx_val >= adx_min)
        entry_signals = (close_prev <= upper_prev) & (close > upper_prev) & trend_ok
        exit_signals = ((close_prev >= exit_lower) & (close < exit_lower)) | (close < ema)

        return entry_signals.fillna(False), exit_signals.fillna(False)

    def parameter_space(self) -> dict:
        return {
            "period": (40, 100, 15),
            "exit_period": (10, 30, 10),
            "adx_min": (20, 30, 5),
            "ema_period": (150, 200, 50),
        }


STRATEGY_CLASS = DonchianRegimeStrategy

STRATEGIES = [
    ("DREG-BTC", DonchianRegimeStrategy, {"_asset": "BTC"}),
    ("DREG-ETH", DonchianRegimeStrategy, {"_asset": "ETH"}),
]
