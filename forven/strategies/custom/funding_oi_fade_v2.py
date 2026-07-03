from __future__ import annotations

"""Funding-Rate Crowding Fades with OI Confirmation (H00012 v2).

HYP-e5423215cbc6 (display H00012). Refines v1 `funding` family
(taxonomy: `funding @ quick_screen: no_metrics_error x3` - signal-starvation).
Mechanism v2 directly addresses that.

ENTRY SHORT: funding>+0.0003 AND OI rises over 24h AND close>VWAP(20)
ENTRY LONG : funding<-0.0003 AND OI rises over 24h AND close<VWAP(20)
EXIT       : funding returns to +/-0.0001 band; 48-bar time stop; 2.5xATR(14).
SIZING     : |funding| in [0.0003, 0.0006] full; >0.0006 half-Kelly.
NO EMA/RSI/MACD/Williams %R filters (LESSONS S00013/S00014 trap).
funding_rate + open_interest are enrichment-joined, .notna() gated.
"""

from typing import ClassVar

import numpy as np
import pandas as pd

from forven.strategies.base import BaseStrategy, Signal, DirectionalSignals


def _vwap(close, high, low, volume, window: int) -> pd.Series:
    if window <= 0:
        return pd.Series(np.nan, index=close.index)
    typ = (high + low + close) / 3.0
    pv = typ * volume
    roll_pv = pv.rolling(window, min_periods=window).sum()
    roll_v = volume.rolling(window, min_periods=window).sum()
    return roll_pv / roll_v.replace(0.0, np.nan)


def _atr(high, low, close, window: int) -> pd.Series:
    if window <= 0:
        return pd.Series(np.nan, index=close.index)
    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_c).abs(),
         (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


class FundingOIFadeRefinedStrategy(BaseStrategy):
    FAMILY: ClassVar[str] = "funding_carry"
    SUPPORTED_ASSETS: ClassVar[tuple[str, ...]] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    SUPPORTED_TIMEFRAMES: ClassVar[tuple[str, ...]] = ("1h",)

    DEFAULT_PARAMS: ClassVar[dict] = {
        "funding_threshold": 0.0003,
        "oi_lookback_bars": 24,
        "vwap_window": 20,
        "funding_neutral_band": 0.0001,
        "time_stop_bars": 48,
        "adverse_atr_multiple": 2.5,
        "atr_window": 14,
        "funding_full_kelly_cap": 0.0006,
        "execution_profile": {
            "sizing_mode": "atr",
            "atr_stop_multiplier": 2.5,
            "time_stop_bars": 48,
        },
    }

    @property
    def name(self) -> str:
        return "funding_oi_fade_v2"

    @property
    def asset(self) -> str:
        return "MULTI_PERP_FUNDING"

    @property
    def strategy_type(self) -> str:
        return "funding_oi_fade_v2"

    @property
    def default_params(self) -> dict:
        return dict(self.DEFAULT_PARAMS)

    @property
    def supported_trade_modes(self) -> set:
        return {"long_only", "short_only", "both"}

    def __init__(self, *args, **kwargs):
        params = None
        if args:
            if isinstance(args[0], dict):
                params = args[0]
            elif len(args) > 1 and isinstance(args[1], dict):
                params = args[1]
        if params is None and "params" in kwargs and isinstance(kwargs["params"], dict):
            params = kwargs["params"]
        elif params is None and kwargs and any(k in self.DEFAULT_PARAMS for k in kwargs):
            params = {k: v for k, v in kwargs.items() if k in self.DEFAULT_PARAMS}
        merged: dict = dict(self.DEFAULT_PARAMS)
        if params:
            merged.update(params)
        if args and isinstance(args[0], str):
            super().__init__(args[0], *args[1:], **kwargs)
        elif "strategy_id" in kwargs:
            super().__init__(**kwargs)
        else:
            super().__init__()
        self.params = merged
        self.config = merged

    def generate_signals(self, df: pd.DataFrame) -> DirectionalSignals:
        params = self.params
        ft = float(params["funding_threshold"])
        cap = float(params["funding_full_kelly_cap"])
        neutral = float(params["funding_neutral_band"])

        funding = df.get("funding_rate")
        oi = df.get("open_interest")
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        if (
            funding is None or oi is None
            or not funding.notna().any() or not oi.notna().any()
        ):
            return DirectionalSignals.empty(df.index)

        vwap = _vwap(close, high, low, volume, int(params["vwap_window"]))

        oi_24h_ago = oi.shift(int(params["oi_lookback_bars"]))
        oi_rising = oi.notna() & oi_24h_ago.notna() & (oi > oi_24h_ago)

        f_abs = funding.abs()
        is_extreme = f_abs.notna() & (f_abs > cap)
        is_full = f_abs.notna() & (f_abs >= ft) & ~is_extreme
        size_qualifies = is_full | is_extreme

        f_above = funding.notna() & (funding > ft)
        f_below = funding.notna() & (funding < -ft)
        ctx_short = close.notna() & vwap.notna() & (close > vwap)
        ctx_long = close.notna() & vwap.notna() & (close < vwap)

        entry_long = f_below & oi_rising & ctx_long & size_qualifies
        entry_short = f_above & oi_rising & ctx_short & size_qualifies

        carry_unwound = funding.notna() & (f_abs <= neutral)
        exit_ = carry_unwound.astype(bool)
        return DirectionalSignals(
            long_entries=entry_long,
            long_exits=exit_,
            short_entries=entry_short,
            short_exits=exit_,
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if df is None or df.empty:
            return Signal(0)
        signals = self.generate_signals(df)
        if signals is None:
            return Signal(0)
        has_long = not signals.long_entries.empty and bool(signals.long_entries.iloc[-1])
        has_short = not signals.short_entries.empty and bool(signals.short_entries.iloc[-1])
        if not has_long and not has_short:
            return Signal(0)
        funding = df.get("funding_rate")
        if funding is None or funding.isna().iloc[-1]:
            return Signal(0)
        f = float(funding.iloc[-1])
        close = df["close"].iloc[-1]
        vwap_series = _vwap(df["close"], df["high"], df["low"], df["volume"],
                            int(self.params["vwap_window"]))
        vwap_val = vwap_series.iloc[-1]
        if pd.isna(vwap_val):
            return Signal(0)
        if f > 0 and close > vwap_val and has_short:
            return Signal(-1)
        if f < 0 and close < vwap_val and has_long:
            return Signal(1)
        return Signal(0)


STRATEGY_CLASS = FundingOIFadeRefinedStrategy
TYPE_NAME = "funding_oi_fade_v2"
