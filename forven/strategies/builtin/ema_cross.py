"""EMA Cross strategy - S016, S018, S00011 variants.

Strategy Container S00011: EMA 20/50 Cross on SOL/USDT
- Entry: Fast EMA (20) crosses above Slow EMA (50)
- Exit: Fast EMA crosses below Slow EMA  
- Filters: ADX >= 20, price above 200 EMA regime filter
- Compatible Regimes: TREND_UP, TREND_DOWN
"""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "ema_cross"


class EMACrossStrategy(BaseStrategy):
    """EMA Crossover strategy with ADX trend filter and 200 EMA regime filter.
    
    Entry: Fast EMA crosses above Slow EMA while ADX >= min and price above regime EMA
    Exit: Fast EMA crosses below Slow EMA
    """

    @property
    def name(self) -> str:
        return f"EMA{self.params.get('ema_fast', 20)}/{self.params.get('ema_slow', 50)} Cross ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "SOL")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {"ema_fast": 20, "ema_slow": 50, "ema_regime": 200,
                "adx_period": 14, "adx_min": 20, "leverage": 3.0}

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (f"Buys when {p['ema_fast']}-bar EMA crosses above {p['ema_slow']}-bar EMA. "
                f"Uses {p['ema_regime']}-bar trend filter.")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import adx, atr
        p = self.params
        close = df["close"]
        
        ema_fast = close.ewm(span=p["ema_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["ema_slow"], adjust=False).mean()
        ema_regime = close.ewm(span=p["ema_regime"], adjust=False).mean()
        adx_val = adx(df, p["adx_period"])
        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_ema_fast = float(ema_fast.iloc[-1])
        prev_ema_fast = float(ema_fast.iloc[-2])
        curr_ema_slow = float(ema_slow.iloc[-1])
        prev_ema_slow = float(ema_slow.iloc[-2])
        curr_ema_regime = float(ema_regime.iloc[-1])
        curr_adx = float(adx_val.iloc[-1])
        curr_atr = float(atr_14.iloc[-1])

        regime_ok = curr_close > curr_ema_regime
        adx_ok = curr_adx >= p["adx_min"]
        cross_up = prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow
        cross_down = prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow
        entry = cross_up and regime_ok and adx_ok
        exit_ = cross_down

        return Signal(entry_signal=bool(entry), exit_signal=bool(exit_),
                     price=round(curr_close, 4), direction="long",
                     confidence=min(1.0, curr_adx / 40) if adx_ok else 0.0,
                     indicators={"ema_fast": round(curr_ema_fast, 4), "ema_slow": round(curr_ema_slow, 4),
                                 "ema_regime": round(curr_ema_regime, 4), "adx": round(curr_adx, 1),
                                 "regime_ok": bool(regime_ok), "atr_14": round(curr_atr, 6)})

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        from forven.strategies.indicators import adx

        p = self.params
        close = df["close"]
        ema_fast = close.ewm(span=p["ema_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["ema_slow"], adjust=False).mean()
        ema_regime = close.ewm(span=p["ema_regime"], adjust=False).mean()
        adx_val = adx(df, p["adx_period"])

        ema_fast_prev = ema_fast.shift(1)
        ema_slow_prev = ema_slow.shift(1)
        cross_up = (ema_fast_prev <= ema_slow_prev) & (ema_fast > ema_slow)
        cross_down = (ema_fast_prev >= ema_slow_prev) & (ema_fast < ema_slow)
        regime_ok = close > ema_regime
        adx_ok = adx_val >= p["adx_min"]

        entry_signals = cross_up & regime_ok & adx_ok
        exit_signals = cross_down

        return entry_signals.fillna(False), exit_signals.fillna(False)

    def parameter_space(self) -> dict:
        return {"ema_fast": (10, 30, 5), "ema_slow": (40, 60, 5), "adx_min": (15, 30, 5)}


STRATEGY_CLASS = EMACrossStrategy

STRATEGIES = [
    ("S016", EMACrossStrategy, {"_asset": "SOL"}),
    ("S018", EMACrossStrategy, {"_asset": "BTC"}),
    ("S00011", EMACrossStrategy, {"_asset": "SOL", "ema_fast": 20, "ema_slow": 50}),
]
