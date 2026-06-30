"""Bollinger Band Breakout strategy — S026."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "bollinger"


class BollingerStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return f"Bollinger Band Breakout ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "ETH")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "bb_period": 20, "bb_std": 2.0,
            "adx_period": 14, "adx_min": 20, "leverage": 3.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "RANGE_BOUND"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buys when price breaks above the upper Bollinger Band "
            f"({p['bb_period']}-period, {p['bb_std']} std dev) while in an uptrend. "
            f"Sells when price falls back to the middle band."
        )

    def generate_signals(self, df: pd.DataFrame):
        """Vectorized twin of generate_signal — the SINGLE source of entry/exit logic."""
        from forven.strategies.indicators import adx
        p = self.params
        bp = p.get("bb_period", 20)

        close = df["close"]
        bb_mid = close.rolling(bp).mean()
        bb_std = close.rolling(bp).std()
        bb_upper = bb_mid + p.get("bb_std", 2.0) * bb_std
        ema200 = close.ewm(span=200, adjust=False).mean()
        adx_val = adx(df, p.get("adx_period", 14))

        prev_close = close.shift(1)
        prev_bb_upper = bb_upper.shift(1)

        breakout = (prev_close <= prev_bb_upper) & (close > bb_upper)
        entry = breakout & (close > ema200) & (adx_val >= p.get("adx_min", 20))
        exit_ = close < bb_mid

        return entry.fillna(False), exit_.fillna(False)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import adx, atr
        p = self.params
        bp = p.get("bb_period", 20)

        close = df["close"]
        bb_mid = close.rolling(bp).mean()
        bb_std = close.rolling(bp).std()
        bb_upper = bb_mid + p.get("bb_std", 2.0) * bb_std
        ema200 = close.ewm(span=200, adjust=False).mean()
        adx_val = adx(df, p.get("adx_period", 14))
        atr_14 = atr(df, 14)

        # Get values for current and previous bar
        curr_close = float(close.iloc[-1])

        curr_bb_upper = float(bb_upper.iloc[-1])
        curr_bb_mid = float(bb_mid.iloc[-1])
        curr_ema200 = float(ema200.iloc[-1])
        curr_adx = float(adx_val.iloc[-1])
        curr_atr = float(atr_14.iloc[-1])

        entries, exits = self.generate_signals(df)
        entry = bool(entries.iloc[-1])
        exit_ = bool(exits.iloc[-1])

        return Signal(
            entry_signal=bool(entry), exit_signal=bool(exit_),
            price=round(curr_close, 4), direction="long",
            confidence=min(1.0, curr_adx / 40) if entry else 0.0,
            indicators={
                "bb_mid": round(curr_bb_mid, 4),
                "bb_upper": round(curr_bb_upper, 4),
                "ema200": round(curr_ema200, 4),
                "adx": round(curr_adx, 1),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "bb_period": (15, 25, 5),
            "bb_std": (1.5, 3.0, 0.5),
            "adx_min": (15, 25, 5),
        }


STRATEGY_CLASS = BollingerStrategy

STRATEGIES = [
    ("S026-BB-ETH", BollingerStrategy, {"_asset": "ETH"}),
]
