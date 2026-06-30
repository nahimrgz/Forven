"""MACD Cross strategy — S030."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "macd"


class MACDStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return f"MACD {self.params.get('fast', 5)}/{self.params.get('slow', 13)}/{self.params.get('signal', 3)} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "ETH")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "fast": 5, "slow": 13, "signal": 3,
            "ema_regime": 200, "adx_min": 20, "leverage": 3.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Uses MACD ({p['fast']}/{p['slow']}/{p['signal']}) to track momentum. "
            f"Buys when MACD crosses above the signal line in an uptrend. "
            f"Sells on the reverse crossover."
        )

    def generate_signals(self, df: pd.DataFrame):
        """Vectorized twin of generate_signal — the SINGLE source of entry/exit logic."""
        from forven.strategies.indicators import adx
        p = self.params
        close = df["close"]

        ema_fast = close.ewm(span=p["fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["slow"], adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=p["signal"], adjust=False).mean()
        ema_regime = close.ewm(span=p.get("ema_regime", 200), adjust=False).mean()
        adx_val = adx(df, p.get("adx_period", 14))

        prev_macd = macd.shift(1)
        prev_macd_signal = macd_signal.shift(1)

        cross_up = (prev_macd <= prev_macd_signal) & (macd > macd_signal)
        cross_down = (prev_macd >= prev_macd_signal) & (macd < macd_signal)
        entry = cross_up & (close > ema_regime) & (adx_val >= p.get("adx_min", 20))
        exit_ = cross_down

        return entry.fillna(False), exit_.fillna(False)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import adx, atr
        p = self.params
        close = df["close"]

        ema_fast = close.ewm(span=p["fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["slow"], adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=p["signal"], adjust=False).mean()
        ema_regime = close.ewm(span=p.get("ema_regime", 200), adjust=False).mean()
        adx_val = adx(df, p.get("adx_period", 14))
        atr_14 = atr(df, 14)

        curr_close = float(close.iloc[-1])
        curr_macd = float(macd.iloc[-1])
        curr_macd_signal = float(macd_signal.iloc[-1])
        curr_ema_regime = float(ema_regime.iloc[-1])
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
                "macd": round(curr_macd, 4),
                "macd_signal": round(curr_macd_signal, 4),
                "ema_regime": round(curr_ema_regime, 4),
                "adx": round(curr_adx, 1),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "fast": (3, 8, 1),
            "slow": (10, 21, 2),
            "signal": (2, 5, 1),
        }


STRATEGY_CLASS = MACDStrategy

STRATEGIES = [
    ("S030-MACD-ETH", MACDStrategy, {"_asset": "ETH"}),
]
