"""Stochastic Oscillator strategy — both long and short signals."""

import pandas as pd

from forven.strategies.base import BaseStrategy, DirectionalSignals, Signal
from forven.strategies.indicators import atr, stochastic

TYPE_NAME = "stochastic"


class StochasticStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        d = self.params.get("direction", "long")
        return f"Stochastic {d.upper()} ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "k_period": 14,
            "d_period": 3,
            "k_oversold": 20,
            "k_overbought": 80,
            "k_exit_oversold": 40,
            "k_exit_overbought": 60,
            "direction": "long",
            "leverage": 3.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN", "RANGE_BOUND"}

    def describe(self) -> str:
        p = self.params
        direction = p.get("direction", "long")
        if direction == "long":
            return (
                f"Buys when the {p['k_period']}-period Stochastic bounces from "
                f"oversold (below {p['k_oversold']}). "
                f"Sells at overbought (above {p['k_overbought']})."
            )
        return (
            f"Shorts when the {p['k_period']}-period Stochastic drops from "
            f"overbought (above {p['k_overbought']}). "
            f"Covers at oversold (below {p['k_oversold']})."
        )

    def generate_signals(self, df: pd.DataFrame) -> DirectionalSignals:
        """Vectorized twin of generate_signal — the SINGLE source of entry/exit logic.

        Mirrors generate_signal's per-bar %K cross logic condition-for-condition for
        BOTH the long and short branches (the class is direction-parametrized). No ADX
        or volume filter is applied, matching generate_signal exactly.
        """
        p = self.params

        if len(df) < 2:
            return DirectionalSignals.empty(df.index)

        stoch = stochastic(df, int(p.get("k_period", 14)), int(p.get("d_period", 3)))
        stoch_k = stoch["stoch_k"]
        prev_stoch_k = stoch_k.shift(1)

        k_oversold = float(p.get("k_oversold", 20))
        k_overbought = float(p.get("k_overbought", 80))
        k_exit_oversold = float(p.get("k_exit_oversold", 40))
        k_exit_overbought = float(p.get("k_exit_overbought", 60))

        # Long branch: entry on %K crossing up through oversold; exit at overbought
        # OR %K crossing down through the long exit-oversold band.
        long_entry = (prev_stoch_k < k_oversold) & (stoch_k >= k_oversold)
        long_exit = (stoch_k >= k_overbought) | (
            (prev_stoch_k >= k_exit_oversold) & (stoch_k < k_exit_oversold)
        )

        # Short branch: entry on %K crossing down through overbought; exit at oversold
        # OR %K crossing up through the short exit-overbought band.
        short_entry = (prev_stoch_k > k_overbought) & (stoch_k <= k_overbought)
        short_exit = (stoch_k <= k_oversold) | (
            (prev_stoch_k <= k_exit_overbought) & (stoch_k > k_exit_overbought)
        )

        return DirectionalSignals(
            long_entries=long_entry.fillna(False),
            long_exits=long_exit.fillna(False),
            short_entries=short_entry.fillna(False),
            short_exits=short_exit.fillna(False),
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        p = self.params

        stoch = stochastic(df, int(p.get("k_period", 14)), int(p.get("d_period", 3)))
        stoch_k = stoch["stoch_k"]
        stoch_d = stoch["stoch_d"]
        atr_14 = atr(df, 14)

        if len(df) < 2:
            return Signal(
                entry_signal=False, exit_signal=False,
                price=round(float(df["close"].iloc[-1]), 4),
                direction=p.get("direction", "long"),
                confidence=0.0, indicators={}
            )

        curr_close = float(df["close"].iloc[-1])
        curr_stoch_k = float(stoch_k.iloc[-1])
        curr_stoch_d = float(stoch_d.iloc[-1])
        curr_atr = float(atr_14.iloc[-1])

        direction = p.get("direction", "long")

        # Decision comes from the vectorized twin — pick the side this instance trades.
        signals = self.generate_signals(df)
        if direction == "long":
            entry = bool(signals.long_entries.iloc[-1])
            exit_ = bool(signals.long_exits.iloc[-1])
        else:
            entry = bool(signals.short_entries.iloc[-1])
            exit_ = bool(signals.short_exits.iloc[-1])

        return Signal(
            entry_signal=bool(entry), exit_signal=bool(exit_),
            price=round(curr_close, 4), direction=direction,
            confidence=min(1.0, abs(curr_stoch_k - curr_stoch_d) / 20) if entry else 0.0,
            indicators={
                "stoch_k": round(curr_stoch_k, 1),
                "stoch_d": round(curr_stoch_d, 1),
                "atr_14": round(curr_atr, 6),
            },
        )

    def parameter_space(self) -> dict:
        return {
            "k_oversold": [10, 30, 5],
            "k_overbought": [70, 90, 5],
            "k_period": [10, 20, 2],
        }


STRATEGY_CLASS = StochasticStrategy

STRATEGIES = [
    ["S020-BTC-LONG", StochasticStrategy, {"_asset": "BTC", "direction": "long"}],
    ["S020-BTC-SHORT", StochasticStrategy, {"_asset": "BTC", "direction": "short"}],
    ["S020-ETH-LONG", StochasticStrategy, {"_asset": "ETH", "direction": "long"}],
    ["S020-ETH-SHORT", StochasticStrategy, {"_asset": "ETH", "direction": "short"}],
    ["S020-SOL-LONG", StochasticStrategy, {"_asset": "SOL", "direction": "long"}],
    ["S020-SOL-SHORT", StochasticStrategy, {"_asset": "SOL", "direction": "short"}],
]
