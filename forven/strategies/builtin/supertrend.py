"""SuperTrend strategy."""
import pandas as pd
from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "supertrend"

class SuperTrendStrategy(BaseStrategy):
    @property
    def name(self) -> str: return f"SuperTrend ({self.asset})"
    @property
    def asset(self) -> str: return self.params.get("_asset", "BTC")
    @property
    def strategy_type(self) -> str: return TYPE_NAME
    @property
    def default_params(self) -> dict:
        return {"atr_period": 10, "multiplier": 3.0, "leverage": 3.0}
    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "TREND_DOWN"}

    def describe(self) -> str:
        p = self.params
        return f"Buys when price crosses above SuperTrend ({p['atr_period']}, {p['multiplier']}), sells when crosses below."

    def generate_signals(self, df: pd.DataFrame):
        """Vectorized twin of generate_signal — the SINGLE source of entry/exit logic.

        Real SuperTrend (final-band trend state machine) per the class's documented
        intent ("buys when price crosses above SuperTrend, sells when it crosses
        below") and the canonical ``indicators._f_supertrend`` implementation: go long
        when the trend flips up, exit when it flips down.

        NOTE (parity overhaul): the previous class body computed a *basic*-band cross
        (``close > hl2 + mult*ATR``) which can essentially never trigger (the band sits
        above the bar's own high), so the class was non-functional and disagreed with
        the backtest's trend-state-machine. This restores the correct, single shared
        algorithm used by both engines and the chart.
        """
        from forven.strategies.indicators import compute_indicator

        p = self.params
        out = compute_indicator(
            df,
            {"id": "st", "kind": "supertrend",
             "params": {"length": p.get("atr_period", 10), "mult": p.get("multiplier", 3.0)}},
        )
        direction = out["st_dir"]
        prev_dir = direction.shift(1)
        entry = (prev_dir < 0) & (direction > 0)
        exit_ = (prev_dir > 0) & (direction < 0)
        return entry.fillna(False), exit_.fillna(False)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr
        p = self.params
        close = df["close"]

        atr_val = atr(df, p["atr_period"])

        curr_close = float(close.iloc[-1])

        entries, exits = self.generate_signals(df)

        return Signal(
            entry_signal=bool(entries.iloc[-1]), exit_signal=bool(exits.iloc[-1]),
            price=round(curr_close, 4), direction="long", confidence=1.0,
            indicators={"atr": round(float(atr_val.iloc[-1]), 4)}
        )

    def parameter_space(self) -> dict:
        return {"atr_period": (10, 20, 5), "multiplier": (2.0, 4.0, 0.5)}

STRATEGY_CLASS = SuperTrendStrategy
STRATEGIES = [("TOMB-SUPERTREND", SuperTrendStrategy, {"_asset": "BTC"})]
