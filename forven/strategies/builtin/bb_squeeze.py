"""Bollinger Band Squeeze Breakout strategy."""
import pandas as pd
from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "bb_squeeze"

class BBSqueezeStrategy(BaseStrategy):
    @property
    def name(self) -> str: return f"BB Squeeze Breakout ({self.asset})"
    @property
    def asset(self) -> str: return self.params.get("_asset", "BTC")
    @property
    def strategy_type(self) -> str: return TYPE_NAME
    @property
    def default_params(self) -> dict:
        return {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_mult": 1.5, "leverage": 3.0}
    @property
    def compatible_regimes(self) -> set[str]:
        return {"RANGE_BOUND", "VOLATILE"}
    def describe(self) -> str:
        return "Detects a volatility squeeze (BB inside KC) and trades the breakout."
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import atr
        p = self.params
        close = df["close"]
        
        bb_mid = close.rolling(p["bb_period"]).mean()
        bb_std = close.rolling(p["bb_period"]).std()
        bb_upper = bb_mid + p["bb_std"] * bb_std
        bb_lower = bb_mid - p["bb_std"] * bb_std
        
        atr_val = atr(df, p["kc_period"])
        kc_upper = bb_mid + p["kc_mult"] * atr_val
        kc_lower = bb_mid - p["kc_mult"] * atr_val
        
        curr_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        
        sqz_on = (bb_upper.iloc[-2] < kc_upper.iloc[-2]) and (bb_lower.iloc[-2] > kc_lower.iloc[-2])
        breakout_up = prev_close <= bb_upper.iloc[-2] and curr_close > bb_upper.iloc[-1]
        
        entry = sqz_on and breakout_up
        exit_ = curr_close < bb_mid.iloc[-1]
        
        return Signal(
            entry_signal=bool(entry), exit_signal=bool(exit_),
            price=round(curr_close, 4), direction="long", confidence=1.0,
            indicators={"squeeze_on": bool(sqz_on)}
        )
    def parameter_space(self) -> dict:
        return {"bb_period": (15, 25, 5), "kc_mult": (1.0, 2.0, 0.5)}

STRATEGY_CLASS = BBSqueezeStrategy
STRATEGIES = [("TOMB-BBSQUEEZE", BBSqueezeStrategy, {"_asset": "BTC"})]
