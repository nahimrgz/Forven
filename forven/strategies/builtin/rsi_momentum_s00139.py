"""RSI Momentum Strategy - S00139.

Strategy: BTC-RSI_MOMENTUM-S00139
Entry: Price above SMA and RSI crosses above rsi_oversold threshold (momentum reversal)
Exit: RSI crosses above rsi_overbought threshold or price below SMA

Based on backtest results:
- Sharpe: 0.97
- Win Rate: 68.6%
- Profit Factor: 1.18
- Max Drawdown: 42.40%
- Fitness: 42.3
"""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal
from forven.strategies.indicators import rsi as compute_rsi, atr


TYPE_NAME = "rsi_momentum"


class RSIMomentumS00139Strategy(BaseStrategy):
    """S00139: RSI Momentum Strategy for BTC."""

    @property
    def name(self) -> str:
        return f"RSI Momentum ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "BTC")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "sma_period": 20,
            "atr_period": 14,
            "leverage": 3.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP", "RANGE_BOUND", "VOLATILE"}

    def describe(self) -> str:
        p = self.params
        return (
            f"BTC S00139: RSI Momentum. Enter long when price is above "
            f"{p['sma_period']}-period SMA and RSI crosses above {p['rsi_oversold']} "
            f"(oversold). Exit when RSI crosses above {p['rsi_overbought']} "
            f"(overbought) or price falls below SMA."
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        p = self.params
        
        rsi_period = p.get("rsi_period", 14)
        rsi_oversold = p.get("rsi_oversold", 25)
        rsi_overbought = p.get("rsi_overbought", 75)
        sma_period = p.get("sma_period", 20)
        
        close = df["close"]
        
        # Calculate indicators
        sma = close.rolling(sma_period).mean()
        rsi_val = compute_rsi(close, rsi_period)
        atr_val = atr(df, p.get("atr_period", 14))
        
        # Need at least sma_period + 1 bars
        if len(df) < sma_period + 2:
            return Signal(
                entry_signal=False,
                exit_signal=False,
                price=round(float(close.iloc[-1]), 4) if len(df) > 0 else 0.0,
                direction="long",
                confidence=0.0,
                indicators={"rsi": 0.0, "sma": 0.0, "atr_14": 0.0},
            )
        
        # Get current and previous values
        curr_close = float(close.iloc[-1])
        curr_sma = float(sma.iloc[-1])
        curr_rsi = float(rsi_val.iloc[-1])
        prev_rsi = float(rsi_val.iloc[-2])
        
        curr_atr = float(atr_val.iloc[-1])
        
        # Entry: price above SMA AND RSI crosses above oversold threshold
        price_above_sma = curr_close > curr_sma
        rsi_cross_up = prev_rsi < rsi_oversold and curr_rsi >= rsi_oversold
        
        entry = price_above_sma and rsi_cross_up
        
        # Exit: RSI crosses above overbought threshold OR price falls below SMA
        rsi_cross_down = prev_rsi < rsi_overbought and curr_rsi >= rsi_overbought
        price_below_sma = curr_close < curr_sma
        
        exit_signal = rsi_cross_down or price_below_sma
        
        # Confidence based on RSI position (0-1 scale normalized to entry zone)
        if entry:
            rsi_buffer = curr_rsi - rsi_oversold
            confidence = min(1.0, rsi_buffer / 10.0)
        else:
            confidence = 0.0
        
        return Signal(
            entry_signal=bool(entry),
            exit_signal=bool(exit_signal),
            price=round(curr_close, 4),
            direction="long",
            confidence=round(confidence, 2),
            indicators={
                "rsi": round(curr_rsi, 1),
                "sma": round(curr_sma, 4),
                "atr_14": round(curr_atr, 6),
                "price_above_sma": price_above_sma,
            },
        )

    def get_stop_loss(self, signal: Signal) -> float | None:
        """Default ATR-based stop loss: 3x ATR below entry price."""
        if "atr_14" in signal.indicators and signal.price > 0:
            atr = signal.indicators.get("atr_14", 0)
            if atr > 0:
                return round(signal.price - 3 * atr, 4)
        return None

    def parameter_space(self) -> dict:
        return {
            "rsi_period": (10, 20, 2),
            "rsi_oversold": (20, 35, 5),
            "rsi_overbought": (65, 80, 5),
            "sma_period": (15, 30, 5),
        }


STRATEGY_CLASS = RSIMomentumS00139Strategy

STRATEGIES = [
    ("S00139-BTC-RSI_MOMENTUM", RSIMomentumS00139Strategy, {"_asset": "BTC"}),
]
