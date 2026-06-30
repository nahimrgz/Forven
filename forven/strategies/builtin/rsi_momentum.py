"""RSI Momentum strategy — S012 variants."""

import pandas as pd

from forven.strategies.base import BaseStrategy, Signal

TYPE_NAME = "rsi_momentum"


class RSIMomentumStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return f"RSI+ADX+EMA50+EMA200 ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "ETH")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "rsi_period": 14, 
            "rsi_entry": 40,  # oversold threshold
            "rsi_exit": 60,   # overbought threshold
            "ema_fast": 50, "ema_slow": 200,
            "adx_period": 14, "adx_min": 0, "leverage": 3.0,
        }

    @property
    def compatible_regimes(self) -> set[str]:
        return {"TREND_UP"}

    def describe(self) -> str:
        p = self.params
        return (
            f"Buys when the {p['rsi_period']}-period RSI bounces up from below {p['rsi_entry']} "
            f"while price is above the {p['ema_fast']} and {p['ema_slow']}-bar moving averages. "
            f"Sells when RSI drops below {p['rsi_exit']}."
        )

    def _get_oversold(self) -> float:
        """Get oversold threshold - supports both 'rsi_entry' and 'oversold' param names."""
        p = self.params
        if "oversold" in p:
            return p["oversold"]
        if "rsi_entry" in p:
            return p["rsi_entry"]
        return self.default_params["rsi_entry"]

    def _get_overbought(self) -> float:
        """Get overbought threshold - supports both 'rsi_exit' and 'overbought' param names."""
        p = self.params
        if "overbought" in p:
            return p["overbought"]
        if "rsi_exit" in p:
            return p["rsi_exit"]
        return self.default_params["rsi_exit"]

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Vectorized twin of ``generate_signal`` — the SINGLE source of entry/exit
        logic so the backtest and the live/paper scanner trade the identical signal
        set. ``generate_signal`` delegates here for its boolean decision.

        Entry: RSI crosses up through the oversold threshold while price is above
        both EMAs and ADX clears its floor. Exit: RSI crosses down through the
        overbought threshold.
        """
        from forven.strategies.indicators import adx, rsi

        p = self.params
        close = df["close"]
        rsi_val = rsi(close, p["rsi_period"])
        ema_fast = close.ewm(span=p["ema_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["ema_slow"], adjust=False).mean()
        adx_val = adx(df, p["adx_period"])

        oversold = self._get_oversold()
        overbought = self._get_overbought()

        prev_rsi = rsi_val.shift(1)
        trend_ok = (close > ema_fast) & (close > ema_slow)
        adx_ok = adx_val >= p["adx_min"]

        entry = (prev_rsi < oversold) & (rsi_val >= oversold) & trend_ok & adx_ok
        exit_ = (prev_rsi >= overbought) & (rsi_val < overbought)
        return entry.fillna(False), exit_.fillna(False)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import adx, atr, rsi
        p = self.params
        close = df["close"]

        rsi_val = rsi(close, p["rsi_period"])
        ema_fast = close.ewm(span=p["ema_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["ema_slow"], adjust=False).mean()
        adx_val = adx(df, p["adx_period"])
        atr_14 = atr(df, 14)

        # Single source of truth for the decision (keeps per-bar and vectorized in lockstep).
        entries, exits = self.generate_signals(df)

        curr_close = float(close.iloc[-1])
        curr_rsi = float(rsi_val.iloc[-1])
        curr_ema_fast = float(ema_fast.iloc[-1])
        curr_ema_slow = float(ema_slow.iloc[-1])
        curr_adx = float(adx_val.iloc[-1])
        curr_atr = float(atr_14.iloc[-1])

        oversold = self._get_oversold()
        overbought = self._get_overbought()
        trend_ok = curr_close > curr_ema_fast and curr_close > curr_ema_slow
        adx_ok = curr_adx >= p["adx_min"]

        return Signal(
            entry_signal=bool(entries.iloc[-1]), exit_signal=bool(exits.iloc[-1]),
            price=round(curr_close, 4), direction="long",
            confidence=min(1.0, curr_adx / 40) if adx_ok else 0.0,
            indicators={
                "rsi": round(curr_rsi, 1),
                "ema_fast": round(curr_ema_fast, 4),
                "ema_slow": round(curr_ema_slow, 4),
                "adx": round(curr_adx, 1),
                "trend_ok": bool(trend_ok),
                "atr_14": round(curr_atr, 6),
                "oversold": oversold,
                "overbought": overbought,
            },
        )

    def parameter_space(self) -> dict:
        return {
            "rsi_entry": (30, 45, 5),
            "rsi_exit": (60, 80, 5),
            "adx_min": (0, 15, 5),
        }


STRATEGY_CLASS = RSIMomentumStrategy

STRATEGIES = [
    ("S012-ETH", RSIMomentumStrategy, {"_asset": "ETH"}),
    ("S012-SOL", RSIMomentumStrategy, {"_asset": "SOL"}),
    ("S012-BTC", RSIMomentumStrategy, {"_asset": "BTC"}),
]
