"""Keltner Channel Breakout strategy - S025, S00019, S00027 variants.

Supports both LONG and SHORT position modes.
"""

import pandas as pd

from forven.strategies.base import BaseStrategy, DirectionalSignals, Signal

TYPE_NAME = "keltner"


class KeltnerStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return f"Keltner Channel Breakout ({self.asset})"

    @property
    def asset(self) -> str:
        return self.params.get("_asset", "ETH")

    @property
    def strategy_type(self) -> str:
        return TYPE_NAME

    @property
    def default_params(self) -> dict:
        return {
            "kc_period": 20, "kc_mult": 2.0,
            "adx_period": 14, "adx_min": 20, "leverage": 3.0,
            "position": "long",  # "long" or "short"
        }

    @property
    def compatible_regimes(self) -> set[str]:
        position = self.params.get("position", "long")
        if position == "short":
            return {"TREND_DOWN"}
        return {"TREND_UP"}

    def describe(self) -> str:
        p = self.params
        kp = p.get("keltner_period") or p.get("keltner_window") or p.get("kc_period", 20)
        km = p.get("keltner_mult") or p.get("keltner_multiplier") or p.get("kc_mult", 2.0)
        position = p.get("position", "long")
        
        if position == "short":
            return (
                f"Shorts when price breaks below the lower Keltner Channel "
                f"({kp}-period, {km}x ATR) in a downtrend. "
                f"Covers when price rises to the middle line."
            )
        return (
            f"Buys when price breaks above the upper Keltner Channel "
            f"({kp}-period, {km}x ATR) in an uptrend. "
            f"Sells when price falls to the middle line."
        )

    def _bands(self, df: pd.DataFrame):
        """Compute the Keltner channel bands (mid/upper/lower) the SAME way the
        per-bar method does, resolving the param-name aliases once."""
        p = self.params
        # Support multiple naming conventions
        kp = (
            p.get("keltner_period") or
            p.get("keltner_window") or
            p.get("kc_period", 20)
        )
        km = (
            p.get("keltner_mult") or
            p.get("keltner_multiplier") or
            p.get("kc_mult", 2.0)
        )
        # Also support atr_multiplier as alias for keltner multiplier
        if p.get("atr_multiplier"):
            km = p.get("atr_multiplier")

        close = df["close"]
        kc_mid = close.ewm(span=kp, adjust=False).mean()
        h, low_p, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - low_p), (h - c.shift()).abs(), (low_p - c.shift()).abs()], axis=1).max(axis=1)
        atr_kc = tr.ewm(span=kp, adjust=False).mean()
        kc_upper = kc_mid + km * atr_kc
        kc_lower = kc_mid - km * atr_kc
        return kp, km, kc_mid, kc_upper, kc_lower

    def generate_signals(self, df: pd.DataFrame) -> DirectionalSignals:
        """Vectorized twin of ``generate_signal`` — the SINGLE source of entry/exit
        logic so the backtest and the live/paper scanner trade the identical signal
        set. ``generate_signal`` delegates here for its boolean decision.

        Long (position="long"): enter when close crosses up through the upper
        Keltner band (prev close at/below, current above) while ADX clears its
        floor; exit when close falls below the channel mid. Short (position="short")
        mirrors it: enter on a cross down through the lower band, cover when close
        rises back above the mid.
        """
        from forven.strategies.indicators import adx

        p = self.params
        adx_period = p.get("adx_period", 14)
        adx_min = p.get("adx_min", 20)
        position = p.get("position", "long")  # "long" or "short"

        close = df["close"]
        kp, _, kc_mid, kc_upper, kc_lower = self._bands(df)

        # Optional: ADX filter for regime detection. When disabled the per-bar
        # method uses curr_adx=50, which always clears adx_min for the defaults,
        # so the vectorized equivalent is an all-True gate.
        use_adx_filter = p.get("use_adx_filter", True)
        if use_adx_filter:
            adx_ok = adx(df, adx_period) >= adx_min
        else:
            adx_ok = pd.Series(50 >= adx_min, index=df.index)

        prev_close = close.shift(1)
        prev_upper = kc_upper.shift(1)
        prev_lower = kc_lower.shift(1)

        # Mirror the per-bar method's warmup guard (`if len(df) < kp + 2: neutral`):
        # the first kp+1 bars (positional index < kp+1) emit no signal at all.
        ready = pd.Series(range(len(df)), index=df.index) >= (kp + 1)

        empty = pd.Series(False, index=df.index, dtype=bool)

        if position == "short":
            short_entries = (close < kc_lower) & (prev_close >= prev_lower) & adx_ok & ready
            short_exits = (close > kc_mid) & ready
            return DirectionalSignals(
                long_entries=empty.copy(),
                long_exits=empty.copy(),
                short_entries=short_entries.fillna(False),
                short_exits=short_exits.fillna(False),
            )

        long_entries = (close > kc_upper) & (prev_close <= prev_upper) & adx_ok & ready
        long_exits = (close < kc_mid) & ready
        return DirectionalSignals(
            long_entries=long_entries.fillna(False),
            long_exits=long_exits.fillna(False),
            short_entries=empty.copy(),
            short_exits=empty.copy(),
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        from forven.strategies.indicators import adx
        p = self.params

        adx_period = p.get("adx_period", 14)
        position = p.get("position", "long")  # "long" or "short"

        close = df["close"]
        kp, km, kc_mid, kc_upper, kc_lower = self._bands(df)

        # Optional: ADX filter for regime detection
        use_adx_filter = p.get("use_adx_filter", True)
        if use_adx_filter:
            adx_val = adx(df, adx_period)
            curr_adx = float(adx_val.iloc[-1])
        else:
            curr_adx = 50  # Default to allowing signals if ADX filter disabled

        # Check if we have enough data
        if len(df) < kp + 2:
            return Signal(
                entry_signal=False, exit_signal=False,
                price=float(close.iloc[-1]), direction=position, confidence=0.0,
            )

        curr_close = close.iloc[-1]
        curr_mid = kc_mid.iloc[-1]

        # Single source of truth for the decision (keeps per-bar and vectorized in lockstep).
        sig = self.generate_signals(df)
        if position == "short":
            entry_now = bool(sig.short_entries.iloc[-1])
            exit_now = bool(sig.short_exits.iloc[-1])
        else:
            entry_now = bool(sig.long_entries.iloc[-1])
            exit_now = bool(sig.long_exits.iloc[-1])

        if position == "short":
            # SHORT SIGNAL: price breaks below lower Keltner channel in downtrend
            if entry_now:
                return Signal(
                    entry_signal=True,
                    exit_signal=False,
                    price=float(curr_close),
                    direction="short",
                    confidence=0.7,
                    indicators={"kc_upper": float(kc_upper.iloc[-1]), "kc_lower": float(kc_lower.iloc[-1]), "kc_mid": float(curr_mid), "adx": curr_adx}
                )

            # EXIT SHORT: price rises above middle line
            if exit_now:
                return Signal(
                    entry_signal=False,
                    exit_signal=True,
                    price=float(curr_close),
                    direction="short",
                    confidence=1.0,
                    indicators={"kc_mid": float(curr_mid)}
                )
        else:
            # LONG SIGNAL: price breaks above upper Keltner channel in uptrend
            if entry_now:
                return Signal(
                    entry_signal=True,
                    exit_signal=False,
                    price=float(curr_close),
                    direction="long",
                    confidence=0.7,
                    indicators={"kc_upper": float(kc_upper.iloc[-1]), "kc_lower": float(kc_lower.iloc[-1]), "kc_mid": float(curr_mid), "adx": curr_adx}
                )

            # EXIT LONG: price falls below middle line
            if exit_now:
                return Signal(
                    entry_signal=False,
                    exit_signal=True,
                    price=float(curr_close),
                    direction="long",
                    confidence=1.0,
                    indicators={"kc_mid": float(curr_mid)}
                )

        return Signal(
            entry_signal=False, exit_signal=False,
            price=float(curr_close), direction=position, confidence=0.0,
        )


STRATEGY_CLASS = KeltnerStrategy
