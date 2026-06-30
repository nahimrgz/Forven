"""Golden-ORACLE tests for the execution engine: deterministic strategies on hand-crafted
OHLCV frames whose every trade — entry bar, fill price, exit price, exit reason and net PnL
— is computed BY HAND below, then asserted against the engine output.

Why this exists (vs the existing parity tests). test_execution_parity /
test_per_bar_kernel_adapter prove the backtest and the paper/live scanner produce the SAME
trades (self-consistency). But if the engine were wrong in a way that hit both paths
identically, those tests would be "consistently wrong" and still pass. These oracles pin the
engine to an EXTERNAL ground truth: the math is worked out on paper, so a regression in fill
timing, any exit type, the PnL convention, leverage, sizing, or the sign of a short is caught.

Every frame is rigged so the numbers are trivial: leverage 2.0, fees = slippage = 0 (drag 0),
and sizing_mode "full" (size_fraction 1.0) ⇒ pnl_pct collapses to (exit-entry)/entry · sign · 2.

Engine conventions exercised (from execution_kernel.simulate):
  * entries fill at the NEXT bar's open (signal on bar i → fill on bar i+1);
  * exit precedence per bar: time-stop → signal exit → stop/trailing → take-profit;
  * a long stop fills at min(open, stop), a short stop at max(open, stop) (gap-through);
  * a take-profit fills AT the target; a time-stop / signal exit fills at the bar OPEN;
  * a trailing stop ratchets on the PRIOR bar's extreme and fills at the level;
  * risk-based 'fraction' sizing deploys size_fraction = risk/(stop·lev), so a stop-out loses
    exactly risk_per_trade of equity — leverage-invariant (the last oracle; the others use
    'full' sizing = size_fraction 1.0 to keep the price-fill math isolated).
"""

from __future__ import annotations

import pandas as pd
import pytest

from forven.strategies import backtest as bt
from forven.strategies import execution_kernel as ek
from forven.strategies import sizing as _sizing
from forven.strategies.base import BaseStrategy, DirectionalSignals, Signal

LEVERAGE = 2.0
WARMUP = 1


def _frame(bars: list[tuple]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [b[0] for b in bars], "high": [b[1] for b in bars],
            "low": [b[2] for b in bars], "close": [b[3] for b in bars], "volume": 1000.0,
        },
        index=idx,
    )


def _signals(index: pd.Index, *, long_e=(), long_x=(), short_e=(), short_x=()) -> DirectionalSignals:
    """Build entry/exit boolean series from explicit bar indices. Positional, so a past bar's
    signal never changes as future bars are appended (prefix-stable) — the property the kernel
    relies on to replay a growing prefix."""
    n = len(index)

    def s(bars):
        out = pd.Series(False, index=index)
        for b in bars:
            if b < n:
                out.iloc[b] = True
        return out

    return DirectionalSignals(long_entries=s(long_e), long_exits=s(long_x),
                              short_entries=s(short_e), short_exits=s(short_x))


def _ec(**profile) -> dict:
    """A 'full'-sizing (size_fraction 1.0) execution profile with the given stop/target/etc."""
    return _sizing.normalize_execution_controls({"sizing_mode": "full", **profile})


def _run_kernel_lev(df: pd.DataFrame, signals: DirectionalSignals, ec: dict, leverage: float):
    res = ek.simulate(
        df, signals, WARMUP, leverage, regimes=None, round_trip_drag=0.0, trade_mode="both",
        allowed_modes=("long", "short"), ec=ec, initial_capital=10000.0,
    )
    trades = ek.force_close(res, df, leverage=leverage, round_trip_drag=0.0, trade_mode="both")
    return trades, res


def _run_kernel(df: pd.DataFrame, signals: DirectionalSignals, ec: dict):
    return _run_kernel_lev(df, signals, ec, LEVERAGE)


def _as_tuples(trades: list[dict]) -> list[tuple]:
    return [
        (t["direction"], int(t["entry_bar"]), float(t["entry_price"]), float(t["exit_price"]),
         t["exit_reason"], round(float(t["pnl_pct"]), 5))
        for t in trades
    ]


# ── Oracle 1: entry timing + take-profit + stop-loss (long & short) ─────────────────────────
#   long @100 → stop 98, target 104 ; short @100 → stop 102, target 96
_STOP_TP_BARS = [
    (100, 101, 99, 100),   # 0  warmup filler
    (100, 101, 99, 100),   # 1  long signal
    (100, 101, 99, 100),   # 2  T1 LONG fills @100
    (101, 105, 100, 104),  # 3  high 105>=104 -> TP @104
    (100, 101, 99, 100),   # 4  short signal
    (100, 101, 99, 100),   # 5  T2 SHORT fills @100
    (100, 101, 95, 96),    # 6  low 95<=96 -> TP @96 (high 101<102, no stop)
    (100, 101, 99, 100),   # 7  long signal
    (100, 101, 99, 100),   # 8  T3 LONG fills @100
    (99, 100, 97, 98),     # 9  low 97<=98 -> STOP @min(99,98)=98
    (100, 101, 99, 100),   # 10 short signal
    (100, 101, 99, 100),   # 11 T4 SHORT fills @100
    (101, 103, 100, 102),  # 12 high 103>=102 -> STOP @max(101,102)=102
    (100, 101, 99, 100),   # 13 filler
    (100, 101, 99, 100),   # 14 filler
]
_STOP_TP_LONG_E = (1, 7)
_STOP_TP_SHORT_E = (4, 10)
_STOP_TP_EXPECTED = [
    ("long", 2, 100.0, 104.0, "take_profit", 0.08),
    ("short", 5, 100.0, 96.0, "take_profit", 0.08),
    ("long", 8, 100.0, 98.0, "stop_loss", -0.04),
    ("short", 11, 100.0, 102.0, "stop_loss", -0.04),
]


def test_kernel_take_profit_and_stop_loss_oracle():
    df = _frame(_STOP_TP_BARS)
    sig = _signals(df.index, long_e=_STOP_TP_LONG_E, short_e=_STOP_TP_SHORT_E)
    trades, res = _run_kernel(df, sig, _ec(stop_loss_pct=2.0, take_profit_pct=4.0))
    assert not res.open_positions, "oracle is crafted so every trade closes before the end"
    assert _as_tuples(trades) == _STOP_TP_EXPECTED


class _OracleStrategy(BaseStrategy):
    """Deterministic 'both'-mode strategy emitting the Oracle-1 schedule — so the FULL pipeline
    (strategy → profile → kernel) is pinned to the same hand-computed trades as the bare kernel."""

    @property
    def name(self) -> str:
        return "oracle"

    @property
    def asset(self) -> str:
        return "BTC"

    @property
    def strategy_type(self) -> str:
        return "oracle_test"

    @property
    def default_params(self) -> dict:
        return {"trade_mode": "both"}

    def generate_signal(self, df: pd.DataFrame) -> Signal:  # required by the ABC; unused (vectorized wins)
        return Signal()

    def generate_signals(self, df: pd.DataFrame) -> DirectionalSignals:
        return _signals(df.index, long_e=_STOP_TP_LONG_E, short_e=_STOP_TP_SHORT_E)


def test_full_pipeline_take_profit_and_stop_loss_oracle(forven_db):
    """The full backtest entry point (run_strategy_execution: strategy → generate_signals →
    profile → kernel) reproduces the SAME hand-computed trades — proving the wiring, not just
    the kernel in isolation."""
    df = _frame(_STOP_TP_BARS)
    params = {"trade_mode": "both",
              "execution_profile": {"sizing_mode": "full", "stop_loss_pct": 2.0, "take_profit_pct": 4.0}}
    res = bt.run_strategy_execution(
        df, _OracleStrategy("ORACLE", params), params=params, warmup=WARMUP, leverage=LEVERAGE,
        fee_bps=0.0, slippage_bps=0.0, regime_gate=False, trade_mode="both",
        execution_controls=bt.execution_controls_from_params(params), initial_capital=10000.0,
        strategy_type="oracle_test",
    )
    assert res is not None
    trades = ek.force_close(res, df, leverage=LEVERAGE, round_trip_drag=0.0, trade_mode="both")
    assert _as_tuples(trades) == _STOP_TP_EXPECTED


def test_oracle_size_fraction_full_and_pure_return():
    """Pin the two simplifications the hand math relies on: full sizing ⇒ size_fraction 1.0,
    and zero costs ⇒ stored pnl_pct is exactly the leveraged price return (no hidden drag)."""
    df = _frame(_STOP_TP_BARS)
    sig = _signals(df.index, long_e=_STOP_TP_LONG_E, short_e=_STOP_TP_SHORT_E)
    _, res = _run_kernel(df, sig, _ec(stop_loss_pct=2.0, take_profit_pct=4.0))
    for t in res.closed_trades:
        assert t["size_fraction"] == 1.0
        sign = 1.0 if t["direction"] == "long" else -1.0
        pure = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * sign * LEVERAGE
        assert t["pnl_pct"] == pytest.approx(pure, abs=1e-9)


# ── Oracle 2: signal-driven exits (fill at the NEXT bar's open) ─────────────────────────────
#   Stops/targets set wide (10%/20%) so they never trigger — the exit is the strategy signal.
_SIGNAL_BARS = [
    (100, 101, 99, 100),   # 0
    (100, 101, 99, 100),   # 1  long entry signal
    (100, 101, 99, 100),   # 2  LONG fills @100
    (100, 101, 99, 100),   # 3  long EXIT signal (decided at close)
    (103, 104, 102, 103),  # 4  long exit fills @ open 103 (reason 'signal')
    (100, 101, 99, 100),   # 5  short entry signal
    (100, 101, 99, 100),   # 6  SHORT fills @100
    (100, 101, 99, 100),   # 7  short EXIT signal
    (98, 99, 97, 98),      # 8  short exit fills @ open 98 (reason 'signal')
    (100, 101, 99, 100),   # 9  filler
]
_SIGNAL_EXPECTED = [
    ("long", 2, 100.0, 103.0, "signal", 0.06),    # (103-100)/100 · 1 · 2
    ("short", 6, 100.0, 98.0, "signal", 0.04),    # (98-100)/100 · -1 · 2
]


def test_kernel_signal_exit_oracle():
    df = _frame(_SIGNAL_BARS)
    sig = _signals(df.index, long_e=(1,), long_x=(3,), short_e=(5,), short_x=(7,))
    trades, res = _run_kernel(df, sig, _ec(stop_loss_pct=10.0, take_profit_pct=20.0))
    assert not res.open_positions
    assert _as_tuples(trades) == _SIGNAL_EXPECTED


# ── Oracle 3: time-stop (checked first; fills at the bar OPEN) ──────────────────────────────
#   time_stop_bars=3 → exits the first bar where (idx - entry_bar) >= 3.
_TIME_BARS = [
    (100, 101, 99, 100),   # 0
    (100, 101, 99, 100),   # 1  long entry signal
    (100, 101, 99, 100),   # 2  LONG fills @100 (entry_bar=2)
    (100, 101, 99, 100),   # 3  held (idx-2=1)
    (100, 101, 99, 100),   # 4  held (idx-2=2)
    (105, 106, 104, 105),  # 5  TIME-STOP @ open 105 (idx-2=3)
    (100, 101, 99, 100),   # 6  filler
    (100, 101, 99, 100),   # 7  short entry signal
    (100, 101, 99, 100),   # 8  SHORT fills @100 (entry_bar=8)
    (100, 101, 99, 100),   # 9  held (idx-8=1)
    (100, 101, 99, 100),   # 10 held (idx-8=2)
    (97, 98, 96, 97),      # 11 TIME-STOP @ open 97 (idx-8=3)
    (100, 101, 99, 100),   # 12 filler
    (100, 101, 99, 100),   # 13 filler
]
_TIME_EXPECTED = [
    ("long", 2, 100.0, 105.0, "time_stop", 0.10),   # (105-100)/100 · 1 · 2
    ("short", 8, 100.0, 97.0, "time_stop", 0.06),    # (97-100)/100 · -1 · 2
]


def test_kernel_time_stop_oracle():
    df = _frame(_TIME_BARS)
    sig = _signals(df.index, long_e=(1,), short_e=(7,))
    trades, res = _run_kernel(df, sig, _ec(time_stop_bars=3))
    assert not res.open_positions
    assert _as_tuples(trades) == _TIME_EXPECTED


# ── Oracle 4: trailing stop (ratchets on the PRIOR bar's extreme; fills at the level) ───────
#   trail 5%. LONG: peak 110 → trail 104.5, price drops through it. SHORT: trough 90 → trail
#   94.5, price rises through it.
_TRAIL_BARS = [
    (100, 101, 99, 100),    # 0
    (100, 101, 99, 100),    # 1  long entry signal
    (100, 101, 99, 100),    # 2  LONG fills @100 (extreme=100)
    (105, 110, 103, 109),   # 3  trail=95, low 103>95 hold; extreme ratchets to 110
    (108, 109, 104, 105),   # 4  trail=110·0.95=104.5, low 104<=104.5 -> TRAIL @104.5
    (100, 101, 99, 100),    # 5  filler
    (100, 101, 99, 100),    # 6  short entry signal
    (100, 101, 99, 100),    # 7  SHORT fills @100 (extreme=100)
    (98, 103, 90, 92),      # 8  trail=100·1.05=105, high 103<105 hold; extreme ratchets to 90
    (92, 96, 91, 95),       # 9  trail=90·1.05=94.5, high 96>=94.5 -> TRAIL @max(92,94.5)=94.5
    (100, 101, 99, 100),    # 10 filler
    (100, 101, 99, 100),    # 11 filler
]
_TRAIL_EXPECTED = [
    ("long", 2, 100.0, 104.5, "trailing_stop", 0.09),    # (104.5-100)/100 · 1 · 2
    ("short", 7, 100.0, 94.5, "trailing_stop", 0.11),     # (94.5-100)/100 · -1 · 2
]


def test_kernel_trailing_stop_oracle():
    df = _frame(_TRAIL_BARS)
    sig = _signals(df.index, long_e=(1,), short_e=(6,))
    trades, res = _run_kernel(df, sig, _ec(trailing_stop_pct=5.0))
    assert not res.open_positions
    assert _as_tuples(trades) == _TRAIL_EXPECTED


# ── Oracle 5: risk-based (fraction) sizing — size_fraction = risk/(stop·lev), and a stop-out
#    loses EXACTLY risk_per_trade of equity, leverage-invariant ───────────────────────────────
#   risk 2%, stop 4% (=> size_fraction 0.02/(0.04·lev)), target 8% (a 2:1 winner => +2·risk).
_SIZING_RISK = 0.02
_SIZING_STOP_PCT = 4.0
_SIZING_BARS = [
    (100, 101, 99, 100),    # 0
    (100, 101, 99, 100),    # 1  long entry signal
    (100, 101, 99, 100),    # 2  LONG fills @100 -> stop 96, target 108
    (99, 100, 95, 96),      # 3  low 95<=96 -> STOP @min(99,96)=96
    (100, 101, 99, 100),    # 4  filler
    (100, 101, 99, 100),    # 5  long entry signal
    (100, 101, 99, 100),    # 6  LONG fills @100
    (101, 109, 100, 108),   # 7  high 109>=108 -> TP @108
    (100, 101, 99, 100),    # 8  filler
]
_SIZING_LONG_E = (1, 5)


def _sizing_ec() -> dict:
    return _ec(sizing_mode="fraction", risk_per_trade=_SIZING_RISK,
               stop_loss_pct=_SIZING_STOP_PCT, take_profit_pct=8.0)


def _expected_size_fraction(leverage: float) -> float:
    return _SIZING_RISK / (_SIZING_STOP_PCT / 100.0 * leverage)


def test_kernel_fraction_sizing_oracle():
    """Risk-based sizing deploys EXACTLY risk/(stop·lev) of equity, and the stop-out loses
    exactly risk_per_trade while a 2:1 winner makes 2·risk — pins the formula and the PnL
    scaling (gross · size_fraction), not just price fills."""
    df = _frame(_SIZING_BARS)
    sig = _signals(df.index, long_e=_SIZING_LONG_E)
    trades, res = _run_kernel(df, sig, _sizing_ec())  # leverage 2.0 -> size_fraction 0.25
    assert not res.open_positions
    assert _as_tuples(trades) == [
        ("long", 2, 100.0, 96.0, "stop_loss", round(-_SIZING_RISK, 5)),       # stop loses exactly -risk
        ("long", 6, 100.0, 108.0, "take_profit", round(2 * _SIZING_RISK, 5)),  # 2:1 winner -> +2·risk
    ]
    sf = round(_expected_size_fraction(LEVERAGE), 4)
    assert sf == 0.25
    for t in res.closed_trades:
        assert t["size_fraction"] == sf
    assert res.closed_trades[0]["pnl_pct"] == pytest.approx(-_SIZING_RISK, abs=1e-9)


def test_fraction_sizing_is_leverage_invariant():
    """Doubling leverage HALVES size_fraction, so the equity PnL is IDENTICAL — the stop still
    loses exactly risk_per_trade. The leverage-invariance the risk-based sizing guarantees (a
    backtest's % returns are achievable regardless of the leverage knob)."""
    df = _frame(_SIZING_BARS)
    sig = _signals(df.index, long_e=_SIZING_LONG_E)
    t2, r2 = _run_kernel_lev(df, sig, _sizing_ec(), 2.0)
    t4, r4 = _run_kernel_lev(df, sig, _sizing_ec(), 4.0)
    # identical equity PnL despite different leverage...
    assert _as_tuples(t2) == _as_tuples(t4)
    # ...because size_fraction scales by 1/leverage (0.25 at 2x, 0.125 at 4x)
    assert r2.closed_trades[0]["size_fraction"] == round(_expected_size_fraction(2.0), 4) == 0.25
    assert r4.closed_trades[0]["size_fraction"] == round(_expected_size_fraction(4.0), 4) == 0.125
    # a stop-out loses exactly risk_per_trade at BOTH leverages
    assert r2.closed_trades[0]["pnl_pct"] == pytest.approx(-_SIZING_RISK, abs=1e-9)
    assert r4.closed_trades[0]["pnl_pct"] == pytest.approx(-_SIZING_RISK, abs=1e-9)
