"""Phase 4: the paper chart bundle (get_paper_session_chart) — real indicators,
full-history triggers, actual trade markers, active SL/TP — driven by the registry
and the strategy's own signal function (no guessed reimplementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forven.api_domains import paper as paper_domain
from forven.strategies.builtin.rsi_momentum import RSIMomentumStrategy


def _frame(n=320, seed=11):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(rng.normal(0.0, 0.02, n).cumsum())
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0},
        index=idx,
    )


def _bars(n=320, seed=11):
    f = _frame(n, seed)
    return [
        {"timestamp": t.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
        for t, r in zip(f.index, f.itertuples())
    ]


def test_indicator_specs_resolution():
    # Builtin → mapped registry specs.
    rsi = paper_domain._chart_indicator_specs("rsi_momentum", {"rsi_period": 14, "ema_fast": 10, "ema_slow": 30})
    assert [s["kind"] for s in rsi] == ["rsi", "ema", "ema"]
    # rule_engine → declared specs passed through.
    declared = [{"id": "a", "kind": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}]
    assert paper_domain._chart_indicator_specs("rule_engine", {"indicators": declared}) == declared
    # Unknown type → no overlay (NOT a guessed reimplementation).
    assert paper_domain._chart_indicator_specs("some_custom_thing", {}) == []


def test_compute_chart_indicators_uses_registry():
    frame = _frame()
    specs = paper_domain._chart_indicator_specs("rsi_momentum", {"rsi_period": 14, "ema_fast": 10, "ema_slow": 30})
    main, sub, warns = paper_domain._compute_chart_indicators(frame, specs)
    assert warns == []
    assert {m["name"] for m in main} == {"ema_fast", "ema_slow"}  # overlays on price
    assert any(s["name"] == "rsi" for s in sub)                    # rsi in its own pane
    # Series carry real values aligned to the bars.
    rsi_series = next(s for s in sub if s["name"] == "rsi")["data"]
    assert len(rsi_series) == len(frame)
    assert any(p["value"] is not None for p in rsi_series)


def test_kernel_triggers_are_sparse_events_not_a_per_bar_band():
    """Regression: triggers must be discrete KERNEL events (one per would-be position
    open/close), NOT a marker on every bar where the raw signal state is True. The old
    bug lit up nearly every candle."""
    frame = _frame(n=600)
    strat = RSIMomentumStrategy("t", {"rsi_period": 14, "rsi_entry": 45, "rsi_exit": 55, "ema_fast": 10, "ema_slow": 30, "adx_period": 14, "adx_min": 0})
    entries, exits = paper_domain._kernel_trigger_markers(
        strat, frame, params=strat.params, leverage=2.0, strategy_type="rsi_momentum", cutoff=None,
    )
    assert entries, "no triggers emitted"
    assert all(m["marker_kind"] == "signal" for m in entries + exits)
    # Sparse: far fewer triggers than bars (events, not a band). A per-bar band would
    # be ~len(frame); real trade events are a small fraction.
    assert len(entries) < len(frame) * 0.2, f"too many trigger entries ({len(entries)}) — looks like a per-bar band"
    # Entries roughly pair with exits (each open eventually closes).
    assert abs(len(entries) - len(exits)) <= 2


def test_kernel_triggers_respect_prelive_cutoff():
    """Triggers only appear BEFORE the first real trade (the live period is the real
    trade markers' job)."""
    frame = _frame(n=600)
    strat = RSIMomentumStrategy("t", {"rsi_period": 14, "rsi_entry": 45, "rsi_exit": 55, "ema_fast": 10, "ema_slow": 30, "adx_period": 14, "adx_min": 0})
    cutoff = frame.index[400]
    entries, exits = paper_domain._kernel_trigger_markers(
        strat, frame, params=strat.params, leverage=2.0, strategy_type="rsi_momentum", cutoff=cutoff,
    )
    for m in entries + exits:
        assert pd.Timestamp(m["timestamp"]) < cutoff


def test_chart_bundle_assembles_everything(monkeypatch):
    """End-to-end shape: bundle has real indicators, triggers, ACTUAL trade markers,
    and the open position's active stop/take-profit."""
    bars = _bars(n=600)
    session = {
        "id": "PAPER-CHART-1", "strategy_id": "PAPER-CHART-1",
        "symbol": "BTC/USDT", "timeframe": "1h",
        "type": "rsi_momentum", "runtime_type": "rsi_momentum",
        "params": {"rsi_period": 14, "rsi_entry": 45, "rsi_exit": 55, "ema_fast": 10, "ema_slow": 30, "adx_period": 14, "adx_min": 0},
        # One real trade late in the series, so there's a pre-live window (bars ~200..560)
        # for would-be triggers to populate.
        "trades": [
            {"id": "E1", "entry_time": bars[560]["timestamp"], "exit_time": bars[580]["timestamp"],
             "entry_price": bars[560]["close"], "exit_price": bars[580]["close"], "side": "long",
             "pnl": 12.0, "pnl_pct": 0.012, "marker_kind": "trade"},
        ],
        "position": {
            "side": "long", "entry_time": bars[590]["timestamp"], "entry_price": bars[590]["close"],
            "stop_loss_price": bars[590]["close"] * 0.97, "take_profit_price": bars[590]["close"] * 1.05,
        },
    }

    strat = RSIMomentumStrategy("PAPER-CHART-1", dict(session["params"], _asset="BTC"))
    monkeypatch.setattr(paper_domain, "_find_compat_paper_session", lambda sid, include_deployed=True: session)
    monkeypatch.setattr(paper_domain, "_load_session_bars", lambda s, limit=2000, timeframe_override=None: bars)
    monkeypatch.setattr("forven.strategies.registry.get_active", lambda: {"PAPER-CHART-1": strat})

    bundle = paper_domain.get_paper_session_chart("PAPER-CHART-1")

    assert len(bundle["bars"]) == len(bars)
    # 1. Real indicators (registry), not guessed.
    assert {m["name"] for m in bundle["main_indicators"]} == {"ema_fast", "ema_slow"}
    assert any(s["name"] == "rsi" for s in bundle["sub_indicators"])
    # 2. Full-history triggers.
    assert bundle["trigger_entries"] and bundle["trigger_exits"]
    # 3. Actual trade markers.
    assert any(m["trade_id"] == "E1" for m in bundle["entry_markers"])
    assert any(m["trade_id"] == "E1" for m in bundle["exit_markers"])
    # 4. Active stop + take-profit from the open position.
    assert bundle["active_levels"]["stop"] and bundle["active_levels"]["take_profit"]
    assert bundle["active_levels"]["stop"][0]["price"] == pytest.approx(bars[590]["close"] * 0.97)
    assert bundle["active_levels"]["take_profit"][0]["price"] == pytest.approx(bars[590]["close"] * 1.05)
