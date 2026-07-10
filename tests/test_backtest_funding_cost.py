"""Tier 0: perp-funding cost in backtest PnL + promotion gate.

Covers the pure helpers (`_apply_funding_to_trades`, `_compute_basic_metrics`
aggregation) and the policy-level funding-completeness gate. These are fast,
deterministic unit tests — they do not run a full backtest.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forven.strategies import backtest as bt


def _funding_df(rates, freq="1h"):
    ts = pd.date_range("2024-01-01", periods=len(rates), freq=freq, tz="UTC")
    return pd.DataFrame({"funding_rate": list(rates)}, index=ts)


def test_long_pays_positive_funding():
    df = _funding_df([0.001] * 10)
    trades = [{"entry_bar": 0, "bars_held": 4, "direction": "long", "pnl_pct": 0.10}]
    out, complete = bt._apply_funding_to_trades(trades, df, leverage=2.0, timeframe="1h")
    # 4 bars * 0.001 * 1h * 2x leverage = 0.008 cost; long pays it.
    assert complete is True
    assert out[0]["funding_cost_pct"] == pytest.approx(-0.008, abs=1e-9)
    assert out[0]["pnl_pct"] == pytest.approx(0.092, abs=1e-9)
    assert out[0]["funding_applied"] is True
    assert out[0]["funding_complete"] is True


def test_funding_cost_prefers_hourly_column_for_parity():
    """Two-column funding convention (zero_trade fix): strategies read the
    canonical PER-8H ``funding_rate``, but cost accrual must use the PER-HOUR
    ``_funding_rate_hourly`` the enrichment provides. The charged cost must be
    IDENTICAL to the legacy per-hour-only frame — parity by construction."""
    ts = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "funding_rate": [0.008] * 10,          # per-8h (strategy-facing, 8x)
            "_funding_rate_hourly": [0.001] * 10,  # per-hour (cost source)
        },
        index=ts,
    )
    trades = [{"entry_bar": 0, "bars_held": 4, "direction": "long", "pnl_pct": 0.10}]
    out, complete = bt._apply_funding_to_trades(trades, df, leverage=2.0, timeframe="1h")
    # Cost is driven by the HOURLY column, so it equals the legacy
    # test_long_pays_positive_funding result exactly: 4*0.001*1h*2x = 0.008.
    assert complete is True
    assert out[0]["funding_cost_pct"] == pytest.approx(-0.008, abs=1e-9)
    assert out[0]["pnl_pct"] == pytest.approx(0.092, abs=1e-9)


def test_funding_cost_falls_back_to_funding_rate_without_hourly():
    """Legacy frames (and unit callers) with only ``funding_rate`` and no
    ``_funding_rate_hourly`` keep the OLD per-hour interpretation — back-compat."""
    df = _funding_df([0.001] * 10)  # only funding_rate, treated as per-hour
    trades = [{"entry_bar": 0, "bars_held": 4, "direction": "long", "pnl_pct": 0.10}]
    out, _ = bt._apply_funding_to_trades(trades, df, leverage=2.0, timeframe="1h")
    assert out[0]["funding_cost_pct"] == pytest.approx(-0.008, abs=1e-9)


def test_short_receives_positive_funding():
    df = _funding_df([0.001] * 10)
    trades = [{"entry_bar": 0, "bars_held": 4, "direction": "short", "pnl_pct": 0.10}]
    out, complete = bt._apply_funding_to_trades(trades, df, leverage=2.0, timeframe="1h")
    # Short receives the funding: +0.008 added to pnl.
    assert out[0]["funding_cost_pct"] == pytest.approx(0.008, abs=1e-9)
    assert out[0]["pnl_pct"] == pytest.approx(0.108, abs=1e-9)
    assert complete is True


def test_funding_scales_with_hours_per_bar():
    df = _funding_df([0.001] * 10, freq="4h")
    trades = [{"entry_bar": 0, "bars_held": 2, "direction": "long", "pnl_pct": 0.0}]
    out, _ = bt._apply_funding_to_trades(trades, df, leverage=1.0, timeframe="4h")
    # 2 bars * 0.001 * 4h/bar * 1x = 0.008 cost.
    assert out[0]["funding_cost_pct"] == pytest.approx(-0.008, abs=1e-9)


def test_missing_funding_column_marks_incomplete():
    df = pd.DataFrame({"close": [1, 2, 3]})
    trades = [{"entry_bar": 0, "bars_held": 2, "direction": "long", "pnl_pct": 0.05}]
    out, complete = bt._apply_funding_to_trades(trades, df, leverage=3.0, timeframe="1h")
    assert complete is False
    assert out[0]["funding_complete"] is False
    assert out[0]["funding_applied"] is True
    # Price PnL is left untouched when funding data is absent.
    assert out[0]["pnl_pct"] == pytest.approx(0.05, abs=1e-9)
    assert out[0]["funding_cost_pct"] == 0.0


def test_nan_funding_in_window_marks_that_trade_incomplete():
    df = _funding_df([0.001, np.nan, 0.001, 0.001])
    trades = [{"entry_bar": 0, "bars_held": 3, "direction": "long", "pnl_pct": 0.0}]
    out, complete = bt._apply_funding_to_trades(trades, df, leverage=1.0, timeframe="1h")
    assert complete is False
    assert out[0]["funding_complete"] is False


def test_no_trades_is_trivially_complete():
    df = _funding_df([0.001] * 5)
    out, complete = bt._apply_funding_to_trades([], df, leverage=1.0, timeframe="1h")
    assert out == []
    assert complete is True


def test_basic_metrics_aggregate_funding_flags():
    trades = [
        {"pnl_pct": 0.05, "bars_held": 2, "direction": "long",
         "funding_applied": True, "funding_complete": True},
        {"pnl_pct": -0.02, "bars_held": 1, "direction": "long",
         "funding_applied": True, "funding_complete": False},
    ]
    metrics = bt._compute_basic_metrics(trades, total_bars=100, timeframe="1h")
    assert metrics["funding_applied"] is True
    assert metrics["funding_complete"] is False  # one trade incomplete poisons the set


def test_basic_metrics_funding_complete_when_all_complete():
    trades = [
        {"pnl_pct": 0.05, "bars_held": 2, "direction": "long",
         "funding_applied": True, "funding_complete": True},
    ]
    metrics = bt._compute_basic_metrics(trades, total_bars=100, timeframe="1h")
    assert metrics["funding_applied"] is True
    assert metrics["funding_complete"] is True


def _patch_promotion_env(monkeypatch, metrics: dict, *, settings):
    import forven.policy as policy

    monkeypatch.setattr(policy, "load_pipeline_config", lambda: {})
    monkeypatch.setattr(policy, "kv_get", lambda *a, **k: settings)
    monkeypatch.setattr(
        policy, "_load_strategy_row_for_gate",
        lambda sid: {"metrics": json.dumps(metrics)},
    )


def test_promotion_blocked_when_funding_incomplete(monkeypatch):
    # Stage-aware: funding-incompleteness blocks ->live (capital) but NOT earlier
    # forward stages — paper (testnet) measures real funding directly ("strict
    # live, achievable paper"), so a winner is no longer deleted over a data gap.
    # The paper-allowed path needs a real DB (symbol gate); it's covered in
    # tests/test_funding_gate_stage_aware.py::test_funding_incomplete_allowed_into_paper.
    import forven.policy as policy

    _patch_promotion_env(
        monkeypatch,
        {"funding_applied": True, "funding_complete": False, "sharpe": 2.0},
        settings={"backtest_include_funding": True},
    )
    passed, reason = policy.evaluate_promotion("S1", "paper", "live_graduated")
    assert passed is False
    assert "Funding data incomplete" in reason


def test_promotion_not_blocked_when_funding_complete(forven_db, monkeypatch):
    import forven.policy as policy

    _patch_promotion_env(
        monkeypatch,
        {"funding_applied": True, "funding_complete": True, "sharpe": 2.0},
        settings={"backtest_include_funding": True},
    )
    _, reason = policy.evaluate_promotion("S1", "quick_screen", "gauntlet")
    assert "Funding data incomplete" not in reason


def test_promotion_not_blocked_when_setting_off(forven_db, monkeypatch):
    import forven.policy as policy

    _patch_promotion_env(
        monkeypatch,
        {"funding_applied": True, "funding_complete": False, "sharpe": 2.0},
        settings={"backtest_include_funding": False},
    )
    _, reason = policy.evaluate_promotion("S1", "quick_screen", "gauntlet")
    assert "Funding data incomplete" not in reason


def test_promotion_not_blocked_for_pre_feature_metrics(forven_db, monkeypatch):
    """Strategies tested before this feature have no funding_applied key."""
    import forven.policy as policy

    _patch_promotion_env(
        monkeypatch,
        {"sharpe": 2.0},  # no funding_* keys
        settings={"backtest_include_funding": True},
    )
    _, reason = policy.evaluate_promotion("S1", "quick_screen", "gauntlet")
    assert "Funding data incomplete" not in reason
