"""The timeframe-sweep "best" selector must not crown a statistically degenerate
slice (few trades / zero in-sample trades). Such a slice yields a lucky high Sharpe
that, in the Sharpe-dominated score, beat an honest 100+-trade run and contaminated
strategies.metrics — the gate then read IS Sharpe 0.00 / <5 trades and re-archived a
healthy strategy on every retry. Generic: any strategy whose off-timeframe sweep
produces such a slice was mis-archived.
"""

from __future__ import annotations

import json

from forven.db import get_db
from forven.gauntlet.tasks import _best_sweep_result
from forven.policy import is_degenerate_backtest_metrics


def _insert_strategy(sid: str):
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO strategies "
            "(id, name, type, symbol, timeframe, params, metrics, status, owner, stage, "
            " stage_changed_at, created_at, updated_at) "
            "VALUES (?, ?, 'rsi_momentum', 'BTC', '1h', '{}', '{}', 'quick_screen', 'brain', "
            "'quick_screen', datetime('now'), datetime('now'), datetime('now'))",
            (sid, sid),
        )


def _insert_bt(sid: str, rid: str, tf: str, *, total_trades, is_trades, sharpe, ret=0.05):
    _insert_strategy(sid)
    metrics = {
        "total_trades": total_trades,
        "sharpe": sharpe,
        "total_return_pct": ret,
        "in_sample": {"total_trades": is_trades, "sharpe": sharpe},
        "out_of_sample": {"total_trades": total_trades, "sharpe": sharpe},
    }
    with get_db() as conn:
        conn.execute(
            "INSERT INTO backtest_results "
            "(result_id, strategy_id, result_type, symbol, timeframe, metrics_json, config_json, created_at) "
            "VALUES (?, ?, 'backtest', 'BTC', ?, ?, '{}', datetime('now'))",
            (rid, sid, tf, json.dumps(metrics)),
        )


def test_is_degenerate_backtest_metrics():
    assert is_degenerate_backtest_metrics({"total_trades": 4, "in_sample": {"total_trades": 0}}) is True
    assert is_degenerate_backtest_metrics({"total_trades": 40, "in_sample": {"total_trades": 0}}) is True   # 0 IS trades
    assert is_degenerate_backtest_metrics({"total_trades": 4, "in_sample": {"total_trades": 20}}) is True    # < 10 total
    assert is_degenerate_backtest_metrics({"total_trades": 40, "in_sample": {"total_trades": 80}}) is False  # healthy
    assert is_degenerate_backtest_metrics({}) is False  # empty -> not flagged (skipped elsewhere)


def test_sweep_rejects_degenerate_slice_even_with_higher_sharpe(forven_db):
    # Honest 1h: 100 OOS / 200 IS trades, Sharpe 1.95.
    # Degenerate 1d: 4 trades / 0 IS trades, Sharpe 2.86 (HIGHER) -> must NOT win.
    _insert_bt("S-X", "bt-1h", "1h", total_trades=100, is_trades=200, sharpe=1.95)
    _insert_bt("S-X", "bt-1d", "1d", total_trades=4, is_trades=0, sharpe=2.86)

    tf, rid, metrics = _best_sweep_result("S-X", "1h")
    assert tf == "1h", "a lucky 4-trade / 0-IS slice must not beat the honest 100-trade run"
    assert rid == "bt-1h"
    assert metrics.get("total_trades") == 100


def test_sweep_falls_back_to_least_degenerate_when_all_degenerate(forven_db):
    # No valid context -> return the MOST-traded (least degenerate), not the highest Sharpe,
    # so a strategy that genuinely can't trade is failed honestly, not crowned by a fluke.
    _insert_bt("S-Y", "bt-a", "1h", total_trades=4, is_trades=0, sharpe=2.86)
    _insert_bt("S-Y", "bt-b", "4h", total_trades=8, is_trades=0, sharpe=1.0)

    _tf, rid, _metrics = _best_sweep_result("S-Y", "1h")
    assert rid == "bt-b"  # 8 trades > 4 trades — least degenerate
