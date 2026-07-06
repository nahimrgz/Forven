"""Two-tier transparency in the paper-readiness report (2026-07-06).

A walk_forward artifact can pass the PAPER tier (fold pass-rate over judgeable
folds) while its strict artifact verdict is FAIL (avg IS/OOS Sharpe +
degradation — the paper->live bar). The readiness report used to say only
"All required artifact rows passed", which next to the artifact's FAIL verdict
read as a false green (an S06126 lean-pass was misdiagnosed as the 2026-07-03
false-green bug). The passing detail must now surface the strict-verdict caveat.
"""

from __future__ import annotations

import json

from forven.db import get_db
from forven.policy import _check_artifact_rows_exist, check_promotion_readiness


def _insert_strategy(conn, sid):
    conn.execute(
        "INSERT INTO strategies (id, name, type, status, stage, owner, timeframe, metrics) "
        "VALUES (?, ?, 'rsi_momentum', 'gauntlet', 'gauntlet', 'brain', '1h', '{}')",
        (sid, sid),
    )


def _wfa_metrics(*, verdict: str) -> dict:
    """A 5-fold artifact: 2 judgeable (>=5 OOS trades) folds, both positive, so
    the paper-tier fold pass-rate is 1.0 — while the strict verdict may be FAIL
    (e.g. one sparse all-loss fold sank the avg OOS Sharpe)."""
    def split(n_trades, sharpe):
        return {
            "in_sample": {"total_trades": 12, "sharpe": 0.5},
            "out_of_sample": {"total_trades": n_trades, "sharpe": sharpe},
        }

    return {
        "status": "succeeded",
        "verdict": verdict,
        "splits": [
            split(2, 1.0),
            split(3, -10.0),
            split(6, 1.2),
            split(4, 0.5),
            split(12, 2.0),
        ],
        "avg_is_sharpe": 0.5,
        "avg_oos_sharpe": -1.06,
        "degradation": 3.12,
        "aggregate_oos": {"total_trades": 27, "sharpe": 1.1},
    }


def _insert_row(conn, sid, result_type, metrics):
    conn.execute(
        "INSERT INTO backtest_results (result_id, strategy_id, result_type, symbol, timeframe, "
        "metrics_json, config_json, created_at) "
        "VALUES (?, ?, ?, 'BTC', '1h', ?, '{}', datetime('now'))",
        (f"{result_type}-{sid}", sid, result_type, json.dumps(metrics)),
    )


def _insert_wfa_row(conn, sid, metrics):
    _insert_row(conn, sid, "walk_forward", metrics)


_JITTER_PASS = {
    "status": "succeeded",
    "verdict": "PASS",
    "pass_rate": 0.8,
    "pct_positive_sharpe": 80.0,
    "n_iterations": 30,
    "iterations_completed": 30,
    "original_sharpe": 2.0,
    "mean_sharpe": 1.5,
}


def test_lean_pass_with_strict_fail_carries_the_caveat(forven_db):
    with get_db() as conn:
        _insert_strategy(conn, "s-two-tier")
        _insert_wfa_row(conn, "s-two-tier", _wfa_metrics(verdict="FAIL"))
        _insert_row(conn, "s-two-tier", "param_jitter", dict(_JITTER_PASS))

    ok, detail = _check_artifact_rows_exist("s-two-tier", ["walk_forward"])
    assert ok, detail  # paper tier: 2 judgeable folds, both positive
    assert "strict artifact verdict is FAIL" in detail, detail
    assert "paper->live" in detail, detail

    # and the caveat reaches the readiness report's passing step
    report = check_promotion_readiness("s-two-tier")
    step = next(s for s in report["steps"] if s["name"] == "validation_artifacts")
    assert step["status"] == "passed"
    assert "strict artifact verdict is FAIL" in step["detail"]


def test_strict_pass_has_no_caveat(forven_db):
    with get_db() as conn:
        _insert_strategy(conn, "s-clean-pass")
        _insert_wfa_row(conn, "s-clean-pass", _wfa_metrics(verdict="PASS"))

    ok, detail = _check_artifact_rows_exist("s-clean-pass", ["walk_forward"])
    assert ok, detail
    assert "strict artifact verdict" not in detail, detail
