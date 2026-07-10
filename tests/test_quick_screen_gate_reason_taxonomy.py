"""Quick-screen gate mislabels merit failures as 'no_metrics_error' (2026-07-06 audit).

Verified against ~/.forven/forven.db: 51 gate_rejections rows carry reason_code
'no_metrics_error' / reason_text "No quick-screen metrics available". For 13 of
the underlying strategies, gauntlet_steps.error_json (joined via
gauntlet_workflows.strategy_id) already contains a REAL merit verdict with full
metrics, e.g. {"message": "profit_factor 0.9 < 1.0", "metrics": {...}} — the
quick-screen backtest WAS judged, but ``run_quick_screen_gate`` (tasks.py) only
persists the winning metrics onto ``strategies.metrics`` via
``_persist_quick_screen_winner`` when ``_best_sweep_result`` finds a usable row;
when it does not, the merit-failing ``run_quick_screen_gate`` return value
(message + metrics) lands ONLY in ``gauntlet_steps.error_json`` via
``block_step``, and ``strategies.metrics`` stays empty. A later re-check
(``routers.robustness._reconcile_stage_after_validation`` -> ``evaluate_promotion``
-> ``_evaluate_quick_screen_gate``) then reads the empty ``strategies.metrics``
and records a false "no metrics" rejection — hiding the true reason AND letting
the strategy dodge the repeated-failure auto-archive counter (no_metrics_error is
counter-exempt by design).

Fix: ``_evaluate_quick_screen_gate`` falls back to the latest persisted
quick-screen step's error_json/output_json metrics blob before concluding
"No quick-screen metrics available" (read-only; no new writes).

Also covers reason-code coverage gaps found during the same audit: two
genuinely-never-ran texts ("... could not be downloaded" from the data-
availability precheck, and "Retries exhausted on a transient block" from the
gauntlet drain sweep) fell through to the generic 'gate_reject' bucket, which
is NOT exempt from the repeated-failure auto-archive counter -- silently
letting infrastructure/data blocks count as merit failures and eventually
demote/archive a strategy that was never actually judged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import forven.policy as policy
from forven.db import get_db
from forven.gauntlet.store import init_gauntlet_schema


def _insert_strategy(strategy_id: str, *, stage: str = "quick_screen", metrics: dict | None = None) -> None:
    now = datetime.now(timezone.utc)
    stage_changed = (now - timedelta(hours=2)).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO strategies
                (id, name, type, symbol, timeframe, params, metrics, status, owner,
                 stage, stage_changed_at, created_at, updated_at)
            VALUES (?, ?, 'rsi_momentum', 'ETH/USDT', '1h', '{}', ?, ?, 'brain', ?, ?, ?, ?)
            """,
            (
                strategy_id,
                strategy_id,
                json.dumps(metrics or {}),
                stage,
                stage,
                stage_changed,
                stage_changed,
                now.isoformat(),
            ),
        )


def _insert_quick_screen_gate_step(
    strategy_id: str,
    *,
    status: str,
    error_payload: dict | None = None,
    output_payload: dict | None = None,
    step_key: str = "quick_screen_gate",
    completed_at: str | None = None,
) -> None:
    """Seed a gauntlet_workflows + gauntlet_steps row directly (unit-level, no
    workflow-engine dependency) mirroring what ``block_step``/``complete_step``
    persist for a real quick_screen_gate step outcome."""
    now = datetime.now(timezone.utc).isoformat()
    workflow_id = f"wf-{strategy_id}"
    step_id = f"step-{strategy_id}-{step_key}"
    with get_db() as conn:
        init_gauntlet_schema(conn)
        conn.execute(
            """
            INSERT INTO gauntlet_workflows
                (id, strategy_id, definition_version, status, current_step_key,
                 settings_snapshot_json, created_by, created_at, updated_at)
            VALUES (?, ?, 1, ?, ?, '{}', 'pytest', ?, ?)
            """,
            (workflow_id, strategy_id, status, step_key, now, now),
        )
        conn.execute(
            """
            INSERT INTO gauntlet_steps
                (id, workflow_id, step_key, order_index, status, output_json,
                 error_json, completed_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                workflow_id,
                step_key,
                status,
                json.dumps(output_payload or {}),
                json.dumps(error_payload) if error_payload is not None else None,
                completed_at or now,
                now,
            ),
        )


# --- root-cause fix: merit-failing quick_screen falls back to step evidence -------


def test_quick_screen_gate_falls_back_to_step_metrics_on_merit_failure(forven_db):
    strategy_id = "s-merit-mislabel"
    _insert_strategy(strategy_id, metrics={})  # strategies.metrics empty (never persisted)
    real_metrics = {
        "sharpe": 1.2,
        "profit_factor": 0.9,
        "total_trades": 50,
    }
    _insert_quick_screen_gate_step(
        strategy_id,
        status="failed_gate",
        error_payload={
            "status": "failed_gate",
            "message": "profit_factor 0.90 < 1.00",
            "metrics": real_metrics,
        },
    )

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert not passed
    assert "no quick-screen metrics available" not in reason.lower(), reason
    assert "profit factor" in reason.lower(), reason
    # The true reason code (S00552 profit-factor guardrail) is a genuine merit
    # failure -- unlike no_metrics_error it must NOT be exempt from the
    # repeated-failure auto-archive counter.
    reason_code = policy._extract_reason_code(reason)
    assert reason_code != "no_metrics_error"
    assert reason_code not in policy._EVIDENCE_ABSENCE_REASON_CODES, reason_code


def test_quick_screen_gate_still_reports_no_metrics_when_never_ran(forven_db):
    strategy_id = "s-genuinely-blocked"
    _insert_strategy(strategy_id, metrics={})
    # A data-availability / db-lock block carries no "metrics" key at all — the
    # backtest never produced a verdict, so the fallback must not fabricate one.
    _insert_quick_screen_gate_step(
        strategy_id,
        status="blocked_data",
        error_payload={
            "status": "blocked_data",
            "message": "backfilling ETH/USDT 1h history (have 10d, need 90d)",
            "retryable": True,
            "reason_code": "awaiting_data_backfill",
        },
    )

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert not passed
    assert reason == "No quick-screen metrics available"
    assert policy._extract_reason_code(reason) == "no_metrics_error"


def test_quick_screen_gate_ignores_non_merit_step_metrics(forven_db):
    strategy_id = "s-drained-transient"
    _insert_strategy(strategy_id, metrics={})
    # engine.drain_exhausted_blocked_steps stamps merit=False on drained transient
    # blocks precisely so downstream consumers never read them as a judged verdict.
    # Even if such a payload someday carries a diagnostic metrics blob, the gate
    # fallback must NOT resurrect it as a real quick-screen verdict.
    _insert_quick_screen_gate_step(
        strategy_id,
        status="failed_gate",
        error_payload={
            "exhausted": True,
            "merit": False,
            "message": "Retries exhausted on a transient block — NOT a merit verdict",
            "metrics": {"sharpe": 1.5, "profit_factor": 2.0, "total_trades": 40},
        },
    )

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert not passed
    assert reason == "No quick-screen metrics available"
    assert policy._extract_reason_code(reason) == "no_metrics_error"


def test_quick_screen_gate_no_step_history_still_reports_no_metrics(forven_db):
    strategy_id = "s-no-workflow-yet"
    _insert_strategy(strategy_id, metrics={})

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert not passed
    assert reason == "No quick-screen metrics available"


# --- reason-code taxonomy: transient/data-unavailable texts are non-merit ---------


def test_extract_reason_code_classifies_data_unavailable_as_non_merit():
    text = (
        "Cannot backtest rsi_momentum on ETH/USDT 1h: strategy requires data "
        "feed(s) funding_rate (columns: funding_rate) that are could not be "
        "downloaded. Backtest aborted rather than run on silently zero-filled data."
    )
    assert policy._extract_reason_code(text) == "data_unavailable"
    assert "data_unavailable" in policy._EVIDENCE_ABSENCE_REASON_CODES

    text_unfetchable = (
        "Cannot backtest rsi_momentum on ETH/USDT 1h: strategy requires data "
        "feed(s) open_interest (columns: open_interest) that are not available "
        "and cannot be auto-downloaded. Backtest aborted rather than run on "
        "silently zero-filled data."
    )
    assert policy._extract_reason_code(text_unfetchable) == "data_unavailable"


def test_extract_reason_code_classifies_retries_exhausted_as_non_merit():
    text = (
        "Retries exhausted on a transient block — NOT a merit verdict; the "
        "strategy was never judged by the gate (last block: db locked)"
    )
    assert policy._extract_reason_code(text) == "retries_exhausted"
    assert "retries_exhausted" in policy._EVIDENCE_ABSENCE_REASON_CODES


def test_data_unavailable_rejections_never_auto_archive(forven_db):
    strategy_id = "s-data-unavailable"
    text = (
        "Cannot backtest rsi_momentum on ETH/USDT 1h: strategy requires data "
        "feed(s) funding_rate (columns: funding_rate) that are could not be "
        "downloaded. Backtest aborted rather than run on silently zero-filled data."
    )
    _insert_strategy(strategy_id, stage="gauntlet")
    with get_db() as conn:
        for _ in range(8):
            conn.execute(
                """
                INSERT INTO gate_rejections
                    (strategy_id, gate, reason_code, reason_text, created_at)
                VALUES (?, 'gauntlet', 'data_unavailable', ?, datetime('now'))
                """,
                (strategy_id, text),
            )

    policy._check_repeated_failure_auto_archive(strategy_id, "gauntlet", "data_unavailable", text)

    with get_db() as conn:
        row = conn.execute("SELECT stage FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    assert str(row["stage"]) == "gauntlet"


def test_retries_exhausted_rejections_never_auto_archive(forven_db):
    strategy_id = "s-retries-exhausted"
    text = (
        "Retries exhausted on a transient block — NOT a merit verdict; the "
        "strategy was never judged by the gate (last block: db locked)"
    )
    _insert_strategy(strategy_id, stage="gauntlet")
    with get_db() as conn:
        for _ in range(8):
            conn.execute(
                """
                INSERT INTO gate_rejections
                    (strategy_id, gate, reason_code, reason_text, created_at)
                VALUES (?, 'gauntlet', 'retries_exhausted', ?, datetime('now'))
                """,
                (strategy_id, text),
            )

    policy._check_repeated_failure_auto_archive(strategy_id, "gauntlet", "retries_exhausted", text)

    with get_db() as conn:
        row = conn.execute("SELECT stage FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    assert str(row["stage"]) == "gauntlet"
