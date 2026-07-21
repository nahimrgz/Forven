"""JOB-SWEEP-1: the orphaned-running-job sweep runs ONLY at API startup in the
main process.

It used to fire at forven.routers.robustness IMPORT time. That module is
imported by every spawn-context pool worker that pickles its module-level
worker functions (_monte_carlo_bootstrap_worker, the regime-split worker), so
every such spawn swept the LIVE job table and marked genuinely-running jobs as
"Server restarted while job was running" mid-flight — the UI showed that error
on every long walk-forward while the job actually completed and persisted fine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _insert_running_row(result_id: str = "rob_walk_forward_testrun") -> None:
    from forven.db import get_db

    config = {"job_id": "job_testrun", "status": "running", "submitted_at": "2026-07-21T00:00:00+00:00"}
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO strategies (id, name, type, symbol, timeframe, params, stage, status, created_at, updated_at)
               VALUES ('S99001', 'S99001', 'rule_engine', 'ETH', '4h', '{}', 'paper', 'paper',
                       '2026-07-21T00:00:00+00:00', '2026-07-21T00:00:00+00:00')"""
        )
        conn.execute(
            """INSERT INTO backtest_results
                   (result_id, strategy_id, result_type, symbol, timeframe, config_json, metrics_json, created_at)
               VALUES (?, 'S99001', 'walk_forward', 'ETH', '4h', ?, '{}', '2026-07-21T00:00:00+00:00')""",
            (result_id, json.dumps(config)),
        )


def _row_status(result_id: str = "rob_walk_forward_testrun") -> str:
    from forven.db import get_db

    with get_db() as conn:
        row = conn.execute(
            "SELECT config_json FROM backtest_results WHERE result_id = ?", (result_id,)
        ).fetchone()
    return str(json.loads(row["config_json"]).get("status"))


def test_child_process_never_sweeps_running_jobs(forven_db, monkeypatch):
    import multiprocessing

    from forven.routers import robustness

    _insert_running_row()
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    robustness._cleanup_orphaned_running_jobs()
    assert _row_status() == "running"  # untouched: a pool child must never sweep


def test_main_process_startup_sweep_still_marks_orphans(forven_db, monkeypatch):
    import multiprocessing

    from forven.routers import robustness

    _insert_running_row()
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
    robustness._cleanup_orphaned_running_jobs()
    assert _row_status() == "failed"  # genuine boot-time orphan is still reaped


def test_sweep_is_not_invoked_at_module_import():
    src = (REPO_ROOT / "forven" / "routers" / "robustness.py").read_text(encoding="utf-8")
    # A bare module-level invocation (column 0) is the regression; calls inside
    # functions/hooks are indented and fine.
    assert not re.search(r"^_cleanup_orphaned_running_jobs\(\)", src, flags=re.MULTILINE)


def test_startup_hook_invokes_sweep():
    src = (REPO_ROOT / "forven" / "api_core.py").read_text(encoding="utf-8")
    hook = src[src.find("async def _on_startup") :]
    assert "_cleanup_orphaned_running_jobs()" in hook
