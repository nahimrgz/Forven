"""Autonomous crucible-discovery dispatcher.

Default OFF (operator-approves). When enabled it dispatches ONE benchmarking
research task carrying the contract that unlocks the discover_*/inspect_* tools.
"""
import json

from forven.crucible_discovery import run_crucible_discovery
from forven.db import get_db, kv_set


def _enable(mode: str = "operator_approves") -> None:
    kv_set(
        "forven:settings",
        {
            "research_settings": {
                "autonomous_discovery": {
                    "enabled": True,
                    "mode": mode,
                    "max_open_discovery_tasks": 1,
                }
            }
        },
    )


def test_discovery_disabled_by_default(forven_db):
    res = run_crucible_discovery()
    assert res["created"] is False
    assert res["reason"] == "disabled"


def test_discovery_enabled_dispatches_benchmarking_task(forven_db):
    _enable()
    res = run_crucible_discovery()
    assert res["created"] is True
    assert res["mode"] == "operator_approves"
    with get_db() as conn:
        row = conn.execute(
            "SELECT input_data FROM agent_tasks WHERE id = ?", (res["task_id"],)
        ).fetchone()
    payload = json.loads(row["input_data"])
    assert payload["origin_mode"] == "crucible_discovery"
    assert payload["discovery_mode"] == "operator_approves"
    # The contract is what makes the discover_*/inspect_* tools reachable.
    assert payload["research_contract"]["lane"] == "benchmarking"
    assert payload["research_contract"]["external_sources_allowed"] is True


def test_discovery_dedups_open_task(forven_db):
    _enable()
    first = run_crucible_discovery()
    assert first["created"] is True
    second = run_crucible_discovery()
    assert second["created"] is False
    assert second["reason"] == "already_open"


def test_discovery_autonomous_mode_is_stamped(forven_db):
    _enable(mode="autonomous")
    res = run_crucible_discovery()
    assert res["created"] is True
    assert res["mode"] == "autonomous"
    with get_db() as conn:
        row = conn.execute(
            "SELECT input_data FROM agent_tasks WHERE id = ?", (res["task_id"],)
        ).fetchone()
    assert json.loads(row["input_data"])["discovery_mode"] == "autonomous"


def test_force_bypasses_disabled_setting(forven_db):
    """Operator demand (force=True) runs even though discovery is disabled by default."""
    res = run_crucible_discovery(force=True)  # NO _enable() — still off in settings
    assert res["created"] is True
    assert res["mode"] == "operator_approves"
    with get_db() as conn:
        row = conn.execute(
            "SELECT input_data FROM agent_tasks WHERE id = ?", (res["task_id"],)
        ).fetchone()
    payload = json.loads(row["input_data"])
    assert payload["origin_mode"] == "crucible_discovery"
    assert payload["research_contract"]["lane"] == "benchmarking"


def test_force_still_dedups(forven_db):
    """force=True bypasses the enabled flag but NOT the open-task dedup."""
    first = run_crucible_discovery(force=True)
    assert first["created"] is True
    second = run_crucible_discovery(force=True)
    assert second["created"] is False
    assert second["reason"] == "already_open"


def test_discovery_task_lists_known_crucibles_in_description(forven_db):
    """Audit B-16: 'Do not duplicate existing crucibles' is only satisfiable if
    the agent can SEE them — active and recently-disproven titles are inlined."""
    from forven.db import get_db as _get_db
    from forven.hypotheses import create_hypothesis

    active = create_hypothesis(
        title="Liquidation Cascade Reversal",
        market_thesis="m", mechanism="x",
        lane="benchmarking", source_type="public_benchmark",
        target_assets=["BTC/USDT"], target_timeframes=["1h"],
    )
    disproven = create_hypothesis(
        title="Funding Rate Mean Reversion",
        market_thesis="m", mechanism="x",
        lane="benchmarking", source_type="public_benchmark",
        target_assets=["BTC/USDT"], target_timeframes=["1h"],
    )
    with _get_db() as conn:
        conn.execute(
            "UPDATE hypotheses SET status = 'disproven', manager_state = 'archived' WHERE id = ?",
            (disproven["id"],),
        )

    _enable()
    res = run_crucible_discovery()
    assert res["created"] is True
    with get_db() as conn:
        row = conn.execute(
            "SELECT description FROM agent_tasks WHERE id = ?", (res["task_id"],)
        ).fetchone()
    assert active["title"] in row["description"]
    assert disproven["title"] in row["description"]


def test_disproven_title_carries_verdict_rationale(forven_db):
    """A disproven title alone only stops the agent re-minting the same NAME —
    the verdict_memo rationale is appended so it also avoids the same failure
    mode under a different name."""
    from forven.hypotheses import create_hypothesis, update_hypothesis_status

    disproven = create_hypothesis(
        title="Funding Rate Mean Reversion",
        market_thesis="m", mechanism="x",
        lane="benchmarking", source_type="public_benchmark",
        target_assets=["BTC/USDT"], target_timeframes=["1h"],
    )
    update_hypothesis_status(
        disproven["id"], new_status="disproven",
        memo={"verdict": "disproven", "rationale": "Funding edge decayed to zero after fees; no regime filter fixed it."},
        by="test",
    )

    _enable()
    res = run_crucible_discovery()
    assert res["created"] is True
    with get_db() as conn:
        row = conn.execute(
            "SELECT description FROM agent_tasks WHERE id = ?", (res["task_id"],)
        ).fetchone()
    assert disproven["title"] in row["description"]
    assert "Funding edge decayed to zero after fees" in row["description"]


def test_disproven_title_without_memo_is_bare(forven_db):
    """Absent verdict_memo -> bare title, no crash."""
    from forven.hypotheses import create_hypothesis

    disproven = create_hypothesis(
        title="No Memo Thesis",
        market_thesis="m", mechanism="x",
        lane="benchmarking", source_type="public_benchmark",
        target_assets=["BTC/USDT"], target_timeframes=["1h"],
    )
    with get_db() as conn:
        conn.execute(
            "UPDATE hypotheses SET status = 'disproven', manager_state = 'archived' WHERE id = ?",
            (disproven["id"],),
        )

    _enable()
    res = run_crucible_discovery()
    assert res["created"] is True
    with get_db() as conn:
        row = conn.execute(
            "SELECT description FROM agent_tasks WHERE id = ?", (res["task_id"],)
        ).fetchone()
    assert disproven["title"] in row["description"]
