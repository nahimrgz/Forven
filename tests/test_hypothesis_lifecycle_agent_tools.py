"""Agent tools for hypothesis lifecycle transitions (P1-C).

Before these tools existed, agents had NO way to set status/manager_state —
update_hypothesis_fields deliberately excludes lifecycle fields and the real
setters (archive_hypothesis / update_hypothesis_status) had no @register_tool
wrapper. Agents "retired" hypotheses by renaming the title ([ARCHIVED-*]
prefixes), which the dispatcher never reads, so archived work was re-dispatched
forever (H01005/H00970 reports, 2026-07-23..26).
"""

from __future__ import annotations

import importlib
import json


def _make_hypothesis(tools_research, title="Lifecycle test hypothesis"):
    created = json.loads(
        tools_research._tool_create_hypothesis(
            {
                "title": title,
                "market_thesis": "thesis",
                "mechanism": "mechanism",
                "why_now": "now",
                "lane": "benchmarking",
                "source_type": "public_benchmark",
                "origin_role": "strategy-developer",
                "target_assets": ["BTC-PERP"],
                "target_timeframes": ["15m"],
            }
        )
    )
    assert created["ok"] is True
    return created["hypothesis"]["id"]


def _as_agent(monkeypatch, tools_research, agent_id="strategy-developer"):
    monkeypatch.setattr(
        tools_research,
        "_current_agent_id_var",
        type("_Var", (), {"get": staticmethod(lambda: agent_id)})(),
        raising=False,
    )


def test_archive_hypothesis_tool_retires_from_dispatcher(forven_db, monkeypatch):
    from forven.system_pause import set_system_mode
    from forven.crucible_planner import _active_crucible_rows
    from forven.hypotheses import get_hypothesis

    tools_research = importlib.import_module("forven.agents.tools_research")
    set_system_mode("auto")
    _as_agent(monkeypatch, tools_research)

    hid = _make_hypothesis(tools_research)
    assert any(r["id"] == hid for r in _active_crucible_rows())

    result = json.loads(
        tools_research._tool_archive_hypothesis(
            {"hypothesis_id": hid, "reason": "lane saturated: funding-carry 0/12 graduated"}
        )
    )

    assert result["ok"] is True
    row = get_hypothesis(hid)
    assert row["manager_state"] == "archived"
    assert row["archived_at"]
    # The actual bug symptom: the dispatcher pool must no longer serve it.
    assert not any(r["id"] == hid for r in _active_crucible_rows())


def test_update_hypothesis_status_tool_writes_verdict_and_audit(forven_db, monkeypatch):
    from forven.system_pause import set_system_mode
    from forven.db import get_db
    from forven.hypotheses import get_hypothesis

    tools_research = importlib.import_module("forven.agents.tools_research")
    set_system_mode("auto")
    _as_agent(monkeypatch, tools_research)

    hid = _make_hypothesis(tools_research)

    result = json.loads(
        tools_research._tool_update_hypothesis_status(
            {
                "hypothesis_id": hid,
                "new_status": "disproven",
                "verdict_summary": "0/8 candidates survived quick_screen; mechanism refuted",
                "evidence_id": "verdict-batch-2026-07-26",
            }
        )
    )

    assert result["ok"] is True
    row = get_hypothesis(hid)
    assert row["status"] == "disproven"
    assert row["verdict_memo_by"] == "agent:strategy-developer"
    with get_db() as conn:
        memos = conn.execute(
            "SELECT payload, written_by FROM hypothesis_verdict_memos WHERE hypothesis_id = ?",
            (hid,),
        ).fetchall()
    assert len(memos) == 1
    assert memos[0]["written_by"] == "agent:strategy-developer"
    assert "mechanism refuted" in memos[0]["payload"]


def test_update_hypothesis_status_tool_rejects_invalid_status(forven_db, monkeypatch):
    from forven.system_pause import set_system_mode

    tools_research = importlib.import_module("forven.agents.tools_research")
    set_system_mode("auto")
    _as_agent(monkeypatch, tools_research)

    hid = _make_hypothesis(tools_research)
    result = json.loads(
        tools_research._tool_update_hypothesis_status(
            {"hypothesis_id": hid, "new_status": "retired", "verdict_summary": "x"}
        )
    )
    assert result["ok"] is False
    assert "invalid status" in result["error"]


def test_archive_tool_requires_reason(forven_db, monkeypatch):
    from forven.system_pause import set_system_mode

    tools_research = importlib.import_module("forven.agents.tools_research")
    set_system_mode("auto")
    _as_agent(monkeypatch, tools_research)

    hid = _make_hypothesis(tools_research)
    result = json.loads(tools_research._tool_archive_hypothesis({"hypothesis_id": hid}))
    assert result["ok"] is False
    assert "reason" in result["error"]


def test_lifecycle_tools_are_categorized_destructive(forven_db):
    """Both tools must carry category='destructive' so the recovery-context
    default-deny applies (update_hypothesis_status doesn't match the archive_
    auto-categorization prefix, so an explicit category is load-bearing)."""
    import forven.agents.tools_research  # noqa: F401 — ensure registration ran
    from forven.agents.tool_registry import _CONTEXT_DEFAULT_DENY, _REGISTRY

    assert _REGISTRY["archive_hypothesis"].category == "destructive"
    assert _REGISTRY["update_hypothesis_status"].category == "destructive"
    assert "destructive" in _CONTEXT_DEFAULT_DENY["recovery"]


def test_lifecycle_tools_refuse_research_context(forven_db, monkeypatch):
    """A prompt-injected research page must not be able to retire hypotheses:
    research tasks ingest untrusted content in the same turn, so both tools
    refuse in that context regardless of category/list filtering."""
    from forven.system_pause import set_system_mode
    from forven.agents.context import _current_tools_context_var
    from forven.hypotheses import get_hypothesis

    tools_research = importlib.import_module("forven.agents.tools_research")
    set_system_mode("auto")
    _as_agent(monkeypatch, tools_research)
    hid = _make_hypothesis(tools_research)

    token = _current_tools_context_var.set("research")
    try:
        archived = json.loads(
            tools_research._tool_archive_hypothesis({"hypothesis_id": hid, "reason": "injected"})
        )
        flipped = json.loads(
            tools_research._tool_update_hypothesis_status(
                {"hypothesis_id": hid, "new_status": "disproven", "verdict_summary": "injected"}
            )
        )
    finally:
        _current_tools_context_var.reset(token)

    assert archived["ok"] is False and "research context" in archived["error"]
    assert flipped["ok"] is False and "research context" in flipped["error"]
    row = get_hypothesis(hid)
    assert row["manager_state"] == "active"
    assert row["status"] != "disproven"


def test_update_hypothesis_fields_still_ignores_lifecycle_fields(forven_db, monkeypatch):
    """Non-regression: the enrichment tool must keep silently ignoring lifecycle
    fields (they belong to the dedicated tools above, with memo + audit)."""
    from forven.system_pause import set_system_mode
    from forven.hypotheses import get_hypothesis

    tools_research = importlib.import_module("forven.agents.tools_research")
    set_system_mode("auto")
    _as_agent(monkeypatch, tools_research)

    hid = _make_hypothesis(tools_research)
    result = json.loads(
        tools_research._tool_update_hypothesis_fields(
            {"hypothesis_id": hid, "title": "renamed", "status": "disproven"}
        )
    )
    assert result["ok"] is True
    row = get_hypothesis(hid)
    assert row["title"] == "renamed"
    assert row["status"] != "disproven"
