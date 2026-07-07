import json
import pytest
from forven.notifications import emit_notification, acknowledge_notification
from forven.brain import escalate_to_engineer
from forven.agents.tools_assistant import _tool_get_ops_overview

def test_ops_overview_unacknowledged_bugs(forven_db, monkeypatch):
    # Mock bot to avoid sending to Discord
    monkeypatch.setattr("forven.bot.send_sync", lambda *args, **kwargs: True)
    
    # Emit a bug report notification
    bug1 = emit_notification(
        event_type="bug_report",
        severity="warn",
        source="agent:triage",
        title="Uncaught ValueError in backtest",
        body="Value error trace",
        metadata={"error_type": "ValueError"}
    )
    
    # Emit another bug report notification and acknowledge it
    bug2 = emit_notification(
        event_type="bug_report",
        severity="critical",
        source="agent:triage",
        title="Database lock timeout",
        body="Database locked",
        metadata={"error_type": "TimeoutError"}
    )
    acknowledge_notification(bug2["id"])
    
    # Call get_ops_overview
    overview_json = _tool_get_ops_overview()
    overview = json.loads(overview_json)
    
    # Assert get_ops_overview contains bug_reports
    assert "bug_reports" in overview, "bug_reports key must be present in the overview"
    bug_reports = overview["bug_reports"]
    
    # bug1 should be present (unacknowledged), bug2 should be absent (acknowledged)
    assert any(b["id"] == bug1["id"] for b in bug_reports), "Unacknowledged bug must be present in bug_reports"
    assert not any(b["id"] == bug2["id"] for b in bug_reports), "Acknowledged bug must not be present in bug_reports"
    
    # Verify shape of bug report in overview
    reported_bug = next(b for b in bug_reports if b["id"] == bug1["id"])
    assert "title" in reported_bug
    assert "severity" in reported_bug
    assert "created_at" in reported_bug


def test_escalate_to_engineer_no_phantom_approval_id(forven_db, monkeypatch):
    monkeypatch.setattr("forven.bot.send_sync", lambda *args, **kwargs: True)
    
    result = escalate_to_engineer(
        title="SyntaxError in custom strategy",
        description="Invalid syntax at line 5",
        severity="high"
    )
    
    assert result.get("status") == "reported"
    assert result.get("queue") == "operator_triage"
    assert "approval_id" not in result, "approval_id must not be present in migrate/escalate result"
