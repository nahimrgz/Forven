import pytest
from forven.brain import escalate_to_engineer
from forven.notification_policy import resolve_notification_policy
from forven.notifications import emit_notification

def test_bug_report_dedupe_keys(forven_db, monkeypatch):
    monkeypatch.setattr("forven.bot.send_sync", lambda *args, **kwargs: True)
    
    # 1. Dedupe by strategy_id
    res1 = escalate_to_engineer(
        title="Execution error in trend strategy",
        description="ZeroDivisionError",
        context={"strategy_id": "S00042"}
    )
    # Let's fetch the notification we just created to verify its dedupe_key
    from forven.notifications import list_notifications
    notifs = list_notifications(event_type="bug_report", limit=1)
    assert notifs[0]["dedupe_key"] == "bug_report:strategy:s00042"
    
    # 2. Dedupe by error_type (no strategy_id)
    res2 = escalate_to_engineer(
        title="Database lock failure",
        description="Database lock timeout after 30s",
        context={"error_type": "SQLite3LockError"}
    )
    notifs = list_notifications(event_type="bug_report", limit=1)
    assert notifs[0]["dedupe_key"] == "bug_report:error:sqlite3lockerror"
    
    # 3. Fallback to normalized title
    res3 = escalate_to_engineer(
        title="[BUG] Connection to exchange lost",
        description="Network connection dropped",
        context={}
    )
    notifs = list_notifications(event_type="bug_report", limit=1)
    assert notifs[0]["dedupe_key"] == "bug_report:title:connection to exchange lost"


def test_bug_report_policy_cooldown():
    event = {
        "event_type": "bug_report",
        "severity": "warn",
        "title": "Some bug title"
    }
    policy = resolve_notification_policy(event)
    assert policy.get("cooldown_seconds") == 3600, "Bug report cooldown must be 3600 seconds"
