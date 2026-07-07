import os
import pytest
from forven.notifications import emit_notification, list_notification_deliveries
from forven.bot import is_discord_configured

def test_discord_graceful_degradation_without_token(forven_db, monkeypatch):
    # Ensure DISCORD_TOKEN is not in the environment and config has no token
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    
    # Mock load_config to return empty config
    monkeypatch.setattr("forven.bot.load_config", lambda: {})
    monkeypatch.setattr("forven.notifications.get_notification_preferences", lambda: {
        "providers": {
            "discord": {
                "enabled": True,
                "token": "",
                "channel_mappings": {
                    "critical": "alerts"
                }
            }
        }
    })
    
    # Assert discord is not configured
    assert not is_discord_configured()
    
    # Emit a critical system notification (which should resolve to Discord delivery)
    item = emit_notification(
        event_type="risk_critical",
        severity="critical",
        title="Margin call warning",
        summary="Equity dropped below margin requirements"
    )
    
    # It should have status 'stored' (not new/failed)
    assert item["status"] == "stored"
    
    # Verify the delivery attempt was recorded as 'skipped'
    deliveries = list_notification_deliveries(item["id"]) or []
    assert len(deliveries) > 0, "A delivery record must be created"
    
    discord_delivery = next((d for d in deliveries if d["target"] == "discord"), None)
    assert discord_delivery is not None, "Discord delivery record must exist"
    assert discord_delivery["status"] == "skipped"
    assert "not configured" in discord_delivery["detail"]
