import os
import pytest
from forven.notifications import emit_notification, list_notification_deliveries
from forven.bot import is_discord_configured

def test_discord_graceful_degradation_without_token(forven_db, monkeypatch):
    # Ensure DISCORD_TOKEN is not in the environment and config has no token
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    
    # Mock load_config to return empty config
    monkeypatch.setattr("forven.bot.load_config", lambda: {})
    monkeypatch.setattr("forven.config.load_config", lambda: {})
    
    import forven.notifications as notifications
    notifications._DISCORD_CONFIGURED_CACHE = None

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
    assert item["delivery_mode"] == "app_only"
    assert item.get("metadata", {}).get("discord_skipped") == "not_configured"
    
    # Verify no delivery attempt was recorded
    deliveries = list_notification_deliveries(item["id"]) or []
    assert len(deliveries) == 0

