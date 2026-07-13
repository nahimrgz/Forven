from __future__ import annotations


def test_live_entry_fill_persistence_retries_until_success(monkeypatch):
    import forven.scanner as scanner

    outcomes = iter([False, False, True])
    calls: list[dict] = []
    sleeps: list[float] = []

    def fake_update(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return next(outcomes)

    monkeypatch.setattr(scanner, "_update_trade_fill", fake_update)
    monkeypatch.setattr(scanner.time, "sleep", lambda seconds: sleeps.append(seconds))

    persisted = scanner._persist_live_entry_fill(
        "T-LIVE-1",
        101.25,
        signal_price=100.0,
        exchange_order_id="OID-1",
        filled_size=0.4,
        mark_price=101.0,
    )

    assert persisted is True
    assert len(calls) == 3
    assert sleeps == [0.05, 0.1]
    assert calls[-1]["args"] == ("T-LIVE-1", 101.25, "entry")
    assert calls[-1]["kwargs"]["filled_size"] == 0.4


def test_live_entry_fill_persistence_failure_pauses_new_opens(monkeypatch):
    import forven.exchange.hyperliquid as hyperliquid
    import forven.exchange.risk as risk
    import forven.scanner as scanner
    import forven.sim.clock as clock

    pauses: list[dict] = []

    monkeypatch.setattr(scanner, "_resolve_hyperliquid_testnet", lambda: True)
    monkeypatch.setattr(scanner, "_resolve_trade_vault_address", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scanner, "_persist_live_entry_fill", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        scanner,
        "_pause_after_live_fill_persistence_failure",
        lambda **kwargs: pauses.append(kwargs),
    )
    monkeypatch.setattr(scanner, "_update_trade_signal_data", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scanner, "log_activity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(clock, "is_sim_active", lambda: False)
    monkeypatch.setattr(risk, "is_trading_allowed", lambda: (True, "ok"))
    monkeypatch.setattr(hyperliquid, "set_leverage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        hyperliquid,
        "market_order",
        lambda **_kwargs: {
            "entry_price": 101.25,
            "filled_size": 0.4,
            "entry_order_id": "OID-1",
        },
    )

    result = scanner._execute_direct(
        action="open",
        trade_id="T-LIVE-1",
        strat_id="S-LIVE",
        asset="SOL",
        direction="long",
        size=0.5,
        price=100.0,
        stop_loss=95.0,
    )

    assert result["fill_persistence_failed"] is True
    assert pauses == [{
        "trade_id": "T-LIVE-1",
        "asset": "SOL",
        "direction": "long",
        "error": "entry fill write failed after 3 attempts",
    }]


def test_fill_persistence_fault_engages_canonical_pause_and_critical_alert(monkeypatch):
    import forven.notifications as notifications
    import forven.scanner as scanner
    import forven.system_pause as system_pause

    pauses: list[tuple[bool, str | None]] = []
    alerts: list[dict] = []
    monkeypatch.setattr(
        system_pause,
        "set_system_paused",
        lambda paused, *, paused_at=None: pauses.append((paused, paused_at)) or {},
    )
    monkeypatch.setattr(
        notifications,
        "emit_notification",
        lambda event_type, **kwargs: alerts.append({"event_type": event_type, **kwargs}),
    )

    scanner._pause_after_live_fill_persistence_failure(
        trade_id="T-LIVE-1",
        asset="SOL",
        direction="long",
        error="database locked",
    )

    assert pauses and pauses[0][0] is True
    assert pauses[0][1]
    assert alerts[0]["event_type"] == "trade_fill_persistence_failed"
    assert alerts[0]["severity"] == "critical"


def test_rejected_stop_rearm_uses_filled_size_and_alerts_on_exception(monkeypatch):
    import forven.exchange.hyperliquid as hyperliquid
    import forven.exchange.risk as risk
    import forven.notifications as notifications
    import forven.scanner as scanner
    import forven.sim.clock as clock

    stop_sizes: list[float] = []
    alerts: list[dict] = []

    monkeypatch.setattr(scanner, "_resolve_hyperliquid_testnet", lambda: True)
    monkeypatch.setattr(scanner, "_resolve_trade_vault_address", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scanner, "_persist_live_entry_fill", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scanner, "_update_trade_signal_data", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scanner, "log_activity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(clock, "is_sim_active", lambda: False)
    monkeypatch.setattr(risk, "is_trading_allowed", lambda: (True, "ok"))
    monkeypatch.setattr(hyperliquid, "set_leverage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        hyperliquid,
        "market_order",
        lambda **_kwargs: {
            "entry_price": 101.25,
            "filled_size": 0.4,
            "entry_order_id": "OID-1",
            "protective_leg_failed": ["stop"],
        },
    )

    def raise_rearm(_asset, _direction, size, _stop_loss, **_kwargs):
        stop_sizes.append(size)
        raise RuntimeError("venue rejected re-arm")

    monkeypatch.setattr(hyperliquid, "place_protective_stop", raise_rearm)
    monkeypatch.setattr(notifications, "emit_notification", lambda event_type, **kwargs: alerts.append({"event_type": event_type, **kwargs}))

    scanner._execute_direct(
        action="open",
        trade_id="T-LIVE-2",
        strat_id="S-LIVE",
        asset="SOL",
        direction="long",
        size=0.5,
        price=100.0,
        stop_loss=95.0,
    )

    assert stop_sizes == [0.4]
    assert alerts[0]["event_type"] == "trade_protective_unarmed"
    assert alerts[0]["severity"] == "critical"


def test_reconcile_finalizes_matched_trade_from_exchange_truth(forven_db):
    import json

    from forven.db import get_db
    from forven.exchange import risk

    trade_id = "T-LIVE-RECONCILE"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades
            (id, strategy, strategy_id, asset, direction, entry_price, size, leverage,
             status, execution_type, opened_at, signal_data)
            VALUES (?, 'S-LIVE', 'S-LIVE', 'SOL', 'long', 100.0, 0.5, 1.0,
                    'OPEN', 'live', datetime('now'), ?)
            """,
            (
                trade_id,
                json.dumps({
                    "pending_open_reconcile": True,
                    "fill_persistence_failed": True,
                    "entry_finalization_state": "reconcile_required",
                }),
            ),
        )
        conn.execute(
            """
            INSERT INTO portfolio_positions
            (trade_id, asset, direction, strategy, strategy_id, risk_pct, entry_price,
             correlation_group, opened_at, execution_type)
            VALUES (?, 'SOL', 'long', 'S-LIVE', 'S-LIVE', 0.01, 100.0,
                    'crypto_major', datetime('now'), 'live')
            """,
            (trade_id,),
        )

    with get_db() as conn:
        trade = dict(conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone())
        action = risk._finalize_matched_open_trade(
            conn,
            trade,
            {"asset": "SOL", "direction": "long", "entry_price": 101.25, "size": 0.4},
        )

    with get_db() as conn:
        row = dict(conn.execute(
            "SELECT entry_price, fill_entry_price, size, signal_data FROM trades WHERE id = ?",
            (trade_id,),
        ).fetchone())
        position = conn.execute(
            "SELECT entry_price FROM portfolio_positions WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()

    signal_data = json.loads(row["signal_data"])
    assert action and action["type"] == "entry_finalized"
    assert row["entry_price"] == 101.25
    assert row["fill_entry_price"] == 101.25
    assert row["size"] == 0.4
    assert position["entry_price"] == 101.25
    assert signal_data["entry_finalization_state"] == "finalized_reconciled"
    assert signal_data["filled_size"] == 0.4
    assert "pending_open_reconcile" not in signal_data
    assert "fill_persistence_failed" not in signal_data
