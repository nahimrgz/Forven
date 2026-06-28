"""Phase 5: gated LIVE kernel parity — live takes the SAME kernel decisions as
paper/backtest but fills at REAL prices via the existing _execute_direct path.

These exercise the routing/gating logic with the real-order call mocked (no exchange):
the applier must place a real order via _execute_direct, honor the can_open safety
gate, and be OFF by default.
"""

from __future__ import annotations

import pytest

import forven.scanner as scanner
from forven.strategies.paper_reconcile import ReconcileAction


def test_is_live_kernel_stage():
    assert scanner._is_live_kernel_stage({"stage": "live_graduated"}) is True
    assert scanner._is_live_kernel_stage({"stage": "deployed"}) is True
    assert scanner._is_live_kernel_stage({"stage": "paper"}) is False
    assert scanner._is_live_kernel_stage({"stage": "gauntlet"}) is False


def test_live_kernel_execution_is_on_by_default(forven_db):
    # Deployed strategies execute on the validated parity kernel by default so live
    # matches the backtest/paper results the promotion gate approved (testnet-bounded
    # unless FORVEN_ALLOW_MAINNET=1). Operators can still opt out via the setting.
    assert scanner._live_kernel_execution_enabled() is True


def test_live_kernel_execution_can_be_disabled(forven_db):
    from forven.db import kv_set

    kv_set("forven:settings", {"live_kernel_execution": False})
    assert scanner._live_kernel_execution_enabled() is False


def test_kernel_open_live_places_real_order(monkeypatch):
    calls = {}
    monkeypatch.setattr("forven.exchange.risk.can_open", lambda *a, **k: (True, 0.01, "ok"))
    monkeypatch.setattr(scanner, "_open_trade_db", lambda *a, **k: "LIVE1")
    monkeypatch.setattr(scanner, "register", lambda *a, **k: None)

    def _fake_execute(action, trade_id, strat_id, asset, direction, size, price, **k):
        calls["x"] = dict(action=action, asset=asset, direction=direction, size=size,
                          stop=k.get("stop_loss"), tp=k.get("take_profit"), leverage=k.get("leverage"))
        return {}

    monkeypatch.setattr(scanner, "_execute_direct", _fake_execute)

    action = ReconcileAction("open", "long", "2024-01-01T00:00:00+00:00",
                             position={"entry_price": 100.0, "size_fraction": 0.5, "stop_price": 97.0,
                                       "target_price": 105.0, "entry_bar": 10})
    msg = scanner._kernel_open_live_trade("S1", {"asset": "BTC", "params": {}}, action,
                                          sizing_equity=10000.0, leverage=2.0)
    assert calls["x"]["action"] == "open"
    assert calls["x"]["direction"] == "long"
    assert calls["x"]["asset"] == "BTC"
    # units = equity*leverage*size_fraction/price = 10000*2*0.5/100 = 100
    assert calls["x"]["size"] == pytest.approx(100.0)
    assert calls["x"]["stop"] == 97.0 and calls["x"]["tp"] == 105.0
    assert "LIVE-KERNEL-OPEN" in msg


def test_kernel_open_live_blocked_by_can_open(monkeypatch):
    placed = {"called": False}
    monkeypatch.setattr("forven.exchange.risk.can_open", lambda *a, **k: (False, 0.0, "kill-switch active"))
    monkeypatch.setattr(scanner, "_execute_direct", lambda *a, **k: placed.__setitem__("called", True))

    action = ReconcileAction("open", "long", "t", position={"entry_price": 100.0, "size_fraction": 0.5})
    msg = scanner._kernel_open_live_trade("S1", {"asset": "BTC", "params": {}}, action,
                                          sizing_equity=10000.0, leverage=2.0)
    assert placed["called"] is False  # no real order placed when gated by can_open
    assert "BLOCKED" in msg and "kill-switch" in msg


def test_kernel_close_live_places_real_reduce_only(monkeypatch):
    calls = {}
    monkeypatch.setattr(scanner, "_execute_direct", lambda action, *a, **k: calls.setdefault("x", action) or {})
    monkeypatch.setattr(scanner, "_close_trade_db", lambda *a, **k: None)
    monkeypatch.setattr(scanner, "_resolve_trade_vault_address", lambda *a, **k: None)
    monkeypatch.setattr(scanner, "_trade_stop_oids", lambda *a, **k: [])
    monkeypatch.setattr(scanner, "_retire_trade_protection_orders", lambda *a, **k: [])
    monkeypatch.setattr(scanner, "release", lambda *a, **k: None)

    row = {"id": "LIVE1", "asset": "BTC", "direction": "long", "size": 100.0,
           "fill_entry_price": 100.0, "leverage": 2.0}
    action = ReconcileAction("close", "long", "t", trade={"exit_price": 110.0, "exit_reason": "take_profit"},
                             recorded={"_row": row})
    msg = scanner._kernel_close_live_trade("S1", {"asset": "BTC"}, action)
    assert calls["x"] == "close"
    assert "LIVE-KERNEL-CLOSE" in msg


def test_kernel_close_live_respects_pending_reconcile(monkeypatch):
    """If the exchange close is unconfirmed, do NOT finalize the trade locally."""
    finalized = {"closed": False}
    monkeypatch.setattr(scanner, "_execute_direct", lambda *a, **k: {"_close_reconcile_state": "pending"})
    monkeypatch.setattr(scanner, "_close_trade_db", lambda *a, **k: finalized.__setitem__("closed", True))
    row = {"id": "LIVE1", "asset": "BTC", "direction": "long", "size": 100.0, "fill_entry_price": 100.0, "leverage": 2.0}
    action = ReconcileAction("close", "long", "t", trade={"exit_price": 110.0}, recorded={"_row": row})
    msg = scanner._kernel_close_live_trade("S1", {"asset": "BTC"}, action)
    assert finalized["closed"] is False
    assert "pending" in msg
