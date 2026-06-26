"""Tests for reconciliation close paths writing exit price + PnL."""

import json
from datetime import datetime, timedelta, timezone

from forven.api_domains.trading import _close_stale_open_trades
from forven.db import get_db
from forven.trade_state import close_trade_record, mark_trade_pending_close_reconcile


def _insert_open_trade(
    trade_id: str,
    *,
    asset: str,
    direction: str,
    entry_price: float,
    size: float,
    leverage: float = 1.0,
    fill_exit_price: float | None = None,
):
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades
            (id, strategy, strategy_id, asset, direction, entry_price, size, leverage, status, opened_at, fill_exit_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """,
            (
                trade_id,
                "test-strategy",
                "test-strategy",
                asset,
                direction,
                entry_price,
                size,
                leverage,
                opened_at,
                fill_exit_price,
            ),
        )


def test_stale_close_uses_price_map_and_writes_pnl(forven_db):
    _insert_open_trade(
        "t-reconcile-1",
        asset="BTC",
        direction="long",
        entry_price=100.0,
        size=1.5,
        leverage=2.0,
    )

    _close_stale_open_trades(
        ["t-reconcile-1"],
        "test stale close",
        price_map={"BTC": 110.0},
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, exit_price, signal_exit_price, pnl_pct, pnl_usd FROM trades WHERE id=?",
            ("t-reconcile-1",),
        ).fetchone()

    assert row["status"] == "CLOSED"
    assert row["exit_price"] == 110.0
    assert row["signal_exit_price"] == 110.0
    assert row["pnl_pct"] == 0.2
    # size embeds leverage → pnl_usd = price_move * units; was 30.0 under the old
    # leverage^2 double-count (leverage=2).
    assert row["pnl_usd"] == 15.0


def test_stale_close_falls_back_to_fill_exit_price(forven_db):
    _insert_open_trade(
        "t-reconcile-2",
        asset="ETH",
        direction="short",
        entry_price=100.0,
        size=2.0,
        leverage=1.0,
        fill_exit_price=97.0,
    )

    _close_stale_open_trades(
        ["t-reconcile-2"],
        "test stale close fallback",
        price_map={},
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, exit_price, signal_exit_price, pnl_pct, pnl_usd FROM trades WHERE id=?",
            ("t-reconcile-2",),
        ).fetchone()

    assert row["status"] == "CLOSED"
    assert row["exit_price"] == 97.0
    assert row["signal_exit_price"] == 97.0
    assert row["pnl_pct"] == 0.03
    assert row["pnl_usd"] == 6.0


def test_stale_close_marks_trade_incomplete_when_no_price_is_available(forven_db):
    _insert_open_trade(
        "t-reconcile-3",
        asset="SOL",
        direction="long",
        entry_price=100.0,
        size=1.0,
    )

    _close_stale_open_trades(
        ["t-reconcile-3"],
        "test stale close incomplete",
        price_map={},
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, exit_price, signal_exit_price, pnl_pct, pnl_usd, signal_data FROM trades WHERE id=?",
            ("t-reconcile-3",),
        ).fetchone()

    assert row["status"] == "CLOSED"
    assert row["exit_price"] is None
    assert row["signal_exit_price"] is None
    assert row["pnl_pct"] is None
    assert row["pnl_usd"] is None
    assert '"close_incomplete": true' in str(row["signal_data"]).lower()
    assert "stale_missing_on_exchange" in str(row["signal_data"])


def test_close_trade_record_preserves_fill_exit_price_over_signal_price(forven_db):
    _insert_open_trade(
        "t-reconcile-4",
        asset="BTC",
        direction="long",
        entry_price=100.0,
        size=1.0,
        leverage=2.0,
        fill_exit_price=103.0,
    )

    closed = close_trade_record(
        "t-reconcile-4",
        signal_exit_price=110.0,
        exit_price=110.0,
        close_reason="signal",
        close_price_source="scanner_signal",
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, exit_price, signal_exit_price, pnl_pct, pnl_usd FROM trades WHERE id=?",
            ("t-reconcile-4",),
        ).fetchone()

    assert closed is not None
    assert row["status"] == "CLOSED"
    assert row["exit_price"] == 103.0
    assert row["signal_exit_price"] == 110.0
    assert row["pnl_pct"] == 0.06
    assert row["pnl_usd"] == 3.0


def test_mark_trade_pending_close_reconcile_keeps_trade_open_until_confirmed(forven_db):
    _insert_open_trade(
        "t-reconcile-5",
        asset="BTC",
        direction="long",
        entry_price=100.0,
        size=1.0,
        leverage=2.0,
    )

    pending = mark_trade_pending_close_reconcile(
        "t-reconcile-5",
        signal_exit_price=109.0,
        close_reason="take_profit",
        close_price_source="scanner_signal",
        extra_signal_data={"exit_exchange_order_id": "close-123"},
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, signal_exit_price, signal_data FROM trades WHERE id=?",
            ("t-reconcile-5",),
        ).fetchone()

    assert pending is not None
    assert pending["updated"] is True
    assert row["status"] == "OPEN"
    assert row["signal_exit_price"] == 109.0
    signal_data = json.loads(row["signal_data"])
    assert signal_data["pending_close_reconcile"] is True
    assert signal_data["pending_close_reason"] == "take_profit"
    assert signal_data["pending_close_price_source"] == "scanner_signal"
    assert signal_data["exit_exchange_order_id"] == "close-123"

    closed = close_trade_record(
        "t-reconcile-5",
        signal_exit_price=110.0,
        exit_price=110.0,
        close_reason="pending_close_reconcile_confirmed",
        close_price_source="exchange_mids",
    )

    with get_db() as conn:
        final_row = conn.execute(
            "SELECT status, signal_data FROM trades WHERE id=?",
            ("t-reconcile-5",),
        ).fetchone()

    assert closed is not None
    assert final_row["status"] == "CLOSED"
    final_signal_data = json.loads(final_row["signal_data"])
    assert "pending_close_reconcile" not in final_signal_data
    assert final_signal_data["close_reason"] == "pending_close_reconcile_confirmed"
