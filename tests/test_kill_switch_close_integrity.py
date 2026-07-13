"""Kill-switch close integrity (the 2026-06-28 E0001/E0002/paper-contamination class).

Two invariants:
  * the booked close price is the REAL fill (exit_price/avgPx) or the honest mid —
    NEVER close_position's ``close_price``, which is the aggressive IOC *limit*
    (mid padded by the emergency slippage tier, 300+ bps past the market);
  * a kill-switch flatten touches only LIVE local rows — a PAPER trade on the same
    asset never reached the exchange, and closing it at another account's fill
    fabricates PnL on the promotion gate's input book.
"""

from __future__ import annotations

import json

from forven.exchange import risk


def test_extract_close_price_prefers_real_fill_over_padded_limit():
    response = {"close_price": 1622.7, "exit_price": 1574.2, "mid": 1573.5}
    assert risk._extract_close_price(response) == 1574.2


def test_extract_close_price_falls_back_to_mid_never_limit():
    # No fill price returned (IOC unfilled / response missing avgPx): the honest
    # mid is bookable; the padded limit is not.
    response = {"close_price": 1622.7, "mid": 1573.5}
    assert risk._extract_close_price(response) == 1573.5
    assert risk._extract_close_price({"close_price": 1622.7}) is None
    assert risk._extract_close_price(None) is None


def _insert_open_trade(conn, trade_id: str, asset: str, execution_type: str, source=None):
    # one strategy per row: the unique-open partial index allows a single OPEN per
    # (strategy, asset, direction)
    sid = f"S-{trade_id}"
    conn.execute(
        "INSERT INTO trades (id, strategy, strategy_id, asset, direction, entry_price, size, "
        "status, execution_type, source, signal_data, opened_at) "
        "VALUES (?, ?, ?, ?, 'long', 100.0, 1.0, 'OPEN', ?, ?, ?, '2026-01-01T00:00:00+00:00')",
        (trade_id, sid, sid, asset, execution_type, source, json.dumps({"kernel_managed": True})),
    )


def test_kill_switch_flatten_spares_paper_rows(forven_db, monkeypatch):
    """close_all_positions flattens a live BTC position; the local sweep must close
    the LIVE row at the real fill and leave the PAPER row (source NULL — the filter
    that only excluded 'bot:%' sources missed it) untouched."""
    from forven.db import get_db

    with get_db() as conn:
        _insert_open_trade(conn, "LIVE-1", "BTC", "live")
        _insert_open_trade(conn, "PAPER-1", "BTC", "paper")          # scanner kernel: source NULL
        _insert_open_trade(conn, "BOT-1", "BTC", "paper", "bot:b1")  # bot factory row

    monkeypatch.setattr(
        "forven.exchange.hyperliquid.get_positions",
        lambda *a, **k: {
            "positions": [{"position": {"coin": "BTC", "szi": 1.0}}],
            "margin_summary": {},
        },
    )
    monkeypatch.setattr(
        "forven.exchange.hyperliquid.close_position",
        lambda *a, **k: {"status": "ok", "mid": 100.0, "close_price": 103.0,
                         "exit_price": 100.2, "requested_size": 1.0, "filled_size": 1.0},
    )
    monkeypatch.setattr(risk, "release", lambda *a, **k: None, raising=False)

    risk.close_all_positions()

    from forven.db import get_db as _gdb
    with _gdb() as conn:
        rows = {r["id"]: dict(r) for r in conn.execute(
            "SELECT id, status, exit_price FROM trades").fetchall()}
    assert rows["LIVE-1"]["status"] == "CLOSED"
    # booked at the REAL fill, not the 3%-padded limit
    assert float(rows["LIVE-1"]["exit_price"]) == 100.2
    assert rows["PAPER-1"]["status"] == "OPEN"
    assert rows["BOT-1"]["status"] == "OPEN"


def test_kill_switch_timeout_window_never_leaves_bare_open(forven_db, monkeypatch):
    """KS-TIMEOUT-1: the daemon wraps close_all_positions in a hard timeout. The
    dangerous window is a position whose exchange close ALREADY SUCCEEDED but whose
    post-loop close_trade_record bookkeeping hasn't run — a timeout landing there
    would leave a bare OPEN row (no marker) that the reconciler ghost-closes at mid,
    fabricating PnL. Simulate the timeout landing IN the bookkeeping gap: both
    exchange closes succeed, but close_trade_record raises for the SECOND asset (as
    the daemon's thread timeout would). The pre-close pending_close_reconcile marker
    must keep that row from ever being a bare OPEN lie; the first asset, whose
    bookkeeping did run, is fully CLOSED with the marker superseded."""
    from forven.db import get_db

    with get_db() as conn:
        _insert_open_trade(conn, "LIVE-BTC", "BTC", "live")
        _insert_open_trade(conn, "LIVE-ETH", "ETH", "live")

    monkeypatch.setattr(
        "forven.exchange.hyperliquid.get_positions",
        lambda *a, **k: {
            "positions": [
                {"position": {"coin": "BTC", "szi": 1.0}},
                {"position": {"coin": "ETH", "szi": 1.0}},
            ],
            "margin_summary": {},
        },
    )
    # BOTH exchange closes SUCCEED — the exchange truth is "flat" for both assets.
    monkeypatch.setattr(
        "forven.exchange.hyperliquid.close_position",
        lambda coin, *a, **k: {
            "status": "ok", "mid": 100.0, "close_price": 103.0,
            "exit_price": (100.2 if str(coin).upper() == "BTC" else 200.4),
            "requested_size": 1.0, "filled_size": 1.0,
        },
    )
    monkeypatch.setattr(risk, "release", lambda *a, **k: None, raising=False)

    # The timeout lands during the post-loop bookkeeping: BTC's close_trade_record
    # commits, then ETH's raises (as a thread-timeout interrupt would). The raise
    # propagates out of close_all_positions exactly as the daemon's timeout wrapper
    # sees it.
    real_close_trade_record = risk.close_trade_record

    def _close_trade_record(trade_id, *a, **k):
        if str(trade_id) == "LIVE-ETH":
            raise TimeoutError("daemon close timeout landed in bookkeeping gap")
        return real_close_trade_record(trade_id, *a, **k)

    monkeypatch.setattr(risk, "close_trade_record", _close_trade_record)

    try:
        risk.close_all_positions()
    except TimeoutError:
        pass  # the daemon's _to_thread_with_timeout swallows this

    with get_db() as conn:
        rows = {r["id"]: dict(r) for r in conn.execute(
            "SELECT id, status, exit_price, signal_data FROM trades").fetchall()}

    # Position 1: exchange close succeeded AND bookkeeping ran → fully CLOSED at the
    # real fill, with the pre-close marker superseded by close_trade_record.
    assert rows["LIVE-BTC"]["status"] == "CLOSED"
    assert float(rows["LIVE-BTC"]["exit_price"]) == 100.2
    btc_sig = json.loads(rows["LIVE-BTC"]["signal_data"])
    assert "pending_close_reconcile" not in btc_sig

    # Position 2: bookkeeping interrupted by the timeout → still OPEN, but carries
    # the pre-close marker, so the reconciler re-verifies exchange truth (recovers
    # the true exit from the fill ledger) instead of ghost-closing a bare OPEN.
    assert rows["LIVE-ETH"]["status"] == "OPEN"
    eth_sig = json.loads(rows["LIVE-ETH"]["signal_data"])
    assert eth_sig.get("pending_close_reconcile") is True
    assert eth_sig.get("pending_close_reason") == "kill_switch"
