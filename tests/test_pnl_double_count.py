"""Phase 5: dollar PnL must not double-count leverage.

``size`` is contract units, which already embed leverage
(position_units = equity*leverage*size_fraction/entry). So dollar P&L is
price_move * units — multiplying by leverage again is a leverage^2 double-count
that overstated pnl_usd and the promotion-gate metrics it feeds.
"""

from __future__ import annotations

import pytest

from forven.api_domains import trading


def test_calculate_closed_trade_pnl_no_leverage_double_count():
    # entry 100, exit 110, 2 units, 3x leverage, long.
    pnl_pct, pnl_usd = trading._calculate_closed_trade_pnl(
        entry_price=100.0, exit_price=110.0, size=2.0, leverage=3.0, direction="long"
    )
    # dollar P&L = price_move * units = 10 * 2 = 20 (NOT 60).
    assert pnl_usd == pytest.approx(20.0)
    # pnl_pct stays return-on-margin (unchanged): 0.10 * 3 = 0.30.
    assert pnl_pct == pytest.approx(0.30)


def test_calculate_closed_trade_pnl_short_no_double_count():
    pnl_pct, pnl_usd = trading._calculate_closed_trade_pnl(
        entry_price=100.0, exit_price=90.0, size=2.0, leverage=3.0, direction="short"
    )
    assert pnl_usd == pytest.approx(20.0)  # 10 * 2, profit on a short
    assert pnl_pct == pytest.approx(0.30)


def test_close_trade_record_pnl_usd_is_units_times_move(forven_db):
    """close_trade_record must store pnl_usd = price_move * size (units), with NO
    extra leverage factor — even when closed without a real fill price."""
    import json
    from forven.db import get_db
    from forven.trade_state import close_trade_record

    with get_db() as conn:
        conn.execute(
            "INSERT INTO trades (id, strategy, strategy_id, asset, direction, entry_price, signal_entry_price, "
            "size, risk_pct, leverage, status, execution_type, signal_data, opened_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("E-DBL", "S1", "S1", "BTC", "long", 100.0, 100.0, 2.0, 0.01, 3.0, "OPEN", "paper", json.dumps({}), "2024-01-01"),
        )
    closed = close_trade_record("E-DBL", exit_price=110.0, close_reason="signal")
    # Modeled close (no fill_exit_price) used to multiply by leverage → 60.0. Now 20.0.
    assert closed["pnl_usd"] == pytest.approx(20.0)
