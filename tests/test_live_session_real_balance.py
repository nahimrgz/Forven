"""Live/deployed sessions must show the REAL Hyperliquid balance, never the
fabricated $10k paper-sandbox base — and live sizing must fail closed rather than
size off the _ACCOUNT_FALLBACK constant.

Covers the fix for: "live trading balances show a fake amount and should show
hyper balances".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from forven.api_domains import paper as paper_domain
from forven.db import get_db, kv_set, live_equity_baseline_kv_key


def _insert_strategy(strategy_id: str, *, stage: str = "paper") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO strategies
            (id, name, type, symbol, timeframe, params, metrics, status, owner, stage,
             stage_changed_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                "Live Balance Strategy",
                "ema_cross",
                "BTC/USDT",
                "1h",
                json.dumps({"fast": 12, "slow": 26}),
                json.dumps({"total_trades": 40}),
                stage,
                "risk-manager",
                stage,
                now,
                now,
                now,
            ),
        )


def _insert_open_live_trade(strategy_id: str, *, entry_price: float = 100.0, size: float = 2.0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades
            (id, strategy, strategy_id, asset, symbol, direction, entry_price, fill_entry_price,
             size, risk_pct, leverage, status, execution_type, signal_data, opened_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{strategy_id}-trade-live",
                strategy_id,
                strategy_id,
                "BTC",
                "BTC/USDT",
                "long",
                entry_price,
                entry_price,
                size,
                0.01,
                1.0,
                "OPEN",
                "live",
                json.dumps({"source": "exchange_sync"}),
                now,
                now,
            ),
        )


def _insert_closed_trade(strategy_id: str, *, entry_price=100.0, exit_price=103.0, size=2.0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades
            (id, strategy, strategy_id, asset, symbol, direction, entry_price, exit_price,
             fill_exit_price, size, leverage, status, execution_type, signal_data, opened_at, closed_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{strategy_id}-trade-closed",
                strategy_id,
                strategy_id,
                "BTC",
                "BTC/USDT",
                "long",
                entry_price,
                None,
                exit_price,
                size,
                1.0,
                "CLOSED",
                "live",
                json.dumps({"close_reason": "signal_exit"}),
                now,
                now,
                now,
            ),
        )


def _set_real_account(account_value: float, *, source: str = "exchange", **extra) -> None:
    kv_set(
        "daemon_state",
        {
            "exchange_account": {
                "accountValue": account_value,
                "withdrawable": extra.get("withdrawable", account_value),
                "totalMarginUsed": extra.get("totalMarginUsed", 0.0),
                "source": source,
                "network": extra.get("network", "testnet"),
                "synced_at": extra.get("synced_at", "2026-06-28T00:00:00+00:00"),
            },
            "account_equity": account_value,
            "last_prices": extra.get("last_prices", {}),
            "exchange_positions": extra.get("exchange_positions", {}),
        },
    )


# ── Display: real balance for live, simulated for paper ──────────────────────


def test_paper_session_capital_is_simulated(forven_db):
    _insert_strategy("real-bal-paper", stage="paper")
    _insert_closed_trade("real-bal-paper")  # +6.0 realized

    session = paper_domain.get_paper_session("compat:strategy:real-bal-paper")

    assert session["compat_kind"] == "paper"
    assert session["balance_source"] == "simulated"
    assert session["account_value"] is None
    # Paper sandbox is unchanged: $10k base + realized PnL.
    assert session["capital"] == pytest.approx(10_006.0)


def test_deployed_session_shows_real_account_value(forven_db):
    _set_real_account(1004.13, source="exchange")
    _insert_strategy("real-bal-live", stage="live_graduated")

    session = paper_domain.get_paper_session("compat:strategy:real-bal-live")

    assert session["compat_kind"] == "deployed"
    assert session["balance_source"] == "exchange"
    assert session["account_value"] == pytest.approx(1004.13)
    # Capital == real wallet equity (NOT 10k + pnl, and NOT double-counting pnl).
    assert session["capital"] == pytest.approx(1004.13)
    assert session["account_network"] == "testnet"


def test_deployed_session_without_real_snapshot_is_unavailable_not_fake_10k(forven_db):
    # source == 'paper' is the creds-missing fallback — must NOT count as real.
    _set_real_account(10_000.0, source="paper")
    _insert_strategy("real-bal-live-unavail", stage="live_graduated")

    session = paper_domain.get_paper_session("compat:strategy:real-bal-live-unavail")

    assert session["compat_kind"] == "deployed"
    assert session["balance_source"] == "unavailable"
    assert session["account_value"] is None  # never present the fake base as real


def test_deployed_unavailable_pct_is_not_anchored_to_fake_10k(forven_db):
    # A deployed session with no real balance snapshot must NOT report a return %
    # computed off the fabricated $10k sandbox base — capital/pct are unavailable.
    _set_real_account(10_000.0, source="paper")  # creds-missing fallback => not real
    _insert_strategy("real-bal-unavail-pct", stage="live_graduated")
    _insert_closed_trade("real-bal-unavail-pct")  # +6.0 realized

    session = paper_domain.get_paper_session("compat:strategy:real-bal-unavail-pct")

    assert session["balance_source"] == "unavailable"
    assert session["account_value"] is None
    assert session["capital"] is None
    assert session["initial_capital"] is None
    # The strategy's own $ PnL is still real and shown; the % is undefined (NOT 0.06%).
    assert session["total_pnl"] == pytest.approx(6.0)
    assert session["total_pnl_pct"] is None


def test_deployed_session_pct_anchors_to_stamped_baseline(forven_db):
    _set_real_account(1006.0, source="exchange")
    _insert_strategy("real-bal-baseline", stage="live_graduated")
    _insert_closed_trade("real-bal-baseline")  # +6.0 realized
    # Stamp a baseline distinct from the derived (equity - pnl == 1000) cost basis.
    kv_set(live_equity_baseline_kv_key("real-bal-baseline"), {"equity": 500.0, "source": "exchange"})

    session = paper_domain.get_paper_session("compat:strategy:real-bal-baseline")

    assert session["account_value"] == pytest.approx(1006.0)
    assert session["total_pnl"] == pytest.approx(6.0)
    # 6.0 / 500.0 * 100 == 1.2% (stamped baseline), NOT 6 / 1000 == 0.6% (derived).
    assert session["total_pnl_pct"] == pytest.approx(1.2)


def test_deployed_position_uses_exchange_unrealized_pnl(forven_db):
    _set_real_account(
        1500.0,
        source="exchange",
        last_prices={"BTC": 110.0},
        exchange_positions={
            "BTC:long": {
                "asset": "BTC",
                "direction": "long",
                "mark_price": 121.0,
                "entry_price": 100.0,
                "unrealized_pnl": 42.0,
                "size": 2.0,
            }
        },
    )
    _insert_strategy("real-bal-pos", stage="live_graduated")
    _insert_open_live_trade("real-bal-pos", entry_price=100.0, size=2.0)

    session = paper_domain.get_paper_session("compat:strategy:real-bal-pos")
    position = session["position"]
    assert position is not None
    # Exchange-reported unrealized (42.0) overrides the local last_prices estimate
    # ((110-100)*2 == 20.0).
    assert position["unrealized_pnl"] == pytest.approx(42.0)


def test_paper_position_ignores_unrelated_exchange_position(forven_db):
    # A PAPER position on the same coin/direction must NOT pick up a live strategy's
    # exchange unrealized — matching is gated to execution_type == 'live'.
    kv_set(
        "daemon_state",
        {
            "last_prices": {"BTC": 110.0},
            "exchange_positions": {
                "BTC:long": {"asset": "BTC", "direction": "long", "unrealized_pnl": 999.0, "entry_price": 100.0, "size": 2.0}
            },
        },
    )
    _insert_strategy("real-bal-paper-pos", stage="paper")
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades
            (id, strategy, strategy_id, asset, symbol, direction, entry_price, fill_entry_price,
             size, leverage, status, execution_type, signal_data, opened_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "real-bal-paper-pos-trade",
                "real-bal-paper-pos",
                "real-bal-paper-pos",
                "BTC",
                "BTC/USDT",
                "long",
                100.0,
                100.0,
                2.0,
                1.0,
                "OPEN",
                "paper_challenger",
                json.dumps({}),
                now,
                now,
            ),
        )

    session = paper_domain.get_paper_session("compat:strategy:real-bal-paper-pos")
    position = session["position"]
    assert position is not None
    # Local estimate (110-100)*2 == 20.0, NOT the live position's 999.0.
    assert position["unrealized_pnl"] == pytest.approx(20.0)


# ── Sizing: live fails closed, never sizes off the fallback constant ─────────


def test_get_real_account_equity_none_when_unavailable(forven_db):
    from forven import scanner

    kv_set("daemon_state", {})
    kv_set("risk_state", {})
    kv_set("daily_risk", {})
    assert scanner._get_real_account_equity() is None
    # The tolerant accessor still returns the fallback for paper/sim callers.
    assert scanner._get_account_equity() == pytest.approx(scanner._ACCOUNT_FALLBACK)


def test_get_real_account_equity_reads_daemon_snapshot(forven_db):
    from forven import scanner

    kv_set("daemon_state", {"account_equity": 2500.0})
    assert scanner._get_real_account_equity() == pytest.approx(2500.0)
    assert scanner._get_account_equity() == pytest.approx(2500.0)


def test_paper_test_mode_forces_local_execution_for_live_strategy(forven_db):
    # The kernel + legacy paths share this carve-out: under paper-test mode a
    # live/deployed strategy must execute LOCALLY (simulated fill, NO real order).
    from forven import scanner

    live_strat = {"stage": "live_graduated", "asset": "BTC"}

    kv_set("forven:settings", {"paper_test_mode_enabled": True, "paper_test_local_execution_only": True})
    assert scanner._paper_test_local_execution_for(live_strat) is True

    # With paper-test mode OFF a genuine live strategy executes for real (not local).
    kv_set(
        "forven:settings",
        {
            "paper_test_mode_enabled": False,
            "paper_stage_local_execution_only": False,
        },
    )
    assert scanner._paper_test_local_execution_for(live_strat) is False
