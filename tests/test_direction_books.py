"""Approach C — direction-book routing for live multi-strategy positions.

Covers the books resolution module, can_open()'s per-book live scoping (the
long-trend + short-scalp on the same asset case), and the reconciler safety
guard that a sub-account position is never ghost-closed by another account's
pass. Paper isolation is covered in test_risk_module.py.
"""

import pytest

from forven.db import get_db, kv_set
from forven.exchange import books
from forven.exchange.risk import (
    _repair_position_protection,
    can_open,
    reconcile_all_books,
    reconcile_exchange_positions,
)


SHORT_ADDR = "0xShortBookSubAccount"
LONG_ADDR = "0xLongBookSubAccount"


def _books_settings(**overrides):
    base = {
        "live_books_enabled": True,
        "hyperliquid_long_book_address": "",
        "hyperliquid_short_book_address": SHORT_ADDR,
        "max_concurrent_positions": 5,
    }
    base.update(overrides)
    kv_set("forven:settings", base)


def _seed_live_position(trade_id, asset, direction, strategy, book, risk_pct=0.01):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_positions
            (trade_id, asset, direction, strategy, strategy_id, risk_pct, entry_price,
             correlation_group, opened_at, execution_type, book)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'live', ?)
            """,
            (trade_id, asset, direction, strategy, strategy, risk_pct, 100.0, "crypto_major", book),
        )


def _seed_open_trade(trade_id, asset, direction, strategy, book, execution_type="live"):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades
            (id, strategy, strategy_id, asset, direction, entry_price, size, risk_pct, leverage,
             status, execution_type, book, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, datetime('now'))
            """,
            (trade_id, strategy, strategy, asset, direction, 100.0, 1.0, 0.01, 1.0, execution_type, book),
        )


class TestBooksResolution:
    def test_disabled_routes_everything_to_main(self, forven_db):
        kv_set("forven:settings", {"live_books_enabled": False})
        assert books.books_enabled() is False
        assert books.resolve_open_book("long") == ("main", None)
        assert books.resolve_open_book("short") == ("main", None)
        assert books.active_book_addresses() == [("main", None)]

    def test_long_routes_to_long_book(self, forven_db):
        _books_settings(hyperliquid_long_book_address=LONG_ADDR)
        book, reason = books.resolve_open_book("long")
        assert book == "long" and reason is None
        assert books.book_address("long") == LONG_ADDR

    def test_short_with_book_routes_to_short(self, forven_db):
        _books_settings()
        book, reason = books.resolve_open_book("short")
        assert book == "short" and reason is None
        assert books.book_address("short") == SHORT_ADDR

    def test_long_only_when_no_short_book(self, forven_db):
        _books_settings(hyperliquid_short_book_address="")
        assert books.is_long_only() is True
        assert books.short_book_available() is False
        book, reason = books.resolve_open_book("short")
        assert book is None
        assert "LONG ONLY" in reason
        # Longs still route fine in long-only mode.
        assert books.resolve_open_book("long")[0] == "long"

    def test_active_addresses_include_both_books(self, forven_db):
        _books_settings(hyperliquid_long_book_address=LONG_ADDR)
        pairs = dict(books.active_book_addresses())
        assert pairs.get("long") == LONG_ADDR
        assert pairs.get("short") == SHORT_ADDR

    def test_status_reports_long_only(self, forven_db):
        _books_settings(hyperliquid_short_book_address="")
        status = books.live_books_status()
        assert status["enabled"] is True
        assert status["long_only"] is True
        assert status["short_book_configured"] is False


class TestCanOpenBookScoping:
    def test_long_and_short_same_asset_coexist_across_books(self, forven_db):
        _books_settings()
        # A long-book BTC long already open (long book = master wallet here).
        _seed_live_position("l-1", "BTC", "long", "trend-strat", "long")
        # A short-scalp on the SAME asset routes to the short book -> allowed.
        allowed, _risk, reason = can_open(
            "BTC", "short", "scalp-strat", risk_pct=0.01, execution_type="live", book="short"
        )
        assert allowed is True, reason

    def test_same_asset_same_book_still_one_net_position(self, forven_db):
        _books_settings()
        _seed_live_position("l-1", "BTC", "long", "trend-strat", "long")
        # A second long on BTC lands in the SAME (long) book -> blocked.
        allowed, _risk, reason = can_open(
            "BTC", "long", "other-strat", risk_pct=0.01, execution_type="live", book="long"
        )
        assert allowed is False
        assert "asset conflict" in reason.lower()

    def test_live_cap_is_per_book(self, forven_db):
        _books_settings(max_concurrent_positions=1)
        # One position in the long book fills the cap for the long book...
        _seed_live_position("l-1", "ETH", "long", "a", "long")
        blocked, _r, reason = can_open("SOL", "long", "b", risk_pct=0.01, execution_type="live", book="long")
        assert blocked is False
        assert "max concurrent positions" in reason.lower()
        # ...but the short book is a separate account, so it is not capped by it.
        allowed, _r2, reason2 = can_open("SOL", "short", "c", risk_pct=0.01, execution_type="live", book="short")
        assert allowed is True, reason2


class TestReconcilerBookSafetyGuard:
    def _empty_snapshot(self, *_args, **_kwargs):
        return {"raw_positions": [], "positions": [], "open_orders": [], "price_map": {}}

    def test_master_pass_does_not_ghost_close_short_book_trade(self, forven_db, monkeypatch):
        _books_settings()
        _seed_open_trade("T-SHORT-1", "BTC", "short", "scalp", "short")
        monkeypatch.setattr(
            "forven.exchange.risk._snapshot_exchange_state", self._empty_snapshot
        )
        # Master pass (account_address=None) must NOT see the short-book trade,
        # so it cannot ghost-close it even though the (empty) master snapshot
        # lacks it.
        result = reconcile_exchange_positions(account_address=None)
        assert "error" not in result
        with get_db() as conn:
            row = conn.execute("SELECT status FROM trades WHERE id = 'T-SHORT-1'").fetchone()
        assert dict(row)["status"] == "OPEN"

    def test_short_pass_considers_short_book_trade(self, forven_db, monkeypatch):
        # confirm_count=1: this test exercises book SCOPING, not the empty-read streak
        # guard (RECONCILE-6, tested separately) — sweep on the first empty read.
        _books_settings(reconcile_empty_read_confirm_count=1)
        _seed_open_trade("T-SHORT-2", "BTC", "short", "scalp", "short")
        monkeypatch.setattr(
            "forven.exchange.risk._snapshot_exchange_state", self._empty_snapshot
        )
        # The short-book pass DOES consider it; with an empty snapshot it is a
        # ghost and gets resolved (closed) — i.e. it is no longer left OPEN.
        result = reconcile_exchange_positions(account_address=SHORT_ADDR, book_label="short")
        assert "error" not in result
        with get_db() as conn:
            row = conn.execute("SELECT status FROM trades WHERE id = 'T-SHORT-2'").fetchone()
        assert dict(row)["status"] != "OPEN"

    def test_disabled_books_still_sweep_leftover_book_trade(self, forven_db, monkeypatch):
        # Books toggled OFF but the short address remains set and a short-book
        # position is still open: reconcile_all_books must still reconcile that
        # sub-account (no orphaned, never-reconciled OPEN row).
        _books_settings(live_books_enabled=False, reconcile_empty_read_confirm_count=1)  # addresses NOT cleared
        _seed_open_trade("T-LEFT-1", "BTC", "short", "scalp", "short")
        monkeypatch.setattr(
            "forven.exchange.risk._snapshot_exchange_state", self._empty_snapshot
        )
        result = reconcile_all_books()
        assert "error" not in result
        with get_db() as conn:
            row = conn.execute("SELECT status FROM trades WHERE id = 'T-LEFT-1'").fetchone()
        assert dict(row)["status"] != "OPEN"


class TestLongOnlyVolumeGateSurfacing:
    def test_status_note_explains_100k_volume_gate(self, forven_db):
        _books_settings(hyperliquid_short_book_address="")
        status = books.live_books_status()
        assert status["long_only"] is True
        assert status["note"] and "100k" in status["note"]
        assert status["subaccount_volume_requirement_usd"] == 100_000

    def test_long_only_notification_is_throttled(self, forven_db, monkeypatch):
        from forven import scanner

        calls = []
        monkeypatch.setattr(
            "forven.notifications.emit_notification",
            lambda *a, **k: calls.append(k.get("dedupe_key")),
        )
        scanner._notify_long_only_mode("BTC")
        scanner._notify_long_only_mode("ETH")  # within 6h window -> suppressed
        assert calls == ["live_long_only_mode"]


class TestSizeQuantization:
    """B1: order size must be rounded DOWN to the asset's szDecimals, fail-closed."""

    def test_rounds_down_to_szdecimals(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr(hl, "_get_sz_decimals", lambda url: {"BTC": 5, "ETH": 4})
        assert hl.quantize_size("BTC", 0.000189999, "u") == 0.00018
        assert hl.quantize_size("ETH", 0.123456, "u") == 0.1234

    def test_fails_closed_on_unknown_asset(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr(hl, "_get_sz_decimals", lambda url: {})
        # Unknown precision (and not in the static fallback) -> 0.0 (refuse order)
        assert hl.quantize_size("FOOBAR", 1.23, "u") == 0.0

    def test_static_fallback_when_meta_unavailable(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr(hl, "_get_sz_decimals", lambda url: {})
        assert hl.quantize_size("SOL", 1.239, "u") == 1.23  # fallback SOL=2

    def test_dust_rounds_to_zero_is_refused(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr(hl, "_get_sz_decimals", lambda url: {"BTC": 5})
        assert hl.quantize_size("BTC", 0.0000009, "u") == 0.0  # below 1e-5 -> 0

    def test_static_fallback_covers_every_tick_asset(self):
        # Any asset with a tick size must also have a szDecimals fallback, so a
        # meta outage can never fail-close an asset we otherwise quote.
        import forven.exchange.hyperliquid as hl
        assert set(hl.TICK_SIZES).issubset(set(hl._SZ_DECIMALS_FALLBACK))

    def test_meta_fetch_failure_is_not_cached(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        hl._SZ_DECIMALS_CACHE.clear()
        calls = {"n": 0}

        class _DC:
            def _post(self, payload):
                calls["n"] += 1
                raise RuntimeError("network blip")

        monkeypatch.setattr(hl, "_get_direct_info_client", lambda url: _DC())
        assert hl._get_sz_decimals("u") == {}
        assert hl._get_sz_decimals("u") == {}
        assert calls["n"] == 2  # retried each time, not poisoned into the cache
        hl._SZ_DECIMALS_CACHE.clear()

    def test_close_fails_open_to_raw_size_when_precision_unknown(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr(hl, "_get_sz_decimals", lambda url: {})  # FOOBAR unknown
        monkeypatch.setattr("forven.sim.clock.is_sim_active", lambda: False)
        monkeypatch.setattr(hl, "_assert_execution_allowed", lambda *a, **k: None)
        captured = {}

        class _Ex:
            base_url = "u"
            def order(self, asset, is_buy, sz, px, otype, reduce_only=False):
                captured["sz"] = sz
                return {"status": "ok", "response": {"data": {"statuses": [{}]}}}

        class _Info:
            def all_mids(self):
                return {"FOOBAR": 10.0}

        monkeypatch.setattr(hl, "_exchange_for_trading", lambda testnet, vault_address=None: (_Ex(), _Info(), "0xabc"))
        monkeypatch.setattr(hl, "_with_breaker", lambda name, br, fn, *a, **k: fn(*a, **k))
        monkeypatch.setattr(hl, "get_all_mids", lambda testnet=True: {"FOOBAR": 10.0})
        res = hl.close_position("FOOBAR", 0.5, "sell", testnet=True)
        assert "non-positive" not in str((res or {}).get("error") or "")
        assert captured.get("sz") == 0.5  # raw size attempted (reduce-only), not refused


class TestPartialCloseH3:
    def test_close_position_returns_filled_size(self, monkeypatch):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr("forven.sim.clock.is_sim_active", lambda: False)
        monkeypatch.setattr(hl, "_assert_execution_allowed", lambda *a, **k: None)
        monkeypatch.setattr(hl, "get_all_mids", lambda testnet=True: {"BTC": 100.0})
        monkeypatch.setattr(hl, "_get_sz_decimals", lambda url: {"BTC": 5})

        class _Ex:
            base_url = "u"
            def order(self, *a, **k):
                return {"status": "ok", "response": {"data": {"statuses": [{"filled": {"avgPx": "100", "totalSz": "0.3"}}]}}}

        monkeypatch.setattr(hl, "_exchange_for_trading", lambda testnet, vault_address=None: (_Ex(), object(), "0xabc"))
        monkeypatch.setattr(hl, "_with_breaker", lambda name, br, fn, *a, **k: fn(*a, **k))
        res = hl.close_position("BTC", 1.0, "sell", testnet=True)
        # Partial fill surfaced so _execute_direct can keep the residual protected.
        assert res.get("filled_size") == 0.3
        assert res.get("requested_size") == 1.0


class TestHTierSafety:
    """H8 (halt re-check at execute), H9 (failure alert), H10 (close routing fail-closed)."""

    def test_h8_open_refused_when_halt_fires_after_can_open(self, monkeypatch):
        import forven.scanner as sc
        monkeypatch.setattr("forven.sim.clock.is_sim_active", lambda: False)
        monkeypatch.setattr(sc, "_resolve_trade_vault_address", lambda tid, strict=False: None)
        monkeypatch.setattr("forven.exchange.risk.is_trading_allowed", lambda: (False, "Kill-switch active"))
        with pytest.raises(RuntimeError, match="halted"):
            sc._execute_direct(action="open", trade_id="X", strat_id="s", asset="BTC",
                               direction="long", size=0.001, price=100.0, stop_loss=97.0)

    def test_h10_strict_resolution_reraises_else_fails_open(self, monkeypatch):
        import forven.scanner as sc

        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(sc, "get_db", _boom)
        with pytest.raises(RuntimeError):
            sc._resolve_trade_vault_address("X", strict=True)
        assert sc._resolve_trade_vault_address("X", strict=False) is None

    def test_h9_emits_trade_failed_notification(self, forven_db, monkeypatch):
        import forven.scanner as sc
        import forven.brain as brain
        calls = []
        monkeypatch.setattr(
            "forven.notifications.emit_notification",
            lambda event_type=None, *a, **k: calls.append((event_type, k.get("dedupe_key"))),
        )
        monkeypatch.setattr(brain, "handoff_execution_failure_to_developer", lambda **k: None)
        sc._report_execution_failure("S1", "open", "T1", "boom")
        assert any(c[0] == "trade_failed" for c in calls)


class TestLeverageB2:
    """B2: leverage + margin mode are set on the exchange before every live open."""

    def test_margin_mode_toggle_defaults_isolated(self, forven_db):
        from forven.db import kv_set
        from forven.exchange.hyperliquid import configured_margin_is_cross
        kv_set("forven:settings", {"hyperliquid_use_cross_margin": False})
        assert configured_margin_is_cross() is False
        kv_set("forven:settings", {"hyperliquid_use_cross_margin": True})
        assert configured_margin_is_cross() is True

    def _patch_exchange(self, monkeypatch, exchange):
        import forven.exchange.hyperliquid as hl
        monkeypatch.setattr("forven.sim.clock.is_sim_active", lambda: False)
        monkeypatch.setattr(hl, "_assert_execution_allowed", lambda *a, **k: None)
        monkeypatch.setattr(hl, "_exchange_for_trading", lambda testnet, vault_address=None: (exchange, object(), "0xabc"))
        monkeypatch.setattr(hl, "_with_breaker", lambda name, br, fn, *a, **k: fn(*a, **k))

    def test_set_leverage_success_passes_int_and_mode(self, monkeypatch):
        import forven.exchange.hyperliquid as hl

        class _Ex:
            base_url = "u"
            captured = None
            def update_leverage(self, lev, asset, cross):
                _Ex.captured = (lev, asset, cross)
                return {"status": "ok"}

        self._patch_exchange(monkeypatch, _Ex())
        res = hl.set_leverage("BTC", 3.0, testnet=True, is_cross=False)
        assert res.get("error") is None
        assert _Ex.captured == (3, "BTC", False)

    def test_set_leverage_fails_closed_on_exception(self, monkeypatch):
        import forven.exchange.hyperliquid as hl

        class _Ex:
            base_url = "u"
            def update_leverage(self, *a):
                raise RuntimeError("boom")

        self._patch_exchange(monkeypatch, _Ex())
        res = hl.set_leverage("BTC", 3.0, testnet=True, is_cross=False)
        assert "boom" in (res.get("error") or "")


class TestProtectionCoverageB3:
    """B3: only stop-loss reduce-only orders count as protective coverage."""

    def _pos(self, **kw):
        base = {"asset": "BTC", "size": 0.001, "direction": "long", "entry_price": 100.0}
        base.update(kw)
        return base

    def test_take_profit_alone_is_not_protected(self):
        from forven.exchange.risk import _summarize_position_protection
        orders = [{"coin": "BTC", "reduceOnly": True, "tpsl": "tp", "sz": 0.001, "oid": 1, "triggerPx": 110.0}]
        s = _summarize_position_protection(self._pos(), orders)
        assert s["status"] == "missing"
        assert s["covered_size"] == 0.0

    def test_stop_loss_is_protected(self):
        from forven.exchange.risk import _summarize_position_protection
        orders = [{"coin": "BTC", "reduceOnly": True, "tpsl": "sl", "sz": 0.001, "oid": 2, "triggerPx": 90.0}]
        s = _summarize_position_protection(self._pos(), orders)
        assert s["fully_protected"] is True

    def test_tp_plus_sl_counts_only_the_stop(self):
        from forven.exchange.risk import _summarize_position_protection
        orders = [
            {"coin": "BTC", "reduceOnly": True, "tpsl": "sl", "sz": 0.001, "oid": 2, "triggerPx": 90.0},
            {"coin": "BTC", "reduceOnly": True, "tpsl": "tp", "sz": 0.001, "oid": 3, "triggerPx": 110.0},
        ]
        s = _summarize_position_protection(self._pos(), orders)
        assert s["covered_size"] == 0.001  # not 0.002
        assert s["fully_protected"] is True

    def test_geometry_fallback_when_no_tpsl(self):
        from forven.exchange.risk import _order_is_stop_loss
        assert _order_is_stop_loss({"reduceOnly": True, "triggerPx": 90}, {"direction": "long", "entry_price": 100}) is True
        assert _order_is_stop_loss({"reduceOnly": True, "triggerPx": 110}, {"direction": "long", "entry_price": 100}) is False
        assert _order_is_stop_loss({"reduceOnly": True, "triggerPx": 110}, {"direction": "short", "entry_price": 100}) is True

    def test_priceless_reduce_only_treated_as_stop(self):
        # A take-profit always carries a price; a price-less reduce-only therefore
        # cannot be a TP, so it counts as protective coverage (real TPs are excluded
        # by the geometry check above).
        from forven.exchange.risk import _order_is_stop_loss
        assert _order_is_stop_loss({"reduceOnly": True, "sz": 1}, {"direction": "long"}) is True
        # A take-profit (profit-side limitPx) is NOT counted as a stop.
        assert _order_is_stop_loss({"reduceOnly": True, "limitPx": 120}, {"direction": "long", "entry_price": 100}) is False


class TestRecoveryStopRouting:
    def test_repair_places_protective_stop_on_the_sub_account(self, forven_db, monkeypatch):
        _books_settings()
        captured = {}

        def _fake_place(asset, direction, size, stop_price, *, testnet=True, vault_address=None):
            captured["vault_address"] = vault_address
            return {"stop_order_id": "stop-1", "order_id": "stop-1"}

        monkeypatch.setattr("forven.exchange.hyperliquid.place_protective_stop", _fake_place)
        position = {"asset": "BTC", "direction": "short", "size": 0.5, "entry_price": 100.0, "leverage": 1.0}
        _repair_position_protection(
            position,
            matched_trade=None,
            open_orders=[],
            price_map={"BTC": 100.0},
            testnet=True,
            account_address=SHORT_ADDR,
        )
        assert captured.get("vault_address") == SHORT_ADDR


class TestFillLedgerRecoveryH4:
    """H4: ghost-close recovers the TRUE exit/fee from the fill ledger, not the
    reconcile-time mid."""

    _OPENED = "2020-01-01T00:00:00+00:00"

    def test_recovers_weighted_exit_and_fee(self, monkeypatch):
        import forven.exchange.risk as risk
        fills = [
            {"coin": "BTC", "dir": "Close Long", "px": "100", "sz": "1", "fee": "0.5", "closedPnl": "10", "time": 1000},
            {"coin": "BTC", "dir": "Close Long", "px": "102", "sz": "1", "fee": "0.5", "closedPnl": "12", "time": 2000},
            {"coin": "ETH", "dir": "Close Long", "px": "50", "sz": "1", "fee": "0.1", "time": 3000},
        ]
        monkeypatch.setattr("forven.exchange.hyperliquid.get_user_fills", lambda *a, **k: fills)
        out = risk._recover_exit_from_fills(
            "BTC", {"direction": "long", "opened_at": self._OPENED, "size": 2.0},
            testnet=True, account_address=None,
        )
        assert out is not None
        assert abs(out["exit_price"] - 101.0) < 1e-9  # size-weighted across the 2 closing fills
        assert abs(out["fee_usd"] - 1.0) < 1e-9
        assert out["fill_count"] == 2

    def test_size_cap_isolates_first_close_from_reopened_position(self, monkeypatch):
        # The coin was closed (this trade, size 1 @ 100), then RE-OPENED and
        # re-closed in the same direction (size 1 @ 200) before reconcile noticed.
        # Recovery must use THIS position's close only, not blend the later one.
        import forven.exchange.risk as risk
        fills = [
            {"coin": "BTC", "dir": "Close Long", "px": "100", "sz": "1", "fee": "0.5", "time": 1000},
            {"coin": "BTC", "dir": "Close Long", "px": "200", "sz": "1", "fee": "0.5", "time": 5000},
        ]
        monkeypatch.setattr("forven.exchange.hyperliquid.get_user_fills", lambda *a, **k: fills)
        out = risk._recover_exit_from_fills(
            "BTC", {"direction": "long", "opened_at": self._OPENED, "size": 1.0},
            testnet=True, account_address=None,
        )
        assert out is not None
        assert abs(out["exit_price"] - 100.0) < 1e-9  # NOT 150 (the blended avg)
        assert out["fill_count"] == 1
        assert out["closed_at"] is not None

    def test_missing_opened_at_bails_to_none(self, monkeypatch):
        # No lower time bound -> cannot isolate this position's close -> bail
        # (caller falls back to the reconcile-time mid).
        import forven.exchange.risk as risk
        fills = [{"coin": "BTC", "dir": "Close Long", "px": "100", "sz": "1", "fee": "0.5", "time": 1000}]
        monkeypatch.setattr("forven.exchange.hyperliquid.get_user_fills", lambda *a, **k: fills)
        out = risk._recover_exit_from_fills("BTC", {"direction": "long", "size": 1.0}, testnet=True, account_address=None)
        assert out is None

    def test_ignores_opposite_direction_close(self, monkeypatch):
        import forven.exchange.risk as risk
        fills = [{"coin": "BTC", "dir": "Close Short", "px": "100", "sz": "1", "fee": "0.5", "time": 1000}]
        monkeypatch.setattr("forven.exchange.hyperliquid.get_user_fills", lambda *a, **k: fills)
        out = risk._recover_exit_from_fills(
            "BTC", {"direction": "long", "opened_at": self._OPENED, "size": 1.0}, testnet=True, account_address=None
        )
        assert out is None

    def test_no_fills_returns_none(self, monkeypatch):
        import forven.exchange.risk as risk
        monkeypatch.setattr("forven.exchange.hyperliquid.get_user_fills", lambda *a, **k: [])
        out = risk._recover_exit_from_fills(
            "BTC", {"direction": "long", "opened_at": self._OPENED, "size": 1.0}, testnet=True, account_address=None
        )
        assert out is None


class TestFundingNetPnlH6:
    """H6: realized funding is folded into net_pnl_pct for live closes."""

    def test_funding_folded_into_net(self, forven_db):
        import json
        import forven.scanner as sc
        _seed_open_trade("FT1", "BTC", "long", "s", None)
        sc._close_trade_db("FT1", 110.0, 0.10, 10.0, funding_usd=2.0)
        with get_db() as conn:
            row = dict(conn.execute(
                "SELECT pnl_pct, net_pnl_pct, fees_pct, signal_data FROM trades WHERE id='FT1'"
            ).fetchone())
        sd = json.loads(row["signal_data"] or "{}")
        assert sd.get("funding_usd") == 2.0
        # margin = entry(100) * size(1) / lev(1) = 100 -> funding_pct = 2/100 = 0.02
        assert abs(sd.get("funding_pct") - 0.02) < 1e-9
        assert abs(row["net_pnl_pct"] - (row["pnl_pct"] - row["fees_pct"] - 0.02)) < 1e-6

    def test_no_funding_leaves_net_at_gross_minus_fees(self, forven_db):
        import json
        import forven.scanner as sc
        _seed_open_trade("FT2", "BTC", "long", "s", None)
        sc._close_trade_db("FT2", 110.0, 0.10, 10.0)
        with get_db() as conn:
            row = dict(conn.execute(
                "SELECT pnl_pct, net_pnl_pct, fees_pct, signal_data FROM trades WHERE id='FT2'"
            ).fetchone())
        sd = json.loads(row["signal_data"] or "{}")
        assert "funding_usd" not in sd
        assert abs(row["net_pnl_pct"] - (row["pnl_pct"] - row["fees_pct"])) < 1e-6

    def test_funding_read_from_signal_data_on_default_close_path(self, forven_db):
        # The default fast-path close discards the execution result dict, so
        # _execute_direct persists funding in signal_data; _close_trade_db must
        # fold it even when the caller passes no funding_usd kwarg (HIGH-#1 fix).
        import json
        import forven.scanner as sc
        _seed_open_trade("FT3", "BTC", "long", "s", None)
        with get_db() as conn:
            conn.execute(
                "UPDATE trades SET signal_data = ? WHERE id = 'FT3'",
                (json.dumps({"close_funding_usd": 2.0}),),
            )
        sc._close_trade_db("FT3", 110.0, 0.10, 10.0)  # NO funding_usd kwarg
        with get_db() as conn:
            row = dict(conn.execute(
                "SELECT pnl_pct, net_pnl_pct, fees_pct, signal_data FROM trades WHERE id='FT3'"
            ).fetchone())
        sd = json.loads(row["signal_data"] or "{}")
        assert sd.get("funding_usd") == 2.0
        assert abs(sd.get("funding_pct") - 0.02) < 1e-9
        assert abs(row["net_pnl_pct"] - (row["pnl_pct"] - row["fees_pct"] - 0.02)) < 1e-6


class TestLiquidationMonitorH7:
    """H7: open positions are watched against their liquidation price each tick."""

    def test_critical_alert_when_close_to_liq(self, forven_db, monkeypatch):
        import forven.daemon as dmn
        calls = []
        monkeypatch.setattr(dmn, "_get_testnet", lambda: True)
        monkeypatch.setattr("forven.exchange.books.books_enabled", lambda: False)
        monkeypatch.setattr(
            "forven.exchange.hyperliquid.get_positions",
            lambda testnet=True, **k: {"positions": [
                {"position": {"coin": "BTC", "szi": "0.1", "liquidationPx": "96"}}
            ]},
        )
        monkeypatch.setattr("forven.exchange.hyperliquid.get_all_mids", lambda testnet=True: {"BTC": 100.0})
        monkeypatch.setattr(
            "forven.notifications.emit_notification",
            lambda event_type=None, *a, **k: calls.append((event_type, k.get("severity"))),
        )
        dmn._check_liquidation_distances()
        # dist = |100-96|/100 = 0.04 <= crit (0.07) -> critical
        assert any(c[1] == "critical" for c in calls)

    def test_no_alert_when_far_from_liq(self, forven_db, monkeypatch):
        import forven.daemon as dmn
        calls = []
        monkeypatch.setattr(dmn, "_get_testnet", lambda: True)
        monkeypatch.setattr("forven.exchange.books.books_enabled", lambda: False)
        monkeypatch.setattr(
            "forven.exchange.hyperliquid.get_positions",
            lambda testnet=True, **k: {"positions": [
                {"position": {"coin": "BTC", "szi": "0.1", "liquidationPx": "50"}}
            ]},
        )
        monkeypatch.setattr("forven.exchange.hyperliquid.get_all_mids", lambda testnet=True: {"BTC": 100.0})
        monkeypatch.setattr(
            "forven.notifications.emit_notification",
            lambda event_type=None, *a, **k: calls.append(event_type),
        )
        dmn._check_liquidation_distances()
        assert calls == []  # dist = 0.5, far from liquidation
