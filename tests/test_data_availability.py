"""Tests for the backtest data-availability precheck (forven.strategies.data_availability).

Regression cover for S05577: a strategy that requires liquidation feeds absent
for its symbol must be BLOCKED rather than run to a silently zero-filled,
degenerate 0-trade backtest.
"""

from __future__ import annotations

import forven.strategies.data_availability as da


class _DummyStrategy:
    """Stand-in class; detection is monkeypatched in most tests."""


def _patch(monkeypatch, *, required, present, fetch=None):
    """Wire evaluate_data_availability's seams: class resolution, detection, availability."""
    import forven.strategies.backtest as backtest_mod

    monkeypatch.setattr(backtest_mod, "_resolve_strategy_class", lambda _t: _DummyStrategy)
    monkeypatch.setattr(da, "infer_required_columns", lambda _cls, _sym: frozenset(required))

    # ``present`` may be a set (static) or a callable returning the current set,
    # so a fetch can flip availability mid-evaluation.
    def _present(_sym, _tf):
        return frozenset(present() if callable(present) else present)

    monkeypatch.setattr(da, "_present_columns", _present)
    if fetch is not None:
        monkeypatch.setattr(da, "_fetch_stream", fetch)


def test_ohlcv_only_is_ok(monkeypatch):
    _patch(monkeypatch, required=set(), present=set())
    res = da.evaluate_data_availability("ohlcv_strat", "BTC/USDT", "1h")
    assert res.ok and not res.blocked
    assert res.required == []


def test_present_feed_is_ok(monkeypatch):
    _patch(monkeypatch, required={"ls_ratio"}, present={"ls_ratio", "funding_rate"})
    res = da.evaluate_data_availability("ls_strat", "BTC/USDT", "1h")
    assert res.ok and not res.blocked
    assert res.present == ["ls_ratio"]


def test_unfetchable_liquidations_blocks(monkeypatch):
    fetch_calls = []
    _patch(
        monkeypatch,
        required={"long_liq_usd", "short_liq_usd", "liq_imbalance"},
        present={"funding_rate", "ls_ratio"},
        fetch=lambda sym, stream: fetch_calls.append(stream) or False,
    )
    res = da.evaluate_data_availability("crowded_flush", "BTC/USDT", "1h", strategy_id="S05577")
    assert res.blocked and not res.ok
    assert set(res.missing_unfetchable) == {"long_liq_usd", "short_liq_usd", "liq_imbalance"}
    assert res.missing_fetchable == []
    assert "liquidations" in res.error
    assert "cannot be auto-downloaded" in res.error
    # Unfetchable feeds must not trigger a (pointless) download attempt.
    assert fetch_calls == []


def test_fetchable_feed_autofetches_then_ok(monkeypatch):
    state = {"present": set()}
    fetched = []

    def _fetch(sym, stream):
        fetched.append(stream)
        state["present"] = {"funding_rate"}  # download lands the feed
        return True

    _patch(monkeypatch, required={"funding_rate"}, present=lambda: state["present"], fetch=_fetch)
    res = da.evaluate_data_availability("fund_strat", "BTC/USDT", "1h")
    assert res.ok and not res.blocked
    assert fetched == ["funding"]
    assert res.fetched_streams == ["funding"]
    assert any("Auto-downloaded" in w for w in res.warnings)


def test_fetchable_feed_fetch_fails_blocks(monkeypatch):
    _patch(
        monkeypatch,
        required={"funding_rate"},
        present=set(),
        fetch=lambda sym, stream: False,  # download fails / no coverage
    )
    res = da.evaluate_data_availability("fund_strat", "BTC/USDT", "1h")
    assert res.blocked and not res.ok
    assert res.missing_fetchable == ["funding_rate"]
    assert "could not be downloaded" in res.error


def test_fail_open_when_class_unresolved(monkeypatch):
    import forven.strategies.backtest as backtest_mod

    monkeypatch.setattr(backtest_mod, "_resolve_strategy_class", lambda _t: None)
    res = da.evaluate_data_availability("unknown_type", "BTC/USDT", "1h")
    assert res.ok and not res.blocked


# --- funding/OI live-source probe -------------------------------------------
# Regression cover for GAP-1878e01612d8 / GAP-27bbe19a01a3: the backtest frame
# gets funding_rate/open_interest from the live market-data fetch
# (_enrich_with_market_data), NOT from the DataManager collector lake — so the
# gate must probe that source, or it blocks runs that would succeed.
def test_funding_oi_presence_uses_live_probe_not_lake(monkeypatch):
    import forven.dataeng.hub as hub
    import forven.strategies.backtest as backtest_mod

    monkeypatch.setattr(backtest_mod, "_resolve_strategy_class", lambda _t: _DummyStrategy)
    monkeypatch.setattr(
        da, "infer_required_columns", lambda _c, _s: frozenset({"funding_rate", "open_interest"})
    )
    monkeypatch.setattr(da, "_AVAIL_CACHE", {})
    # Lake has no funding/oi parquet at all…
    monkeypatch.setattr(hub, "_available_enrichment_specs", lambda *a, **k: [])
    # …but the live source (the backtest's actual provider) has data.
    monkeypatch.setattr(
        backtest_mod,
        "_resolve_market_data_series",
        lambda _asset, _s, _e: ([(1, 0.0001)], [(1, 123.0)]),
    )

    res = da.evaluate_data_availability("funding_oi_fade_v2", "BTC/USDT", "4h")
    assert res.ok and not res.blocked
    assert res.present == ["funding_rate", "open_interest"]


def test_lake_lookup_excludes_funding_oi_streams(monkeypatch):
    import forven.dataeng.hub as hub

    captured = {}

    def _spy(_symbol, _timeframe, *, include_macro, exclude_streams):
        captured["exclude_streams"] = set(exclude_streams)
        return []

    monkeypatch.setattr(hub, "_available_enrichment_specs", _spy)
    monkeypatch.setattr(da, "_probe_market_data_columns", lambda _s: frozenset())
    monkeypatch.setattr(da, "_AVAIL_CACHE", {})

    da._present_columns("ETH/USDT", "1h")
    assert captured["exclude_streams"] == {"funding", "oi"}


def test_probe_market_data_columns_reflects_series(monkeypatch):
    import forven.strategies.backtest as backtest_mod

    monkeypatch.setattr(
        backtest_mod,
        "_resolve_market_data_series",
        lambda _asset, _s, _e: ([(1, 0.0001)], []),  # funding yes, OI no
    )
    assert da._probe_market_data_columns("BTC/USDT") == frozenset({"funding_rate"})


def test_probe_market_data_columns_fails_open(monkeypatch):
    import forven.strategies.backtest as backtest_mod

    def _boom(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(backtest_mod, "_resolve_market_data_series", _boom)
    assert da._probe_market_data_columns("BTC/USDT") == frozenset(
        {"funding_rate", "open_interest"}
    )


# --- detection ------------------------------------------------------------
def test_declared_columns_detected(monkeypatch):
    # Isolate the declared path (source-scan reads the whole test module).
    monkeypatch.setattr(da, "_scan_source_columns", lambda _cls: set())

    class Declared(_DummyStrategy):
        def __init__(self, *_a, **_k):
            pass

        def data_requirements(self):
            return [{"asset": "BTC", "columns": ["funding_rate", "open_interest"]}]

    cols = da.infer_required_columns(Declared, "BTC/USDT")
    assert cols == frozenset({"funding_rate", "open_interest"})


def test_declared_ignores_unknown_columns(monkeypatch):
    monkeypatch.setattr(da, "_scan_source_columns", lambda _cls: set())

    class Declared(_DummyStrategy):
        def __init__(self, *_a, **_k):
            pass

        def data_requirements(self):
            return [{"asset": "BTC", "columns": ["close", "made_up_column", "ls_ratio"]}]

    cols = da.infer_required_columns(Declared, "BTC/USDT")
    assert cols == frozenset({"ls_ratio"})  # unknown/non-feed columns dropped


def test_columns_in_source_matches_quoted_literals_only():
    src = '''
        long = df.get("long_liq_usd", 0.0)
        f = df['funding_rate']
        # basis appears here as a bare word but not as a column literal
        note = "this mentions basis in prose"
        close = df["close"]  # not a feed column
    '''
    found = da._columns_in_source(src)
    assert "long_liq_usd" in found
    assert "funding_rate" in found
    assert "basis" not in found  # bare word / prose, not a quoted column literal
    assert "close" not in found  # OHLCV column, not in feed vocabulary
