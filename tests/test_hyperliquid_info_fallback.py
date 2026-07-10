"""Regression coverage for HyperLiquid Info bootstrap fallbacks."""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


class _DummyHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture
def hl_module(monkeypatch):
    root = types.ModuleType("hyperliquid")
    root.__path__ = []

    exchange_mod = types.ModuleType("hyperliquid.exchange")

    class _DummyExchange:
        def __init__(self, account, url, **kwargs):
            self.wallet = account
            self.base_url = url
            self.kwargs = kwargs

    exchange_mod.Exchange = _DummyExchange

    info_mod = types.ModuleType("hyperliquid.info")

    class _BrokenInfo:
        def __init__(self, _url, skip_ws=True, **_kwargs):
            assert skip_ws is True
            raise IndexError("list index out of range")

    info_mod.Info = _BrokenInfo

    utils_mod = types.ModuleType("hyperliquid.utils")
    utils_mod.__path__ = []
    constants_mod = types.ModuleType("hyperliquid.utils.constants")
    constants_mod.TESTNET_API_URL = "https://test.hyperliquid.local"
    constants_mod.MAINNET_API_URL = "https://main.hyperliquid.local"
    utils_mod.constants = constants_mod

    types_mod = types.ModuleType("hyperliquid.utils.types")
    class _DummyCloid:
        @classmethod
        def from_str(cls, val):
            return val
    types_mod.Cloid = _DummyCloid
    utils_mod.types = types_mod

    root.exchange = exchange_mod
    root.info = info_mod
    root.utils = utils_mod

    monkeypatch.setitem(sys.modules, "hyperliquid", root)
    monkeypatch.setitem(sys.modules, "hyperliquid.exchange", exchange_mod)
    monkeypatch.setitem(sys.modules, "hyperliquid.info", info_mod)
    monkeypatch.setitem(sys.modules, "hyperliquid.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "hyperliquid.utils.constants", constants_mod)
    monkeypatch.setitem(sys.modules, "hyperliquid.utils.types", types_mod)

    _orig_hl_module = sys.modules.get("forven.exchange.hyperliquid")
    sys.modules.pop("forven.exchange.hyperliquid", None)
    import forven.exchange.hyperliquid as hl

    module = importlib.reload(hl)
    yield module
    if _orig_hl_module is not None:
        sys.modules["forven.exchange.hyperliquid"] = _orig_hl_module
    else:
        sys.modules.pop("forven.exchange.hyperliquid", None)


def test_get_account_value_uses_direct_info_fallback_when_sdk_bootstrap_breaks(hl_module, monkeypatch):
    hl = hl_module

    def _kv_get(key, default=None):
        if key == "forven:settings":
            return {
                "hyperliquid_wallet": "0xabc123",
                "hyperliquid_testnet": True,
            }
        if key == "forven:settings:secrets":
            return {}
        return default

    def _fake_urlopen(request, timeout=15):
        assert timeout == 15
        payload = json.loads(request.data.decode("utf-8"))
        if payload["type"] == "clearinghouseState":
            return _DummyHttpResponse(
                {
                    "marginSummary": {
                        "accountValue": "0",
                        "totalMarginUsed": "0",
                        "totalNtlPos": "0",
                        "totalRawUsd": "0",
                    }
                }
            )
        if payload["type"] == "spotClearinghouseState":
            return _DummyHttpResponse(
                {"balances": [{"coin": "USDC", "total": "1002.68", "hold": "0"}]}
            )
        if payload["type"] == "spotMeta":
            return _DummyHttpResponse({"tokens": [], "universe": []})
        raise AssertionError(f"Unexpected HyperLiquid info payload: {payload}")

    monkeypatch.setattr("forven.sim.clock.is_sim_active", lambda: False)
    monkeypatch.setattr(hl, "kv_get", _kv_get)
    monkeypatch.setattr(hl, "_with_breaker", lambda _name, _breaker, fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(hl.urllib.request, "urlopen", _fake_urlopen)

    account = hl.get_account_value(testnet=True, require_connection=True)

    assert account["accountValue"] == 1002.68
    assert account["totalRawUsd"] == 1002.68
    assert account["withdrawable"] == 1002.68


def test_build_info_client_logs_fallback_warning_once_per_process(hl_module, monkeypatch):
    hl = hl_module
    warnings: list[str] = []

    def _fake_warning(message, *args):
        warnings.append(message % args if args else str(message))

    monkeypatch.setattr(hl.log, "info", _fake_warning)

    first = hl._build_info_client("https://test.hyperliquid.local")
    second = hl._build_info_client("https://test.hyperliquid.local")

    assert first is second
    assert first.__class__.__name__ == "_HyperliquidDirectInfoClient"
    assert len(warnings) == 1
