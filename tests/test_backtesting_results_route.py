"""The forven_get_results agent tool calls GET /api/backtesting/results/{result_id}
(BacktestingClient.get_results -> '/backtesting/results/{id}' joined onto the '/api'
base). That route was missing — the results endpoint only existed at
/api/results/{result_id} — so every forven_get_results returned HTTP 404 for valid,
locally-persisted result IDs (e.g. S00246-btc-1783359759629), blocking audits.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from forven.api_security import require_operator_access
from forven.routers.strategies import router as strategies_router


def _client(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_get_backtest_result(result_id, remote_skip=False):
        captured["result_id"] = result_id
        captured["remote_skip"] = remote_skip
        return {"result_id": result_id, "ok": True}

    monkeypatch.setattr("forven.api_core.get_backtest_result", _fake_get_backtest_result)

    app = FastAPI()
    app.include_router(strategies_router)
    app.dependency_overrides[require_operator_access] = lambda: None
    return TestClient(app), captured


def test_backtesting_results_route_exists_and_delegates(monkeypatch):
    client, captured = _client(monkeypatch)

    resp = client.get("/api/backtesting/results/S00246-btc-1783359759629")

    assert resp.status_code == 200
    assert resp.json()["result_id"] == "S00246-btc-1783359759629"
    assert captured["result_id"] == "S00246-btc-1783359759629"


def test_backtesting_results_route_accepts_client_query_params(monkeypatch):
    # BacktestingClient.get_results sends include_trades / include_equity_curve as
    # query params — the route must accept them (no 422); they don't alter the call.
    client, captured = _client(monkeypatch)

    resp = client.get(
        "/api/backtesting/results/S00246-btc-1783359759629",
        params={"include_trades": "true", "include_equity_curve": "true"},
    )

    assert resp.status_code == 200
    assert captured["result_id"] == "S00246-btc-1783359759629"
