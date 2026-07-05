"""list_hypotheses tool exposes a `view` so the Brain can see archived/graduated
crucibles (BUG id=241).

The active-only default made a disproven-and-archived hypothesis (e.g. H00031)
look "unregistered" to the Brain even though it resolves fine by id. These tests
pin that the tool threads `view` through to list_hypotheses_page and pulls in
disproven rows for non-active views, without hitting the DB for a bad view.
"""
from __future__ import annotations

import forven.agents.tools_assistant as ta
import forven.api_domains.hypotheses as hyp_api


def test_archived_view_threads_view_and_includes_disproven(monkeypatch):
    captured = {}

    def _fake_page(**kwargs):
        captured.update(kwargs)
        return {"hypotheses": [], "total": 0}

    monkeypatch.setattr(hyp_api, "list_hypotheses_page", _fake_page)
    ta._tool_list_hypotheses(view="archived")

    assert captured["view"] == "archived"
    assert captured["include_disproven"] is True


def test_default_view_is_active_without_disproven(monkeypatch):
    captured = {}

    def _fake_page(**kwargs):
        captured.update(kwargs)
        return {"hypotheses": [], "total": 0}

    monkeypatch.setattr(hyp_api, "list_hypotheses_page", _fake_page)
    ta._tool_list_hypotheses()

    assert captured.get("view") in (None, "active")
    assert captured.get("include_disproven", False) is False


def test_unknown_view_errors_without_db_call(monkeypatch):
    calls = {"n": 0}

    def _fake_page(**kwargs):
        calls["n"] += 1
        return {"hypotheses": [], "total": 0}

    monkeypatch.setattr(hyp_api, "list_hypotheses_page", _fake_page)
    out = ta._tool_list_hypotheses(view="bogus")

    assert "bogus" in out.lower()
    assert calls["n"] == 0
