"""write_file returns distinct, actionable envelopes (BUG-139).

Before this fix the agent could not tell a policy rejection ("stop, you're not
allowed") from a transient FS error ("hiccup, safe to retry"), and a successful
write was mislabeled by the tool-execution ledger. These tests pin the three
envelope shapes: REJECTED (do not retry), FAILED transient (safe to retry), and
success ("Appended to" / "Wrote").
"""
from __future__ import annotations

import pytest

from forven.agents.tools_core import _tool_write_file


def test_write_file_rejects_disallowed_path():
    result = _tool_write_file("secrets.txt", "x")
    assert result.startswith("REJECTED:")
    assert "Do not retry" in result


def test_write_file_rejects_protected_file():
    result = _tool_write_file("SOUL.md", "x")
    assert result.startswith("REJECTED:")
    assert "Do not retry" in result


def test_write_file_success_append():
    result = _tool_write_file("memory/plan_check.md", "hello", append=True)
    assert result.startswith("Appended to")
    assert "memory/plan_check.md" in result


def test_write_file_success_overwrite():
    result = _tool_write_file("memory/plan_check.md", "hello", append=False)
    assert result.startswith("Wrote")
    assert "memory/plan_check.md" in result


def test_write_file_transient_error_is_retryable(monkeypatch):
    def _boom(path, content):
        raise OSError("disk full")

    monkeypatch.setattr("forven.agents.tools_core.append_workspace", _boom)
    result = _tool_write_file("memory/plan_check.md", "hello", append=True)
    assert result.startswith("FAILED (transient)")
    assert "Safe to retry" in result
    assert "disk full" in result
