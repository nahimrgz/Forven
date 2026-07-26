"""Regression tests for read_file pagination + per-tool output caps.

The old read_file silently sliced content[:10000] with no marker: agents read
truncated files believing they were complete, recomposed them, and overwrote
the originals (2026-07-23 report #5246 — 277KB of LESSONS.md lost that way).
"""

from __future__ import annotations

import asyncio

import pytest

import forven.workspace as ws_mod
from forven.agents.tools_core import _tool_read_file
from forven.agents.tool_registry import TRUNCATION_LINE_MARKER, execute_tool


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "notes").mkdir(exist_ok=True)
    monkeypatch.setattr(ws_mod, "WORKSPACE_DIR", ws_dir)
    monkeypatch.setattr(ws_mod, "LEGACY_WORKSPACE_DIR", ws_dir)
    return ws_dir


def test_small_file_returned_verbatim(workspace):
    (workspace / "notes" / "small.md").write_text("hello\nworld\n", encoding="utf-8")
    assert _tool_read_file("notes/small.md") == "hello\nworld\n"


def test_large_file_first_chunk_carries_explicit_truncation_marker(workspace):
    content = "".join(f"line-{i:05d}\n" for i in range(2500))  # ~27.5KB
    (workspace / "notes" / "big.md").write_text(content, encoding="utf-8")

    result = _tool_read_file("notes/big.md")

    assert "TRUNCATED" in result
    assert str(len(content)) in result  # total size disclosed
    assert "offset=" in result  # tells the agent how to continue
    assert "line-00000" in result


def test_large_file_chunks_reassemble_to_original(workspace):
    content = "".join(f"line-{i:05d}\n" for i in range(2500))
    (workspace / "notes" / "big.md").write_text(content, encoding="utf-8")

    chunks: list[str] = []
    offset = 0
    for _ in range(10):  # safety bound
        result = _tool_read_file("notes/big.md", offset=offset)
        assert f"chars {offset}-" in result
        body = result.split("]\n", 1)[1]
        if "TRUNCATED" in body:
            body, _marker = body.rsplit("\n\n…[TRUNCATED", 1)
            chunks.append(body)
            offset += len(body)
        else:
            body = body.rsplit("\n\n[end of file", 1)[0]
            chunks.append(body)
            break

    assert "".join(chunks) == content


def test_malformed_offset_is_an_explicit_error_not_a_silent_restart(workspace):
    (workspace / "notes" / "small.md").write_text("hello", encoding="utf-8")
    result = _tool_read_file("notes/small.md", offset="10.5")
    assert result.startswith("Error")
    assert "offset" in result


def test_offset_beyond_eof_reports_size(workspace):
    (workspace / "notes" / "small.md").write_text("short", encoding="utf-8")
    result = _tool_read_file("notes/small.md", offset=999)
    assert result.startswith("Error")
    assert "5" in result  # discloses the real size


def test_read_file_long_lines_survive_global_per_line_cap(workspace):
    # A single 5000-char line: the global 2000-chars-per-line cap used to
    # corrupt code/JSON reads mid-line with no way around it.
    (workspace / "notes" / "wide.md").write_text("x" * 5000, encoding="utf-8")

    result = asyncio.run(execute_tool("read_file", {"path": "notes/wide.md"}))

    assert TRUNCATION_LINE_MARKER not in result
    assert "x" * 5000 in result


def test_write_workspace_takes_append_lock(workspace, monkeypatch):
    acquired: list[bool] = []
    real_lock = ws_mod._APPEND_LOCK

    class SpyLock:
        def __enter__(self):
            acquired.append(True)
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    monkeypatch.setattr(ws_mod, "_APPEND_LOCK", SpyLock())
    ws_mod.write_workspace("notes/locked.md", "content")
    assert acquired, "write_workspace must serialize against append_workspace via _APPEND_LOCK"
