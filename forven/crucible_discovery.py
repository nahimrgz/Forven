"""Autonomous crucible discovery.

Dispatches a benchmarking research task that harvests NEW crucibles from external
sources (YouTube / Reddit / forums / blogs / podcasts / github). It reuses the
benchmarking research_contract so the discover_*/inspect_* tools are actually
reachable for the dispatched task (without a contract the runner defaults to the
exploration lane and those tools hard-reject).

Gated by the ``autonomous_discovery`` setting (default OFF = operator-approves).
This module only creates the work item; the agent's harvesting behaviour and the
disposition of discovered crucibles (review vs auto-advance) are governed by the
mode it stamps on the task.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from forven.db import get_db

log = logging.getLogger(__name__)

_DISCOVERY_ORIGIN = "crucible_discovery"
_VALID_MODES = {"operator_approves", "autonomous"}

# How many known-crucible titles to inline in the discovery task description.
# Bounded so a large pool can't blow up the prompt.
_KNOWN_TITLE_DIGEST_LIMIT = 40
# One-line rationale appended to each disproven title (see _disproven_rationale).
_DISPROVEN_RATIONALE_TRUNCATE_CHARS = 200


def _disproven_rationale(verdict_memo_raw: Any) -> str:
    """One-line reason a disproven hypothesis died, from its verdict_memo.

    verdict_memo (hypotheses.verdict_memo, written by
    hypothesis_verdict.write_verdict_memo) carries the LLM-authored rationale
    for why a hypothesis was disproven. Surfacing it next to the bare title lets
    the discovery agent avoid re-proposing the same FAILURE MODE, not just the
    same name. Best-effort: absent/malformed/legacy-empty memo -> "" so the
    digest degrades to the bare title instead of failing.
    """
    try:
        memo = verdict_memo_raw
        if isinstance(memo, str):
            memo = json.loads(memo)
        if not isinstance(memo, dict):
            return ""
        rationale = str(memo.get("rationale") or "").strip()
        if not rationale:
            return ""
        rationale = " ".join(rationale.split())
        return rationale[:_DISPROVEN_RATIONALE_TRUNCATE_CHARS]
    except Exception:
        return ""


def _known_crucible_titles(limit: int = _KNOWN_TITLE_DIGEST_LIMIT) -> list[str]:
    """Titles the discovery agent must not re-propose: the active pool plus
    crucibles disproven within the dedup lookback.

    Before this digest existed, the prompt said "Do not duplicate existing
    crucibles" but the agent had no tool to list them — the instruction was
    unsatisfiable, and create_hypothesis had no dedup either, so the same theses
    were re-minted every cycle (audit B-16). The hard gate lives in the
    create_hypothesis tool; this digest just stops the agent wasting a harvest
    on theses the gate will reject. Best-effort: failures return [].
    """
    try:
        from forven.research_contract import get_hypothesis_discipline_settings

        lookback_days = int(
            get_hypothesis_discipline_settings()["disproven_dedup_lookback_days"]
        )
    except Exception:  # pragma: no cover — defence in depth
        lookback_days = 30
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(0, lookback_days))).isoformat()
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT title, status, verdict_memo
                FROM hypotheses
                WHERE (manager_state = 'active' AND status IN ('proposed', 'researching', 'proven'))
                   OR (status = 'disproven' AND COALESCE(verdict_memo_at, updated_at, created_at) >= ?)
                ORDER BY COALESCE(updated_at, created_at) DESC
                LIMIT ?
                """,
                (cutoff, max(1, int(limit))),
            ).fetchall()
    except Exception as exc:  # pragma: no cover — defence in depth
        log.warning("could not build known-crucible digest: %s", exc)
        return []
    seen: set[str] = set()
    titles: list[str] = []
    for row in rows:
        title = str(row["title"] or "").strip()
        key = title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        entry = title
        if str(row["status"] or "").strip().lower() == "disproven":
            rationale = _disproven_rationale(row["verdict_memo"])
            if rationale:
                entry = f"{title} (died: {rationale})"
        titles.append(entry)
    return titles


def _discovery_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(settings, dict):
        effective = settings
    else:
        from forven.research_contract import get_effective_research_settings

        effective = get_effective_research_settings()
    block = effective.get("autonomous_discovery")
    return dict(block) if isinstance(block, dict) else {}


def _open_discovery_task_count() -> int:
    """Count pending/running discovery tasks (so we don't pile up duplicates)."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT input_data
            FROM agent_tasks
            WHERE status NOT IN ('done', 'completed', 'reviewed', 'failed', 'cancelled', 'expired')
              AND input_data LIKE ?
            """,
            (f"%{_DISCOVERY_ORIGIN}%",),
        ).fetchall()
    count = 0
    for row in rows:
        try:
            payload = json.loads(row["input_data"] or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("origin_mode") == _DISCOVERY_ORIGIN:
            count += 1
    return count


def run_crucible_discovery(*, settings: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """Dispatch one external-source discovery task.

    Scheduled use honors the ``autonomous_discovery.enabled`` flag; operator demand
    passes ``force=True`` to run regardless of the flag (the dedup against an open
    discovery task still applies, so the operator can't pile up duplicates).
    Returns {"created": bool, "reason"?: str, "task_id"?: int, "mode"?: str}.
    Never raises — failures are reported in the return value.
    """
    block = _discovery_settings(settings)
    if not force and not bool(block.get("enabled", False)):
        return {"created": False, "reason": "disabled"}

    mode = str(block.get("mode") or "operator_approves").strip().lower()
    if mode not in _VALID_MODES:
        mode = "operator_approves"

    try:
        max_open = int(block.get("max_open_discovery_tasks") or 1)
    except (TypeError, ValueError):
        max_open = 1
    if _open_discovery_task_count() >= max(1, max_open):
        return {"created": False, "reason": "already_open"}

    from forven.research_contract import build_research_contract, get_effective_research_settings

    try:
        contract = build_research_contract(
            lane="benchmarking",
            settings=get_effective_research_settings(),
            available_datasets=[],
        ).to_dict()
    except Exception as exc:  # pragma: no cover — defence in depth
        log.warning("could not build discovery research_contract: %s", exc)
        contract = {}

    if mode == "operator_approves":
        review_clause = (
            "Operator-approves mode: create each new crucible as 'proposed' "
            "(origin=harvested) for operator review — do NOT push it onward yourself."
        )
    else:
        review_clause = (
            "Autonomous mode: proposed crucibles may proceed through the normal pipeline."
        )

    known_titles = _known_crucible_titles()
    dedup_clause = ""
    if known_titles:
        dedup_clause = (
            "KNOWN crucibles (active or recently disproven) — do NOT re-propose these "
            "or near-duplicates; create_hypothesis will reject them: "
            + "; ".join(known_titles)
            + ". "
        )

    description = (
        "Discover NEW trading-idea crucibles by harvesting external sources "
        "(YouTube/Reddit/forums/blogs/podcasts/github). Use the discover_* and inspect_* "
        "research tools to find materially novel, testable theses, then call create_hypothesis "
        "for each with explicit target assets/timeframes, mechanism, and the source's "
        "claimed_edge captured as an artifact. Do not duplicate existing crucibles. "
        f"{dedup_clause}"
        f"{review_clause}"
    )

    try:
        from forven.brain import assign_task

        task_id = assign_task(
            agent_id="strategy-developer",
            task_type="research",
            title="Discover new crucibles from external sources",
            description=description,
            input_data={
                "origin_mode": _DISCOVERY_ORIGIN,
                "discovery_mode": mode,
                "research_contract": contract,
            },
            priority=1,
            source="user" if force else "system",
        )
    except Exception as exc:
        log.exception("crucible discovery dispatch failed")
        return {"created": False, "reason": f"dispatch_failed: {exc}"}

    return {"created": True, "task_id": int(task_id) if task_id else None, "mode": mode}
