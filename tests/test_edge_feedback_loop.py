"""Outcome→generation feedback loop (2026-07-02).

Before this fix set the pipeline reacted to failure only per-strategy: the
structured failure taxonomy (query_failure_taxonomy) had zero callers, the
diversity guard steered by generation frequency alone, cited_skills was never
written so skill outcome closure never fired, and ideation (research context)
never saw the quant-skills KB. These tests pin the closed loop:

- family_outcome_stats + survivor-weighted diversity guard (dead vs live regions)
- render_failure_taxonomy consuming gate_rejections
- register-time citation persistence (agent_tasks.strategy_id backfill +
  output_data merge) feeding skill_outcomes closure
- taxonomy + learned-skills injection into agent and research contexts
"""
from __future__ import annotations

import json
import re

import forven.strategy_diversity as sd
from forven.db import get_db


# ── helpers ──────────────────────────────────────────────────────────────────


def _insert_strategy(sid: str, name: str, *, stage: str = "archived", created_days_ago: int = 1):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO strategies (id, name, type, stage, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now', ?))",
            (sid, name, name, stage, f"-{int(created_days_ago)} days"),
        )


def _insert_event(sid: str, to_state: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO strategy_events (strategy_id, to_state) VALUES (?, ?)",
            (sid, to_state),
        )


def _insert_rejection(gate: str, reason_code: str, strategy_type: str, regime: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO gate_rejections (strategy_id, gate, reason_code, reason_text, strategy_type, regime_context) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("S0001", gate, reason_code, f"{reason_code} details", strategy_type, regime),
        )


# ── family_outcome_stats ─────────────────────────────────────────────────────


def test_family_outcome_stats_counts_attempts_and_survivors(forven_db):
    _insert_strategy("S1", "rsi_reversal_btc")
    _insert_strategy("S2", "rsi_divergence_eth")
    _insert_strategy("S3", "rsi_pullback_sol")
    _insert_event("S1", "paper")  # one RSI survivor via events
    _insert_strategy("S4", "funding_carry_btc")
    _insert_strategy("S5", "funding_squeeze_eth")
    # Survivor detectable from current stage even without an event row.
    _insert_strategy("S6", "macd_trend_btc", stage="paper")
    # Outside the window — must not count.
    _insert_strategy("S7", "rsi_ancient", created_days_ago=200)

    stats = sd.family_outcome_stats(days=90)

    assert stats["rsi"] == {"attempts": 3, "survivors": 1}
    assert stats["funding"] == {"attempts": 2, "survivors": 0}
    assert stats["macd"] == {"attempts": 1, "survivors": 1}


def test_family_outcome_stats_fails_soft_without_db(monkeypatch):
    def _boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(sd, "get_db", _boom)
    assert sd.family_outcome_stats() == {}


# ── survivor-weighted diversity guard ────────────────────────────────────────


def test_guard_flags_dead_regions_and_survivors(monkeypatch):
    monkeypatch.setattr(sd, "saturated_strategy_families", lambda **k: [])
    monkeypatch.setattr(
        sd,
        "family_outcome_stats",
        lambda days=90: {
            "bollinger": {"attempts": 12, "survivors": 0},  # dead region
            "funding": {"attempts": 5, "survivors": 2},     # live region
            "other": {"attempts": 30, "survivors": 0},      # never flagged dead
            "vwap": {"attempts": 3, "survivors": 0},        # below attempt floor
        },
    )

    guard = sd.render_strategy_diversity_guard()

    assert "Proven-dead regions" in guard
    assert "Bollinger / band mean reversion: 12 candidates, 0 survivors" in guard
    assert "funding/carry (2/5 reached paper)" in guard
    assert "30 candidates" not in guard  # 'other' bucket excluded from dead list
    assert "VWAP" not in guard           # under DEAD_FAMILY_MIN_ATTEMPTS


def test_guard_empty_when_no_signal(monkeypatch):
    monkeypatch.setattr(sd, "saturated_strategy_families", lambda **k: [])
    monkeypatch.setattr(sd, "family_outcome_stats", lambda days=90: {})
    assert sd.render_strategy_diversity_guard() == ""


def test_guard_keeps_saturation_lines_alongside_outcomes(monkeypatch):
    monkeypatch.setattr(
        sd,
        "saturated_strategy_families",
        lambda **k: [
            {"family": "rsi", "label": "RSI / oscillator momentum", "count": 40,
             "share": 0.5, "total": 80, "severity": "hard"}
        ],
    )
    monkeypatch.setattr(
        sd,
        "family_outcome_stats",
        lambda days=90: {"rsi": {"attempts": 40, "survivors": 0}},
    )

    guard = sd.render_strategy_diversity_guard()

    assert "Prefer families outside the saturated set" in guard
    assert "Proven-dead regions" in guard


# ── failure taxonomy rendering ───────────────────────────────────────────────


def test_render_failure_taxonomy_from_gate_rejections(forven_db):
    for _ in range(3):
        _insert_rejection("gauntlet", "wfa_degradation", "momentum", "trending")
    _insert_rejection("paper_promotion", "min_trades", "mean_reversion", "ranging")

    block = sd.render_failure_taxonomy(days=30)

    assert block.startswith("# FAILURE TAXONOMY")
    assert "momentum @ gauntlet: wfa_degradation ×3 (trending)" in block
    assert "mean_reversion @ paper_promotion: min_trades ×1 (ranging)" in block


def test_render_failure_taxonomy_empty_without_rows(forven_db):
    assert sd.render_failure_taxonomy() == ""


def _insert_rejection_with_sid(sid: str, gate: str, reason_code: str, strategy_type: str, regime: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO gate_rejections (strategy_id, gate, reason_code, reason_text, strategy_type, regime_context) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, gate, reason_code, f"{reason_code} details", strategy_type, regime),
        )


def test_render_failure_taxonomy_includes_example_strategy_ids(forven_db):
    for i in range(4):
        _insert_rejection_with_sid(f"S{i:04d}", "gauntlet", "wfa_degradation", "momentum", "trending")

    block = sd.render_failure_taxonomy(days=30)

    assert "momentum @ gauntlet: wfa_degradation ×4 (trending)" in block
    match = re.search(r"\[e\.g\. ([^\]]+)\]", block)
    assert match, block
    examples = [x.strip() for x in match.group(1).split(",")]
    # Capped at 3 examples even though 4 rejections exist.
    assert len(examples) == 3
    assert set(examples) <= {"S0000", "S0001", "S0002", "S0003"}


# ── cited_skills persistence + outcome-closure integration ───────────────────


def test_persist_task_strategy_link_backfills_and_merges(forven_db):
    from forven.agents.context import _current_task_display_id_var
    from forven.agents.tools_backtesting import _persist_task_strategy_link

    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO agent_tasks (agent_id, type, display_id, status, output_data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "strategy-developer",
                "develop_candidate",
                "T0042",
                "running",
                json.dumps({"cited_skills": ["existing-skill"]}),
            ),
        )
        task_pk = int(cur.lastrowid)
    _insert_strategy("S100", "rsi_citation_test")

    token = _current_task_display_id_var.set("T0042")
    try:
        _persist_task_strategy_link("S100", ["regime-trend-rsi", "regime-trend-rsi", "  "])
        # A second registration in the same task must not steal the link.
        _persist_task_strategy_link("S999", [])
    finally:
        _current_task_display_id_var.reset(token)

    with get_db() as conn:
        row = conn.execute(
            "SELECT strategy_id, output_data FROM agent_tasks WHERE id = ?", (task_pk,)
        ).fetchone()
    assert row["strategy_id"] == "S100"  # COALESCE keeps the first link
    payload = json.loads(row["output_data"])
    assert payload["cited_skills"] == ["existing-skill", "regime-trend-rsi"]

    # The whole point: outcome closure can now find the citations.
    from forven import skill_outcomes as so

    cited = [name for name, _task in so._find_skills_for_strategy("S100")]
    assert "regime-trend-rsi" in cited


def test_persist_task_strategy_link_noop_without_task_context(forven_db):
    from forven.agents.context import _current_task_display_id_var
    from forven.agents.tools_backtesting import _persist_task_strategy_link

    token = _current_task_display_id_var.set("")
    try:
        _persist_task_strategy_link("S100", ["some-skill"])  # must not raise
    finally:
        _current_task_display_id_var.reset(token)


# ── context injection ────────────────────────────────────────────────────────


def test_agent_context_includes_failure_taxonomy(monkeypatch):
    import forven.context as ctx

    monkeypatch.setattr(ctx, "read_workspace", lambda *a, **k: None)
    monkeypatch.setattr(ctx, "_get_recent_task_context", lambda agent_id: "")
    monkeypatch.setattr(ctx, "render_strategy_diversity_guard", lambda **k: "")
    monkeypatch.setattr(ctx, "get_learned_skills_context", lambda: "")
    monkeypatch.setattr(
        ctx,
        "render_failure_taxonomy",
        lambda **k: "# FAILURE TAXONOMY (last 30d)\n- momentum @ gauntlet: wfa ×3",
    )

    out = ctx.build_agent_context("strategy-developer", "You create strategies.")

    assert "# FAILURE TAXONOMY" in out


def test_learned_skills_block_instructs_citation(monkeypatch):
    import forven.context as ctx

    monkeypatch.setattr(
        "forven.quant_skills.get_ideation_context",
        lambda regime=None, limit=5: "## Learned Knowledge (1 total insights)\n- alpha",
    )

    block = ctx.get_learned_skills_context()

    assert "# LEARNED KNOWLEDGE" in block
    assert "cited_skills" in block


def test_research_context_includes_taxonomy_and_learned_skills(monkeypatch):
    import forven.context as ctx
    import forven.research_context as rc

    monkeypatch.setattr(rc, "render_strategy_diversity_guard", lambda **k: "")
    monkeypatch.setattr(
        rc,
        "render_failure_taxonomy",
        lambda **k: "# FAILURE TAXONOMY (last 30d)\n- momentum @ gauntlet: wfa ×3",
    )
    monkeypatch.setattr(
        ctx,
        "get_learned_skills_context",
        lambda: "# LEARNED KNOWLEDGE (from past outcomes)\n- alpha works",
    )

    contract = rc.coerce_research_contract({"lane": "exploration"})
    out = rc.build_research_context(
        agent_id="quant-researcher",
        role_md="You research.",
        task_description="find an edge",
        contract=contract,
    )

    assert "# FAILURE TAXONOMY" in out
    assert "# LEARNED KNOWLEDGE" in out


# ── near-miss digest ─────────────────────────────────────────────────────────


def _insert_rejection_with_metrics(
    sid: str, gate: str, reason_code: str, strategy_type: str, metrics_snapshot: dict
):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO gate_rejections (strategy_id, gate, reason_code, reason_text, "
            "strategy_type, metrics_snapshot) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, gate, reason_code, f"{reason_code} details", strategy_type, json.dumps(metrics_snapshot)),
        )


def test_query_near_miss_rejections_flat_and_nested_shapes(forven_db):
    from forven.db import query_near_miss_rejections

    # Nested OOS leg clears the bar.
    _insert_rejection_with_metrics(
        "S0201", "quick_screen", "overfit_reject", "keltner",
        {"out_of_sample": {"sharpe": 2.8, "profit_factor": 2.96, "total_trades": 19, "max_drawdown_pct": 0.02}},
    )
    # Flat shape clears the bar too.
    _insert_rejection_with_metrics(
        "S0202", "quick_screen", "overfit_reject", "rsi",
        {"sharpe": 1.5, "profit_factor": 1.4, "total_trades": 20, "max_drawdown_pct": 0.1},
    )
    # Genuinely weak — must be excluded.
    _insert_rejection_with_metrics(
        "S0203", "quick_screen", "overfit_reject", "keltner",
        {"sharpe": -1.6, "profit_factor": 0.45, "total_trades": 15, "max_drawdown_pct": 0.07},
    )

    rows = query_near_miss_rejections(days=90, limit=6)
    ids = {row["strategy_id"] for row in rows}
    assert "S0201" in ids
    assert "S0202" in ids
    assert "S0203" not in ids


def test_query_near_miss_rejections_survives_malformed_json_row(forven_db):
    """One malformed metrics_snapshot row must only lose ITSELF, never zero the
    whole digest: json_extract RAISES on invalid JSON (it does not return NULL),
    which would abort .fetchall() and hit the blanket except -> []. The
    json_valid CASE guard in the query keeps extraction off invalid rows."""
    from forven.db import query_near_miss_rejections

    # A genuine near-miss...
    _insert_rejection_with_metrics(
        "S0210", "quick_screen", "overfit_reject", "keltner",
        {"sharpe": 2.1, "profit_factor": 2.0, "total_trades": 25, "max_drawdown_pct": 0.05},
    )
    # ...next to a row whose snapshot is NOT valid JSON (bypass json.dumps).
    with get_db() as conn:
        conn.execute(
            "INSERT INTO gate_rejections (strategy_id, gate, reason_code, reason_text, "
            "strategy_type, metrics_snapshot) VALUES (?, ?, ?, ?, ?, ?)",
            ("S0211", "quick_screen", "overfit_reject", "corrupt row", "rsi", "not valid json{{{"),
        )

    rows = query_near_miss_rejections(days=90, limit=6)
    ids = {row["strategy_id"] for row in rows}
    assert "S0210" in ids  # the good row survives
    assert "S0211" not in ids  # the bad row just fails the filter


def test_query_near_miss_rejections_fails_soft_without_db(monkeypatch):
    from forven.db import query_near_miss_rejections
    import forven.db as db_mod

    def _boom(*a, **k):
        raise RuntimeError("no db")

    monkeypatch.setattr(db_mod, "get_db", _boom)
    assert query_near_miss_rejections() == []


def test_render_near_miss_digest_includes_strategy_and_reason(forven_db):
    _insert_rejection_with_metrics(
        "S0201", "quick_screen", "overfit_reject", "keltner",
        {"out_of_sample": {"sharpe": 2.8, "profit_factor": 2.96, "total_trades": 19, "max_drawdown_pct": 0.02}},
    )

    block = sd.render_near_miss_digest(days=90)

    assert block.startswith("# NEAR-MISS DIGEST")
    assert "S0201" in block
    assert "overfit_reject" in block


def test_render_near_miss_digest_excludes_weak_rejections(forven_db):
    _insert_rejection_with_metrics(
        "S0203", "quick_screen", "overfit_reject", "keltner",
        {"sharpe": -1.6, "profit_factor": 0.45, "total_trades": 15, "max_drawdown_pct": 0.07},
    )

    assert sd.render_near_miss_digest(days=90) == ""


def test_render_near_miss_digest_empty_on_db_error(monkeypatch):
    def _boom(**k):
        raise RuntimeError("db down")

    monkeypatch.setattr("forven.db.query_near_miss_rejections", _boom)
    assert sd.render_near_miss_digest() == ""


def test_agent_context_includes_near_miss_digest(monkeypatch):
    import forven.context as ctx

    monkeypatch.setattr(ctx, "read_workspace", lambda *a, **k: None)
    monkeypatch.setattr(ctx, "_get_recent_task_context", lambda agent_id: "")
    monkeypatch.setattr(ctx, "render_strategy_diversity_guard", lambda **k: "")
    monkeypatch.setattr(ctx, "get_learned_skills_context", lambda: "")
    monkeypatch.setattr(ctx, "render_failure_taxonomy", lambda **k: "")
    monkeypatch.setattr(
        ctx,
        "render_near_miss_digest",
        lambda **k: "# NEAR-MISS DIGEST (last 90d)\n- S0201 (keltner): died on quick_screen/overfit_reject",
    )

    out = ctx.build_agent_context("strategy-developer", "You create strategies.")

    assert "# NEAR-MISS DIGEST" in out


def test_research_context_includes_near_miss_digest(monkeypatch):
    import forven.context as ctx
    import forven.research_context as rc

    monkeypatch.setattr(rc, "render_strategy_diversity_guard", lambda **k: "")
    monkeypatch.setattr(rc, "render_failure_taxonomy", lambda **k: "")
    monkeypatch.setattr(
        rc,
        "render_near_miss_digest",
        lambda **k: "# NEAR-MISS DIGEST (last 90d)\n- S0201 (keltner): died on quick_screen/overfit_reject",
    )
    monkeypatch.setattr(ctx, "get_learned_skills_context", lambda: "")

    contract = rc.coerce_research_contract({"lane": "exploration"})
    out = rc.build_research_context(
        agent_id="quant-researcher",
        role_md="You research.",
        task_description="find an edge",
        contract=contract,
    )

    assert "# NEAR-MISS DIGEST" in out


def test_research_context_skills_respect_inspiration_off(monkeypatch):
    import forven.context as ctx
    import forven.research_context as rc

    monkeypatch.setattr(rc, "render_strategy_diversity_guard", lambda **k: "")
    monkeypatch.setattr(
        rc,
        "render_failure_taxonomy",
        lambda **k: "# FAILURE TAXONOMY (last 30d)\n- x",
    )
    monkeypatch.setattr(
        ctx,
        "get_learned_skills_context",
        lambda: "# LEARNED KNOWLEDGE (from past outcomes)\n- alpha works",
    )

    contract = rc.coerce_research_contract(
        {"lane": "exploration", "memory_mode": {"inspiration_memory": "off"}}
    )
    out = rc.build_research_context(
        agent_id="quant-researcher",
        role_md="You research.",
        task_description="find an edge",
        contract=contract,
    )

    # Taxonomy is constraint-class (always on); skills are inspiration-class (gated).
    assert "# FAILURE TAXONOMY" in out
    assert "# LEARNED KNOWLEDGE" not in out
