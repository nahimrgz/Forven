"""Phase 5: lineage helpers + parent_strategy_id validation."""


import pytest

from forven.db import create_strategy_container, get_db
from forven.hypothesis_lineage import (
    build_canonical_coverage_map,
    build_sibling_table,
)
from forven.hypotheses import create_hypothesis


def _hyp(idx: int = 0) -> dict:
    return create_hypothesis(
        title=f"H{idx}", market_thesis="m", mechanism="x", why_now=None,
        lane="benchmarking", source_type="agent_original",
        origin_agent_id="a", origin_role="strategy-developer",
        target_assets=["BTC"], target_timeframes=["1h"],
    )


def _make_strategy(
    hypothesis_id: str,
    *,
    sid_seed: int,
    symbol: str = "BTC",
    timeframe: str = "1h",
    stage: str = "quick_screen",
    canonical: bool = False,
    parent_strategy_id: str | None = None,
) -> str:
    with get_db() as conn:
        sid, _, _ = create_strategy_container(
            conn,
            name=f"strat-{sid_seed}",
            type_="rsi",
            symbol=symbol,
            timeframe=timeframe,
            params={"regime_filter": "trending"},
            stage=stage,
            hypothesis_id=hypothesis_id,
            strategy_id=f"S{40000 + sid_seed:05d}",
            parent_strategy_id=parent_strategy_id,
        )
        if canonical:
            conn.execute(
                "UPDATE strategies SET canonical = 1 WHERE id = ?",
                (sid,),
            )
        conn.commit()
    return sid


# ---- sibling table ----


def test_sibling_table_returns_active_children(forven_db):
    h = _hyp()
    s1 = _make_strategy(h["id"], sid_seed=1)
    s2 = _make_strategy(h["id"], sid_seed=2, symbol="ETH")
    s3 = _make_strategy(h["id"], sid_seed=3, symbol="SOL", parent_strategy_id=s1)

    table = build_sibling_table(h["id"])
    ids = [row["strategy_id"] for row in table]
    assert s1 in ids and s2 in ids and s3 in ids
    by_id = {row["strategy_id"]: row for row in table}
    assert by_id[s3]["parent_strategy_id"] == s1
    # Bare base assets ("BTC") are repaired to canonical pair form
    # ("BTC/USDT") by ``_normalize_strategy_symbol`` — see
    # ``test_strategy_symbol_normalization``.
    assert by_id[s1]["asset"] == "BTC/USDT"
    assert by_id[s2]["asset"] == "ETH/USDT"
    assert by_id[s1]["regime_filter"] == "trending"


def test_sibling_table_excludes_archived_and_rejected(forven_db):
    h = _hyp()
    keep = _make_strategy(h["id"], sid_seed=1)
    _make_strategy(h["id"], sid_seed=2, stage="archived")
    _make_strategy(h["id"], sid_seed=3, stage="rejected")

    table = build_sibling_table(h["id"])
    assert len(table) == 1
    assert table[0]["strategy_id"] == keep


def test_sibling_table_empty_when_no_children(forven_db):
    h = _hyp()
    assert build_sibling_table(h["id"]) == []


def _insert_gate_rejection(
    strategy_id: str,
    *,
    gate: str = "gauntlet",
    reason_code: str = "wfa_degradation",
    reason_text: str = "reason detail",
    created_at: str | None = None,
) -> None:
    with get_db() as conn:
        if created_at:
            conn.execute(
                "INSERT INTO gate_rejections "
                "(strategy_id, gate, reason_code, reason_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (strategy_id, gate, reason_code, reason_text, created_at),
            )
        else:
            conn.execute(
                "INSERT INTO gate_rejections "
                "(strategy_id, gate, reason_code, reason_text) "
                "VALUES (?, ?, ?, ?)",
                (strategy_id, gate, reason_code, reason_text),
            )
        conn.commit()


def test_sibling_table_carries_latest_rejection_reason(forven_db):
    h = _hyp()
    s1 = _make_strategy(h["id"], sid_seed=1)
    _insert_gate_rejection(
        s1, gate="quick_screen", reason_code="low_sharpe",
        reason_text="older rejection", created_at="2026-01-01T00:00:00+00:00",
    )
    _insert_gate_rejection(
        s1, gate="gauntlet", reason_code="wfa_degradation",
        reason_text="x" * 300, created_at="2026-06-01T00:00:00+00:00",
    )

    table = build_sibling_table(h["id"])
    by_id = {row["strategy_id"]: row for row in table}
    rejection = by_id[s1]["last_rejection"]

    assert rejection is not None
    # Most recent rejection wins, not the first one inserted.
    assert rejection["gate"] == "gauntlet"
    assert rejection["reason_code"] == "wfa_degradation"
    assert rejection["reason"] == ("x" * 300)[:200]


def test_sibling_table_no_rejection_is_none(forven_db):
    h = _hyp()
    s1 = _make_strategy(h["id"], sid_seed=1)

    table = build_sibling_table(h["id"])
    by_id = {row["strategy_id"]: row for row in table}

    assert by_id[s1].get("last_rejection") is None


def test_sibling_table_rejection_window_scoped_to_own_hypothesis(forven_db):
    """Rejections belonging to a strategy from a DIFFERENT hypothesis must
    never leak into this hypothesis's sibling table. Guards the windowed
    subquery scoping (performance fix) didn't change join semantics."""
    h1 = _hyp(0)
    h2 = _hyp(1)
    s1 = _make_strategy(h1["id"], sid_seed=1)
    s2 = _make_strategy(h2["id"], sid_seed=2)
    _insert_gate_rejection(
        s2, gate="gauntlet", reason_code="other_hypothesis_rejection",
        reason_text="belongs to a different hypothesis",
    )

    table = build_sibling_table(h1["id"])
    by_id = {row["strategy_id"]: row for row in table}

    assert by_id[s1].get("last_rejection") is None


def test_sibling_table_survives_missing_gate_rejections_table(forven_db):
    h = _hyp()
    s1 = _make_strategy(h["id"], sid_seed=1)
    with get_db() as conn:
        conn.execute("DROP TABLE gate_rejections")
        conn.commit()

    table = build_sibling_table(h["id"])
    ids = [row["strategy_id"] for row in table]

    assert s1 in ids
    by_id = {row["strategy_id"]: row for row in table}
    assert by_id[s1].get("last_rejection") is None


# ---- canonical coverage map ----


def test_canonical_coverage_map_only_counts_canonicals(forven_db):
    h = _hyp()
    _make_strategy(h["id"], sid_seed=1, symbol="BTC")  # not canonical
    s_canon = _make_strategy(h["id"], sid_seed=2, symbol="ETH", canonical=True)
    _make_strategy(h["id"], sid_seed=3, symbol="SOL")  # not canonical

    coverage = build_canonical_coverage_map(h["id"])
    # Bare base assets ("ETH") are repaired to canonical pair form
    # ("ETH/USDT"); coverage map keys it as f"{symbol}:{timeframe}".
    assert "ETH/USDT:1h" in coverage
    assert "BTC/USDT:1h" not in coverage
    assert "SOL/USDT:1h" not in coverage
    assert coverage["ETH/USDT:1h"]["strategy_id"] == s_canon


def test_canonical_coverage_map_empty_for_no_canonicals(forven_db):
    h = _hyp()
    _make_strategy(h["id"], sid_seed=1)
    _make_strategy(h["id"], sid_seed=2, symbol="ETH")
    assert build_canonical_coverage_map(h["id"]) == {}


# ---- create_strategy_container parent validation ----


def test_create_strategy_rejects_parent_from_different_hypothesis(forven_db):
    h_a = _hyp(0)
    h_b = _hyp(1)
    parent = _make_strategy(h_a["id"], sid_seed=1)
    with pytest.raises(ValueError, match="cannot cross hypotheses"):
        with get_db() as conn:
            create_strategy_container(
                conn,
                name="child",
                type_="rsi",
                symbol="BTC",
                timeframe="1h",
                params={},
                stage="quick_screen",
                hypothesis_id=h_b["id"],  # different hypothesis
                parent_strategy_id=parent,
            )


def test_create_strategy_rejects_unknown_parent(forven_db):
    h = _hyp()
    with pytest.raises(ValueError, match="not found"):
        with get_db() as conn:
            create_strategy_container(
                conn,
                name="child",
                type_="rsi",
                symbol="BTC",
                timeframe="1h",
                params={},
                stage="quick_screen",
                hypothesis_id=h["id"],
                parent_strategy_id="S99999_BOGUS",
            )


def test_create_strategy_accepts_parent_in_same_hypothesis(forven_db):
    h = _hyp()
    parent = _make_strategy(h["id"], sid_seed=1)
    child = _make_strategy(h["id"], sid_seed=2, parent_strategy_id=parent)
    table = build_sibling_table(h["id"])
    by_id = {row["strategy_id"]: row for row in table}
    assert by_id[child]["parent_strategy_id"] == parent
