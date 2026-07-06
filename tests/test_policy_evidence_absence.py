"""B-35 (2026-06-09 audit): evidence-absence gate rejections must not auto-archive.

The gauntlet->paper gate emits several rejections that mean "the tests have not
been run or persisted YET" (work queued, optimization in flight, validation
awaiting a post-optimization re-run) — absence of evidence, not evidence of a
bad edge. Evolution polls the gate up to 3x per cycle, so if these fed the
repeated-failure counter an in-flight strategy was terminally archived in ~2-5
cycles before its edge was ever measured. They now get dedicated reason codes
that return early from the auto-archive counter (mirroring no_metrics_error);
genuine ran-and-failed quality rejections keep counting.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import forven.policy as policy
from forven.db import get_db


def _insert_strategy(strategy_id: str, *, stage: str = "gauntlet") -> None:
    now = datetime.now(timezone.utc)
    stage_changed = (now - timedelta(days=1)).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO strategies
                (id, name, type, symbol, timeframe, params, metrics, status, owner,
                 stage, stage_changed_at, created_at, updated_at)
            VALUES (?, ?, 'rsi_momentum', 'ETH', '1h', '{}', ?, ?, 'brain', ?, ?, ?, ?)
            """,
            (
                strategy_id,
                strategy_id,
                json.dumps({"fitness": 50.0}),
                stage,
                stage,
                stage_changed,
                stage_changed,
                now.isoformat(),
            ),
        )


def _insert_rejections(strategy_id: str, reason_code: str, reason_text: str, count: int) -> None:
    with get_db() as conn:
        for _ in range(count):
            conn.execute(
                """
                INSERT INTO gate_rejections
                    (strategy_id, gate, reason_code, reason_text, created_at)
                VALUES (?, 'gauntlet', ?, ?, datetime('now'))
                """,
                (strategy_id, reason_code, reason_text),
            )


def _stage(strategy_id: str) -> str:
    with get_db() as conn:
        row = conn.execute("SELECT stage FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    return str(row["stage"])


# --- reason-code classification ------------------------------------------------


def test_extract_reason_code_separates_evidence_absence_from_quality_failures():
    assert (
        policy._extract_reason_code(
            "Gauntlet requires at least one persisted optimization or walk-forward run before promotion to paper"
        )
        == "artifacts_pending"
    )
    assert (
        policy._extract_reason_code("Stale validation tests (run before latest optimization): walk_forward")
        == "stale_validation"
    )
    assert (
        policy._extract_reason_code(
            "Ordering violation: walk_forward was run before optimization — re-run after optimization"
        )
        == "stale_validation"
    )
    assert (
        policy._extract_reason_code("Gauntlet missing required verdict tests: monte_carlo")
        == "missing_evidence"
    )
    # Genuine quality failures keep their codes and still count toward auto-archive.
    assert policy._extract_reason_code("Walk-forward fold pass rate 0.33 below floor") == "wfa_reject"
    assert policy._extract_reason_code("Gauntlet robustness too low: 12.0/100") == "robustness_reject"


# --- auto-archive counter exemption ---------------------------------------------


def test_artifacts_pending_rejections_never_auto_archive(forven_db):
    strategy_id = "s-artifacts-pending"
    text = "Gauntlet requires at least one persisted optimization or walk-forward run before promotion to paper"
    _insert_strategy(strategy_id)
    _insert_rejections(strategy_id, "artifacts_pending", text, count=8)

    policy._check_repeated_failure_auto_archive(strategy_id, "gauntlet", "artifacts_pending", text)

    assert _stage(strategy_id) == "gauntlet"


def test_stale_validation_rejections_never_auto_archive(forven_db):
    strategy_id = "s-stale-validation"
    text = "Stale validation tests (run before latest optimization): walk_forward, monte_carlo"
    _insert_strategy(strategy_id)
    _insert_rejections(strategy_id, "stale_validation", text, count=8)

    policy._check_repeated_failure_auto_archive(strategy_id, "gauntlet", "stale_validation", text)

    assert _stage(strategy_id) == "gauntlet"


def test_missing_evidence_rejections_never_auto_archive(forven_db):
    strategy_id = "s-missing-evidence"
    text = "Gauntlet missing required verdict tests: monte_carlo"
    _insert_strategy(strategy_id)
    _insert_rejections(strategy_id, "missing_evidence", text, count=8)

    policy._check_repeated_failure_auto_archive(strategy_id, "gauntlet", "missing_evidence", text)

    assert _stage(strategy_id) == "gauntlet"


def test_quality_failure_rejections_still_auto_archive(forven_db):
    strategy_id = "s-quality-failure"
    text = "Walk-forward fold pass rate 0.20 below floor"
    _insert_strategy(strategy_id)
    _insert_rejections(strategy_id, "wfa_reject", text, count=5)

    policy._check_repeated_failure_auto_archive(strategy_id, "gauntlet", "wfa_reject", text)

    assert _stage(strategy_id) == "archived"


# --- L-21 (2026-06-09 audit): insufficient-paper-evidence reason code ------------
# Paper warm-up rejections ("Insufficient paper duration/sample/trades") used to be
# carved out of the dethrone counter by brittle startswith/SQL-NOT-LIKE text
# matching. They now classify to a dedicated evidence-absence reason code.


def test_extract_reason_code_classifies_insufficient_paper_evidence():
    for text in (
        "Insufficient paper duration: 3/14 days",
        "Insufficient paper sample: 12/50 closed trades",
        "Insufficient paper trades: 0/50",
    ):
        assert policy._extract_reason_code(text) == "insufficient_paper_evidence", text
    assert "insufficient_paper_evidence" in policy._EVIDENCE_ABSENCE_REASON_CODES
    # The legacy text-matching carve-outs are gone for good.
    assert not hasattr(policy, "_is_insufficient_paper_evidence_reason")


def test_insufficient_paper_evidence_rejections_never_auto_archive_or_dethrone(forven_db):
    strategy_id = "s-paper-warmup"
    text = "Insufficient paper duration: 2/14 days"
    _insert_strategy(strategy_id, stage="paper")
    with get_db() as conn:
        for _ in range(8):
            conn.execute(
                """
                INSERT INTO gate_rejections
                    (strategy_id, gate, reason_code, reason_text, created_at)
                VALUES (?, 'paper', 'insufficient_paper_evidence', ?, datetime('now'))
                """,
                (strategy_id, text),
            )

    policy._check_repeated_failure_auto_archive(
        strategy_id, "paper", "insufficient_paper_evidence", text
    )

    assert _stage(strategy_id) == "paper"
    with get_db() as conn:
        approvals = conn.execute(
            """
            SELECT COUNT(*) AS c FROM approvals
            WHERE approval_type = 'strategy_dethrone_recommendation' AND target_id = ?
            """,
            (strategy_id,),
        ).fetchone()["c"]
    assert int(approvals) == 0


# --- WFA fold-density shortfalls (S06127 2026-07-06) -----------------------------
# "Walk-forward has N folds, requires minimum M" and "walk-forward window
# insufficient" mean the window was too small for the strategy's trade cadence to
# produce judgeable folds — evidence-insufficiency, not merit. Five background
# polls of one unchanged sparse-fold artifact auto-archived a strategy whose
# retirement snapshot read robustness 100/100 and which PASSED once the window
# was sized to its cadence. The fold PASS-RATE reject stays a counting failure.


def test_extract_reason_code_classifies_wfa_fold_density_as_evidence_absence():
    for text in (
        "S00552 REJECT: Walk-forward has 1 folds, requires minimum 2; re-run WFA on the trade-frequency-aware window",
        "S00552 REJECT: Walk-forward has 1 folds, requires minimum 2",
        "S00552 BLOCK: walk-forward window insufficient — no OOS fold reached 5 trades; re-run WFA on the trade-frequency-aware window",
    ):
        assert policy._extract_reason_code(text) == "wfa_window_insufficient", text
    assert "wfa_window_insufficient" in policy._EVIDENCE_ABSENCE_REASON_CODES
    # The fold pass-rate reject is a genuine ran-and-failed outcome and keeps counting.
    assert (
        policy._extract_reason_code("S00552 REJECT: Walk-forward pass rate 20% below 40% minimum")
        == "s00552_reject"
    )


def test_wfa_fold_density_rejections_never_auto_archive(forven_db):
    strategy_id = "s-wfa-fold-density"
    text = (
        "S00552 REJECT: Walk-forward has 1 folds, requires minimum 2; "
        "re-run WFA on the trade-frequency-aware window"
    )
    _insert_strategy(strategy_id)
    _insert_rejections(strategy_id, "wfa_window_insufficient", text, count=8)

    policy._check_repeated_failure_auto_archive(
        strategy_id, "gauntlet", "wfa_window_insufficient", text
    )

    assert _stage(strategy_id) == "gauntlet"


def test_paper_quality_failures_do_not_auto_queue_dethrone_for_operator_owned(forven_db):
    # Updated contract (paper param/metric lock): paper/live are operator-owned, so
    # background gate re-evaluations must NOT auto-queue a paper->gauntlet dethrone
    # recommendation even on a genuine ran-and-failed quality code. The strategy
    # stays in paper (not auto-archived); legitimate demotion is operator action +
    # decay_tracker paper_live_drift (both untouched).
    strategy_id = "s-paper-quality"
    text = "Paper drawdown too high: 22.00% (maximum 15.00%)"
    _insert_strategy(strategy_id, stage="paper")
    with get_db() as conn:
        for _ in range(5):
            conn.execute(
                """
                INSERT INTO gate_rejections
                    (strategy_id, gate, reason_code, reason_text, created_at)
                VALUES (?, 'paper', 'drawdown_reject', ?, datetime('now'))
                """,
                (strategy_id, text),
            )

    policy._check_repeated_failure_auto_archive(strategy_id, "paper", "drawdown_reject", text)

    assert _stage(strategy_id) == "paper"  # not archived
    with get_db() as conn:
        approvals = conn.execute(
            """
            SELECT COUNT(*) AS c FROM approvals
            WHERE approval_type = 'strategy_dethrone_recommendation' AND target_id = ?
            """,
            (strategy_id,),
        ).fetchone()["c"]
    assert int(approvals) == 0  # suppressed for operator-owned strategy
