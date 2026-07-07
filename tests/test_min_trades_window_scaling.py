"""Window-aware min-trades floor (2026-07-06 audit, ~/.forven/forven.db evidence).

The quick_screen/gauntlet min_trades gates enforced an ABSOLUTE trade-count floor
(default 20) regardless of the backtest window (app setting
`backtest_duration_days`, default 730d). A selective low-frequency edge (e.g.
1-2 trades/week) mathematically cannot reach 20 trades on a short window --
the floor silently converted "short window" into "reject good strategies".
Verified casualties with GOOD metrics killed purely by this absolute floor:
S00014/S00221 (Sharpe 1.45, PF 2.01, DD 2.4%, 14 trades -> rejected), S00046
(Sharpe 1.85, PF 2.28, 18 trades -> rejected).

Fix: the floor now SCALES with the actual backtest window:

    effective_min_trades = max(hard_statistical_floor, ceil(rate_per_30d * window_days / 30))

`hard_statistical_floor` (new knob, default 10) is the absolute floor below
which Sharpe/PF are NEVER trusted, regardless of window. `rate_per_30d` is
either the explicit `min_trades_per_30d` knob (0 = unset) or DERIVED from the
gate's `min_trades` value, anchored at the constant default window
(DEFAULT_BACKTEST_DURATION_DAYS = 730d) -- so an existing `min_trades=20`
(or a preset/custom override) keeps its current meaning ("N trades at the
default window") and scales proportionally at other windows. At the default
window this is mathematically forced back to exactly the configured
`min_trades`, so existing behavior is unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import forven.policy as policy
from forven.db import get_db, kv_set


GOOD_QS_METRICS_TEMPLATE = {
    "sharpe": 1.45,
    "profit_factor": 2.01,
    "max_drawdown_pct": 0.024,
    "total_return_pct": 12.0,
    "win_rate": 42.0,
}


def _insert_strategy(strategy_id: str, *, stage: str = "quick_screen", metrics: dict | None = None) -> None:
    now = datetime.now(timezone.utc)
    stage_changed = (now - timedelta(hours=2)).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO strategies
                (id, name, type, symbol, timeframe, params, metrics, status, owner,
                 stage, stage_changed_at, created_at, updated_at)
            VALUES (?, ?, 'rsi_momentum', 'ETH/USDT', '1h', '{}', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                strategy_id,
                json.dumps(metrics or {}),
                "quick_screen" if stage == "quick_screen" else "active",
                "brain",
                stage,
                stage_changed,
                stage_changed,
                now.isoformat(),
            ),
        )


def _qs_metrics(total_trades: int) -> dict:
    payload = dict(GOOD_QS_METRICS_TEMPLATE)
    payload["total_trades"] = total_trades
    return payload


def _gauntlet_metrics(total_trades: int) -> dict:
    return {
        "total_trades": total_trades,
        "sharpe": 1.6,
        "profit_factor": 1.8,
        "max_drawdown_pct": 0.10,
        "total_return_pct": 15.0,
        "win_rate": 48.0,
        "composite_robustness_score": 60,
        "robustness_score": 60,
    }


def _insert_backtest_result(
    strategy_id: str,
    result_type: str = "backtest",
    result_id: str | None = None,
    metrics: dict | None = None,
    config: dict | None = None,
) -> None:
    """Insert a minimal backtest_results row for gate satisfaction (mirrors the
    helper in test_pipeline_hardening.py)."""
    rid = result_id or f"auto-{result_type}-{strategy_id}-{int(datetime.now(timezone.utc).timestamp() * 1e6)}"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO backtest_results
               (result_id, strategy_id, result_type, symbol, timeframe, metrics_json, config_json, created_at)
               VALUES (?, ?, ?, 'BTC', '1h', ?, ?, datetime('now'))""",
            (rid, strategy_id, result_type, json.dumps(metrics or {}), json.dumps(config or {})),
        )


def _validation_metrics(result_type: str, *, passing: bool = True) -> dict:
    verdict = "PASS" if passing else "FAIL"
    if result_type == "walk_forward":
        return {
            "verdict": verdict,
            "splits": [
                {"out_of_sample": {"sharpe": 1.2 if passing else -0.2}},
                {"out_of_sample": {"sharpe": 1.0 if passing else -0.1}},
                {"out_of_sample": {"sharpe": 0.8 if passing else -0.3}},
            ],
        }
    if result_type == "monte_carlo":
        return {
            "verdict": verdict,
            "n_simulations": 1000,
            "n_trades": 30,
            "drawdown_distribution": {"p95": 15.0 if passing else 40.0},
            "max_dd_p95_ratio": 0.15 if passing else 0.40,
        }
    if result_type == "param_jitter":
        return {"verdict": verdict, "n_iterations": 50, "pct_positive_sharpe": 82.0 if passing else 30.0}
    if result_type == "cost_stress":
        return {"verdict": verdict, "degradation_pct": 18.0 if passing else 70.0}
    if result_type == "regime_split":
        return {"verdict": verdict, "n_regimes": 3 if passing else 1}
    return {"verdict": verdict}


def _insert_validation_result(strategy_id: str, result_type: str, *, passing: bool = True) -> None:
    _insert_backtest_result(
        strategy_id,
        result_type=result_type,
        metrics=_validation_metrics(result_type, passing=passing),
        config={"status": "succeeded"},
    )


def _insert_required_validation_results(strategy_id: str, *, failing: str | None = None) -> None:
    for result_type in ("walk_forward", "monte_carlo", "param_jitter", "cost_stress", "regime_split"):
        _insert_validation_result(strategy_id, result_type, passing=result_type != failing)
    _insert_backtest_result(strategy_id, result_type="optimization")


# --- (a) default window: effective floor stays exactly 20 (no behavior shift) -----


def test_effective_min_trades_floor_helper_unchanged_at_default_window():
    defaults = policy.DEFAULT_PIPELINE_CONFIG["quick_screen"]
    effective, rate, hard_floor = policy._effective_min_trades_floor(defaults, defaults, window_days=730)
    assert effective == 20
    assert hard_floor == 10


def test_quick_screen_gate_still_rejects_14_trades_at_default_730d_window(forven_db):
    strategy_id = "s-default-window-still-rejects"
    _insert_strategy(strategy_id, metrics=_qs_metrics(14))

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert passed is False
    assert "14 trades" in reason
    assert "20 effective minimum" in reason or "20 minimum" in reason


def test_quick_screen_gate_passes_20_trades_at_default_window(forven_db):
    strategy_id = "s-default-window-boundary-pass"
    _insert_strategy(strategy_id, metrics=_qs_metrics(20))

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert passed is True, reason


# --- (b) short window: a good 14-trade strategy now PASSES ------------------------


def test_quick_screen_gate_passes_14_trades_at_30d_window(forven_db):
    kv_set("forven:settings", {"quick_screen_duration_days": 30})
    strategy_id = "s-short-window-passes"
    _insert_strategy(strategy_id, metrics=_qs_metrics(14))

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert passed is True, reason


# --- (c) hard statistical floor (10) still rejects below it at ANY window ---------


def test_quick_screen_gate_rejects_below_hard_floor_even_at_short_window(forven_db):
    kv_set("forven:settings", {"quick_screen_duration_days": 7})
    strategy_id = "s-hard-floor-rejects"
    _insert_strategy(strategy_id, metrics=_qs_metrics(9))

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert passed is False
    assert "statistical floor 10" in reason


def test_quick_screen_gate_passes_at_hard_floor_boundary_short_window(forven_db):
    kv_set("forven:settings", {"quick_screen_duration_days": 7})
    strategy_id = "s-hard-floor-boundary-pass"
    _insert_strategy(strategy_id, metrics=_qs_metrics(10))

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, policy.load_pipeline_config())

    assert passed is True, reason


# --- (d) reason text carries window context and still classifies as gate_reject ---


def test_quick_screen_min_trades_reason_mentions_window_and_classifies_gate_reject(forven_db):
    kv_set("forven:settings", {"quick_screen_duration_days": 56})
    strategy_id = "s-reason-text-window"
    _insert_strategy(strategy_id, metrics=_qs_metrics(12))

    # Explicit rate knob: 8/30d over 56d -> ceil(8 * 56/30) = 15 effective floor,
    # so 12 trades rejects. (The DERIVED default rate — 20 anchored at 730d —
    # collapses to the hard floor of 10 at 56d, which 12 trades would PASS;
    # the explicit knob is what makes this rejection scenario reachable.)
    config = policy.load_pipeline_config()
    config["quick_screen"]["min_trades_per_30d"] = 8

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, config)

    assert passed is False
    assert "12 trades" in reason
    assert "56d" in reason
    assert "window-scaled" in reason
    assert "statistical floor" in reason
    # Keeps "trades"/"minimum" so the existing taxonomy classification is stable
    # (falls through to the generic gate_reject bucket, same as before this fix).
    assert policy._extract_reason_code(reason) == "gate_reject"


# --- (e) preset scaling behaves sanely (relaxed=5, strict=30) ---------------------


def test_relaxed_preset_min_trades_scaling_sane(forven_db):
    kv_set("forven:pipeline_thresholds", {"pipeline_preset": "relaxed"})
    config = policy.load_pipeline_config()
    gate = config["quick_screen"]
    defaults = policy.DEFAULT_PIPELINE_CONFIG["quick_screen"]
    assert gate["min_trades"] == 5

    # At the (constant) default window the relaxed floor is exactly its configured 5.
    effective_default, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=730)
    assert effective_default == 5

    # The relaxed preset relaxes the BEDROCK too (hard_min_trades_floor: 5) —
    # otherwise the default bedrock (10) would make "relaxed" silently STRICTER
    # than its own configured min_trades at every window. At a very short window
    # the derived rate collapses toward 0 and the preset's own hard floor (5)
    # takes over — relaxed can never go below 5, at any window.
    effective_short, _, hard_floor = policy._effective_min_trades_floor(gate, defaults, window_days=7)
    assert hard_floor == 5
    assert effective_short == 5


def test_strict_preset_min_trades_scaling_sane(forven_db):
    kv_set("forven:pipeline_thresholds", {"pipeline_preset": "strict"})
    config = policy.load_pipeline_config()
    gate = config["quick_screen"]
    defaults = policy.DEFAULT_PIPELINE_CONFIG["quick_screen"]
    assert gate["min_trades"] == 30

    effective_default, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=730)
    assert effective_default == 30

    # Double the window -> roughly double the effective floor (linear scaling).
    effective_double, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=1460)
    assert effective_double == 60


def test_explicit_low_min_trades_override_is_not_raised_by_bedrock():
    """An operator's explicit `min_trades: 1` (previously floored only by the
    safety_floors rail at the gate call sites) must NOT be silently raised to
    the default bedrock (10) by a knob they never touched — the bedrock guards
    against window-scaling collapse, not against deliberate operator choice.
    Regression: test_gauntlet_paper_bypass_fix::test_trade_count_floor_allows_modest_sample."""
    defaults = policy.DEFAULT_PIPELINE_CONFIG["gauntlet"]
    gate = dict(defaults)
    gate["min_trades"] = 1

    effective, _, hard_floor = policy._effective_min_trades_floor(gate, defaults, window_days=730)
    assert hard_floor == 10  # the knob itself is untouched...
    assert effective == 1    # ...but it cannot bind above the explicit min_trades


def test_custom_min_trades_override_scales_proportionally():
    """A plain custom `min_trades` override (no explicit min_trades_per_30d) is
    honored as the value AT THE DEFAULT WINDOW and scales proportionally --
    it is not silently ignored in favor of a decoupled rate knob."""
    defaults = policy.DEFAULT_PIPELINE_CONFIG["quick_screen"]
    gate = dict(defaults)
    gate["min_trades"] = 45

    effective_default, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=730)
    assert effective_default == 45

    effective_half, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=365)
    assert effective_half in (22, 23)  # ~half of 45, ceil-rounded


# --- gauntlet gate: same window-scaling applied to its min_trades floor -----------


def test_gauntlet_min_trades_floor_helper_unchanged_at_default_window():
    defaults = policy.DEFAULT_PIPELINE_CONFIG["gauntlet"]
    effective, _, hard_floor = policy._effective_min_trades_floor(defaults, defaults, window_days=730)
    assert effective == 20
    assert hard_floor == 10


def test_gauntlet_gate_rejects_short_of_scaled_floor_at_short_window(forven_db):
    kv_set("forven:settings", {"confirmation_duration_days": 30})
    strategy_id = "s-gauntlet-short-window-reject"
    _insert_strategy(strategy_id, stage="gauntlet", metrics=_gauntlet_metrics(3))
    _insert_required_validation_results(strategy_id)

    passed, reason = policy._evaluate_gauntlet_gate(strategy_id, policy.load_pipeline_config())

    assert passed is False
    assert "trades" in reason.lower()
    assert "minimum" in reason.lower()


def test_min_trades_zero_disables_floor_when_no_rate_set():
    """An explicit `min_trades: 0` (with no min_trades_per_30d) must DISABLE the
    floor entirely — pre-existing semantics that quick_screen's `if
    min_trades_floor > 0` guard and the gauntlet's `max(0, safety_floors)`
    clamp both rely on. Regression: a min_trades=0 operator override must not
    be silently re-floored to hard_min_trades_floor (10)."""
    defaults = policy.DEFAULT_PIPELINE_CONFIG["quick_screen"]
    gate = dict(defaults)
    gate["min_trades"] = 0

    effective_default, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=730)
    assert effective_default == 0

    effective_short, _, _ = policy._effective_min_trades_floor(gate, defaults, window_days=7)
    assert effective_short == 0


def test_min_trades_zero_with_explicit_rate_uses_rate_only():
    """With min_trades disabled (0) but an explicit min_trades_per_30d set, the
    rate is the SOLE driver — the disabled absolute floor must not clamp it."""
    defaults = policy.DEFAULT_PIPELINE_CONFIG["quick_screen"]
    gate = dict(defaults)
    gate["min_trades"] = 0
    gate["min_trades_per_30d"] = 8

    effective, rate, _ = policy._effective_min_trades_floor(gate, defaults, window_days=56)
    assert rate == 8
    assert effective == 15  # ceil(8 * 56/30) = ceil(14.93) = 15


def test_quick_screen_gate_min_trades_zero_disables_floor_end_to_end(forven_db):
    """End-to-end: quick_screen.min_trades=0 must let a thin (5-trade) sample
    pass the trade-count check — it may still fail LATER, unrelated gates,
    but must never be rejected for trade count."""
    strategy_id = "s-min-trades-zero-e2e"
    _insert_strategy(strategy_id, metrics=_qs_metrics(5))
    config = policy.load_pipeline_config()
    config["quick_screen"]["min_trades"] = 0

    passed, reason = policy._evaluate_quick_screen_gate(strategy_id, config)

    if not passed:
        assert "trades" not in reason.lower(), reason


def test_gauntlet_gate_passes_reduced_trades_at_short_window(forven_db):
    kv_set("forven:settings", {"confirmation_duration_days": 30})
    strategy_id = "s-gauntlet-short-window-pass"
    # Default gauntlet.min_trades=20 anchored at 730d -> rate ~0.822/30d -> at a
    # 30d window the scaled floor collapses to the hard statistical floor (10).
    _insert_strategy(strategy_id, stage="gauntlet", metrics=_gauntlet_metrics(11))
    _insert_required_validation_results(strategy_id)

    passed, reason = policy._evaluate_gauntlet_gate(strategy_id, policy.load_pipeline_config())

    # This test's contract is the TRADE floor only: 11 trades must clear the
    # window-scaled floor (10) that the old absolute floor (20) rejected. The
    # gate may still fail on LATER, unrelated evidence gates this fixture does
    # not seed (e.g. the multi-timeframe sweep) — that is out of scope here, so
    # assert specifically that the trade floor was not the rejection.
    if not passed:
        assert "trades" not in reason.lower(), reason
