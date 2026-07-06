"""RISK-PARITY-1: declared risk controls must be enforced or loudly flagged.

Top-level risk params were silently inert everywhere (the engine only reads the
nested ``execution_profile``), while ``validate_backtest_risk_controls`` also
false-blocked strategies whose control WAS enforced via the profile. Now:

- ``time_stop_bars`` (unit-unambiguous: integer bars) is lifted into the
  execution_profile at mint time (create_strategy_container), so new strategies
  get real backtest/paper/live enforcement;
- percent-unit fields are never lifted (0.04 could mean 4% or 0.04%); creation
  surfaces a warning instead;
- the validator skips fields the profile already covers.
"""

from __future__ import annotations

import json

from forven.db import create_strategy_container, get_db
from forven.strategies.backtest import validate_backtest_risk_controls
from forven.strategies.sizing import lift_unambiguous_risk_params


# ----------------------------------------------------------------- lift helper


def test_lift_moves_time_stop_bars_into_profile():
    out = lift_unambiguous_risk_params({"time_stop_bars": 48, "rsi_period": 14})
    assert out["execution_profile"]["time_stop_bars"] == 48
    assert out["time_stop_bars"] == 48  # top-level kept — strategy code may read it
    assert out["rsi_period"] == 14


def test_lift_never_clobbers_existing_profile_value():
    out = lift_unambiguous_risk_params(
        {"time_stop_bars": 48, "execution_profile": {"time_stop_bars": 24}}
    )
    assert out["execution_profile"]["time_stop_bars"] == 24


def test_lift_preserves_other_profile_fields():
    out = lift_unambiguous_risk_params(
        {"time_stop_bars": 48, "execution_profile": {"stop_loss_pct": 3.0}}
    )
    assert out["execution_profile"] == {"stop_loss_pct": 3.0, "time_stop_bars": 48}


def test_lift_ignores_invalid_and_nonpositive_values():
    assert "execution_profile" not in lift_unambiguous_risk_params({"time_stop_bars": 0})
    assert "execution_profile" not in lift_unambiguous_risk_params({"time_stop_bars": -5})
    assert "execution_profile" not in lift_unambiguous_risk_params({"time_stop_bars": "soon"})
    assert "execution_profile" not in lift_unambiguous_risk_params({"time_stop_bars": None})


def test_lift_never_touches_percent_unit_fields():
    out = lift_unambiguous_risk_params({"stop_loss_pct": 0.04, "risk_pct": 1.0})
    assert "execution_profile" not in out


def test_lift_does_not_mutate_input():
    src = {"time_stop_bars": 48}
    lift_unambiguous_risk_params(src)
    assert src == {"time_stop_bars": 48}


# ------------------------------------------------------------------- validator


def test_validator_still_flags_uncovered_fields():
    warning = validate_backtest_risk_controls({"stop_loss_pct": 4.0})
    assert warning and "stop_loss_pct" in warning


def test_validator_skips_profile_covered_fields():
    params = {
        "time_stop_bars": 48,
        "execution_profile": {"time_stop_bars": 48},
    }
    assert validate_backtest_risk_controls(params) is None


def test_validator_profile_coverage_is_per_field():
    params = {
        "time_stop_bars": 48,
        "stop_loss_pct": 4.0,
        "execution_profile": {"time_stop_bars": 48},
    }
    warning = validate_backtest_risk_controls(params)
    assert warning and "stop_loss_pct" in warning and "time_stop_bars" not in warning


def test_validator_portfolio_guards_never_covered():
    # Portfolio-level guards have no simulator implementation; a profile cannot
    # cover them.
    params = {
        "max_daily_loss_pct": 5.0,
        "execution_profile": {"time_stop_bars": 48},
    }
    warning = validate_backtest_risk_controls(params)
    assert warning and "max_daily_loss_pct" in warning


# ------------------------------------------------------------- mint-time lift


def test_create_strategy_container_lifts_time_stop(forven_db):
    with get_db() as conn:
        sid, _display, _base = create_strategy_container(
            conn=conn,
            name="test",
            type_="rsi_momentum",
            symbol="ETH",
            timeframe="1h",
            params={"rsi_period": 14, "time_stop_bars": 36},
        )
    with get_db() as conn:
        row = conn.execute("SELECT params FROM strategies WHERE id = ?", (sid,)).fetchone()
    stored = json.loads(row["params"])
    assert stored["execution_profile"]["time_stop_bars"] == 36
    assert stored["time_stop_bars"] == 36
    # And the stored shape passes the validator — no false "unenforced" block.
    assert validate_backtest_risk_controls(stored) is None


def test_create_strategy_container_without_time_stop_unchanged(forven_db):
    with get_db() as conn:
        sid, _display, _base = create_strategy_container(
            conn=conn,
            name="test2",
            type_="rsi_momentum",
            symbol="BTC",
            timeframe="1h",
            params={"rsi_period": 14},
        )
    with get_db() as conn:
        row = conn.execute("SELECT params FROM strategies WHERE id = ?", (sid,)).fetchone()
    stored = json.loads(row["params"])
    assert "execution_profile" not in stored
