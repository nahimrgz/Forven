import json
import pytest
from forven.db import get_db


@pytest.fixture
def tmp_strategy(forven_db):
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO strategies (id, name, params) VALUES (?, ?, ?)",
            ("S77001", "test", json.dumps({"rsi_period": 14, "rsi_threshold": 30})),
        )
        conn.commit()
    return "S77001"


def test_update_existing_params(tmp_strategy):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    set_deepdive_strategy(tmp_strategy)
    _update_default_params(params={"rsi_period": 21}, rationale="longer lookback", thread_id="dd_t")
    with get_db() as conn:
        row = conn.execute(
            "SELECT params FROM strategies WHERE id = ?", (tmp_strategy,)
        ).fetchone()
    merged = json.loads(row[0])
    assert merged["rsi_period"] == 21
    assert merged["rsi_threshold"] == 30  # unchanged


def test_update_unknown_key_rejected(tmp_strategy):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    set_deepdive_strategy(tmp_strategy)
    with pytest.raises(ValueError, match="unknown param"):
        _update_default_params(params={"made_up_key": 1}, rationale="x", thread_id="dd_t")


def test_update_logs_to_activity(tmp_strategy):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    set_deepdive_strategy(tmp_strategy)
    _update_default_params(params={"rsi_period": 21}, rationale="r", thread_id="dd_p")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT source, message FROM activity_log "
            "WHERE source = 'deepdive_agent:dd_p' ORDER BY id DESC LIMIT 1"
        ).fetchall()
    assert rows


# ---------------------------------------------------------------------------
# execution_profile updates post-mint (P1-D).
#
# The engine honors risk controls ONLY from the nested params['execution_profile']
# dict (sizing.extract_execution_profile — deliberately no top-level fallback).
# The unknown-key guard used to make that dict impossible to CREATE after mint,
# and flat top-level writes persisted but were operationally inert (S01027
# time_stop_bars reports, 2026-07-25).
# ---------------------------------------------------------------------------


def _params_of(sid: str) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT params FROM strategies WHERE id = ?", (sid,)).fetchone()
    return json.loads(row[0])


def test_execution_profile_can_be_created_post_mint(tmp_strategy):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    from forven.strategies.sizing import extract_execution_profile

    set_deepdive_strategy(tmp_strategy)
    _update_default_params(
        params={"execution_profile": {"time_stop_bars": 96}},
        rationale="add time stop",
        thread_id="dd_ep",
    )
    merged = _params_of(tmp_strategy)
    # The engine must actually honor it — nested, not flat.
    assert extract_execution_profile(merged) == {"time_stop_bars": 96}


def test_execution_profile_deep_merges_with_existing(forven_db):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    from forven.strategies.sizing import extract_execution_profile

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO strategies (id, name, params) VALUES (?, ?, ?)",
            ("S77002", "test", json.dumps({"rsi_period": 14, "execution_profile": {"risk_per_trade": 0.01}})),
        )
    set_deepdive_strategy("S77002")
    _update_default_params(
        params={"execution_profile": {"time_stop_bars": 96}},
        rationale="add time stop, keep sizing",
        thread_id="dd_ep",
    )
    profile = extract_execution_profile(_params_of("S77002"))
    assert profile == {"risk_per_trade": 0.01, "time_stop_bars": 96}


def test_execution_profile_none_clears_a_field(forven_db):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    from forven.strategies.sizing import extract_execution_profile

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO strategies (id, name, params) VALUES (?, ?, ?)",
            ("S77003", "test", json.dumps({"execution_profile": {"time_stop_bars": 96, "risk_per_trade": 0.01}})),
        )
    set_deepdive_strategy("S77003")
    _update_default_params(
        params={"execution_profile": {"time_stop_bars": None}},
        rationale="drop time stop",
        thread_id="dd_ep",
    )
    assert extract_execution_profile(_params_of("S77003")) == {"risk_per_trade": 0.01}


def test_execution_profile_null_clears_the_whole_profile(forven_db):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy
    from forven.strategies.sizing import extract_execution_profile

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO strategies (id, name, params) VALUES (?, ?, ?)",
            ("S77006", "test", json.dumps({"execution_profile": {"time_stop_bars": 96}})),
        )
    set_deepdive_strategy("S77006")
    _update_default_params(
        params={"execution_profile": None}, rationale="revert to default sizing", thread_id="dd_ep"
    )
    assert extract_execution_profile(_params_of("S77006")) == {}


def test_execution_profile_rejects_unknown_subkeys(tmp_strategy):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy

    set_deepdive_strategy(tmp_strategy)
    with pytest.raises(ValueError, match="time_stop_bars"):
        # The error must list the honored fields so the agent can self-correct.
        _update_default_params(
            params={"execution_profile": {"bogus_control": 1}},
            rationale="x",
            thread_id="dd_ep",
        )


def test_execution_profile_must_be_a_dict(tmp_strategy):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy

    set_deepdive_strategy(tmp_strategy)
    with pytest.raises(ValueError, match="execution_profile"):
        _update_default_params(params={"execution_profile": 96}, rationale="x", thread_id="dd_ep")


def test_flat_honored_key_write_warns_it_is_inert(forven_db):
    from forven.agents.tools_deepdive import _update_default_params, set_deepdive_strategy

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO strategies (id, name, params) VALUES (?, ?, ?)",
            ("S77004", "test", json.dumps({"time_stop_bars": 0})),
        )
    set_deepdive_strategy("S77004")
    result = _update_default_params(
        params={"time_stop_bars": 96}, rationale="x", thread_id="dd_ep"
    )
    # Still allowed (the key exists), but the agent must be told the engine
    # ignores it there.
    assert "execution_profile" in result
    assert _params_of("S77004")["time_stop_bars"] == 96


def test_update_strategy_default_params_certifies_sandbox_rows_via_runtime_type(forven_db, monkeypatch):
    """A sandbox-only row carries a bare type with no parent-registry class; the
    certification gate must resolve the namespaced runtime_type or every param
    update on an imported strategy 422s."""
    import forven.api_core as api_core

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO strategies (id, name, type, runtime_type, params) VALUES (?, ?, ?, ?, ?)",
            (
                "S77005",
                "imported paper strategy",
                "custom_taker_absorption",
                "imported__dropzone_custom_taker_absorption_ab12cd34ef56",
                json.dumps({"window": 20}),
            ),
        )
    monkeypatch.setattr(
        "forven.strategies.registry.imported_module_exists", lambda *_a, **_k: True
    )

    res = api_core.update_strategy_default_params(
        "S77005", {"execution_profile": {"time_stop_bars": 96}}, actor="user"
    )
    assert res["ok"] is True
    from forven.strategies.sizing import extract_execution_profile

    assert extract_execution_profile(_params_of("S77005")) == {"time_stop_bars": 96}
