"""Prompt input-data budget (2026-07-06).

runner.py used to serialize task ``input_data`` with a blind
``json.dumps(input_data, indent=2)[:3000]`` slice. Once the Phase 5 lineage
context (sibling table + per-sibling rejection reasons) grew past a few
entries, that slice routinely cut the JSON mid-token, feeding the model
invalid JSON. These tests pin the replacement: a budget-aware serializer
that stays valid JSON, shortens the longest fields first (sibling rejection
reasons), then drops siblings from the tail with an explicit omission
marker — and is byte-identical to the old output for small payloads.
"""
from __future__ import annotations

import json

from forven.agents.runner import (
    PROMPT_INPUT_DATA_BUDGET_CHARS,
    _truncate_input_data_for_prompt,
)


def _flatten_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten_values(v)
    else:
        yield obj


def test_small_payload_is_byte_identical_to_plain_json_dumps():
    payload = {"a": 1, "siblings": [{"strategy_id": "S1", "name": "x"}]}
    expected = json.dumps(payload, indent=2, default=str)

    assert _truncate_input_data_for_prompt(payload) == expected


def test_default_budget_is_generous_but_bounded():
    # Sanity check on the chosen constant itself (documents the decision).
    assert PROMPT_INPUT_DATA_BUDGET_CHARS >= 8000
    assert PROMPT_INPUT_DATA_BUDGET_CHARS <= 20000


def test_over_budget_payload_stays_valid_json_with_omission_marker():
    siblings = [
        {
            "strategy_id": f"S{i:05d}",
            "last_rejection": {
                "gate": "gauntlet",
                "reason_code": "wfa_degradation",
                "reason": "x" * 500,
            },
        }
        for i in range(200)
    ]
    payload = {"hypothesis_id": "H1", "siblings": siblings}

    naive = json.dumps(payload, indent=2, default=str)
    serialized = _truncate_input_data_for_prompt(payload, budget=2000)

    # Must remain parseable JSON — the whole point of this fix.
    parsed = json.loads(serialized)
    assert len(serialized) < len(naive)
    assert any("omitted" in str(v) for v in _flatten_values(parsed))


def test_reason_truncated_before_dropping_siblings():
    siblings = [
        {
            "strategy_id": f"S{i}",
            "last_rejection": {"gate": "g", "reason_code": "r", "reason": "y" * 1000},
        }
        for i in range(3)
    ]
    payload = {"siblings": siblings}

    serialized = _truncate_input_data_for_prompt(payload, budget=1500)
    parsed = json.loads(serialized)

    for sibling in parsed.get("siblings", []):
        assert len(sibling["last_rejection"]["reason"]) <= 120


def test_non_sibling_payload_still_falls_back_safely():
    # A payload with no "siblings" key at all — the generic last-resort
    # fallback must still produce valid, budget-respecting JSON. The prior
    # version of this test only asserted the result was SMALLER than the
    # naive dump, not that it actually fit the requested budget — the real
    # invariant the function promises.
    payload = {"blob": "z" * 50000, "hypothesis_id": "H2", "action_kind": "develop_candidate"}

    serialized = _truncate_input_data_for_prompt(payload, budget=500)
    parsed = json.loads(serialized)

    assert parsed.get("hypothesis_id") == "H2"
    assert len(serialized) < len(json.dumps(payload, indent=2, default=str))
    assert len(serialized) <= 500


def test_oversized_id_field_is_capped_in_fallback():
    """A corrupted/oversized whitelisted ID field (e.g. hypothesis_display_id)
    must not sail through the last-resort fallback verbatim — it must be
    capped so the fallback's OWN output still respects the budget."""
    payload = {
        "hypothesis_display_id": "H" * 30_000,
        "action_kind": "develop_candidate",
    }

    serialized = _truncate_input_data_for_prompt(payload)
    parsed = json.loads(serialized)

    assert len(serialized) <= PROMPT_INPUT_DATA_BUDGET_CHARS
    assert parsed.get("action_kind") == "develop_candidate"


def test_massive_keys_present_list_is_capped_in_fallback():
    """A payload with thousands of top-level keys must not blow the budget
    via an unbounded ``keys_present`` list in the fallback envelope."""
    payload = {f"key_{i}": "v" for i in range(2000)}

    serialized = _truncate_input_data_for_prompt(payload)
    parsed = json.loads(serialized)

    assert len(serialized) <= PROMPT_INPUT_DATA_BUDGET_CHARS
    assert isinstance(parsed, dict)
