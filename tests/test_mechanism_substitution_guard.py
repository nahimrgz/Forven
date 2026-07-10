from __future__ import annotations

import importlib
import sys
import os
import re
from datetime import datetime, timezone

import pytest

import forven.brain as brain_mod
from forven.db import get_db, create_approval
from forven.hypotheses import create_hypothesis, get_hypothesis
from forven.strategies import custom as custom_pkg
from forven.strategies import intake as intake_mod
from forven.strategies import registry


def _write_custom_strategy(
    path,
    *,
    class_name: str = "AIDropzoneWave",
    type_name: str = "ai_dropzone_wave_test",
    body_marker: str = "",
) -> None:
    # body_marker injects a REAL statement into the logic so two strategies can
    # share a class name yet have distinct bodies (a legit _v2 iteration).
    lines = [
        "import pandas as pd",
        "from forven.strategies.base import BaseStrategy, Signal",
        "",
        f"class {class_name}(BaseStrategy):",
        "    @property",
        "    def name(self) -> str:",
        f"        return '{class_name} test'",
        "",
        "    @property",
        "    def asset(self) -> str:",
        "        return 'BTC'",
        "",
        "    @property",
        "    def strategy_type(self) -> str:",
        "        return TYPE_NAME",
        "",
        "    @property",
        "    def default_params(self) -> dict:",
        "        return {'risk_pct': 0.01, 'leverage': 1.0}",
        "",
        "    def generate_signal(self, df: pd.DataFrame) -> Signal:",
        f"        _variant = {body_marker!r}",
        "        price = float(df['close'].iloc[-1]) if 'close' in df and len(df.index) else 0.0",
        "        return Signal(price=price)",
        "",
        f"STRATEGY_CLASS = {class_name}",
        f"TYPE_NAME = '{type_name}'",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_substitution_uses_token_overlap_not_exact_substring():
    """Fragility fix: the old exact-substring match required the class name to
    appear VERBATIM (normalized) in the hypothesis prose, so descriptively-named
    classes were mass-forbidden. A class whose MECHANISM TOKENS appear in the
    hypothesis (even if the concatenated name never does) is NOT a substitution;
    a genuinely different edge (zero meaningful-token overlap) still is."""
    hypothesis = {
        "title": "Lead-lag cross-asset momentum",
        "mechanism": "Exploit lead-lag momentum between correlated crypto assets.",
    }
    # Concept matches (lead/lag/asset/momentum all present) but the concatenated
    # class name never appears verbatim — must NOT be flagged.
    assert intake_mod._is_mechanism_substituted(
        "LeadLagXAssetMomentumStrategy", hypothesis, []
    ) is False
    # A different mechanism (no shared mechanism tokens) is still caught.
    assert intake_mod._is_mechanism_substituted(
        "RsiOscillatorReversion", hypothesis, []
    ) is True
    # Noise-only / generic names are unjudgeable — stay permissive (don't block).
    assert intake_mod._is_mechanism_substituted(
        "StrategyV2", hypothesis, []
    ) is False


def test_duplicate_guard_matches_body_not_class_name(forven_db, monkeypatch, tmp_path):
    """Fix: matching duplicates on class NAME mass-fired (a strategy matched its
    own row on retry; _v2 iterations reused a class name over different logic).
    Now only an IDENTICAL normalized body under the same hypothesis/lane is a
    duplicate — a reused class name with different code is allowed."""
    temp_custom_dir = tmp_path / "custom"
    temp_custom_dir.mkdir()
    monkeypatch.setattr(custom_pkg, "__path__", [str(temp_custom_dir)])
    monkeypatch.setattr(custom_pkg, "__file__", str(temp_custom_dir / "__init__.py"))
    registry.reset()
    importlib.invalidate_caches()
    for t in ("dup_a", "dup_b", "dup_c"):
        sys.modules.pop(f"forven.strategies.custom.{t}", None)

    hyp = create_hypothesis(
        title="Momentum Overlay edge",
        market_thesis="Momentum overlay on trend.",
        mechanism="A momentum overlay strategy.",
        why_now="now",
        lane="benchmarking",
        source_type="test",
        target_assets=["BTC/USDT"],
        target_timeframes=["1h"],
    )
    hyp_id = str(hyp["id"])

    # A: class MomentumOverlay, body 'aaa'
    fa = temp_custom_dir / "dup_a.py"
    _write_custom_strategy(fa, class_name="MomentumOverlay", type_name="dup_a", body_marker="aaa")
    ra = intake_mod.register_custom_strategy_file(file_path=str(fa), hypothesis_id=hyp_id)
    assert ra["strategy_id"] is not None

    # B: SAME class name, DIFFERENT body -> a legit iteration, must be ALLOWED.
    fb = temp_custom_dir / "dup_b.py"
    _write_custom_strategy(fb, class_name="MomentumOverlay", type_name="dup_b", body_marker="bbb_different_logic")
    rb = intake_mod.register_custom_strategy_file(file_path=str(fb), hypothesis_id=hyp_id)
    assert rb["strategy_id"] is not None

    # C: SAME class name, IDENTICAL body to A -> true duplicate, must be FORBIDDEN.
    fc = temp_custom_dir / "dup_c.py"
    _write_custom_strategy(fc, class_name="MomentumOverlay", type_name="dup_c", body_marker="aaa")
    with pytest.raises(ValueError) as ei:
        intake_mod.register_custom_strategy_file(file_path=str(fc), hypothesis_id=hyp_id)
    assert "Duplicate registration forbidden" in str(ei.value)


def test_mechanism_substitution_guards(forven_db, monkeypatch, tmp_path):
    # Setup temporary custom directory
    temp_custom_dir = tmp_path / "custom"
    temp_custom_dir.mkdir()
    monkeypatch.setattr(custom_pkg, "__path__", [str(temp_custom_dir)])
    monkeypatch.setattr(custom_pkg, "__file__", str(temp_custom_dir / "__init__.py"))

    registry.reset()
    importlib.invalidate_caches()
    sys.modules.pop("forven.strategies.custom.btc_ai_dropzone_wave_test", None)
    sys.modules.pop("forven.strategies.custom.taker_flow_div_test", None)
    sys.modules.pop("forven.strategies.custom.lsr_rev_test", None)

    # 1. Create a parent hypothesis
    hyp = create_hypothesis(
        title="TakerFlowDivergence strategy design",
        market_thesis="Taker flow diverges on high volume.",
        mechanism="TakerFlowDivergence mechanism should trigger on volume spikes.",
        why_now="Recent market regimes show high volume.",
        lane="funding/carry",
        source_type="test",
        target_assets=["BTC/USDT"],
        target_timeframes=["1h"],
    )
    hyp_id = str(hyp["id"])

    # 2. Try registering a strategy class that does NOT match the hypothesis spec
    # Expected: fails closed with ValueError, and queues a pending approval
    bad_strat_file = temp_custom_dir / "btc_ai_dropzone_wave_test.py"
    _write_custom_strategy(bad_strat_file, class_name="AIDropzoneWave", type_name="btc_ai_dropzone_wave_test")

    with pytest.raises(ValueError) as excinfo:
        intake_mod.register_custom_strategy_file(file_path=str(bad_strat_file), hypothesis_id=hyp_id)
    assert "Mechanism substitution forbidden" in str(excinfo.value)

    # Verify a pending approval was created
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, status, reason, payload FROM approvals
            WHERE approval_type = 'mechanism_substitution_approval'
              AND target_id = ?
            """,
            (hyp_id,),
        ).fetchone()
    assert row is not None
    assert row["status"] == "pending_approval"
    assert "AIDropzoneWave" in row["reason"]

    # 3. Register a strategy class that DOES match the hypothesis spec
    # Expected: succeeds!
    good_strat_file = temp_custom_dir / "taker_flow_div_test.py"
    _write_custom_strategy(good_strat_file, class_name="TakerFlowDivergence", type_name="taker_flow_div_test")

    result = intake_mod.register_custom_strategy_file(file_path=str(good_strat_file), hypothesis_id=hyp_id)
    assert result["strategy_id"] is not None

    # 4. Attempt to register a duplicate under the same hypothesis family/lane
    # Expected: fails with Duplicate registration forbidden
    dup_strat_file = temp_custom_dir / "taker_flow_div_test_dup.py"
    _write_custom_strategy(dup_strat_file, class_name="TakerFlowDivergence", type_name="taker_flow_div_test_dup")

    with pytest.raises(ValueError) as excinfo:
        intake_mod.register_custom_strategy_file(file_path=str(dup_strat_file), hypothesis_id=hyp_id)
    assert "Duplicate registration forbidden" in str(excinfo.value)

    # 5. Approve an override for a substituted class (e.g. LsrReversionShort)
    # Expected: registration of the substituted class succeeds!
    override_strat_file = temp_custom_dir / "lsr_rev_test.py"
    _write_custom_strategy(override_strat_file, class_name="LsrReversionShort", type_name="lsr_rev_test")

    # Queue and approve override
    approval_id = create_approval(
        approval_type="mechanism_substitution_approval",
        target_type="hypothesis",
        target_id=hyp_id,
        requested_status="override",
        status="approved",
        actor="operator",
        reason="Approved by operator override",
        payload={
            "hypothesis_id": hyp_id,
            "class_name": "LsrReversionShort",
            "type_name": "lsr_rev_test",
        }
    )

    result_override = intake_mod.register_custom_strategy_file(file_path=str(override_strat_file), hypothesis_id=hyp_id)
    assert result_override["strategy_id"] is not None
