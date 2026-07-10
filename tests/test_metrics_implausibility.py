"""Implausible-metrics quarantine (the look-ahead / data-leak fingerprint).

`check_metrics_integrity` now flags clamped/implausible Sharpe, PF, and return
values and routes them to the non-terminal `DataQualityHold` quarantine. This
runs on the persist path + both brain transition guardrails, which are NOT
bypassed under `testing_mode` (the policy gate's plausibility check is), so a
clamped-Sharpe leak can no longer enqueue while gates are relaxed.
"""
from __future__ import annotations

from forven.metrics_integrity import (
    check_metrics_integrity,
    data_quality_hold_reason,
    DATA_QUALITY_HOLD_PREFIX,
)


def _metrics(is_sharpe, oos_sharpe, *, is_tr=60, oos_tr=30, pf=1.5, ret=0.05):
    def leg(s, t):
        return {"sharpe": s, "total_trades": t, "profit_factor": pf, "total_return_pct": ret}
    return {"in_sample": leg(is_sharpe, is_tr), "out_of_sample": leg(oos_sharpe, oos_tr)}


def test_clamped_sharpe_is_quarantined_non_terminally():
    anomalies = check_metrics_integrity(_metrics(10.0, 9.99))
    assert anomalies and any("clamp" in a for a in anomalies)
    reason = data_quality_hold_reason(anomalies)
    assert reason.startswith(DATA_QUALITY_HOLD_PREFIX)
    assert "(reject)" not in reason  # must be NON-terminal (held, not archived)


def test_negative_clamp_is_quarantined():
    assert check_metrics_integrity(_metrics(-10.0, 1.0))


def test_implausibly_high_but_unclamped_sharpe_flagged():
    # The exact S02940 leak fingerprint (IS 6.82 / OOS 5.52) -- caught at |Sharpe|>=6.
    assert check_metrics_integrity(_metrics(6.82, 5.52))


def test_high_pf_on_real_sample_flagged():
    assert check_metrics_integrity(_metrics(1.0, 1.0, pf=12.0))


def test_high_pf_on_tiny_sample_is_noise_not_flagged():
    # A high PF on a handful of trades is small-sample noise, not a leak.
    assert check_metrics_integrity(_metrics(1.0, 1.0, is_tr=3, oos_tr=2, pf=12.0)) == []


def test_absurd_return_flagged():
    assert check_metrics_integrity(_metrics(1.0, 1.0, ret=24000.0))  # millions-% leak


def test_large_negative_unclamped_sharpe_is_not_flagged_as_leak():
    # S00239: OOS Sharpe -8.64 (unclamped) on a genuinely losing strategy is NOT a
    # data leak — a leak makes performance implausibly GOOD (positive). A large
    # negative Sharpe is a consistent loser / low-dispersion small sample the gates
    # reject normally. It must not fire the data-quality "data leak" quarantine.
    assert check_metrics_integrity(_metrics(-0.12, -8.64)) == []


def test_s00239_realistic_losing_payload_is_clean():
    # The real persisted S00239 legs — losing in both, no anomaly to quarantine.
    metrics = {
        "in_sample": {"sharpe": -0.12, "total_trades": 12, "profit_factor": 0.869, "total_return_pct": -0.02184},
        "out_of_sample": {"sharpe": -8.64, "total_trades": 23, "profit_factor": 0.054, "total_return_pct": -0.22452},
    }
    assert check_metrics_integrity(metrics) == []


def test_negative_unclamped_sharpe_does_not_mask_a_positive_leak_leg():
    # A losing OOS leg must not be flagged, but an implausibly-high POSITIVE in_sample
    # leg on the same payload still is — direction-awareness is per leg.
    anomalies = check_metrics_integrity(_metrics(7.5, -8.64))
    assert any("in_sample" in a and "leak" in a for a in anomalies)
    assert not any("out_of_sample" in a for a in anomalies)


def test_normal_metrics_pass_clean():
    assert check_metrics_integrity(_metrics(1.2, 0.9, pf=1.6, ret=0.08)) == []
    assert check_metrics_integrity(_metrics(0.07, 2.19, pf=1.96, ret=0.30)) == []  # real R2 winner S02754


def test_existing_zero_trade_anomaly_preserved():
    assert check_metrics_integrity(_metrics(0.0, 0.0, is_tr=0, oos_tr=30))
