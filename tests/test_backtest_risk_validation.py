from forven.strategies.backtest import backtest_strategy, walk_forward


def test_backtest_strategy_rejects_unsupported_risk_controls(forven_db):
    result = backtest_strategy(
        strategy_id="bt-risk-validation",
        asset="BTC",
        strategy_type="rsi_momentum",
        params={"stop_loss_pct": 2.0, "risk_pct": 0.01},
        bars=240,
    )

    warning = str(result.get("warning") or result.get("error") or "")
    assert "stop_loss_pct" in warning
    assert "risk_pct" in warning


def test_walk_forward_rejects_unsupported_risk_controls(forven_db):
    result = walk_forward(
        strategy_id="wf-risk-validation",
        asset="BTC",
        strategy_type="rsi_momentum",
        params={"min_risk_reward_ratio": 2.0},
        total_bars=500,
        n_splits=2,
    )

    warning = str(result.get("warning") or result.get("error") or "")
    assert "min_risk_reward_ratio" in warning


def test_validate_backtest_risk_controls_exclusions():
    from forven.strategies.backtest import validate_backtest_risk_controls
    
    # Only in params (should warn)
    warn = validate_backtest_risk_controls({"stop_loss_pct": 2.0})
    assert warn is not None and "stop_loss_pct" in warn
    
    # In params + execution_profile (should not warn)
    assert validate_backtest_risk_controls({
        "stop_loss_pct": 2.0,
        "execution_profile": {"stop_loss_pct": 2.0}
    }) is None
    
    # In params + extra_controls (should not warn)
    assert validate_backtest_risk_controls({"stop_loss_pct": 2.0}, extra_controls={"stop_loss_pct": 2.0}) is None
    
    # Verify that time_stop_bars in execution_profile does not warn
    assert validate_backtest_risk_controls({
        "time_stop_bars": 48,
        "execution_profile": {"time_stop_bars": 48}
    }) is None


def test_backtest_strategy_ignores_risk_controls_in_execution_profile(forven_db):
    result = backtest_strategy(
        strategy_id="bt-risk-execution-profile",
        asset="BTC",
        strategy_type="rsi_momentum",
        params={
            "execution_profile": {
                "time_stop_bars": 48,
                "sizing_mode": "atr"
            }
        },
        bars=240,
    )
    warning = str(result.get("warning") or result.get("error") or "")
    assert "time_stop_bars" not in warning
    assert "sizing_mode" not in warning


def test_backtest_strategy_ignores_risk_controls_in_extra_controls(forven_db):
    result = backtest_strategy(
        strategy_id="bt-risk-extra-controls",
        asset="BTC",
        strategy_type="rsi_momentum",
        params={"stop_loss_pct": 2.0},
        bars=240,
        execution_controls={"stop_loss_pct": 2.0}
    )
    warning = str(result.get("warning") or result.get("error") or "")
    assert "stop_loss_pct" not in warning

