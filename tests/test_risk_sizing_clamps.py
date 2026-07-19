from unittest.mock import patch
from forven.db import kv_set, get_db
from forven.exchange.risk import _get_risk_limits, _budget_pct_setting
from forven.scanner import _get_paper_strategy_equity

def test_get_risk_limits_clamped_mainnet(forven_db):
    # GIVEN mainnet mode and override of 5%
    kv_set("forven:settings", {"max_risk_per_trade_pct": 5.0})
    with patch("forven.config.get_execution_mode", return_value="mainnet"):
        limits = _get_risk_limits()
    # THEN limits should be clamped to 1%
    assert limits["max_risk_per_trade"] == 0.01

def test_get_risk_limits_legacy_clamped_mainnet(forven_db):
    # GIVEN mainnet mode and legacy override of 5%
    kv_set("forven:settings", {"max_position_size_pct": 5.0})
    with patch("forven.config.get_execution_mode", return_value="mainnet"):
        limits = _get_risk_limits()
    # THEN limits should be clamped to 1%
    assert limits["max_risk_per_trade"] == 0.01

def test_get_risk_limits_clamped_testnet(forven_db):
    # GIVEN paper/testnet mode and override of 5%
    kv_set("forven:settings", {"max_risk_per_trade_pct": 5.0})
    with patch("forven.config.get_execution_mode", return_value="paper"):
        limits = _get_risk_limits()
    # THEN limits should be clamped to 2%
    assert limits["max_risk_per_trade"] == 0.02

def test_get_risk_limits_unclamped_mainnet(forven_db):
    # GIVEN mainnet mode and override of 0.5% (below cap)
    kv_set("forven:settings", {"max_risk_per_trade_pct": 0.5})
    with patch("forven.config.get_execution_mode", return_value="mainnet"):
        limits = _get_risk_limits()
    # THEN limits should not be clamped
    assert limits["max_risk_per_trade"] == 0.005

def test_get_risk_limits_unclamped_testnet(forven_db):
    # GIVEN paper/testnet mode and override of 1.5% (below cap)
    kv_set("forven:settings", {"max_risk_per_trade_pct": 1.5})
    with patch("forven.config.get_execution_mode", return_value="paper"):
        limits = _get_risk_limits()
    # THEN limits should not be clamped
    assert limits["max_risk_per_trade"] == 0.015

def test_budget_pct_setting_clamped_mainnet(forven_db):
    # GIVEN mainnet mode and override of 3%
    settings = {"live_hard_max_per_trade_risk_pct": 3.0}
    with patch("forven.config.get_execution_mode", return_value="mainnet"):
        val = _budget_pct_setting(settings, "live_hard_max_per_trade_risk_pct")
    # THEN value should be clamped to 1%
    assert val == 1.0

def test_budget_pct_setting_default_mainnet(forven_db):
    # GIVEN mainnet mode and NO override
    settings = {}
    with patch("forven.config.get_execution_mode", return_value="mainnet"):
        val = _budget_pct_setting(settings, "live_hard_max_per_trade_risk_pct")
    # THEN value should default to 1%
    assert val == 1.0

def test_budget_pct_setting_clamped_testnet(forven_db):
    # GIVEN paper mode and override of 3%
    settings = {"live_hard_max_per_trade_risk_pct": 3.0}
    with patch("forven.config.get_execution_mode", return_value="paper"):
        val = _budget_pct_setting(settings, "live_hard_max_per_trade_risk_pct")
    # THEN value should be clamped to 2%
    assert val == 2.0

def test_budget_pct_setting_default_testnet(forven_db):
    # GIVEN paper mode and NO override
    settings = {}
    with patch("forven.config.get_execution_mode", return_value="paper"):
        val = _budget_pct_setting(settings, "live_hard_max_per_trade_risk_pct")
    # THEN value should default to 2%
    assert val == 2.0

def test_get_paper_strategy_equity_uses_account_equity(forven_db):
    # GIVEN account equity is $150k
    with patch("forven.scanner._get_account_equity", return_value=150000.0):
        # WHEN resolving paper strategy equity
        eq = _get_paper_strategy_equity("strategy-1")
    # THEN should resolve to $150k
    assert eq == 150000.0

def test_get_paper_strategy_equity_with_realized_pnl(forven_db):
    # GIVEN account equity is $150k and there is closed trade PnL of $1000
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO trades (id, strategy, strategy_id, asset, direction, entry_price, signal_entry_price, size, risk_pct, leverage, status, execution_type, pnl_usd, opened_at)
            VALUES ('T-TEST-1', 'strategy-1', 'strategy-1', 'BTC', 'long', 100.0, 100.0, 1.0, 0.01, 2.0, 'CLOSED', 'paper', 1000.0, '2026-07-14 00:00:00')
            """
        )
    with patch("forven.scanner._get_account_equity", return_value=150000.0):
        # WHEN resolving paper strategy equity
        eq = _get_paper_strategy_equity("strategy-1")
    # THEN should resolve to $151k
    assert eq == 151000.0
