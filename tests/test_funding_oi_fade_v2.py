import os
import pandas as pd
import numpy as np
import pytest
from forven.strategies.backtest import backtest_strategy

def test_funding_oi_fade_v2_default_params():
    os.environ["FORVEN_INCLUDE_ARCHIVED_CUSTOM_STRATEGIES"] = "1"
    import forven.strategies.registry as reg
    reg._custom_discovered = False
    reg._discovered = False
    reg.discover()
    print("DEFAULT PARAMS - Is funding_oi_fade_v2 in _TYPE_MAP?", "funding_oi_fade_v2" in reg._TYPE_MAP)
    
    from forven.strategies.custom.funding_oi_fade_v2 import FundingOIFadeRefinedStrategy
    strategy = FundingOIFadeRefinedStrategy("test-strat")
    assert "execution_profile" in strategy.DEFAULT_PARAMS
    profile = strategy.DEFAULT_PARAMS["execution_profile"]
    assert profile.get("sizing_mode") == "atr"
    assert profile.get("atr_stop_multiplier") == 2.5
    assert profile.get("time_stop_bars") == 48


def test_funding_oi_fade_v2_time_stop(forven_db):
    os.environ["FORVEN_BACKTEST_PROCESS_ISOLATION"] = "0"
    os.environ["FORVEN_ISOLATED_STRATEGY_EXEC"] = "0"
    os.environ["FORVEN_INCLUDE_ARCHIVED_CUSTOM_STRATEGIES"] = "1"
    
    import forven.strategies.registry as reg
    reg._custom_discovered = False
    reg._discovered = False
    reg.discover()
    print("TIME STOP - Is funding_oi_fade_v2 in _TYPE_MAP?", "funding_oi_fade_v2" in reg._TYPE_MAP)
    
    # Create 300 hourly bars
    dates = pd.date_range(start="2026-06-01", periods=300, freq="1h", tz="UTC")
    close = 100.0 + np.arange(300) * 0.01
    high = close + 1.0
    low = close - 1.0
    volume = np.ones(300) * 1000.0
    open_interest = np.linspace(1000.0, 2000.0, 300)
    funding_rate = np.zeros(300)
    funding_rate[220:] = 0.0004
    
    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "funding_rate": funding_rate,
        "open_interest": open_interest
    }, index=dates)
    
    # Run backtest
    result = backtest_strategy(
        strategy_id="test-oi-fade-v2",
        asset="BTC/USDT",
        strategy_type="funding_oi_fade_v2",
        params={
            "funding_threshold": 0.0003,
            "funding_neutral_band": 0.0001,
            "oi_lookback_bars": 24,
            "vwap_window": 20,
            "regime_filter": ["TREND_UP", "RANGE_BOUND"],
            "execution_profile": {
                "sizing_mode": "atr",
                "atr_stop_multiplier": 2.5,
                "time_stop_bars": 48,
            }
        },
        bars=300,
        candles_df=df,
        trade_mode="both",
    )
    
    assert not result.get("error"), f"Backtest errored: {result.get('error')}"
    trades = result.get("trades") or []
    assert len(trades) > 0, f"No trades were entered. Result: {result}"
    
    trade = trades[0]
    print("Trade details:", trade)
    assert trade.get("exit_reason") == "time_stop"
