"""zero_trade fix (B): the pre-creation dry-run guard must count entry signals
from a 4-series ``generate_signals`` payload the SAME way the backtest engine
consumes it — ``(long_entries, long_exits, short_entries, short_exits)`` — so
entries are payload[0] + payload[2].

Before this, the vectorized path only handled a 2-tuple, so a 4-series strategy
fell through to signal_count=None → the scalar sweep → "-1 unable to validate",
letting genuinely over-tight strategies be created unguarded (they then die at
quick_screen as zero_trade, burning a backtest slot and generation quota).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forven.strategies.base import BaseStrategy, Signal
import forven.strategy_validation as sv


class _FourSeriesStrategy(BaseStrategy):
    """Emits the engine's 4-series contract: long_entries, long_exits,
    short_entries, short_exits. 6 long entries + 4 short entries = 10 entries."""

    @property
    def name(self) -> str:
        return "_four_series_probe"

    @property
    def asset(self) -> str:
        return "BTC"

    @property
    def strategy_type(self) -> str:
        return "_four_series_probe"

    @property
    def default_params(self) -> dict:
        return {}

    def generate_signal(self, df) -> Signal:
        return Signal(0)

    def generate_signals(self, df):
        idx = df.index
        false = pd.Series(False, index=idx)
        long_entries = false.copy()
        short_entries = false.copy()
        long_entries.iloc[10:16] = True   # 6 long entries
        short_entries.iloc[20:24] = True  # 4 short entries
        return long_entries, false.copy(), short_entries, false.copy()


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
        }
    )


def test_dry_run_counts_four_series_entries(monkeypatch):
    from forven.strategies import registry

    monkeypatch.setitem(registry._TYPE_MAP, "_four_series_probe", _FourSeriesStrategy)
    monkeypatch.setattr("forven.data.load_parquet", lambda *a, **k: _synthetic_ohlcv())
    # Keep the dry-run offline/deterministic — no enrichment network/parquet.
    monkeypatch.setattr(
        "forven.data_manager.data_manager.enrich",
        lambda df, *a, **k: df,
    )

    is_valid, reason, count = sv.dry_run_signal_validation(
        "_four_series_probe", params={}, symbol="BTC/USDT", timeframe="1h"
    )

    # 6 long + 4 short = 10 entries (>= MIN_SIGNALS_FOR_CREATION), and crucially
    # NOT the -1 "unable to validate" the 2-tuple-only path returned before.
    assert count == 10
    assert is_valid is True


def test_dry_run_rejects_over_tight_four_series(monkeypatch):
    """A 4-series strategy under the signal floor is now REJECTED at creation
    instead of slipping through as -1."""
    from forven.strategies import registry

    class _TooTight(_FourSeriesStrategy):
        def generate_signals(self, df):
            idx = df.index
            false = pd.Series(False, index=idx)
            long_entries = false.copy()
            long_entries.iloc[5:7] = True  # only 2 entries, below MIN (5)
            return long_entries, false.copy(), false.copy(), false.copy()

    monkeypatch.setitem(registry._TYPE_MAP, "_too_tight_probe", _TooTight)
    monkeypatch.setattr("forven.data.load_parquet", lambda *a, **k: _synthetic_ohlcv())
    monkeypatch.setattr("forven.data_manager.data_manager.enrich", lambda df, *a, **k: df)

    is_valid, reason, count = sv.dry_run_signal_validation(
        "_too_tight_probe", params={}, symbol="BTC/USDT", timeframe="1h"
    )

    assert count == 2
    assert is_valid is False
    assert "signals" in reason.lower()
