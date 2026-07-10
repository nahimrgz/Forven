from __future__ import annotations

import pandas as pd

from forven.strategies.builtin.funding import FundingStrategy


def _sample_ohlcv(rows: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series([100.0 + i * 0.5 for i in range(rows)], index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series([max(o, c) + 0.2 for o, c in zip(open_, close)], index=idx)
    low = pd.Series([min(o, c) - 0.2 for o, c in zip(open_, close)], index=idx)
    volume = pd.Series([1000.0 + i for i in range(rows)], index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_funding_strategy_uses_funding_rates():
    df = _sample_ohlcv()
    df["funding_rate"] = -0.0002

    strategy = FundingStrategy("S027-FUND-BTC", {"_asset": "BTC"})
    signal = strategy.generate_signal(df)

    assert signal.price > 0
    assert signal.entry_signal is True
    assert signal.exit_signal is False
    assert float(signal.indicators.get("funding")) == -0.0002

