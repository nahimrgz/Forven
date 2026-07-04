"""The `pandas_ta` import alias must resolve to `pandas_ta_classic`.

The original `pandas-ta` distribution was pulled from PyPI, but the strategy
corpus, the sandbox allowlist (`ALLOWED_IMPORTS`), and LLM-generated strategy
sources all say `import pandas_ta`. Bug report #121 (2026-07-04) showed the
registration pipeline hard-failing with `ModuleNotFoundError: pandas_ta` for
every indicator-based strategy. The fix ships the maintained fork
`pandas-ta-classic` and bridges the name in `forven/__init__.py` via a
meta-path finder appended to `sys.meta_path` (so a genuine `pandas_ta`
install, if one ever reappears, still wins).

If this test fails, do NOT point strategy code at `pandas_ta_classic`
directly — fix the alias, because the sandboxed strategy corpus (DB-stored
sources included) cannot be rewritten retroactively.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import forven  # noqa: F401 — installs the alias finder as an import side effect


def test_import_pandas_ta_resolves_to_classic_fork() -> None:
    import pandas_ta

    assert pandas_ta.__name__ == "pandas_ta_classic"


def test_alias_is_the_same_module_object() -> None:
    import pandas_ta
    import pandas_ta_classic

    assert pandas_ta is pandas_ta_classic


def test_from_import_works_through_alias() -> None:
    from pandas_ta import ema, rsi  # noqa: F401


def test_alias_computes_indicators() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    import pandas_ta as ta

    rng = np.random.default_rng(7)
    n = 200
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)

    adx = ta.adx(high, low, close)
    assert adx is not None and not adx.dropna().empty
    assert ta.rsi(close).dropna().between(0, 100).all()


def test_ast_guard_allows_both_spellings() -> None:
    from forven.sandbox.ast_guard import scan_source

    assert scan_source("import pandas_ta as ta\n").ok
    assert scan_source("import pandas_ta_classic as ta\n").ok


def test_alias_available_in_fresh_subprocess() -> None:
    """The worker sandbox imports forven in a fresh process; the alias must be
    installed there too, not just in the pytest process."""
    code = (
        "import forven, pandas_ta; "
        "assert pandas_ta.__name__ == 'pandas_ta_classic'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
