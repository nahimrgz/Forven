# SPDX-FileCopyrightText: 2026 Judder <judder@forven.app>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Forven — Algorithmic trading operations framework."""

__version__ = "0.1.34"


def _install_ta_import_tripwire() -> None:
    """Raise ModuleNotFoundError if anything tries to `import ta`.

    The `ta` library (https://github.com/bukosabino/ta) is permanently banned
    from this codebase. See `tests/test_no_ta_imports.py` for the history:
    ~150 strategy files silently depending on it produced fake "successful"
    backtests for months.

    This tripwire runs at `forven` package import time and installs a
    `MetaPathFinder` that blocks any attempt to import `ta` or its submodules.
    The error message points at the banned-imports guidance so anyone hitting
    it (human or LLM) knows what to do instead.

    The tripwire is intentionally run unconditionally — even if the real `ta`
    package is installed on the machine (e.g. as a transitive dep of something
    else), attempts to import it from within forven code will fail loudly.
    """
    import sys
    from importlib.abc import MetaPathFinder

    _BANNED_ROOTS = frozenset({"ta"})

    class _BannedTaImportFinder(MetaPathFinder):
        """Refuses to resolve `ta` or any `ta.*` submodule."""

        def find_spec(self, fullname, path=None, target=None):  # noqa: D401
            root = fullname.split(".")[0]
            if root in _BANNED_ROOTS:
                raise ModuleNotFoundError(
                    f"Import of '{fullname}' is blocked. The `ta` library is "
                    "permanently banned in forven — use native pandas/numpy "
                    "instead. See forven/strategies/STRATEGY_TEMPLATE.md and "
                    "tests/test_no_ta_imports.py for the full history."
                )
            return None  # Defer to the next finder.

    # Insert at the front so nothing else can resolve `ta` before us.
    # Idempotent: only install once per process.
    if not any(isinstance(f, _BannedTaImportFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _BannedTaImportFinder())


def _install_pandas_ta_alias() -> None:
    """Resolve `import pandas_ta` to the `pandas_ta_classic` fork.

    The original `pandas-ta` distribution was pulled from PyPI, but the strategy
    corpus, the sandbox allowlist (`forven.sandbox.ast_guard.ALLOWED_IMPORTS`),
    and LLM-generated strategy sources all say `import pandas_ta`. We ship the
    maintained community fork `pandas-ta-classic` (numpy>=2 compatible), whose
    top-level module is `pandas_ta_classic`, and bridge the name gap here.

    The finder is APPENDED to `sys.meta_path`, so a genuine `pandas_ta`
    distribution — if one ever reappears in the environment — always resolves
    first; this alias only fires when the normal import machinery has already
    failed to find `pandas_ta`. Loading is lazy: `pandas_ta_classic` (and the
    pandas/numpy stack it drags in) is only imported when strategy code first
    asks for `pandas_ta`.
    """
    import sys
    from importlib import import_module
    from importlib.abc import Loader, MetaPathFinder
    from importlib.util import spec_from_loader

    _ALIAS = "pandas_ta"
    _REAL = "pandas_ta_classic"

    class _PandasTaAliasLoader(Loader):
        """Hands the already-imported real module to the import machinery."""

        def __init__(self, real_name: str) -> None:
            self._real_name = real_name

        def create_module(self, spec):  # noqa: D401, ANN001, ANN201
            return import_module(self._real_name)

        def exec_module(self, module) -> None:  # noqa: ANN001
            pass  # create_module returned a fully-initialized module.

    class _PandasTaAliasFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # noqa: D401, ANN001
            if fullname != _ALIAS and not fullname.startswith(_ALIAS + "."):
                return None
            real_name = _REAL + fullname[len(_ALIAS):]
            try:
                real_module = import_module(real_name)
            except ModuleNotFoundError:
                return None  # fork not installed — fail like any missing module
            spec = spec_from_loader(fullname, _PandasTaAliasLoader(real_name))
            search = getattr(real_module, "__path__", None)
            if search is not None:
                spec.submodule_search_locations = list(search)
            return spec

    # Appended, NOT inserted at the front: a real `pandas_ta` install wins.
    # Idempotent: only install once per process.
    if not any(isinstance(f, _PandasTaAliasFinder) for f in sys.meta_path):
        sys.meta_path.append(_PandasTaAliasFinder())


_install_ta_import_tripwire()
_install_pandas_ta_alias()
