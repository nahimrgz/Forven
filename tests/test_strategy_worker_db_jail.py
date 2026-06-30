"""R3 containment regression: the isolated strategy worker is DB-jailed.

Untrusted imported/custom strategy code runs ONLY inside the out-of-process
strategy worker (forven.sandbox.strategy_worker, which sets
FORVEN_IN_STRATEGY_WORKER). The 2026-06-29 strategy-share audit found that the
worker could still reach the production DB *by path* through the still-allowlisted
confused-deputy modules (forven.scanner re-exports get_db/kv_get/kv_set;
forven.data/data_manager), letting in-worker code read/tamper state and decrypt
secrets. The boundary is the DB connection factory itself: it refuses every
connection while the worker flag is set. These tests assert that invariant and
that the parent is unaffected.

See docs/strategy-share-security-audit-2026-06-29.md (R3).
"""

import pytest

import forven.db as db

WORKER_FLAG = "FORVEN_IN_STRATEGY_WORKER"
_FACTORIES = ("get_db", "get_db_best_effort", "get_db_immediate")


@pytest.mark.parametrize("factory_name", _FACTORIES)
def test_db_factories_refuse_inside_worker(monkeypatch, factory_name):
    monkeypatch.setenv(WORKER_FLAG, "1")
    factory = getattr(db, factory_name)
    with pytest.raises(RuntimeError, match="strategy sandbox"):
        with factory():
            pass


def test_scanner_get_db_reexport_is_jailed(monkeypatch):
    """forven.scanner.get_db is the SAME function object as forven.db.get_db, so a
    monkeypatch on the parent alias would miss it — the in-body env guard does not."""
    import forven.scanner as scanner

    assert scanner.get_db is db.get_db
    monkeypatch.setenv(WORKER_FLAG, "1")
    with pytest.raises(RuntimeError, match="strategy sandbox"):
        with scanner.get_db():
            pass


def test_parent_db_access_is_unaffected(monkeypatch):
    """Outside the sandbox the guard must be a pure no-op (the flag is never set in
    the trusted parent)."""
    monkeypatch.delenv(WORKER_FLAG, raising=False)
    db._assert_db_access_allowed()  # must not raise
    with db.get_db() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_active_strategy_sweep_is_skipped_in_worker(monkeypatch):
    """registry.discover()'s parent-side DB sweep is the worker's only legitimate DB
    touch; it must early-return in the worker so startup never hits the jail."""
    from forven.strategies import registry

    monkeypatch.setenv(WORKER_FLAG, "1")
    # A clean no-op: if the sweep reached the jailed get_db this would surface the
    # RuntimeError (the sweep's broad except would swallow it, but the explicit skip
    # is what we assert — so patch get_db to a tripwire that must NOT be called).
    def _tripwire(*_a, **_k):
        raise AssertionError("worker DB sweep must not open a connection")

    monkeypatch.setattr(db, "get_db", _tripwire)
    registry._ensure_active_db_strategy_modules()  # must return without calling get_db
