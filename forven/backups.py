"""Managed SQLite backup creation for destructive operator workflows."""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from forven.config import FORVEN_HOME
from forven.db import backup_db

log = logging.getLogger("forven.backups")

DEFAULT_DB_BACKUP_RETENTION = 3
_BACKUP_LOCK = threading.Lock()
_SAFE_REASON_RE = re.compile(r"[^a-z0-9_-]+")


def _retention_limit(value: int | None = None) -> int:
    if value is not None:
        return max(1, int(value))
    raw = str(os.environ.get("FORVEN_DB_BACKUP_RETENTION") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_DB_BACKUP_RETENTION
    except ValueError:
        return DEFAULT_DB_BACKUP_RETENTION


def _safe_reason(reason: str) -> str:
    normalized = _SAFE_REASON_RE.sub("-", str(reason or "manual").strip().lower()).strip("-_")
    return normalized or "manual"


def _backup_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _prune_managed_backups(backup_dir: Path, *, retain: int) -> list[Path]:
    """Delete only backups created by this module, keeping the newest N."""
    root = backup_dir.resolve()
    candidates = sorted(
        (path for path in backup_dir.glob("forven-*.db") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    removed: list[Path] = []
    for path in candidates[retain:]:
        resolved = path.resolve()
        if resolved.parent != root:
            continue
        try:
            resolved.unlink()
            removed.append(resolved)
        except OSError as exc:
            log.warning("Could not prune managed database backup %s: %s", resolved, exc)
    return removed


def create_managed_db_backup(
    reason: str,
    *,
    backup_root: str | Path | None = None,
    retain: int | None = None,
) -> Path:
    """Create a consistent database snapshot and enforce managed retention.

    Snapshots live under ``FORVEN_HOME/backups/database`` by default. Retention
    applies only to the ``forven-*.db`` files created here; legacy ``.bak`` files
    and operator-selected recovery directories are never touched.
    """
    backup_dir = Path(backup_root) if backup_root is not None else FORVEN_HOME / "backups" / "database"
    backup_dir = backup_dir.expanduser().resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"forven-{_safe_reason(reason)}-{_backup_timestamp()}.db"

    process_lock = FileLock(str(backup_dir / ".database-backup.lock"), timeout=60)
    with _BACKUP_LOCK, process_lock:
        try:
            created = backup_db(target)
        except Exception:
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
            raise
        _prune_managed_backups(backup_dir, retain=_retention_limit(retain))
    return created


__all__ = [
    "DEFAULT_DB_BACKUP_RETENTION",
    "create_managed_db_backup",
]
