from __future__ import annotations

from pathlib import Path


def test_managed_backup_uses_online_primitive_and_prunes_only_managed_files(tmp_path, monkeypatch):
    import forven.backups as backups

    timestamps = iter([
        "20260101T000001000000Z",
        "20260101T000002000000Z",
        "20260101T000003000000Z",
        "20260101T000004000000Z",
    ])
    created: list[Path] = []

    def fake_backup(destination):
        target = Path(destination)
        target.write_text("sqlite snapshot", encoding="utf-8")
        created.append(target)
        return target

    monkeypatch.setattr(backups, "backup_db", fake_backup)
    monkeypatch.setattr(backups, "_backup_timestamp", lambda: next(timestamps))
    legacy = tmp_path / "forven.db.bak-legacy"
    legacy.write_text("preserve", encoding="utf-8")

    for _ in range(4):
        backups.create_managed_db_backup("reset paper", backup_root=tmp_path, retain=2)

    remaining = sorted(tmp_path.glob("forven-*.db"))
    assert len(remaining) == 2
    assert remaining == created[-2:]
    assert legacy.read_text(encoding="utf-8") == "preserve"


def test_managed_backup_removes_partial_target_when_backup_fails(tmp_path, monkeypatch):
    import forven.backups as backups

    def fail_backup(destination):
        Path(destination).write_text("partial", encoding="utf-8")
        raise RuntimeError("backup failed")

    monkeypatch.setattr(backups, "backup_db", fail_backup)
    monkeypatch.setattr(backups, "_backup_timestamp", lambda: "20260101T000001000000Z")

    try:
        backups.create_managed_db_backup("failure", backup_root=tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "backup failed"
    else:
        raise AssertionError("backup failure should propagate")

    assert list(tmp_path.glob("forven-*.db")) == []
