"""Append-only consent registry."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

import config
from tsl.privacy.schemas import (
    ALL_CONSENT_SCOPES,
    CONSENT_VERSION,
    ConsentRecord,
    ConsentScope,
    ConsentSource,
)

REGISTRY_FILENAME = "consent_registry.jsonl"
LOCK_FILENAME = ".consent.lock"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ConsentStore:
    """Persist consent grant/withdrawal events per pseudonymous user."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root = Path(root_dir or config.CONSENT_REGISTRY_DIR).resolve()
        self.registry_path = self.root / REGISTRY_FILENAME
        self.lock_path = self.root / LOCK_FILENAME

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _store_lock(self) -> FileLock:
        self.ensure_dirs()
        return FileLock(str(self.lock_path), timeout=30)

    def _append_record(self, record: ConsentRecord) -> None:
        payload = {
            "user_hash": record.user_hash,
            "consent_version": record.consent_version,
            "scope": record.scope,
            "granted": record.granted,
            "recorded_at": record.recorded_at,
            "source": record.source,
        }
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def iter_records(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        rows: list[dict] = []
        with self.registry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def record_consent(
        self,
        *,
        user_hash: str,
        scope: ConsentScope,
        granted: bool,
        source: ConsentSource,
        consent_version: str = CONSENT_VERSION,
    ) -> ConsentRecord:
        record = ConsentRecord(
            user_hash=user_hash,
            consent_version=consent_version,
            scope=scope,
            granted=granted,
            recorded_at=_utc_now_iso(),
            source=source,
        )
        with self._store_lock():
            self.ensure_dirs()
            self._append_record(record)
        return record

    def current_status(self, user_hash: str) -> dict[str, bool]:
        """Latest granted state per scope for a user."""
        status: dict[str, bool] = {scope: False for scope in ALL_CONSENT_SCOPES}
        for row in self.iter_records():
            if row.get("user_hash") != user_hash:
                continue
            scope = row.get("scope")
            if scope in status:
                status[str(scope)] = bool(row.get("granted"))
        return status

    def has_scope(self, user_hash: str, scope: ConsentScope) -> bool:
        return self.current_status(user_hash).get(scope, False)

    def record_withdrawal(
        self,
        *,
        user_hash: str,
        scopes: list[ConsentScope],
        consent_version: str = CONSENT_VERSION,
    ) -> list[ConsentRecord]:
        records: list[ConsentRecord] = []
        with self._store_lock():
            self.ensure_dirs()
            for scope in scopes:
                record = ConsentRecord(
                    user_hash=user_hash,
                    consent_version=consent_version,
                    scope=scope,
                    granted=False,
                    recorded_at=_utc_now_iso(),
                    source="withdrawal",
                )
                self._append_record(record)
                records.append(record)
        return records
