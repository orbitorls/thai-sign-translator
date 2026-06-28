"""Encrypted storage for opt-in research video blobs."""
from __future__ import annotations

import base64
import base64
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from filelock import FileLock

import config

LOCK_FILENAME = ".video.lock"
_DEV_FALLBACK_FERNET_KEY = base64.urlsafe_b64encode(b"\x00" * 32)


def _fernet() -> Fernet:
    key = config.TSL_DATA_ENCRYPTION_KEY
    if key:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    return Fernet(_DEV_FALLBACK_FERNET_KEY)


class VideoStore:
    """Encrypt and persist user research video clips."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root = Path(root_dir or config.USER_CONTRIBUTIONS_DIR).resolve()
        self.videos_dir = self.root / "videos"
        self.lock_path = self.root / LOCK_FILENAME

    def ensure_dirs(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def _store_lock(self) -> FileLock:
        self.ensure_dirs()
        return FileLock(str(self.lock_path), timeout=30)

    def encrypted_path(self, segment_id: str) -> Path:
        return self.videos_dir / f"{segment_id}.webm.enc"

    def save_encrypted(self, segment_id: str, payload: bytes) -> str:
        if len(payload) > config.MAX_FEEDBACK_VIDEO_BYTES:
            raise ValueError(
                f"video exceeds {config.MAX_FEEDBACK_VIDEO_BYTES} bytes"
            )
        rel = f"videos/{segment_id}.webm.enc"
        path = self.encrypted_path(segment_id)
        token = _fernet().encrypt(payload)
        with self._store_lock():
            self.ensure_dirs()
            path.write_bytes(token)
        return rel

    def delete_encrypted(self, segment_id: str) -> bool:
        path = self.encrypted_path(segment_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def decrypt(self, segment_id: str) -> bytes:
        path = self.encrypted_path(segment_id)
        if not path.is_file():
            raise FileNotFoundError(segment_id)
        try:
            return _fernet().decrypt(path.read_bytes())
        except InvalidToken as exc:
            raise ValueError("video decryption failed") from exc

    def purge_expired(self, retention_days: int | None = None) -> int:
        days = retention_days if retention_days is not None else config.VIDEO_RETENTION_DAYS
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(days, 1))
        removed = 0
        if not self.videos_dir.is_dir():
            return 0
        with self._store_lock():
            for path in self.videos_dir.glob("*.webm.enc"):
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed


def generate_fernet_key() -> str:
    """Helper for ops/docs: returns a url-safe base64 Fernet key."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
