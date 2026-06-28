"""Persist user feedback as PoseT5-compatible training examples."""
from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from filelock import FileLock

import config
from tsl.features.normalize import FEATURE_LAYOUT_VERSION, normalize_sequence
from tsl.feedback.schemas import ContributionKind, ContributionMeta
from tsl.privacy.schemas import CONSENT_VERSION

MANIFEST_COLUMNS = (
    "segment_id",
    "npy_path",
    "text",
    "feature_layout_version",
    "source",
    "split",
    "kind",
)

MIN_FRAMES = 12
MAX_FRAMES = 180
MAX_TEXT_TOKENS = 128
SOURCE_NAME = "user_feedback"
META_FILENAME = "meta.jsonl"
MANIFEST_FILENAME = "manifest.csv"
RETRAIN_LOG_FILENAME = "retrain_log.json"
RETRAIN_HISTORY_FILENAME = "retrain_history.jsonl"
LOCK_FILENAME = ".store.lock"


class ContributionValidationError(ValueError):
    """Raised when a submission fails validation."""


class DuplicateContributionError(ValueError):
    """Raised when an identical contribution already exists."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _count_tokens(text: str) -> int:
    return len(text.split())


def _dedup_hash(features: np.ndarray, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(features.tobytes())
    digest.update(text.strip().encode("utf-8"))
    return digest.hexdigest()


class ContributionStore:
    """Store normalized pose features and manifest rows from user feedback."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root = Path(root_dir or config.USER_CONTRIBUTIONS_DIR).resolve()
        self.features_dir = self.root / "features"
        self.videos_dir = self.root / "videos"
        self.manifest_path = self.root / MANIFEST_FILENAME
        self.meta_path = self.root / META_FILENAME
        self.retrain_log_path = self.root / RETRAIN_LOG_FILENAME
        self.retrain_history_path = self.root / RETRAIN_HISTORY_FILENAME
        self.lock_path = self.root / LOCK_FILENAME
        self._known_hashes: set[str] | None = None

    def ensure_dirs(self) -> None:
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            with self.manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()

    def _store_lock(self) -> FileLock:
        self.ensure_dirs()
        return FileLock(str(self.lock_path), timeout=30)

    def _load_known_hashes(self) -> set[str]:
        if self._known_hashes is not None:
            return self._known_hashes
        known: set[str] = set()
        if not self.meta_path.exists():
            self._known_hashes = known
            return known
        with self.meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                dedup = payload.get("dedup_hash")
                if isinstance(dedup, str) and dedup:
                    known.add(dedup)
        self._known_hashes = known
        return known

    def _invalidate_hash_cache(self) -> None:
        self._known_hashes = None

    def _validate_frames_and_text(self, frames: np.ndarray, text: str) -> None:
        if frames.ndim != 3 or frames.shape[1] != 543 or frames.shape[2] not in (3, 4):
            raise ContributionValidationError(
                f"frames must have shape (T, 543, 3) or (T, 543, 4); got {tuple(frames.shape)}"
            )
        if frames.shape[0] < MIN_FRAMES:
            raise ContributionValidationError(
                f"at least {MIN_FRAMES} frames required; got {frames.shape[0]}"
            )
        if frames.shape[0] > MAX_FRAMES:
            raise ContributionValidationError(
                f"at most {MAX_FRAMES} frames allowed; got {frames.shape[0]}"
            )
        cleaned = text.strip()
        if not cleaned:
            raise ContributionValidationError("text must not be empty")
        if _count_tokens(cleaned) > MAX_TEXT_TOKENS:
            raise ContributionValidationError(
                f"text exceeds {MAX_TEXT_TOKENS} tokens"
            )

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        raw = np.asarray(frames, dtype=np.float32)
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]
        return normalize_sequence(raw)

    def _next_segment_id(self, kind: ContributionKind) -> str:
        return f"uf_{kind}_{uuid.uuid4().hex[:12]}"

    def _append_manifest_row(
        self,
        *,
        segment_id: str,
        npy_rel: str,
        text: str,
        kind: ContributionKind,
    ) -> None:
        row = {
            "segment_id": segment_id,
            "npy_path": npy_rel,
            "text": text.strip(),
            "feature_layout_version": FEATURE_LAYOUT_VERSION,
            "source": SOURCE_NAME,
            "split": "train",
            "kind": kind,
        }
        with self.manifest_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writerow(row)

    def _append_meta(self, meta: ContributionMeta) -> None:
        payload = {
            "segment_id": meta.segment_id,
            "kind": meta.kind,
            "text": meta.text,
            "status": meta.status,
            "user_hash": meta.user_hash,
            "consent_version": meta.consent_version,
            "consent_scope": meta.consent_scope,
            "created_at": meta.created_at,
            "original_text": meta.original_text,
            "model": meta.model,
            "score": meta.score,
            "dedup_hash": meta.dedup_hash,
            "capture_quality": meta.capture_quality,
            "train_allowed": meta.train_allowed,
            "delete_requested": meta.delete_requested,
            "video_path": meta.video_path,
            "environment_tag": meta.environment_tag,
        }
        with self.meta_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def is_eligible_for_training(row: dict) -> bool:
        if row.get("delete_requested"):
            return False
        if row.get("train_allowed") is False:
            return False
        scopes = row.get("consent_scope")
        if scopes is not None and "model_improvement" not in scopes:
            return False
        return True

    def _environment_tag_from_capture_quality(
        self, capture_quality: dict | None
    ) -> dict | None:
        if not capture_quality:
            return None
        tag: dict = {}
        if capture_quality.get("camera_facing"):
            tag["camera_angle"] = capture_quality["camera_facing"]
        fps = capture_quality.get("fps")
        if fps is not None:
            bucket = "under-15"
            if fps >= 25:
                bucket = "25-30"
            elif fps >= 15:
                bucket = "15-24"
            tag["fps_bucket"] = bucket
        if capture_quality.get("lighting_ok") is True:
            tag["lighting"] = "indoor"
        elif capture_quality.get("lighting_ok") is False:
            tag["lighting"] = "low"
        return tag or None

    def _write_meta_rows(self, rows: list[dict]) -> None:
        lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        self.meta_path.write_text(lines, encoding="utf-8")

    def save_submission(
        self,
        *,
        frames: list | np.ndarray,
        text: str,
        kind: ContributionKind,
        user_hash: str,
        consent_version: str = CONSENT_VERSION,
        consent_scope: list[str] | None = None,
        original_text: str | None = None,
        model: str | None = None,
        score: float | None = None,
        capture_quality: dict | None = None,
    ) -> ContributionMeta:
        raw = np.asarray(frames, dtype=np.float32)
        self._validate_frames_and_text(raw, text)
        scopes = consent_scope or ["model_improvement"]

        with self._store_lock():
            self.ensure_dirs()
            features = self._normalize(raw)
            dedup = _dedup_hash(features, text)
            if dedup in self._load_known_hashes():
                raise DuplicateContributionError("duplicate contribution")

            segment_id = self._next_segment_id(kind)
            npy_name = f"{segment_id}.npy"
            npy_rel = f"features/{npy_name}"
            npy_path = self.features_dir / npy_name
            np.save(npy_path, features)

            created_at = _utc_now_iso()
            meta = ContributionMeta(
                segment_id=segment_id,
                kind=kind,
                text=text.strip(),
                status="pending",
                user_hash=user_hash,
                consent_version=consent_version,
                consent_scope=list(scopes),
                created_at=created_at,
                original_text=original_text,
                model=model,
                score=score,
                dedup_hash=dedup,
                capture_quality=capture_quality,
                environment_tag=self._environment_tag_from_capture_quality(capture_quality),
            )
            self._append_manifest_row(
                segment_id=segment_id,
                npy_rel=npy_rel,
                text=text,
                kind=kind,
            )
            self._append_meta(meta)
            self._invalidate_hash_cache()
            return meta

    def attach_video_path(self, segment_id: str, video_path: str) -> bool:
        with self._store_lock():
            rows = self.iter_meta()
            if not rows:
                return False
            updated = False
            new_rows: list[dict] = []
            for row in rows:
                if str(row.get("segment_id")) == segment_id:
                    row = {**row, "video_path": video_path}
                    updated = True
                new_rows.append(row)
            if updated:
                self._write_meta_rows(new_rows)
            return updated

    def save_correction(
        self,
        *,
        frames: list | np.ndarray,
        predicted_text: str,
        corrected_text: str,
        user_hash: str,
        consent_version: str = CONSENT_VERSION,
        consent_scope: list[str] | None = None,
        model: str | None = None,
        score: float | None = None,
        capture_quality: dict | None = None,
    ) -> ContributionMeta:
        return self.save_submission(
            frames=frames,
            text=corrected_text,
            kind="correction",
            user_hash=user_hash,
            consent_version=consent_version,
            consent_scope=consent_scope,
            original_text=predicted_text,
            model=model,
            score=score,
            capture_quality=capture_quality,
        )

    def save_teach(
        self,
        *,
        frames: list | np.ndarray,
        label_text: str,
        user_hash: str,
        consent_version: str = CONSENT_VERSION,
        consent_scope: list[str] | None = None,
        capture_quality: dict | None = None,
    ) -> ContributionMeta:
        return self.save_submission(
            frames=frames,
            text=label_text,
            kind="teach",
            user_hash=user_hash,
            consent_version=consent_version,
            consent_scope=consent_scope,
            capture_quality=capture_quality,
        )

    def iter_meta(self) -> list[dict]:
        if not self.meta_path.exists():
            return []
        rows: list[dict] = []
        with self.meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def pending_segment_ids(self) -> set[str]:
        return {
            str(row["segment_id"])
            for row in self.iter_meta()
            if row.get("status") == "pending"
            and row.get("segment_id")
            and self.is_eligible_for_training(row)
        }

    def pending_count(self) -> int:
        return sum(
            1
            for row in self.iter_meta()
            if row.get("status") == "pending" and self.is_eligible_for_training(row)
        )

    def total_count(self) -> int:
        return len(self.iter_meta())

    def mark_used(self, segment_ids: set[str] | None = None) -> int:
        with self._store_lock():
            rows = self.iter_meta()
            if not rows:
                return 0
            updated = 0
            new_rows: list[dict] = []
            for row in rows:
                seg_id = str(row.get("segment_id", ""))
                if row.get("status") == "pending" and (segment_ids is None or seg_id in segment_ids):
                    row = {**row, "status": "used"}
                    updated += 1
                new_rows.append(row)
            self._write_meta_rows(new_rows)
            self._invalidate_hash_cache()
            return updated

    def rebuild_manifest(self) -> int:
        """Regenerate manifest.csv from eligible meta rows."""
        with self._store_lock():
            return self._rebuild_manifest_unlocked()

    def _rebuild_manifest_unlocked(self) -> int:
        rows = [
            row
            for row in self.iter_meta()
            if self.is_eligible_for_training(row)
        ]
        with self.manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                segment_id = str(row.get("segment_id", ""))
                if not segment_id:
                    continue
                writer.writerow(
                    {
                        "segment_id": segment_id,
                        "npy_path": f"features/{segment_id}.npy",
                        "text": str(row.get("text", "")).strip(),
                        "feature_layout_version": FEATURE_LAYOUT_VERSION,
                        "source": SOURCE_NAME,
                        "split": "train",
                        "kind": str(row.get("kind", "teach")),
                    }
                )
        return len(rows)

    def delete_by_user_hash(self, user_hash: str) -> int:
        """Remove all contributions and videos for a pseudonymous user."""
        with self._store_lock():
            rows = self.iter_meta()
            if not rows:
                return 0
            removed = 0
            kept: list[dict] = []
            for row in rows:
                if row.get("user_hash") == user_hash:
                    segment_id = str(row.get("segment_id", ""))
                    if segment_id:
                        npy = self.features_dir / f"{segment_id}.npy"
                        if npy.is_file():
                            npy.unlink()
                        video_rel = row.get("video_path")
                        if video_rel:
                            video_path = self.root / str(video_rel)
                            if video_path.is_file():
                                video_path.unlink()
                        enc_path = self.videos_dir / f"{segment_id}.webm.enc"
                        if enc_path.is_file():
                            enc_path.unlink()
                    removed += 1
                else:
                    kept.append(row)
            self._write_meta_rows(kept)
            self._rebuild_manifest_unlocked()
            self._invalidate_hash_cache()
            return removed

    def mark_train_disallowed(self, user_hash: str) -> int:
        """Flag remaining rows as not trainable after consent withdrawal."""
        with self._store_lock():
            rows = self.iter_meta()
            if not rows:
                return 0
            updated = 0
            new_rows: list[dict] = []
            for row in rows:
                if row.get("user_hash") == user_hash:
                    row = {**row, "train_allowed": False, "delete_requested": True}
                    updated += 1
                new_rows.append(row)
            self._write_meta_rows(new_rows)
            self._rebuild_manifest_unlocked()
            self._invalidate_hash_cache()
            return updated

    def get_segment_meta(self, segment_id: str) -> dict | None:
        for row in self.iter_meta():
            if str(row.get("segment_id")) == segment_id:
                return row
        return None

    def _read_retrain_log(self) -> dict | None:
        if not self.retrain_log_path.exists():
            return None
        try:
            payload = json.loads(self.retrain_log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def last_retrain_at(self) -> str | None:
        payload = self._read_retrain_log()
        if not payload or payload.get("status") != "completed":
            return None
        value = payload.get("completed_at")
        return str(value) if value else None

    def last_attempt_at(self) -> str | None:
        payload = self._read_retrain_log()
        if payload:
            for key in ("last_attempt_at", "completed_at"):
                value = payload.get(key)
                if value:
                    return str(value)
        if not self.retrain_history_path.exists():
            return None
        last_line = ""
        with self.retrain_history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if not last_line:
            return None
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            return None
        value = payload.get("completed_at")
        return str(value) if value else None

    def feedback_version(self) -> str | None:
        payload = self._read_retrain_log()
        if not payload or payload.get("status") != "completed":
            return None
        value = payload.get("model_version")
        return str(value) if value else None

    def write_retrain_log(self, payload: dict) -> None:
        self.ensure_dirs()
        self.retrain_log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def append_retrain_history(self, payload: dict) -> None:
        self.ensure_dirs()
        with self.retrain_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        return {
            "pending_count": self.pending_count(),
            "total_count": self.total_count(),
            "last_retrain_at": self.last_retrain_at(),
            "last_attempt_at": self.last_attempt_at(),
            "feedback_version": self.feedback_version(),
        }
