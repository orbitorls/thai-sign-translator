from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the latest Colab checkpoint to a Kaggle dataset."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-title", default="Thai Sign Ckpt")
    parser.add_argument("--staging-dir", default="/content/kaggle_ckpt_publish")
    parser.add_argument("--interval-sec", type=int, default=120)
    parser.add_argument("--verify-retries", type=int, default=12)
    parser.add_argument("--verify-delay-sec", type=int, default=10)
    parser.add_argument("--state-path", default="")
    parser.add_argument("--once", action="store_true")
    return parser


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.split("step", 1)[1])


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("ckpt_step*.pt"), key=_checkpoint_step)
    if not checkpoints:
        return None
    return checkpoints[-1]


def _best_checkpoint(checkpoint_dir: Path) -> Path | None:
    best_path: Path | None = None
    best_val = float("-inf")
    for ckpt in checkpoint_dir.glob("ckpt_step*.pt"):
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        val = payload.get("metrics", {}).get("val_chrf")
        if val is None:
            continue
        if float(val) > best_val:
            best_val = float(val)
            best_path = ckpt
    return best_path


def _ensure_metadata(staging_dir: Path, dataset_id: str, dataset_title: str) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = staging_dir / "dataset-metadata.json"
    if metadata_path.exists():
        return
    metadata = {
        "title": dataset_title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)

def _sync_staging(checkpoint_dir: Path, staging_dir: Path, latest_ckpt: Path) -> None:
    best_ckpt = _best_checkpoint(checkpoint_dir)
    keep_names = {latest_ckpt.name}
    if best_ckpt is not None:
        keep_names.add(best_ckpt.name)
    for old in staging_dir.glob("ckpt_step*.pt"):
        if old.name not in keep_names:
            old.unlink()
    for ckpt in [latest_ckpt, best_ckpt]:
        if ckpt is None:
            continue
        target_ckpt = staging_dir / ckpt.name
        if not target_ckpt.exists() or target_ckpt.stat().st_size != ckpt.stat().st_size:
            shutil.copy2(ckpt, target_ckpt)
    _copy_if_exists(checkpoint_dir / "train.log", staging_dir / "train.log")
    _copy_if_exists(checkpoint_dir / "train_metrics.json", staging_dir / "train_metrics.json")
    _copy_if_exists(checkpoint_dir / "launch.json", staging_dir / "launch.json")
    (staging_dir / "latest_checkpoint.txt").write_text(latest_ckpt.name, encoding="utf-8")
    if best_ckpt is not None:
        (staging_dir / "best_checkpoint.txt").write_text(best_ckpt.name, encoding="utf-8")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def _publish_dataset(staging_dir: Path, latest_ckpt: Path) -> None:
    create_result = _run(
        [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(staging_dir),
            "--dir-mode",
            "skip",
            "-q",
        ]
    )
    if create_result.returncode == 0:
        return

    version_result = _run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(staging_dir),
            "-m",
            f"Sync {latest_ckpt.name}",
            "--dir-mode",
            "skip",
            "-q",
        ]
    )
    if version_result.returncode != 0:
        raise RuntimeError(version_result.stderr.strip() or version_result.stdout.strip())


def _dataset_contains_checkpoint(dataset_id: str, checkpoint_name: str) -> bool:
    result = _run(["kaggle", "datasets", "files", "-d", dataset_id])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return checkpoint_name in result.stdout


def _verify_published_checkpoint(
    dataset_id: str,
    checkpoint_name: str,
    *,
    retries: int,
    delay_sec: int,
) -> None:
    for attempt in range(1, retries + 1):
        if _dataset_contains_checkpoint(dataset_id, checkpoint_name):
            return
        if attempt < retries:
            time.sleep(delay_sec)
    raise RuntimeError(f"Published dataset does not expose {checkpoint_name} yet: {dataset_id}")


def _write_state(
    state_path: Path,
    *,
    latest_checkpoint: str | None,
    pending_checkpoint: str | None,
) -> None:
    payload = {
        "latest_checkpoint": latest_checkpoint,
        "pending_checkpoint": pending_checkpoint,
        "published_at": time.time(),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir)
    staging_dir = Path(args.staging_dir)
    state_path = Path(args.state_path) if args.state_path else staging_dir / "publisher_state.json"

    dataset_id = args.dataset_id
    dataset_title = args.dataset_title

    _ensure_metadata(staging_dir, dataset_id, dataset_title)

    last_published = ""
    pending_checkpoint = ""
    while True:
        latest_ckpt = _latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            if pending_checkpoint and pending_checkpoint != latest_ckpt.name:
                pending_step = _checkpoint_step(Path(pending_checkpoint))
                latest_step = _checkpoint_step(latest_ckpt)
                if latest_step > pending_step:
                    print(
                        f"[publisher] newer checkpoint available; replacing pending {pending_checkpoint} with {latest_ckpt.name}",
                        flush=True,
                    )
                    pending_checkpoint = ""
            if pending_checkpoint:
                print(f"[publisher] verifying pending checkpoint: {pending_checkpoint}", flush=True)
                try:
                    _verify_published_checkpoint(
                        dataset_id,
                        pending_checkpoint,
                        retries=args.verify_retries,
                        delay_sec=args.verify_delay_sec,
                    )
                except RuntimeError:
                    if args.once:
                        return 1
                    time.sleep(args.interval_sec)
                    continue
                last_published = pending_checkpoint
                pending_checkpoint = ""
                print(f"[publisher] checkpoint visible in dataset: {last_published}", flush=True)
                _write_state(
                    state_path,
                    latest_checkpoint=last_published,
                    pending_checkpoint=None,
                )

            if latest_ckpt.name != last_published:
                print(f"[publisher] staging checkpoint: {latest_ckpt.name}", flush=True)
                _sync_staging(checkpoint_dir, staging_dir, latest_ckpt)
                print(f"[publisher] publishing checkpoint: {latest_ckpt.name}", flush=True)
                _publish_dataset(staging_dir, latest_ckpt)
                try:
                    _verify_published_checkpoint(
                        dataset_id,
                        latest_ckpt.name,
                        retries=args.verify_retries,
                        delay_sec=args.verify_delay_sec,
                    )
                except RuntimeError:
                    pending_checkpoint = latest_ckpt.name
                    print(
                        f"[publisher] checkpoint not visible yet; pending retry: {pending_checkpoint}",
                        flush=True,
                    )
                    _write_state(
                        state_path,
                        latest_checkpoint=last_published or None,
                        pending_checkpoint=pending_checkpoint,
                    )
                    if args.once:
                        return 1
                    time.sleep(args.interval_sec)
                    continue

                last_published = latest_ckpt.name
                pending_checkpoint = ""
                print(f"[publisher] checkpoint visible in dataset: {last_published}", flush=True)
                _write_state(
                    state_path,
                    latest_checkpoint=last_published,
                    pending_checkpoint=None,
                )

        if args.once:
            return 0
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
