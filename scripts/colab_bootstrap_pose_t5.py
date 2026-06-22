from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_CONFIG_PATH = "/content/thai-sign-colab-config.json"
DEFAULT_REPO_ROOT = "/content/thai-sign-translator"
DEFAULT_REPO_ZIP = "/content/thai-sign-code.zip"
DEFAULT_ACCESS_TOKEN_PATH = "/content/access_token"
_RESET_PATTERNS = (
    "ckpt_step*.pt",
    "*.tmp",
    "train.log",
    "train.pid",
    "train_metrics.json",
    "launch.json",
    "publisher.log",
    "publisher.pid",
    "publisher_state.json",
    "latest_checkpoint.txt",
    "best_checkpoint.txt",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pose_encoder.pt",
    "pose_t5_config.json",
)
_RESUME_RESTORE_STALE_PATTERNS = tuple(
    pattern for pattern in _RESET_PATTERNS if pattern not in {"ckpt_step*.pt", "*.tmp"}
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap managed PoseToTextT5 training inside a Colab VM."
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--repo-zip", default=DEFAULT_REPO_ZIP)
    parser.add_argument("--access-token-path", default=DEFAULT_ACCESS_TOKEN_PATH)
    return parser


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in config: {path}")
    return data


def _ensure_repo_available(repo_root: str, repo_zip: str) -> Path:
    repo_root_path = Path(repo_root)
    train_script = repo_root_path / "scripts" / "colab_train_pose_t5.py"
    zip_path = Path(repo_zip)
    if zip_path.is_file():
        if repo_root_path.exists():
            shutil.rmtree(repo_root_path)
        repo_root_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(repo_root_path)
        if not train_script.is_file():
            raise FileNotFoundError(f"Missing remote train entrypoint after extract: {train_script}")
        return repo_root_path
    if train_script.is_file():
        return repo_root_path
    raise FileNotFoundError(f"Repo archive not found and train entrypoint is missing: {repo_zip}")


def _ensure_kaggle_access_token(access_token_path: str) -> None:
    source = Path(access_token_path)
    if not source.is_file():
        raise FileNotFoundError(f"Kaggle access token not found: {access_token_path}")

    kaggle_dir = Path("/root/.kaggle")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = kaggle_dir / "access_token"
    shutil.copy2(source, target)
    os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)

    legacy = kaggle_dir / "kaggle.json"
    if legacy.exists():
        legacy.unlink()


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _ensure_pip_package(spec: str) -> None:
    _run([sys.executable, "-m", "pip", "install", "-q", spec])


def _download_dataset(slug: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(target_dir), "--unzip"])


def _materialize_archived_dataset(target_dir: Path) -> Path:
    manifest = target_dir / "manifest.csv"
    features_archive = target_dir / "features.zip"
    if not manifest.is_file() or not features_archive.is_file():
        return _normalize_feature_layout(target_dir)
    feature_root = target_dir / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(features_archive, "r") as archive:
        archive.extractall(feature_root)
    return _normalize_feature_layout(target_dir)


def _normalize_feature_layout(target_dir: Path) -> Path:
    nested_root = target_dir / "features" / "features"
    feature_root = target_dir / "features"
    if nested_root.is_dir():
        feature_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(nested_root.rglob("*")):
            if not path.is_file():
                continue
            destination = feature_root / path.relative_to(nested_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
        shutil.rmtree(nested_root)
    return target_dir


def _manifest_has_resolvable_features(target_dir: Path, max_rows: int = 16) -> bool:
    manifest = target_dir / "manifest.csv"
    if not manifest.is_file():
        return False
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            raw_value = str(row.get("npy_path", "")).replace("\\", "/").strip()
            if not raw_value:
                continue
            candidate = Path(raw_value)
            if not candidate.is_absolute():
                candidate = target_dir / raw_value
            if candidate.is_file():
                return True
            if index + 1 >= max_rows:
                break
    return False


def _ensure_manifest_dataset(target_dir: Path, slug: str) -> Path:
    manifest = target_dir / "manifest.csv"
    if manifest.is_file():
        _materialize_archived_dataset(target_dir)
        if _manifest_has_resolvable_features(target_dir):
            return target_dir
        shutil.rmtree(target_dir)
    _ensure_pip_package("kaggle>=2.2.2")
    _download_dataset(slug, target_dir)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found after downloading {slug}: {manifest}")
    _materialize_archived_dataset(target_dir)
    if not _manifest_has_resolvable_features(target_dir):
        raise RuntimeError(f"Downloaded dataset {slug} but manifest feature paths are still unresolved in {target_dir}")
    return target_dir


def _extract_local_data_bundle(bundle_zip: str, target_dir: Path) -> Path:
    bundle_path = Path(bundle_zip)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Local data bundle not found: {bundle_zip}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "r") as archive:
        archive.extractall(target_dir)
    manifest = target_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found after extracting local data bundle {bundle_zip}: {manifest}"
        )
    return _materialize_archived_dataset(target_dir)


def _restore_resume_checkpoint(out_dir: Path, checkpoint_slug: str) -> None:
    if list(out_dir.glob("ckpt_step*.pt")):
        print("[resume] Checkpoints already present in out_dir; skipping dataset restore.", flush=True)
        return
    _ensure_pip_package("kaggle>=2.2.2")
    try:
        print(f"[resume] Downloading checkpoint dataset: {checkpoint_slug}", flush=True)
        _download_dataset(checkpoint_slug, out_dir)
    except subprocess.CalledProcessError:
        print(f"[resume] Checkpoint dataset not available yet: {checkpoint_slug}", flush=True)
        return

    restored = sorted(path.name for path in out_dir.glob("ckpt_step*.pt"))
    if restored:
        print(f"[resume] Restored checkpoints: {restored}", flush=True)
    else:
        print(f"[resume] Download finished but no ckpt_step*.pt files were found in {out_dir}", flush=True)


def _prune_restored_resume_artifacts(out_dir: Path) -> list[str]:
    removed: list[str] = []
    for pattern in _RESUME_RESTORE_STALE_PATTERNS:
        for path in out_dir.glob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            removed.append(path.name)
    return sorted(set(removed))


def _reset_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in _RESET_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _build_train_command(repo_root: Path, config: dict, data_roots: list[str], out_dir: Path) -> list[str]:
    train_script = repo_root / "scripts" / "colab_train_pose_t5.py"
    cmd = [
        sys.executable,
        "-u",
        str(train_script),
        "--repo-root",
        str(repo_root),
        "--repo-zip",
        str(config.get("repo_zip", DEFAULT_REPO_ZIP)),
        "--out-dir",
        str(out_dir),
        "--data-roots",
        ",".join(data_roots),
        "--base-model",
        str(config.get("base_model", "google/mt5-small")),
        "--epochs",
        str(config.get("epochs", 300)),
        "--batch-size",
        str(config.get("batch_size", 8)),
        "--grad-accum",
        str(config.get("grad_accum", 4)),
        "--lr",
        str(config.get("lr", 1e-4)),
        "--dropout",
        str(config.get("dropout", 0.3)),
        "--weight-decay",
        str(config.get("weight_decay", 0.05)),
        "--max-src-len",
        str(config.get("max_src_len", 512)),
        "--downsample-factor",
        str(config.get("downsample_factor", 4)),
        "--num-encoder-layers",
        str(config.get("num_encoder_layers", 2)),
        "--amp",
        str(config.get("amp", "auto")),
        "--resume",
        str(config.get("resume", "auto")),
        "--max-runtime-min",
        str(config.get("max_runtime_min", 690)),
        "--keep-checkpoints",
        str(config.get("keep_checkpoints", 3)),
        "--seed",
        str(config.get("seed", 42)),
        "--eval-steps",
        str(config.get("eval_steps", 100)),
        "--checkpoint-steps",
        str(config.get("checkpoint_steps", 500)),
        "--early-stopping-patience",
        str(config.get("early_stopping_patience", 10)),
        "--early-stopping-min-delta",
        str(config.get("early_stopping_min_delta", 0.0)),
        "--early-stopping-metric",
        str(config.get("early_stopping_metric", "val_chrf")),
        "--num-workers",
        str(config.get("num_workers", 2)),
        "--required-sources",
        str(config.get("required_sources", "tsl51,thaisignvis")),
        "--manifest-quality-sources",
        str(config.get("manifest_quality_sources", "")),
        "--fail-on-manifest-quality",
        str(config.get("fail_on_manifest_quality", "true")),
        "--allow-noop-resume",
        str(config.get("allow_noop_resume", "false")),
    ]
    if bool(config.get("reset_progress_history", False)):
        cmd.append("--reset-progress-history")
    return cmd


def _start_remote_publisher(repo_root: Path, config: dict, out_dir: Path) -> None:
    checkpoint_slug = str(config.get("checkpoint_dataset_slug", "")).strip()
    if not checkpoint_slug:
        return

    publisher_script = repo_root / "scripts" / "colab_publish_checkpoints.py"
    if not publisher_script.is_file():
        return

    publisher_log = out_dir / "publisher.log"
    publisher_pid = out_dir / "publisher.pid"
    publisher_cmd = [
        sys.executable,
        "-u",
        str(publisher_script),
        "--checkpoint-dir",
        str(out_dir),
        "--dataset-id",
        checkpoint_slug,
        "--dataset-title",
        str(config.get("checkpoint_dataset_title", "Thai Sign Ckpt")),
        "--staging-dir",
        str(config.get("checkpoint_publish_dir", "/content/kaggle_ckpt_publish")),
        "--interval-sec",
        str(config.get("checkpoint_publish_interval_sec", 120)),
    ]

    with publisher_log.open("ab", buffering=0) as handle:
        handle.write(b"[publisher] starting background publish loop\n")
        proc = subprocess.Popen(
            publisher_cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    publisher_pid.write_text(str(proc.pid), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = _build_parser()
    args, _unknown = parser.parse_known_args(argv)
    config = _load_config(args.config_path)
    repo_root = _ensure_repo_available(args.repo_root, args.repo_zip)
    _ensure_kaggle_access_token(args.access_token_path)

    data_roots = []
    for folder_name, slug in config.get("data_datasets", {}).items():
        target_dir = Path("/content/kaggle") / folder_name
        data_roots.append(str(_ensure_manifest_dataset(target_dir, slug)))
    for bundle in config.get("local_data_bundles", []):
        if not isinstance(bundle, dict):
            continue
        bundle_zip = str(bundle.get("remote_zip", "")).strip()
        out_dir = str(bundle.get("out_dir", "")).strip()
        if not bundle_zip or not out_dir:
            continue
        data_roots.append(str(_extract_local_data_bundle(bundle_zip, Path(out_dir))))

    out_dir = Path(str(config.get("out_dir", "/content/checkpoints/pose_t5_v3_colab")))
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_mode = str(config.get("resume", "auto")).strip().lower()
    if bool(config.get("reset_out_dir", False)):
        _reset_out_dir(out_dir)

    checkpoint_slug = str(config.get("checkpoint_dataset_slug", "")).strip()
    if checkpoint_slug and resume_mode != "none":
        _restore_resume_checkpoint(out_dir, checkpoint_slug)
        if bool(config.get("require_resume_checkpoint", False)) and not list(out_dir.glob("ckpt_step*.pt")):
            raise RuntimeError(
                "Resume was requested, but no checkpoint files were restored into "
                f"{out_dir}. Aborting to avoid an unintended scratch run."
            )
        removed = _prune_restored_resume_artifacts(out_dir)
        if removed:
            print(f"[resume] Removed stale restored artifacts: {removed}", flush=True)

    cmd = _build_train_command(repo_root, config, data_roots, out_dir)
    log_path = out_dir / "train.log"
    pid_path = out_dir / "train.pid"
    launch_path = out_dir / "launch.json"

    with log_path.open("ab", buffering=0) as handle:
        handle.write(b"[launcher] starting background training\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    launch_path.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "cmd": cmd,
                "data_roots": data_roots,
                "checkpoint_dataset_slug": checkpoint_slug,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _start_remote_publisher(repo_root, config, out_dir)

    result = {
        "status": "started",
        "pid": proc.pid,
        "out_dir": str(out_dir),
        "data_roots": data_roots,
        "checkpoint_dataset_slug": checkpoint_slug,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
