from __future__ import annotations

import argparse
import atexit
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path
import shlex

import torch


DEFAULT_DATASET_ID = "orbitorls/thai-sign-ckpt"
DEFAULT_DATASET_TITLE = "Thai Sign Ckpt"
_FINAL_EXPORT_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pose_encoder.pt",
    "pose_t5_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
}
_EXEC_TEXT_FALLBACK_FILES = {
    "train.log",
    "train_metrics.json",
    "launch.json",
    "latest_checkpoint.txt",
    "best_checkpoint.txt",
    "publisher.log",
}


def _sanitize_exec_text_output(text: str) -> str:
    compact_suffix = re.match(r"(?s)^(.*[}\]])(\d+)\r?\n?$", text)
    if compact_suffix:
        return compact_suffix.group(1)
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    stripped = lines[-1].strip()
    if stripped.isdigit():
        return "".join(lines[:-1])
    return text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirror remote Colab checkpoints locally and publish resumable Kaggle dataset versions."
    )
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--remote-out-dir", required=True)
    parser.add_argument("--local-mirror-dir", required=True)
    parser.add_argument("--kaggle-dataset-dir", required=True)
    parser.add_argument("--kaggle-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--kaggle-dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument("--colab-bin", default="/root/.venvs/colabcli/bin/colab")
    parser.add_argument("--interval-sec", type=int, default=90)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    return parser


def parse_ls_output(text: str) -> list[str]:
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[colab]"):
            continue
        if line.endswith("/"):
            continue
        names.append(line)
    return names


def parse_status_output(text: str) -> dict[str, str | None]:
    status_line = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            status_line = line
            break

    result: dict[str, str | None] = {
        "raw": status_line or None,
        "session": None,
        "hardware": None,
        "variant": None,
        "status": None,
        "last_execution": None,
    }
    if not status_line:
        return result
    if status_line.startswith("[colab] Session ") and " not found." in status_line:
        return result

    head, _, tail = status_line.partition(" | ")
    session_match = re.match(r"^\[(?P<session>[^\]]+)\]\s+(?P<instance>.+)$", head)
    if session_match:
        result["session"] = session_match.group("session")
        result["instance"] = session_match.group("instance")

    hardware_match = re.search(r"Hardware:\s*([^|]+)", tail)
    variant_match = re.search(r"Variant:\s*([^|]+)", tail)
    state_match = re.search(r"Status:\s*([^|]+)", tail)
    if hardware_match:
        result["hardware"] = hardware_match.group(1).strip()
    if variant_match:
        result["variant"] = variant_match.group(1).strip()
    if state_match:
        result["status"] = state_match.group(1).strip()

    for raw_line in text.splitlines()[1:]:
        line = raw_line.strip()
        if line.startswith("Last Execution:"):
            result["last_execution"] = line.split(":", 1)[1].strip()
            break
    return result


def checkpoint_step(name: str) -> int:
    return int(Path(name).stem.split("step", 1)[1])


def latest_checkpoint_name(names: list[str]) -> str | None:
    checkpoints = [name for name in names if name.startswith("ckpt_step") and name.endswith(".pt")]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def best_checkpoint_name(paths: list[Path]) -> str | None:
    best_name: str | None = None
    best_val = float("-inf")
    for path in paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        val = payload.get("metrics", {}).get("val_chrf")
        if val is None:
            continue
        if float(val) > best_val:
            best_val = float(val)
            best_name = path.name
    return best_name


def ensure_dataset_metadata(dataset_dir: Path, dataset_id: str, title: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "dataset-metadata.json"
    if metadata_path.exists():
        return
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_local(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _run_colab(colab_bin: str, args: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    quoted = " ".join(shlex.quote(part) for part in [colab_bin, *args])
    return _run_local(["wsl.exe", "bash", "-lc", quoted], capture_output=capture_output)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = _run_local(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True)
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _release_sync_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
    except OSError:
        pass


def _acquire_sync_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            pid = int(payload.get("pid", 0) or 0)
            if pid and _pid_is_running(pid):
                raise RuntimeError(f"Another sync process is already running for {lock_path.parent} (pid {pid})")
            lock_path.unlink(missing_ok=True)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, handle)
            atexit.register(_release_sync_lock, lock_path)
            return
    raise RuntimeError(f"Unable to acquire sync lock at {lock_path}")


def _to_wsl_path(path: Path | str) -> str:
    raw_path = os.fspath(path)
    result = _run_local(
        ["wsl.exe", "bash", "-lc", f"wslpath -a {shlex.quote(raw_path)}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"wslpath failed for {raw_path}")
    return result.stdout.strip()


def _list_remote_files(colab_bin: str, session_name: str, remote_out_dir: str) -> list[str]:
    result = _run_colab(
        colab_bin,
        ["ls", "-s", session_name, remote_out_dir],
        capture_output=True,
    )
    if result.returncode == 0:
        return parse_ls_output(result.stdout)
    fallback = _list_remote_files_via_exec(colab_bin, session_name, remote_out_dir)
    if fallback is not None:
        return fallback
    raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "colab ls failed")


def _list_remote_files_via_exec(colab_bin: str, session_name: str, remote_out_dir: str) -> list[str] | None:
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            script_path = Path(handle.name)
            handle.write(
                "from pathlib import Path\n"
                f"root = Path({remote_out_dir!r})\n"
                "if not root.exists():\n"
                "    raise SystemExit(3)\n"
                "for child in sorted(root.iterdir()):\n"
                "    suffix = '/' if child.is_dir() else ''\n"
                "    print(child.name + suffix)\n"
            )
        script_wsl = _to_wsl_path(script_path)
        result = _run_colab(
            colab_bin,
            ["exec", "-s", session_name, "-f", script_wsl, "--timeout", "120"],
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        return parse_ls_output(result.stdout)
    finally:
        if script_path and script_path.exists():
            script_path.unlink()


def _get_session_status(colab_bin: str, session_name: str) -> dict[str, str | None]:
    result = _run_colab(
        colab_bin,
        ["status", "-s", session_name],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "colab status failed")
    return parse_status_output(result.stdout)


def _download_file(colab_bin: str, session_name: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path_wsl = _to_wsl_path(local_path.resolve())
    result = _run_colab(
        colab_bin,
        ["download", "-s", session_name, remote_path, local_path_wsl],
        capture_output=True,
    )
    if result.returncode == 0:
        return
    if Path(remote_path).name in _EXEC_TEXT_FALLBACK_FILES:
        if _download_text_file_via_exec(colab_bin, session_name, remote_path, local_path):
            return
    raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "colab download failed")


def _download_text_file_via_exec(colab_bin: str, session_name: str, remote_path: str, local_path: Path) -> bool:
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            script_path = Path(handle.name)
            handle.write(
                "from pathlib import Path\n"
                "import sys\n"
                f"path = Path({remote_path!r})\n"
                "if not path.exists():\n"
                "    raise SystemExit(4)\n"
                "sys.stdout.write(path.read_text(encoding='utf-8', errors='ignore'))\n"
            )
        script_wsl = _to_wsl_path(script_path)
        result = _run_colab(
            colab_bin,
            ["exec", "-s", session_name, "-f", script_wsl, "--timeout", "120"],
            capture_output=True,
        )
        if result.returncode != 0:
            return False
        local_path.write_text(_sanitize_exec_text_output(result.stdout), encoding="utf-8")
        return True
    finally:
        if script_path and script_path.exists():
            script_path.unlink()


def _download_priority(remote_names: list[str]) -> list[str]:
    names = [name for name in remote_names if not name.endswith(".tmp")]
    metadata_first = [name for name in ("launch.json", "train.log", "train_metrics.json") if name in names]
    checkpoints = sorted(
        [name for name in names if name.startswith("ckpt_step") and name.endswith(".pt")],
        key=checkpoint_step,
        reverse=True,
    )
    export_files = [name for name in names if name in _FINAL_EXPORT_FILES]
    remaining = [
        name for name in names
        if name not in set(metadata_first) and name not in set(checkpoints) and name not in set(export_files)
    ]
    return [*metadata_first, *checkpoints, *export_files, *remaining]


def _sync_remote_files(
    colab_bin: str,
    session_name: str,
    remote_out_dir: str,
    local_mirror_dir: Path,
    remote_names: list[str],
    *,
    download_retries: int,
    retry_delay_sec: int,
    on_progress=None,
) -> tuple[list[str], dict[str, str]]:
    downloaded: list[str] = []
    failures: dict[str, str] = {}
    for name in _download_priority(remote_names):
        local_path = local_mirror_dir / name
        remote_path = f"{remote_out_dir.rstrip('/')}/{name}"
        should_download = False
        if name.startswith("ckpt_step") and name.endswith(".pt"):
            should_download = not local_path.exists()
        elif name in {"train_metrics.json", "train.log", "launch.json"}:
            should_download = True
        elif name in _FINAL_EXPORT_FILES:
            should_download = True
        if not should_download:
            continue
        last_error = ""
        for attempt in range(1, download_retries + 1):
            try:
                _download_file(colab_bin, session_name, remote_path, local_path)
                downloaded.append(name)
                last_error = ""
                if on_progress is not None:
                    on_progress(downloaded, failures)
                break
            except RuntimeError as exc:
                last_error = str(exc)
                if attempt < download_retries:
                    time.sleep(retry_delay_sec)
        if last_error:
            failures[name] = last_error
            if on_progress is not None:
                on_progress(downloaded, failures)
    return downloaded, failures


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def _copy_dataset_snapshot(snapshot_dir: Path, dataset_dir: Path) -> list[str]:
    copied: list[str] = []
    tracked_names = {
        "latest_checkpoint.txt",
        "best_checkpoint.txt",
        "train_metrics.json",
        "train.log",
        *_FINAL_EXPORT_FILES,
    }
    for source in snapshot_dir.iterdir():
        name = source.name
        if not source.is_file():
            continue
        if not ((name.startswith("ckpt_step") and name.endswith(".pt")) or name in tracked_names):
            continue
        target = dataset_dir / name
        if target.exists() and target.stat().st_size == source.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(name)
    return sorted(copied)


def _refresh_dataset_dir_from_kaggle(dataset_dir: Path, dataset_id: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="thai-sign-ckpt-refresh-") as temp_dir:
        result = _run_local(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_id,
                "-p",
                temp_dir,
                "--unzip",
                "-q",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or f"kaggle datasets download failed for {dataset_id}"
            )
        return _copy_dataset_snapshot(Path(temp_dir), dataset_dir)


def _seed_mirror_from_dataset_dir(mirror_dir: Path, dataset_dir: Path, remote_names: list[str]) -> list[str]:
    seeded: list[str] = []
    for name in remote_names:
        if not (name.startswith("ckpt_step") and name.endswith(".pt")):
            continue
        target = mirror_dir / name
        if target.exists():
            continue
        source = dataset_dir / name
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        seeded.append(name)
    return seeded


def _write_sync_state(
    mirror_dir: Path,
    *,
    downloaded: list[str],
    failures: dict[str, str],
    latest_checkpoint: str | None,
    latest_remote_checkpoint: str | None,
    session_status: dict[str, str | None] | None,
) -> None:
    state = {
        "downloaded": downloaded,
        "failed": failures,
        "latest_checkpoint": latest_checkpoint,
        "latest_remote_checkpoint": latest_remote_checkpoint,
        "session_status": session_status,
        "timestamp": time.time(),
    }
    (mirror_dir / "sync_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def publish_latest_checkpoint(mirror_dir: Path, dataset_dir: Path, dataset_id: str, title: str) -> str | None:
    ensure_dataset_metadata(dataset_dir, dataset_id, title)

    ckpts = sorted(mirror_dir.glob("ckpt_step*.pt"), key=lambda path: checkpoint_step(path.name))
    if not ckpts:
        return None

    latest_ckpt = ckpts[-1]
    best_name = best_checkpoint_name(ckpts)
    keep_names = {latest_ckpt.name}
    if best_name:
        keep_names.add(best_name)
    for old in dataset_dir.glob("ckpt_step*.pt"):
        if old.name not in keep_names:
            old.unlink()

    for ckpt in ckpts:
        if ckpt.name not in keep_names:
            continue
        target_ckpt = dataset_dir / ckpt.name
        if not target_ckpt.exists() or target_ckpt.stat().st_size != ckpt.stat().st_size:
            shutil.copy2(ckpt, target_ckpt)

    _copy_if_exists(mirror_dir / "train_metrics.json", dataset_dir / "train_metrics.json")
    _copy_if_exists(mirror_dir / "train.log", dataset_dir / "train.log")
    for name in _FINAL_EXPORT_FILES:
        _copy_if_exists(mirror_dir / name, dataset_dir / name)
    (dataset_dir / "latest_checkpoint.txt").write_text(latest_ckpt.name, encoding="utf-8")
    if best_name:
        (dataset_dir / "best_checkpoint.txt").write_text(best_name, encoding="utf-8")

    return latest_ckpt.name


def create_or_version_dataset(dataset_dir: Path, message: str) -> None:
    create_cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(dataset_dir),
        "--dir-mode",
        "skip",
        "-q",
    ]
    create_result = _run_local(create_cmd, capture_output=True)
    if create_result.returncode == 0:
        return

    version_cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(dataset_dir),
        "-m",
        message,
        "--dir-mode",
        "skip",
        "-q",
    ]
    version_result = _run_local(version_cmd, capture_output=True)
    if version_result.returncode != 0:
        raise RuntimeError(version_result.stderr.strip() or version_result.stdout.strip() or "kaggle datasets version failed")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mirror_dir = Path(args.local_mirror_dir)
    dataset_dir = Path(args.kaggle_dataset_dir)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset_metadata(dataset_dir, args.kaggle_dataset_id, args.kaggle_dataset_title)
    lock_path = mirror_dir / ".sync.lock"
    _acquire_sync_lock(lock_path)

    last_published_name = ""
    errors = 0

    try:
        while True:
            session_status = _get_session_status(args.colab_bin, args.session_name)
            remote_names = _list_remote_files(args.colab_bin, args.session_name, args.remote_out_dir)
            remote_latest = latest_checkpoint_name(remote_names)
            seeded = _seed_mirror_from_dataset_dir(mirror_dir, dataset_dir, remote_names)

            def _progress(downloaded: list[str], failures: dict[str, str]) -> None:
                _write_sync_state(
                    mirror_dir,
                    downloaded=[*seeded, *downloaded],
                    failures=failures,
                    latest_checkpoint=None,
                    latest_remote_checkpoint=remote_latest,
                    session_status=session_status,
                )

            downloaded, failures = _sync_remote_files(
                args.colab_bin,
                args.session_name,
                args.remote_out_dir,
                mirror_dir,
                remote_names,
                download_retries=args.download_retries,
                retry_delay_sec=args.retry_delay_sec,
                on_progress=_progress,
            )
            refreshed: list[str] = []
            checkpoint_failures = [
                name for name in failures if name.startswith("ckpt_step") and name.endswith(".pt")
            ]
            if remote_latest and (
                checkpoint_failures or not (mirror_dir / remote_latest).exists()
            ):
                try:
                    refreshed = _refresh_dataset_dir_from_kaggle(dataset_dir, args.kaggle_dataset_id)
                except Exception as exc:
                    failures["_kaggle_refresh"] = str(exc)
                else:
                    seeded.extend(
                        _seed_mirror_from_dataset_dir(mirror_dir, dataset_dir, remote_names)
                    )

            latest_name = publish_latest_checkpoint(
                mirror_dir,
                dataset_dir,
                args.kaggle_dataset_id,
                args.kaggle_dataset_title,
            )
            if latest_name and latest_name != last_published_name:
                create_or_version_dataset(dataset_dir, f"Sync {latest_name}")
                last_published_name = latest_name

            _write_sync_state(
                mirror_dir,
                downloaded=[*seeded, *downloaded, *refreshed],
                failures=failures,
                latest_checkpoint=latest_name,
                latest_remote_checkpoint=remote_latest,
                session_status=session_status,
            )
            error_log = mirror_dir / "sync_error.log"
            if failures:
                error_log.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
            elif error_log.exists():
                error_log.unlink()

            errors = 0
            if args.once:
                return 0
            time.sleep(args.interval_sec)
    except Exception as exc:
        errors += 1
        (mirror_dir / "sync_error.log").write_text(str(exc), encoding="utf-8")
        if args.once or errors >= args.max_errors:
            return 1
        time.sleep(args.interval_sec)
    finally:
        _release_sync_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
