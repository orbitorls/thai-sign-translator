from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.publish_kaggle_dataset import create_or_version_dataset


_SEED_ARTIFACT_FILES = {
    "best_model_state.pt",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pose_encoder.pt",
    "pose_t5_config.json",
    "pytorch_model.bin",
    "run_status.json",
    "runtime_metadata.json",
    "source_sampling.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "train_metrics.json",
}
_REQUIRED_SEED_FILES = {
    "best_model_state.pt",
    "pose_t5_config.json",
}
_REQUIRED_TOKENIZER_SEED_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
}
_MIXED_ALL_OUTPUT_EXPORT_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pose_encoder.pt",
    "pose_t5_config.json",
    "runtime_metadata.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
_MIXED_ALL_OUTPUT_ROOT_FILES = {
    "best_model_state.pt",
    "source_sampling.json",
    "train_metrics.json",
    "verified_eval.json",
    "verified_samples.json",
}
_MIXED_ALL_OUTPUT_DECODE_FILES = {
    "decoding_best_eval.json",
    "decoding_best_samples.json",
    "decoding_search.json",
}
_REQUIRED_MIXED_ALL_OUTPUT_FILES = {
    "best_model_state.pt",
    "model.safetensors",
    "pose_encoder.pt",
    "pose_t5_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
_CHECKPOINT_RE = re.compile(r"^ckpt_step(\d+)\.pt$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor the How2Sign Kaggle pretrain lane, publish its output as a seed "
            "dataset, then launch the mixed-all Kaggle finetune lane."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-kernel", default="orbitorls/thai-sign-how2sign-pretrain")
    parser.add_argument("--downstream-kernel", default="orbitorls/thai-sign-how2sign-mixed-all")
    parser.add_argument("--downstream-kernel-path", default="kaggle_upload/how2sign_to_mixed_all_notebook")
    parser.add_argument("--seed-dataset-slug", default="thai-sign-how2sign-pretrain-output")
    parser.add_argument("--seed-dataset-id", default="orbitorls/thai-sign-how2sign-pretrain-output")
    parser.add_argument("--seed-dataset-title", default="Thai Sign How2Sign Pretrain Output")
    parser.add_argument("--staging-root", default="kaggle_upload")
    parser.add_argument("--download-root", default="tmp/kaggle_how2sign_pretrain_output")
    parser.add_argument("--downstream-download-root", default="tmp/kaggle_how2sign_to_mixed_all_output")
    parser.add_argument("--kaggle-temp-dir", default="tmp/kaggle_cli_temp")
    parser.add_argument("--message", default="Refresh How2Sign pretrain output seed dataset")
    parser.add_argument("--downstream-output-message", default="Refresh mixed-all output dataset from How2Sign finetune")
    parser.add_argument("--accelerator", default="t4")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--wait-downstream-timeout-seconds", type=int, default=0)
    parser.add_argument("--push-downstream", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--push-eval", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--eval-kernel", default="orbitorls/thai-sign-mixed-all-eval")
    parser.add_argument("--eval-kernel-path", default="kaggle_upload/mixed_all_eval_notebook")
    parser.add_argument("--downstream-output-dataset-slug", default="thai-sign-mixed-all-output")
    parser.add_argument("--downstream-output-dataset-id", default="orbitorls/thai-sign-mixed-all-output")
    parser.add_argument("--downstream-output-dataset-title", default="Thai Sign Mixed All Output")
    parser.add_argument("--log-tail-chars", type=int, default=4000)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--visible-stale-seconds-threshold", type=int, default=1800)
    parser.add_argument("--visible-stale-polls-threshold", type=int, default=10)
    return parser


def _resolve_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_kernel_status(payload: dict | str | None) -> str:
    status_attr = getattr(payload, "status", None)
    if status_attr is not None:
        return str(status_attr).split(".")[-1]
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed
    if isinstance(payload, dict):
        status = payload.get("status")
        if status is not None:
            return str(status)
    if payload is None:
        return ""
    return str(payload)


def _kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def _kernel_status(api: KaggleApi, kernel: str) -> dict[str, Any]:
    payload = api.kernels_status(kernel)
    if hasattr(payload, "to_dict"):
        data = payload.to_dict()
        if isinstance(data, dict):
            normalized = dict(data)
            status_attr = getattr(payload, "status", None)
            if status_attr is not None:
                normalized["status"] = str(status_attr).split(".")[-1]
            return normalized
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    if isinstance(payload, dict):
        return payload
    return {"status": str(payload)}


def _try_kernel_status(api: KaggleApi, kernel: str) -> tuple[dict[str, Any], str | None]:
    try:
        payload = _kernel_status(api, kernel)
    except Exception as exc:
        return {"status": "UNKNOWN", "error": str(exc)}, str(exc)
    return payload, None


def _kernel_failure_message(status_payload: dict[str, Any]) -> str | None:
    for key in ("failureMessage", "failure_message"):
        value = status_payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _kernel_files(api: KaggleApi, kernel: str) -> list[str]:
    payload = api.kernels_list_files(kernel)
    if hasattr(payload, "to_dict"):
        data = payload.to_dict()
    elif isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
    elif isinstance(payload, dict):
        data = payload
    else:
        data = {}

    files = data.get("files", []) if isinstance(data, dict) else []
    result: list[str] = []
    for item in files:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("path") or item.get("fileName")
            if name is not None and str(name).strip():
                result.append(str(name).strip())
        else:
            name = getattr(item, "name", None) or getattr(item, "path", None)
            if name is not None and str(name).strip():
                result.append(str(name).strip())
    return sorted(result)


def _try_kernel_files(api: KaggleApi, kernel: str) -> tuple[list[str], str | None]:
    try:
        return _kernel_files(api, kernel), None
    except Exception as exc:
        return [], str(exc)


def _kernel_log_tail(api: KaggleApi, kernel: str, max_chars: int) -> str:
    payload = api.kernels_logs(kernel)
    text = str(payload or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _try_kernel_log_tail(api: KaggleApi, kernel: str, max_chars: int) -> tuple[str, str | None]:
    try:
        return _kernel_log_tail(api, kernel, max_chars), None
    except Exception as exc:
        return "", str(exc)


def _write_status_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_status_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path).resolve()
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_progress_key(payload: dict[str, Any]) -> str:
    summary = payload.get("source_file_summary")
    if not isinstance(summary, dict):
        summary = {}
    key_payload = {
        "source_status": payload.get("source_status"),
        "source_failure_message": payload.get("source_failure_message"),
        "latest_visible_checkpoint_step": summary.get("latest_visible_checkpoint_step"),
        "source_files": payload.get("source_files", []),
        "source_log_tail": payload.get("source_log_tail", ""),
    }
    return json.dumps(key_payload, ensure_ascii=False, sort_keys=True)


def _stamp_observation(payload: dict[str, Any], *, status_json_path: str) -> dict[str, Any]:
    observed_at = _now_utc()
    observed_at_utc = observed_at.isoformat()
    payload["observed_at_utc"] = observed_at_utc

    previous = _read_status_json(status_json_path)
    current_key = _source_progress_key(payload)
    payload["source_progress_key"] = current_key

    if not isinstance(previous, dict):
        payload["source_progress_changed"] = True
        payload["source_last_progress_at_utc"] = observed_at_utc
        payload["source_no_visible_progress_seconds"] = 0.0
        payload["source_no_visible_progress_polls"] = 0
        return payload

    previous_key = str(previous.get("source_progress_key", ""))
    previous_progress_at = _parse_utc(previous.get("source_last_progress_at_utc"))
    previous_observed_at = _parse_utc(previous.get("observed_at_utc"))
    previous_no_progress_polls = int(previous.get("source_no_visible_progress_polls", 0) or 0)

    if current_key != previous_key:
        payload["source_progress_changed"] = True
        payload["source_last_progress_at_utc"] = observed_at_utc
        payload["source_no_visible_progress_seconds"] = 0.0
        payload["source_no_visible_progress_polls"] = 0
        return payload

    last_progress_at = previous_progress_at or previous_observed_at or observed_at
    no_progress_seconds = max(0.0, (observed_at - last_progress_at).total_seconds())
    payload["source_progress_changed"] = False
    payload["source_last_progress_at_utc"] = last_progress_at.isoformat()
    payload["source_no_visible_progress_seconds"] = round(no_progress_seconds, 3)
    payload["source_no_visible_progress_polls"] = previous_no_progress_polls + 1
    return payload


def _summarize_kernel_files(files: list[str]) -> dict[str, Any]:
    file_set = {str(item).strip() for item in files if str(item).strip()}
    checkpoint_steps: list[int] = []
    for name in file_set:
        match = _CHECKPOINT_RE.match(Path(name).name)
        if match is None:
            continue
        checkpoint_steps.append(int(match.group(1)))
    checkpoint_steps.sort()

    visible_seed_artifacts = sorted(file_set & _SEED_ARTIFACT_FILES)
    missing_seed_artifacts = sorted(_REQUIRED_SEED_FILES - file_set)
    visible_tokenizer_artifacts = sorted(_REQUIRED_TOKENIZER_SEED_FILES & file_set)
    return {
        "visible_seed_artifacts": visible_seed_artifacts,
        "missing_required_seed_artifacts": missing_seed_artifacts,
        "visible_tokenizer_artifacts": visible_tokenizer_artifacts,
        "seed_artifacts_visible": not missing_seed_artifacts and bool(visible_tokenizer_artifacts),
        "checkpoint_steps": checkpoint_steps,
        "latest_visible_checkpoint_step": checkpoint_steps[-1] if checkpoint_steps else None,
        "best_checkpoint_ref_visible": "best_checkpoint.txt" in file_set,
        "latest_checkpoint_ref_visible": "latest_checkpoint.txt" in file_set,
        "train_metrics_visible": "train_metrics.json" in file_set,
    }


def _attach_incomplete_source_state(
    result: dict[str, Any],
    *,
    api: KaggleApi,
    source_kernel: str,
    source_status_payload: dict[str, Any],
    log_tail_chars: int,
) -> dict[str, Any]:
    result["source_failure_message"] = _kernel_failure_message(source_status_payload)
    source_files, source_files_error = _try_kernel_files(api, source_kernel)
    result["source_files"] = source_files
    if source_files_error is not None:
        result["source_files_error"] = source_files_error
    result["source_file_summary"] = _summarize_kernel_files(result["source_files"])
    source_log_tail, source_log_tail_error = _try_kernel_log_tail(
        api,
        source_kernel,
        max(0, int(log_tail_chars)),
    )
    result["source_log_tail"] = source_log_tail
    if source_log_tail_error is not None:
        result["source_log_tail_error"] = source_log_tail_error
    return result


def _annotate_progress_signal(
    payload: dict[str, Any],
    *,
    visible_stale_seconds_threshold: int,
    visible_stale_polls_threshold: int,
) -> dict[str, Any]:
    source_status = str(payload.get("source_status", "")).strip().upper()
    if source_status == "RUNNING":
        payload["source_progress_signal"] = "weak_committed_snapshot"
        payload["source_progress_signal_note"] = (
            "Kaggle status/output for running notebook sessions may expose only the committed "
            "snapshot rather than live draft progress."
        )
    else:
        payload["source_progress_signal"] = "final_artifact_snapshot"
        payload["source_progress_signal_note"] = (
            "Source artifact status is based on the current committed kernel output listing."
        )

    no_progress_seconds = float(payload.get("source_no_visible_progress_seconds", 0.0) or 0.0)
    no_progress_polls = int(payload.get("source_no_visible_progress_polls", 0) or 0)
    payload["source_visible_stale_seconds_threshold"] = max(0, int(visible_stale_seconds_threshold))
    payload["source_visible_stale_polls_threshold"] = max(0, int(visible_stale_polls_threshold))
    payload["source_visible_stale_suspected"] = (
        no_progress_seconds >= payload["source_visible_stale_seconds_threshold"]
        or no_progress_polls >= payload["source_visible_stale_polls_threshold"]
    )
    return payload


def _download_kernel_output(
    api: KaggleApi,
    *,
    kernel: str,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_token: str | None = None
    downloaded: list[str] = []
    while True:
        page_files, page_token = api.kernels_output(
            kernel,
            path=str(output_dir),
            force=True,
            quiet=True,
            page_size=100,
            page_token=page_token,
        )
        downloaded.extend(str(item) for item in page_files)
        if not page_token:
            break

    files = sorted(
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    return {
        "output_dir": str(output_dir),
        "downloaded": downloaded,
        "files": files,
    }


def _write_dataset_metadata(dataset_dir: Path, dataset_id: str, title: str) -> None:
    metadata = {
        "title": title,
        "id": dataset_id,
        "isPrivate": True,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stage_seed_dataset(
    source_output_dir: Path,
    *,
    staging_dir: Path,
    dataset_id: str,
    title: str,
) -> dict[str, Any]:
    if not source_output_dir.is_dir():
        raise FileNotFoundError(f"source output dir not found: {source_output_dir}")
    best_state = source_output_dir / "best_model_state.pt"
    if not best_state.is_file():
        raise FileNotFoundError(f"best_model_state.pt not found under {source_output_dir}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_metadata(staging_dir, dataset_id, title)

    copied: list[str] = []
    skipped: list[str] = []
    for src in sorted(source_output_dir.iterdir(), key=lambda path: path.name):
        if src.name not in _SEED_ARTIFACT_FILES:
            skipped.append(src.name)
            continue
        dst = staging_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)

    copied_set = set(copied)
    missing_required = sorted(_REQUIRED_SEED_FILES - copied_set)
    if missing_required:
        raise FileNotFoundError(
            "seed dataset is missing required artifacts: " + ", ".join(missing_required)
        )
    if not (_REQUIRED_TOKENIZER_SEED_FILES & copied_set):
        raise FileNotFoundError(
            "seed dataset is missing tokenizer artifacts; expected one of: "
            + ", ".join(sorted(_REQUIRED_TOKENIZER_SEED_FILES))
        )

    return {
        "staging_dir": str(staging_dir),
        "copied": copied,
        "skipped": skipped,
        "has_best_state": best_state.is_file(),
    }


def stage_mixed_all_output_dataset(
    source_output_dir: Path,
    *,
    staging_dir: Path,
    dataset_id: str,
    title: str,
) -> dict[str, Any]:
    if not source_output_dir.is_dir():
        raise FileNotFoundError(f"source output dir not found: {source_output_dir}")
    verified_export_dir = source_output_dir / "verified_export"
    if not verified_export_dir.is_dir():
        raise FileNotFoundError(f"verified_export not found under {source_output_dir}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_metadata(staging_dir, dataset_id, title)

    copied: list[str] = []
    skipped: list[str] = []

    def _copy_if_selected(src: Path, dst_name: str, *, allowed: set[str]) -> None:
        if src.name not in allowed:
            skipped.append(str(src.relative_to(source_output_dir)).replace("\\", "/"))
            return
        shutil.copy2(src, staging_dir / dst_name)
        copied.append(dst_name)

    for src in sorted(source_output_dir.iterdir(), key=lambda path: path.name):
        if src.is_dir():
            if src.name != "verified_export":
                skipped.append(src.name + "/")
            continue
        _copy_if_selected(src, src.name, allowed=_MIXED_ALL_OUTPUT_ROOT_FILES | _MIXED_ALL_OUTPUT_DECODE_FILES)

    for src in sorted(verified_export_dir.iterdir(), key=lambda path: path.name):
        if src.is_dir():
            skipped.append("verified_export/" + src.name + "/")
            continue
        _copy_if_selected(
            src,
            src.name,
            allowed=_MIXED_ALL_OUTPUT_EXPORT_FILES | _MIXED_ALL_OUTPUT_DECODE_FILES,
        )

    copied_set = set(copied)
    missing_required = sorted(_REQUIRED_MIXED_ALL_OUTPUT_FILES - copied_set)
    if missing_required:
        raise FileNotFoundError(
            "mixed-all output dataset is missing required artifacts: " + ", ".join(missing_required)
        )

    return {
        "staging_dir": str(staging_dir),
        "copied": copied,
        "skipped": skipped,
        "verified_export_dir": str(verified_export_dir),
    }


def sync_downstream_kernel_metadata(
    metadata_path: Path,
    *,
    seed_dataset_id: str,
) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_sources = [
        str(item).strip()
        for item in payload.get("dataset_sources", [])
        if str(item).strip()
    ]
    if seed_dataset_id not in dataset_sources:
        dataset_sources.append(seed_dataset_id)
    payload["dataset_sources"] = dataset_sources
    payload["kernel_sources"] = []
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _push_downstream_kernel(
    api: KaggleApi,
    *,
    kernel_path: Path,
    accelerator: str,
) -> dict[str, Any]:
    response = api.kernels_push(str(kernel_path), acc=accelerator)
    return {
        "ref": getattr(response, "ref", None),
        "version_number": getattr(response, "version_number", None),
        "url": getattr(response, "url", None),
    }


def _wait_for_kernel_completion(
    api: KaggleApi,
    *,
    kernel: str,
    poll_seconds: int,
    wait_timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.time() + max(0, int(wait_timeout_seconds)) if int(wait_timeout_seconds) > 0 else None
    status_payload, status_error = _try_kernel_status(api, kernel)
    status = _normalize_kernel_status(status_payload)
    history: list[dict[str, Any]] = []
    while True:
        history.append(
            {
                "status": status,
                "payload": status_payload,
                "error": status_error,
                "observed_at_utc": _now_utc().isoformat(),
            }
        )
        if status in {"COMPLETE", "ERROR"}:
            break
        if deadline is not None and time.time() >= deadline:
            break
        time.sleep(max(1, int(poll_seconds)))
        status_payload, status_error = _try_kernel_status(api, kernel)
        status = _normalize_kernel_status(status_payload)
    return {
        "kernel": kernel,
        "status": status,
        "status_payload": status_payload,
        "status_error": status_error,
        "history": history,
        "timed_out": status not in {"COMPLETE", "ERROR"} and deadline is not None and time.time() >= deadline,
    }


def orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    api = _kaggle_api()
    source_kernel = str(args.source_kernel).strip()
    downstream_kernel = str(args.downstream_kernel).strip()
    eval_kernel = str(args.eval_kernel).strip()
    repo_root = _REPO_ROOT
    download_root = (repo_root / args.download_root).resolve()
    downstream_download_root = (repo_root / args.downstream_download_root).resolve()
    staging_root = (repo_root / args.staging_root).resolve()
    staging_dir = staging_root / args.seed_dataset_slug
    downstream_kernel_path = (repo_root / args.downstream_kernel_path).resolve()
    downstream_metadata_path = downstream_kernel_path / "kernel-metadata.json"
    downstream_output_staging_dir = staging_root / args.downstream_output_dataset_slug
    eval_kernel_path = (repo_root / args.eval_kernel_path).resolve()

    wait_timeout_seconds = max(0, int(args.wait_timeout_seconds))
    wait_downstream_timeout_seconds = max(0, int(args.wait_downstream_timeout_seconds))
    poll_seconds = max(1, int(args.poll_seconds))
    deadline = time.time() + wait_timeout_seconds if wait_timeout_seconds > 0 else None

    source_status_payload, source_status_error = _try_kernel_status(api, source_kernel)
    source_status = _normalize_kernel_status(source_status_payload)
    downstream_status_payload, downstream_status_error = _try_kernel_status(api, downstream_kernel)
    downstream_status = _normalize_kernel_status(downstream_status_payload)
    result: dict[str, Any] = {
        "source_kernel": source_kernel,
        "source_status": source_status,
        "source_status_payload": source_status_payload,
        "seed_dataset_id": args.seed_dataset_id,
        "downstream_kernel": downstream_kernel,
        "downstream_status": downstream_status,
        "downstream_status_payload": downstream_status_payload,
    }
    if source_status_error is not None:
        result["source_status_error"] = source_status_error
    if downstream_status_error is not None:
        result["downstream_status_error"] = downstream_status_error
    if source_status != "COMPLETE":
        result = _attach_incomplete_source_state(
            result,
            api=api,
            source_kernel=source_kernel,
            source_status_payload=source_status_payload,
            log_tail_chars=args.log_tail_chars,
        )
    result = _stamp_observation(result, status_json_path=args.status_json)
    result = _annotate_progress_signal(
        result,
        visible_stale_seconds_threshold=args.visible_stale_seconds_threshold,
        visible_stale_polls_threshold=args.visible_stale_polls_threshold,
    )
    _write_status_json(args.status_json, result)
    while deadline is not None and time.time() < deadline and (
        source_status == "RUNNING" or source_status_error is not None
    ):
        time.sleep(poll_seconds)
        next_source_status_payload, next_source_status_error = _try_kernel_status(api, source_kernel)
        if next_source_status_error is None:
            source_status_payload = next_source_status_payload
            source_status = _normalize_kernel_status(source_status_payload)
        source_status_error = next_source_status_error
        downstream_status_payload, downstream_status_error = _try_kernel_status(api, downstream_kernel)
        downstream_status = _normalize_kernel_status(downstream_status_payload)
        result = {
            "source_kernel": source_kernel,
            "source_status": source_status,
            "source_status_payload": source_status_payload,
            "seed_dataset_id": args.seed_dataset_id,
            "downstream_kernel": downstream_kernel,
            "downstream_status": downstream_status,
            "downstream_status_payload": downstream_status_payload,
        }
        if source_status_error is not None:
            result["source_status_error"] = source_status_error
        if downstream_status_error is not None:
            result["downstream_status_error"] = downstream_status_error
        if source_status != "COMPLETE":
            result = _attach_incomplete_source_state(
                result,
                api=api,
                source_kernel=source_kernel,
                source_status_payload=source_status_payload,
                log_tail_chars=args.log_tail_chars,
            )
        result = _stamp_observation(result, status_json_path=args.status_json)
        result = _annotate_progress_signal(
            result,
            visible_stale_seconds_threshold=args.visible_stale_seconds_threshold,
            visible_stale_polls_threshold=args.visible_stale_polls_threshold,
        )
        _write_status_json(args.status_json, result)
    if source_status != "COMPLETE":
        return result

    download_report = _download_kernel_output(api, kernel=source_kernel, output_dir=download_root)
    result["download"] = download_report
    if "best_model_state.pt" not in set(download_report["files"]):
        result["source_status"] = "COMPLETE_MISSING_ARTIFACTS"
        result = _stamp_observation(result, status_json_path=args.status_json)
        result = _annotate_progress_signal(
            result,
            visible_stale_seconds_threshold=args.visible_stale_seconds_threshold,
            visible_stale_polls_threshold=args.visible_stale_polls_threshold,
        )
        _write_status_json(args.status_json, result)
        return result

    stage_report = stage_seed_dataset(
        download_root,
        staging_dir=staging_dir,
        dataset_id=args.seed_dataset_id,
        title=args.seed_dataset_title,
    )
    result["seed_dataset_stage"] = stage_report
    publish_result = create_or_version_dataset(
        staging_dir,
        message=args.message,
        dir_mode="skip",
        temp_dir=args.kaggle_temp_dir,
    )
    result["seed_dataset_publish"] = publish_result

    metadata_payload = sync_downstream_kernel_metadata(
        downstream_metadata_path,
        seed_dataset_id=args.seed_dataset_id,
    )
    result["downstream_metadata"] = {
        "path": str(downstream_metadata_path),
        "dataset_sources": metadata_payload.get("dataset_sources", []),
        "kernel_sources": metadata_payload.get("kernel_sources", []),
    }

    if not _resolve_bool_flag(args.push_downstream):
        result = _stamp_observation(result, status_json_path=args.status_json)
        result = _annotate_progress_signal(
            result,
            visible_stale_seconds_threshold=args.visible_stale_seconds_threshold,
            visible_stale_polls_threshold=args.visible_stale_polls_threshold,
        )
        _write_status_json(args.status_json, result)
        return result

    result["downstream_status_before_push"] = downstream_status_payload
    if downstream_status == "RUNNING":
        result["downstream_push"] = {"skipped": True, "reason": "already running"}
        result = _stamp_observation(result, status_json_path=args.status_json)
        result = _annotate_progress_signal(
            result,
            visible_stale_seconds_threshold=args.visible_stale_seconds_threshold,
            visible_stale_polls_threshold=args.visible_stale_polls_threshold,
        )
        _write_status_json(args.status_json, result)
        return result

    push_result = _push_downstream_kernel(
        api,
        kernel_path=downstream_kernel_path,
        accelerator=args.accelerator,
    )
    result["downstream_push"] = push_result
    downstream_kernel_ref = str(push_result.get("ref") or downstream_kernel).strip()

    if wait_downstream_timeout_seconds > 0:
        downstream_followup = _wait_for_kernel_completion(
            api,
            kernel=downstream_kernel_ref,
            poll_seconds=poll_seconds,
            wait_timeout_seconds=wait_downstream_timeout_seconds,
        )
        result["downstream_followup"] = downstream_followup
        if downstream_followup["status"] == "COMPLETE":
            downstream_download = _download_kernel_output(
                api,
                kernel=downstream_kernel_ref,
                output_dir=downstream_download_root,
            )
            result["downstream_download"] = downstream_download
            mixed_output_stage = stage_mixed_all_output_dataset(
                downstream_download_root,
                staging_dir=downstream_output_staging_dir,
                dataset_id=args.downstream_output_dataset_id,
                title=args.downstream_output_dataset_title,
            )
            result["downstream_output_stage"] = mixed_output_stage
            mixed_output_publish = create_or_version_dataset(
                downstream_output_staging_dir,
                message=args.downstream_output_message,
                dir_mode="skip",
                temp_dir=args.kaggle_temp_dir,
            )
            result["downstream_output_publish"] = mixed_output_publish

            if _resolve_bool_flag(args.push_eval):
                eval_status_payload, eval_status_error = _try_kernel_status(api, eval_kernel)
                eval_status = _normalize_kernel_status(eval_status_payload)
                result["eval_status_before_push"] = {
                    "status": eval_status,
                    "payload": eval_status_payload,
                    "error": eval_status_error,
                }
                if eval_status == "RUNNING":
                    result["eval_push"] = {"skipped": True, "reason": "already running"}
                else:
                    result["eval_push"] = _push_downstream_kernel(
                        api,
                        kernel_path=eval_kernel_path,
                        accelerator=args.accelerator,
                    )

    result = _stamp_observation(result, status_json_path=args.status_json)
    result = _annotate_progress_signal(
        result,
        visible_stale_seconds_threshold=args.visible_stale_seconds_threshold,
        visible_stale_polls_threshold=args.visible_stale_polls_threshold,
    )
    _write_status_json(args.status_json, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = orchestrate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
