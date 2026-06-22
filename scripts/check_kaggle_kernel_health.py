from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionLogsStreamRequest,
    ApiListKernelSessionOutputRequest,
)


def _normalize_status(payload: Any) -> str:
    status = getattr(payload, "status", None)
    if status is None:
        return str(payload or "").strip().upper()
    return str(status).split(".")[-1].upper()


def _preview_kernel_files(api: KaggleApi, kernel_ref: str, limit: int = 8) -> list[dict[str, Any]]:
    payload = api.kernels_list_files(kernel_ref)
    data = payload.to_dict() if hasattr(payload, "to_dict") else payload
    files = data.get("files", []) if isinstance(data, dict) else []
    preview: list[dict[str, Any]] = []
    for item in files[:limit]:
        if not isinstance(item, dict):
            preview.append({"name": str(item)})
            continue
        preview.append(
            {
                "name": item.get("name"),
                "size": item.get("size"),
                "creationDate": item.get("creationDate"),
            }
        )
    return preview


def _list_session_output(api: KaggleApi, owner: str, slug: str, version_label: str) -> dict[str, Any]:
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        req.page_size = 20
        if version_label:
            req.version_label = version_label
        try:
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        except HTTPError as exc:
            return {
                "error_type": type(exc).__name__,
                "status_code": exc.response.status_code if exc.response is not None else None,
                "message": str(exc),
            }
    return response.to_dict() if hasattr(response, "to_dict") else {"repr": repr(response)}


def _probe_logs_stream(
    api: KaggleApi,
    owner: str,
    slug: str,
    version_label: str,
    *,
    wait_for_logs_url_seconds: int,
    connect_timeout_seconds: int,
    read_timeout_seconds: int,
) -> dict[str, Any]:
    with api.build_kaggle_client() as kaggle:
        client = kaggle.kernels.kernels_api_client._client
        client._init_session()
        req = ApiGetKernelSessionLogsStreamRequest()
        req.user_name = owner
        req.kernel_slug = slug
        req.wait_for_logs_url_seconds = wait_for_logs_url_seconds
        if version_label:
            req.version_label = version_label
        prepared = client._prepare_request(
            "kernels.KernelsApiService",
            "GetKernelSessionLogsStream",
            req,
        )
        try:
            response = client._session.send(
                prepared,
                timeout=(connect_timeout_seconds, read_timeout_seconds),
                stream=False,
            )
        except requests.exceptions.ReadTimeout as exc:
            return {
                "classification": "queued_or_no_live_logs",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        except requests.exceptions.RequestException as exc:
            message = str(exc)
            if "Read timed out" in message:
                return {
                    "classification": "queued_or_no_live_logs",
                    "error_type": type(exc).__name__,
                    "message": message,
                }
            return {
                "classification": "request_error",
                "error_type": type(exc).__name__,
                "message": message,
            }
    return {
        "classification": "response",
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body_preview": response.text[:2000],
    }


def inspect_kernel_health(
    kernel_ref: str,
    *,
    wait_for_logs_url_seconds: int = 5,
    connect_timeout_seconds: int = 10,
    read_timeout_seconds: int = 15,
) -> dict[str, Any]:
    api = KaggleApi()
    api.authenticate()
    owner, slug, version = api.parse_kernel_string(kernel_ref)
    status_payload = api.kernels_status(kernel_ref)
    status = _normalize_status(status_payload)
    output_preview = _preview_kernel_files(api, kernel_ref)
    session_output = _list_session_output(api, owner, slug, version or "")
    logs_probe = _probe_logs_stream(
        api,
        owner,
        slug,
        version or "",
        wait_for_logs_url_seconds=wait_for_logs_url_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
    )

    classification = "unknown"
    if status == "ERROR":
        classification = "error"
    elif status == "RUNNING":
        if logs_probe.get("classification") == "queued_or_no_live_logs":
            classification = "queued_or_worker_unallocated"
        elif (
            logs_probe.get("classification") == "response"
            and int(logs_probe.get("status_code") or 0) == 404
        ):
            classification = "version_propagation_lag_or_missing_session"
        elif logs_probe.get("classification") == "response" and logs_probe.get("body_preview"):
            classification = "running_with_live_or_persisted_logs"
        elif session_output.get("log"):
            classification = "running_with_session_output"
        else:
            classification = "running_but_no_visible_output"

    return {
        "kernel_ref": kernel_ref,
        "parsed": {
            "owner": owner,
            "slug": slug,
            "version_label": version or "",
        },
        "status": status,
        "status_payload": status_payload.to_dict() if hasattr(status_payload, "to_dict") else str(status_payload),
        "output_preview": output_preview,
        "session_output": session_output,
        "logs_probe": logs_probe,
        "classification": classification,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Kaggle kernel health with low-level session output and log-stream probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--wait-for-logs-url-seconds", type=int, default=5)
    parser.add_argument("--connect-timeout-seconds", type=int, default=10)
    parser.add_argument("--read-timeout-seconds", type=int, default=15)
    parser.add_argument("--report-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = inspect_kernel_health(
        args.kernel_ref,
        wait_for_logs_url_seconds=int(args.wait_for_logs_url_seconds),
        connect_timeout_seconds=int(args.connect_timeout_seconds),
        read_timeout_seconds=int(args.read_timeout_seconds),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if str(args.report_json).strip():
        target = Path(args.report_json).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
