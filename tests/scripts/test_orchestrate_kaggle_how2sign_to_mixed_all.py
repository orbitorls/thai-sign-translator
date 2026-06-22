from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import scripts.orchestrate_kaggle_how2sign_to_mixed_all as orchestrator_module
from scripts.orchestrate_kaggle_how2sign_to_mixed_all import (
    orchestrate,
    stage_mixed_all_output_dataset,
    stage_seed_dataset,
    sync_downstream_kernel_metadata,
)


def _write_pretrain_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "best_model_state.pt").write_bytes(b"state")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "pose_t5_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "train_metrics.json").write_text("{}", encoding="utf-8")
    (root / "session.log").write_text("ignore me", encoding="utf-8")


def _write_mixed_all_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    verified_export = root / "verified_export"
    verified_export.mkdir(parents=True, exist_ok=True)
    (root / "best_model_state.pt").write_bytes(b"best")
    (root / "source_sampling.json").write_text("{}", encoding="utf-8")
    (root / "train_metrics.json").write_text("{}", encoding="utf-8")
    (root / "verified_eval.json").write_text("{}", encoding="utf-8")
    (root / "verified_samples.json").write_text("[]", encoding="utf-8")
    for name in [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pose_encoder.pt",
        "pose_t5_config.json",
        "runtime_metadata.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "decoding_search.json",
        "decoding_best_eval.json",
        "decoding_best_samples.json",
    ]:
        (verified_export / name).write_text("{}", encoding="utf-8")


def test_stage_seed_dataset_copies_required_output(tmp_path):
    source_dir = tmp_path / "source"
    staging_dir = tmp_path / "staged"
    _write_pretrain_output(source_dir)

    result = stage_seed_dataset(
        source_dir,
        staging_dir=staging_dir,
        dataset_id="orbitorls/demo-seed",
        title="Demo Seed",
    )

    assert result["has_best_state"] is True
    assert (staging_dir / "best_model_state.pt").is_file()
    assert (staging_dir / "config.json").is_file()
    assert (staging_dir / "pose_t5_config.json").is_file()
    assert (staging_dir / "tokenizer_config.json").is_file()
    assert not (staging_dir / "session.log").exists()
    assert "session.log" in result["skipped"]
    metadata = json.loads((staging_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "orbitorls/demo-seed"


def test_stage_mixed_all_output_dataset_copies_export_and_sidecars(tmp_path):
    source_dir = tmp_path / "source"
    staging_dir = tmp_path / "staged"
    _write_mixed_all_output(source_dir)

    result = stage_mixed_all_output_dataset(
        source_dir,
        staging_dir=staging_dir,
        dataset_id="orbitorls/mixed-output",
        title="Mixed Output",
    )

    assert (staging_dir / "best_model_state.pt").is_file()
    assert (staging_dir / "model.safetensors").is_file()
    assert (staging_dir / "pose_encoder.pt").is_file()
    assert (staging_dir / "pose_t5_config.json").is_file()
    assert (staging_dir / "tokenizer.json").is_file()
    assert (staging_dir / "verified_eval.json").is_file()
    assert (staging_dir / "decoding_best_eval.json").is_file()
    assert result["verified_export_dir"].endswith("verified_export")
    metadata = json.loads((staging_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "orbitorls/mixed-output"


def test_stage_seed_dataset_requires_pose_and_tokenizer_artifacts(tmp_path):
    source_dir = tmp_path / "source"
    staging_dir = tmp_path / "staged"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "best_model_state.pt").write_bytes(b"state")

    try:
        stage_seed_dataset(
            source_dir,
            staging_dir=staging_dir,
            dataset_id="orbitorls/demo-seed",
            title="Demo Seed",
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    assert "required artifacts" in message or "tokenizer artifacts" in message


def test_sync_downstream_kernel_metadata_adds_seed_dataset(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "orbitorls/downstream",
                "dataset_sources": ["orbitorls/thai-sign-code"],
                "kernel_sources": ["orbitorls/old-kernel"],
            }
        ),
        encoding="utf-8",
    )

    payload = sync_downstream_kernel_metadata(
        metadata_path,
        seed_dataset_id="orbitorls/thai-sign-how2sign-pretrain-output",
    )

    assert "orbitorls/thai-sign-how2sign-pretrain-output" in payload["dataset_sources"]
    assert payload["kernel_sources"] == []


def test_orchestrate_returns_early_when_source_not_complete(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    class _FakeApi:
        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                return {"status": "COMPLETE", "failureMessage": None}
            return {"status": "RUNNING", "failureMessage": None}

        def kernels_list_files(self, kernel):
            return {
                "files": [
                    {"name": "best_checkpoint.txt"},
                    {"name": "best_model_state.pt"},
                    {"name": "ckpt_step00003800.pt"},
                    {"name": "pose_t5_config.json"},
                    {"name": "tokenizer.json"},
                    {"name": "tokenizer_config.json"},
                    {"name": "train_metrics.json"},
                ]
            }

        def kernels_logs(self, kernel):
            return "line-1\nline-2\n"

    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: _FakeApi(),
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )

    result = orchestrate(
        SimpleNamespace(
            source_kernel="orbitorls/source",
            downstream_kernel="orbitorls/downstream",
            downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
            seed_dataset_slug="seed-dataset",
            seed_dataset_id="orbitorls/seed-dataset",
            seed_dataset_title="Seed Dataset",
            staging_root="kaggle_upload",
            download_root="tmp/source-output",
            downstream_download_root="tmp/downstream-output",
            kaggle_temp_dir="tmp/kaggle",
            message="demo",
            downstream_output_message="demo-output",
            accelerator="t4",
            poll_seconds=1,
            wait_timeout_seconds=0,
            wait_downstream_timeout_seconds=0,
            push_downstream="true",
            push_eval="false",
            eval_kernel="orbitorls/eval",
            eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
            downstream_output_dataset_slug="thai-sign-mixed-all-output",
            downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
            downstream_output_dataset_title="Thai Sign Mixed All Output",
            log_tail_chars=200,
            status_json=str(tmp_path / "status.json"),
            visible_stale_seconds_threshold=1800,
            visible_stale_polls_threshold=10,
        )
    )

    assert result["source_status"] == "RUNNING"
    assert result["downstream_status"] == "COMPLETE"
    assert result["source_progress_signal"] == "weak_committed_snapshot"
    assert result["source_visible_stale_suspected"] is False
    assert "seed_dataset_stage" not in result
    assert result["source_files"] == [
        "best_checkpoint.txt",
        "best_model_state.pt",
        "ckpt_step00003800.pt",
        "pose_t5_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "train_metrics.json",
    ]
    assert result["source_file_summary"]["seed_artifacts_visible"] is True
    assert result["source_file_summary"]["latest_visible_checkpoint_step"] == 3800
    assert result["source_file_summary"]["best_checkpoint_ref_visible"] is True
    assert result["source_file_summary"]["visible_tokenizer_artifacts"] == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert result["source_log_tail"] == "line-1\nline-2\n"
    status_payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status_payload["source_status"] == "RUNNING"
    assert status_payload["downstream_status"] == "COMPLETE"


def test_orchestrate_surfaces_failure_message_and_log_tail(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    class _FakeApi:
        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                return {"status": "RUNNING", "failureMessage": None}
            return {"status": "ERROR", "failureMessage": "GPU quota exceeded"}

        def kernels_list_files(self, kernel):
            return {"files": []}

        def kernels_logs(self, kernel):
            return "a" * 50

    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: _FakeApi(),
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )

    result = orchestrate(
        SimpleNamespace(
            source_kernel="orbitorls/source",
            downstream_kernel="orbitorls/downstream",
            downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
            seed_dataset_slug="seed-dataset",
            seed_dataset_id="orbitorls/seed-dataset",
            seed_dataset_title="Seed Dataset",
            staging_root="kaggle_upload",
            download_root="tmp/source-output",
            downstream_download_root="tmp/downstream-output",
            kaggle_temp_dir="tmp/kaggle",
            message="demo",
            downstream_output_message="demo-output",
            accelerator="t4",
            poll_seconds=1,
            wait_timeout_seconds=0,
            wait_downstream_timeout_seconds=0,
            push_downstream="true",
            push_eval="false",
            eval_kernel="orbitorls/eval",
            eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
            downstream_output_dataset_slug="thai-sign-mixed-all-output",
            downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
            downstream_output_dataset_title="Thai Sign Mixed All Output",
            log_tail_chars=12,
            status_json=str(tmp_path / "status.json"),
            visible_stale_seconds_threshold=1800,
            visible_stale_polls_threshold=10,
        )
    )

    assert result["source_status"] == "ERROR"
    assert result["downstream_status"] == "RUNNING"
    assert result["source_progress_signal"] == "final_artifact_snapshot"
    assert result["source_failure_message"] == "GPU quota exceeded"
    assert result["source_log_tail"] == "aaaaaaaaaaaa"
    status_payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status_payload["source_failure_message"] == "GPU quota exceeded"
    assert status_payload["downstream_status"] == "RUNNING"


def test_orchestrate_tolerates_inaccessible_downstream_status(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    class _FakeApi:
        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                raise ValueError("Permission 'kernels.get' was denied")
            return {"status": "RUNNING", "failureMessage": None}

        def kernels_list_files(self, kernel):
            return {"files": []}

        def kernels_logs(self, kernel):
            return ""

    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: _FakeApi(),
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )

    result = orchestrate(
        SimpleNamespace(
            source_kernel="orbitorls/source",
            downstream_kernel="orbitorls/downstream",
            downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
            seed_dataset_slug="seed-dataset",
            seed_dataset_id="orbitorls/seed-dataset",
            seed_dataset_title="Seed Dataset",
            staging_root="kaggle_upload",
            download_root="tmp/source-output",
            downstream_download_root="tmp/downstream-output",
            kaggle_temp_dir="tmp/kaggle",
            message="demo",
            downstream_output_message="demo-output",
            accelerator="t4",
            poll_seconds=1,
            wait_timeout_seconds=0,
            wait_downstream_timeout_seconds=0,
            push_downstream="false",
            push_eval="false",
            eval_kernel="orbitorls/eval",
            eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
            downstream_output_dataset_slug="thai-sign-mixed-all-output",
            downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
            downstream_output_dataset_title="Thai Sign Mixed All Output",
            log_tail_chars=200,
            status_json=str(tmp_path / "status.json"),
            visible_stale_seconds_threshold=1800,
            visible_stale_polls_threshold=10,
        )
    )

    assert result["source_status"] == "RUNNING"
    assert result["downstream_status"] == "UNKNOWN"
    assert "Permission 'kernels.get' was denied" in result["downstream_status_error"]


def test_orchestrate_tracks_visible_source_staleness_across_polls(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    class _FakeApi:
        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                return {"status": "COMPLETE", "failureMessage": None}
            return {"status": "RUNNING", "failureMessage": None}

        def kernels_list_files(self, kernel):
            return {
                "files": [
                    {"name": "best_checkpoint.txt"},
                    {"name": "best_model_state.pt"},
                    {"name": "ckpt_step00003800.pt"},
                    {"name": "pose_t5_config.json"},
                    {"name": "tokenizer.json"},
                    {"name": "tokenizer_config.json"},
                    {"name": "train_metrics.json"},
                ]
            }

        def kernels_logs(self, kernel):
            return ""

    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: _FakeApi(),
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )

    clock = {"value": datetime(2026, 6, 21, 10, 0, 0, tzinfo=timezone.utc)}

    def _fake_now_utc():
        return clock["value"]

    monkeypatch.setattr(orchestrator_module, "_now_utc", _fake_now_utc)

    args = SimpleNamespace(
        source_kernel="orbitorls/source",
        downstream_kernel="orbitorls/downstream",
        downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
        seed_dataset_slug="seed-dataset",
        seed_dataset_id="orbitorls/seed-dataset",
        seed_dataset_title="Seed Dataset",
        staging_root="kaggle_upload",
        download_root="tmp/source-output",
        downstream_download_root="tmp/downstream-output",
        kaggle_temp_dir="tmp/kaggle",
        message="demo",
        downstream_output_message="demo-output",
        accelerator="t4",
        poll_seconds=1,
        wait_timeout_seconds=0,
        wait_downstream_timeout_seconds=0,
        push_downstream="false",
        push_eval="false",
        eval_kernel="orbitorls/eval",
        eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
        downstream_output_dataset_slug="thai-sign-mixed-all-output",
        downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
        downstream_output_dataset_title="Thai Sign Mixed All Output",
        log_tail_chars=200,
        status_json=str(tmp_path / "status.json"),
        visible_stale_seconds_threshold=1800,
        visible_stale_polls_threshold=10,
    )

    first = orchestrate(args)
    assert first["source_progress_changed"] is True
    assert first["source_progress_signal"] == "weak_committed_snapshot"
    assert first["source_no_visible_progress_seconds"] == 0.0
    assert first["source_no_visible_progress_polls"] == 0

    clock["value"] = clock["value"] + timedelta(seconds=75)
    second = orchestrate(args)
    assert second["source_progress_changed"] is False
    assert second["source_last_progress_at_utc"] == first["source_last_progress_at_utc"]
    assert second["source_no_visible_progress_seconds"] == 75.0
    assert second["source_no_visible_progress_polls"] == 1
    assert second["source_visible_stale_suspected"] is False

    args.visible_stale_seconds_threshold = 60
    args.visible_stale_polls_threshold = 2
    third = orchestrate(args)
    assert third["source_visible_stale_suspected"] is True


def test_orchestrate_retries_after_transient_source_status_error(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    class _FakeApi:
        def __init__(self):
            self.source_calls = 0

        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                return {"status": "COMPLETE", "failureMessage": None}
            self.source_calls += 1
            if self.source_calls == 1:
                return {"status": "RUNNING", "failureMessage": None}
            if self.source_calls == 2:
                raise ConnectionError("temporary kaggle outage")
            return {"status": "RUNNING", "failureMessage": None}

        def kernels_list_files(self, kernel):
            return {"files": []}

        def kernels_logs(self, kernel):
            return ""

    fake_api = _FakeApi()
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: fake_api,
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )

    time_values = iter([0.0, 0.1, 1.1, 2.1])
    monkeypatch.setattr(orchestrator_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda _seconds: None)

    result = orchestrate(
        SimpleNamespace(
            source_kernel="orbitorls/source",
            downstream_kernel="orbitorls/downstream",
            downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
            seed_dataset_slug="seed-dataset",
            seed_dataset_id="orbitorls/seed-dataset",
            seed_dataset_title="Seed Dataset",
            staging_root="kaggle_upload",
            download_root="tmp/source-output",
            downstream_download_root="tmp/downstream-output",
            kaggle_temp_dir="tmp/kaggle",
            message="demo",
            downstream_output_message="demo-output",
            accelerator="t4",
            poll_seconds=1,
            wait_timeout_seconds=2,
            wait_downstream_timeout_seconds=0,
            push_downstream="false",
            push_eval="false",
            eval_kernel="orbitorls/eval",
            eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
            downstream_output_dataset_slug="thai-sign-mixed-all-output",
            downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
            downstream_output_dataset_title="Thai Sign Mixed All Output",
            log_tail_chars=200,
            status_json=str(tmp_path / "status.json"),
            visible_stale_seconds_threshold=1800,
            visible_stale_polls_threshold=10,
        )
    )

    assert result["source_status"] == "RUNNING"
    assert fake_api.source_calls == 3
    assert "source_status_error" not in result


def test_orchestrate_tolerates_source_snapshot_errors(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    class _FakeApi:
        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                return {"status": "COMPLETE", "failureMessage": None}
            return {"status": "RUNNING", "failureMessage": None}

        def kernels_list_files(self, kernel):
            raise TimeoutError("files api timed out")

        def kernels_logs(self, kernel):
            raise TimeoutError("logs api timed out")

    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: _FakeApi(),
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )

    result = orchestrate(
        SimpleNamespace(
            source_kernel="orbitorls/source",
            downstream_kernel="orbitorls/downstream",
            downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
            seed_dataset_slug="seed-dataset",
            seed_dataset_id="orbitorls/seed-dataset",
            seed_dataset_title="Seed Dataset",
            staging_root="kaggle_upload",
            download_root="tmp/source-output",
            downstream_download_root="tmp/downstream-output",
            kaggle_temp_dir="tmp/kaggle",
            message="demo",
            downstream_output_message="demo-output",
            accelerator="t4",
            poll_seconds=1,
            wait_timeout_seconds=0,
            wait_downstream_timeout_seconds=0,
            push_downstream="false",
            push_eval="false",
            eval_kernel="orbitorls/eval",
            eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
            downstream_output_dataset_slug="thai-sign-mixed-all-output",
            downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
            downstream_output_dataset_title="Thai Sign Mixed All Output",
            log_tail_chars=200,
            status_json=str(tmp_path / "status.json"),
            visible_stale_seconds_threshold=1800,
            visible_stale_polls_threshold=10,
        )
    )

    assert result["source_status"] == "RUNNING"
    assert result["source_files"] == []
    assert "files api timed out" in result["source_files_error"]
    assert result["source_log_tail"] == ""
    assert "logs api timed out" in result["source_log_tail_error"]


def test_orchestrate_can_publish_downstream_output_and_push_eval(tmp_path, monkeypatch):
    downstream_kernel_dir = tmp_path / "kaggle_upload" / "how2sign_to_mixed_all_notebook"
    downstream_kernel_dir.mkdir(parents=True)
    (downstream_kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-how2sign-mixed-all",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )
    eval_kernel_dir = tmp_path / "kaggle_upload" / "mixed_all_eval_notebook"
    eval_kernel_dir.mkdir(parents=True)
    (eval_kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "orbitorls/thai-sign-mixed-all-eval",
                "dataset_sources": [],
                "kernel_sources": [],
            }
        ),
        encoding="utf-8",
    )

    source_download_root = tmp_path / "tmp" / "source-output"
    _write_pretrain_output(source_download_root)
    downstream_download_root = tmp_path / "tmp" / "downstream-output"
    _write_mixed_all_output(downstream_download_root)

    pushed: list[str] = []
    published: list[Path] = []

    class _FakeApi:
        def kernels_status(self, kernel):
            if kernel == "orbitorls/downstream":
                return {"status": "READY", "failureMessage": None}
            if kernel == "orbitorls/downstream-created":
                return {"status": "COMPLETE", "failureMessage": None}
            if kernel == "orbitorls/eval":
                return {"status": "COMPLETE", "failureMessage": None}
            return {"status": "COMPLETE", "failureMessage": None}

        def kernels_push(self, kernel_path, acc=None):
            pushed.append(str(kernel_path))
            if "mixed_all_eval_notebook" in str(kernel_path):
                return SimpleNamespace(ref="orbitorls/eval", version_number=2, url="https://example.com/eval")
            return SimpleNamespace(ref="orbitorls/downstream-created", version_number=1, url="https://example.com/downstream")

    def _fake_download(api, *, kernel, output_dir):
        if kernel == "orbitorls/source":
            return {
                "output_dir": str(source_download_root),
                "downloaded": [],
                "files": sorted(path.name for path in source_download_root.iterdir() if path.is_file()),
            }
        if kernel == "orbitorls/downstream-created":
            return {
                "output_dir": str(downstream_download_root),
                "downloaded": [],
                "files": sorted(
                    str(path.relative_to(downstream_download_root)).replace("\\", "/")
                    for path in downstream_download_root.rglob("*")
                    if path.is_file()
                ),
            }
        raise AssertionError(kernel)

    def _fake_publish(dataset_dir, *, message, dir_mode="skip", temp_dir="", quiet=False):
        published.append(Path(dataset_dir))
        return {"action": "version", "dataset_dir": str(dataset_dir), "message": message}

    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._kaggle_api",
        lambda: _FakeApi(),
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._REPO_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all._download_kernel_output",
        _fake_download,
    )
    monkeypatch.setattr(
        "scripts.orchestrate_kaggle_how2sign_to_mixed_all.create_or_version_dataset",
        _fake_publish,
    )

    result = orchestrate(
        SimpleNamespace(
            source_kernel="orbitorls/source",
            downstream_kernel="orbitorls/downstream",
            downstream_kernel_path="kaggle_upload/how2sign_to_mixed_all_notebook",
            seed_dataset_slug="seed-dataset",
            seed_dataset_id="orbitorls/seed-dataset",
            seed_dataset_title="Seed Dataset",
            staging_root="kaggle_upload",
            download_root="tmp/source-output",
            downstream_download_root="tmp/downstream-output",
            kaggle_temp_dir="tmp/kaggle",
            message="demo-seed",
            downstream_output_message="demo-output",
            accelerator="t4",
            poll_seconds=1,
            wait_timeout_seconds=0,
            wait_downstream_timeout_seconds=1,
            push_downstream="true",
            push_eval="true",
            eval_kernel="orbitorls/eval",
            eval_kernel_path="kaggle_upload/mixed_all_eval_notebook",
            downstream_output_dataset_slug="thai-sign-mixed-all-output",
            downstream_output_dataset_id="orbitorls/thai-sign-mixed-all-output",
            downstream_output_dataset_title="Thai Sign Mixed All Output",
            log_tail_chars=200,
            status_json=str(tmp_path / "status.json"),
            visible_stale_seconds_threshold=1800,
            visible_stale_polls_threshold=10,
        )
    )

    assert result["downstream_followup"]["status"] == "COMPLETE"
    assert result["downstream_output_publish"]["action"] == "version"
    assert result["eval_push"]["ref"] == "orbitorls/eval"
    assert len(published) == 2
    assert any("how2sign_to_mixed_all_notebook" in path for path in pushed)
    assert any("mixed_all_eval_notebook" in path for path in pushed)
