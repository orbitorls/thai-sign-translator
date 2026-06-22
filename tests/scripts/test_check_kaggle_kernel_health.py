from __future__ import annotations

from types import SimpleNamespace

from scripts.check_kaggle_kernel_health import inspect_kernel_health


class _FakeStatus:
    def __init__(self, status: str):
        self.status = status

    def to_dict(self):
        return {"status": self.status}


class _FakeFiles:
    def to_dict(self):
        return {
            "files": [
                {"name": "manifest.csv", "size": 100, "creationDate": "now"},
                {"name": "train_metrics.json", "size": 200, "creationDate": "now"},
            ]
        }


class _FakeApi:
    def authenticate(self):
        return None

    def parse_kernel_string(self, kernel_ref: str):
        parts = kernel_ref.split("/")
        owner = parts[0]
        slug = parts[1]
        version = parts[2] if len(parts) > 2 else ""
        return owner, slug, version

    def kernels_status(self, _kernel_ref: str):
        return _FakeStatus("RUNNING")

    def kernels_list_files(self, _kernel_ref: str):
        return _FakeFiles()


def test_inspect_kernel_health_classifies_queued_when_logs_probe_times_out(monkeypatch):
    monkeypatch.setattr("scripts.check_kaggle_kernel_health.KaggleApi", _FakeApi)
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._list_session_output",
        lambda *_a, **_k: {"log": ""},
    )
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._probe_logs_stream",
        lambda *_a, **_k: {
            "classification": "queued_or_no_live_logs",
            "error_type": "ReadTimeout",
            "message": "timed out",
        },
    )

    report = inspect_kernel_health("orbitorls/thai-sign-mixed-all-v6-train/36")

    assert report["classification"] == "queued_or_worker_unallocated"
    assert report["parsed"]["version_label"] == "36"


def test_inspect_kernel_health_classifies_running_with_logs(monkeypatch):
    monkeypatch.setattr("scripts.check_kaggle_kernel_health.KaggleApi", _FakeApi)
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._list_session_output",
        lambda *_a, **_k: {"log": "step=200"},
    )
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._probe_logs_stream",
        lambda *_a, **_k: {
            "classification": "response",
            "status_code": 200,
            "body_preview": "step 200",
        },
    )

    report = inspect_kernel_health("orbitorls/thai-sign-mixed-all-v6-train")

    assert report["classification"] == "running_with_live_or_persisted_logs"


def test_inspect_kernel_health_classifies_version_propagation_lag(monkeypatch):
    monkeypatch.setattr("scripts.check_kaggle_kernel_health.KaggleApi", _FakeApi)
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._list_session_output",
        lambda *_a, **_k: {"error_type": "HTTPError", "status_code": 404},
    )
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._probe_logs_stream",
        lambda *_a, **_k: {
            "classification": "response",
            "status_code": 404,
            "body_preview": '{"error":{"message":"Kernel version with label 37 not found."}}',
        },
    )

    report = inspect_kernel_health("orbitorls/thai-sign-mixed-all-v6-train/37")

    assert report["classification"] == "version_propagation_lag_or_missing_session"


def test_inspect_kernel_health_treats_read_timeout_wrapped_in_connection_error_as_queue(monkeypatch):
    monkeypatch.setattr("scripts.check_kaggle_kernel_health.KaggleApi", _FakeApi)
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._list_session_output",
        lambda *_a, **_k: {"log": ""},
    )
    monkeypatch.setattr(
        "scripts.check_kaggle_kernel_health._probe_logs_stream",
        lambda *_a, **_k: {
            "classification": "queued_or_no_live_logs",
            "error_type": "ConnectionError",
            "message": "HTTPSConnectionPool(host='api.kaggle.com', port=443): Read timed out.",
        },
    )

    report = inspect_kernel_health("orbitorls/thai-sign-mixed-all-v6-train")

    assert report["classification"] == "queued_or_worker_unallocated"
