from __future__ import annotations

from unittest.mock import mock_open

import pytest
import torch

from tsl.train.runtime import is_wsl, resolve_device


def test_is_wsl_false_without_signals(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)

    def fake_open(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("builtins.open", fake_open, raising=False)

    assert is_wsl() is False


def test_is_wsl_true_from_env(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    assert is_wsl() is True


def test_is_wsl_true_from_proc_version(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(
        "builtins.open",
        mock_open(read_data="Linux version 5.15.90.1-microsoft-standard-WSL2"),
        raising=False,
    )

    assert is_wsl() is True


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = resolve_device(None, require_gpu=True)

    assert device.type == "cuda"


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = resolve_device("auto", require_gpu=False)

    assert device.type == "cpu"


def test_resolve_device_requires_gpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_device("auto", require_gpu=True)


def test_resolve_device_rejects_explicit_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_device("cuda", require_gpu=False)


def test_resolve_device_rejects_explicit_cpu_when_require_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError):
        resolve_device("cpu", require_gpu=True)
