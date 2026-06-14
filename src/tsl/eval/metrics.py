"""Evaluation metrics for the Thai Sign Language translator."""
from __future__ import annotations

import time as _time

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} pred")
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def topk_accuracy(y_true, topk_preds, k: int) -> float:
    if len(y_true) != len(topk_preds):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(topk_preds)} preds")
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, preds in zip(y_true, topk_preds) if t in list(preds)[:k])
    return correct / len(y_true)


def confusion_matrix_fig(y_true, y_pred, labels: list[str], out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    cm = _sk_confusion_matrix(y_true, y_pred, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_png, format="png", dpi=100)
    plt.close(fig)


def kshot_curve(eval_fn, shots: list[int]) -> dict[int, float]:
    return {int(shot): float(eval_fn(shot)) for shot in shots}


def measure_latency(fn, *args, repeats: int = 20) -> dict:
    times_ms: list[float] = []
    for _ in range(repeats):
        start = _time.perf_counter()
        fn(*args)
        end = _time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    times_ms.sort()
    n = len(times_ms)
    mean_ms = sum(times_ms) / n if n else 0.0
    import math

    rank = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
    p95_ms = times_ms[rank] if n else 0.0
    return {"mean_ms": mean_ms, "p95_ms": p95_ms}
