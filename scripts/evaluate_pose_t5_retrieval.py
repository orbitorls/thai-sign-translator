"""Evaluate a PoseT5 runtime export as a text-retrieval baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from tsl.data.unified import load_features, load_manifest
from tsl.eval.build_splits import split_by_video
from tsl.inference.pose_t5_translator import PoseT5Translator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a PoseT5 runtime export as a retrieval baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-roots", default="data/tsl51_v3,data/thaisignvis_v3_probe")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subset-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", default="1,3,5")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--samples-json", default=None)
    return parser


def _default_report_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent / f"{checkpoint_dir.name}_retrieval_eval.json"


def _default_samples_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent / f"{checkpoint_dir.name}_retrieval_samples.json"


def _parse_topk(value: str) -> list[int]:
    topk = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not topk or any(k <= 0 for k in topk):
        raise ValueError("top-k must contain positive integers")
    return topk


def _load_examples(data_roots_arg: str) -> list:
    examples = []
    for root in data_roots_arg.split(","):
        root = root.strip()
        if not root:
            continue
        examples.extend(load_manifest(root))
    return examples


def _select_stratified_val_subset(examples: list, subset_size: int) -> list:
    if subset_size <= 0 or len(examples) <= subset_size:
        return list(examples)

    grouped: dict[str, list] = defaultdict(list)
    for ex in examples:
        grouped[ex.source].append(ex)

    selected: list = []
    offsets = {source: 0 for source in grouped}
    ordered_sources = list(grouped)
    while len(selected) < subset_size:
        progressed = False
        for source in ordered_sources:
            offset = offsets[source]
            bucket = grouped[source]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[source] = offset + 1
            progressed = True
            if len(selected) >= subset_size:
                break
        if not progressed:
            break
    return selected


def _encode_feature_batch(model, features_batch: list, device: str) -> torch.Tensor:
    max_t = max(int(features.shape[0]) for features in features_batch)
    feat_dim = int(features_batch[0].shape[1])
    src = torch.zeros((len(features_batch), max_t, feat_dim), dtype=torch.float32, device=device)
    lengths = torch.zeros((len(features_batch),), dtype=torch.long, device=device)
    for row_idx, features in enumerate(features_batch):
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        src[row_idx, : tensor.shape[0]] = tensor
        lengths[row_idx] = tensor.shape[0]
    with torch.no_grad():
        return model.encode_pooled(src, lengths, normalize=True)


def _encode_examples(model, examples: list, device: str, batch_size: int) -> torch.Tensor:
    batches = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        features_batch = [load_features(ex.features_path) for ex in chunk]
        batches.append(_encode_feature_batch(model, features_batch, device))
    if not batches:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.cat(batches, dim=0)


def _build_prototypes(examples: list, embeddings: torch.Tensor) -> tuple[list[str], torch.Tensor]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for ex, embedding in zip(examples, embeddings):
        grouped[ex.target_text].append(embedding)

    labels = sorted(grouped)
    prototypes = []
    for label in labels:
        proto = torch.stack(grouped[label], dim=0).mean(dim=0)
        proto = torch.nn.functional.normalize(proto, p=2.0, dim=0)
        prototypes.append(proto)
    return labels, torch.stack(prototypes, dim=0)


def _query_report(
    labels: list[str],
    prototypes: torch.Tensor,
    val_examples: list,
    val_embeddings: torch.Tensor,
    topk: list[int],
) -> tuple[dict, list[dict]]:
    if not val_examples:
        return {"n": 0, "mrr": 0.0}, []

    scores = val_embeddings @ prototypes.T
    max_k = max(topk)
    samples = []
    totals = {k: 0 for k in topk}
    reciprocal_rank_sum = 0.0
    source_counts: dict[str, int] = defaultdict(int)
    source_hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    source_rr: dict[str, float] = defaultdict(float)

    for row_idx, ex in enumerate(val_examples):
        ranked_idx = torch.argsort(scores[row_idx], descending=True).tolist()
        ranked_labels = [labels[idx] for idx in ranked_idx]
        rank = next((i + 1 for i, label in enumerate(ranked_labels) if label == ex.target_text), None)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

        source_counts[ex.source] += 1
        if rank is not None:
            source_rr[ex.source] += 1.0 / rank

        for k in topk:
            hit = int(ex.target_text in ranked_labels[:k])
            totals[k] += hit
            source_hits[ex.source][k] += hit

        samples.append(
            {
                "example_id": ex.example_id,
                "source": ex.source,
                "reference": ex.target_text,
                "rank": rank,
                "top_predictions": ranked_labels[:max_k],
            }
        )

    report = {
        "n": len(val_examples),
        "mrr": round(reciprocal_rank_sum / len(val_examples), 4),
        "label_count": len(labels),
        "source_counts": dict(source_counts),
        "source_metrics": {},
    }
    for k in topk:
        report[f"top{k}_exact"] = round(totals[k] / len(val_examples), 4)

    for source in sorted(source_counts):
        source_n = source_counts[source]
        metrics = {
            "n": source_n,
            "mrr": round(source_rr[source] / source_n, 4) if source_n else 0.0,
        }
        for k in topk:
            metrics[f"top{k}_exact"] = round(source_hits[source][k] / source_n, 4) if source_n else 0.0
        report["source_metrics"][source] = metrics

    return report, samples


def _evaluate(args: argparse.Namespace) -> dict:
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    report_path = Path(args.report_json).resolve() if args.report_json else _default_report_path(checkpoint_dir)
    samples_path = Path(args.samples_json).resolve() if args.samples_json else _default_samples_path(checkpoint_dir)
    topk = _parse_topk(args.top_k)

    all_examples = _load_examples(args.data_roots)
    splits = split_by_video(all_examples, fracs={"train": 0.9, "val": 0.1}, seed=args.seed)
    train_examples = splits["train"]
    val_examples = _select_stratified_val_subset(splits["val"], args.val_subset_size)
    if not train_examples:
        raise ValueError("retrieval eval requires at least one train example")

    translator = PoseT5Translator.from_checkpoint_dir(str(checkpoint_dir), device=args.device)
    train_embeddings = _encode_examples(translator.model, train_examples, translator.device, args.batch_size)
    val_embeddings = _encode_examples(translator.model, val_examples, translator.device, args.batch_size)

    labels, prototypes = _build_prototypes(train_examples, train_embeddings)
    report, samples = _query_report(labels, prototypes, val_examples, val_embeddings, topk)
    report["checkpoint_dir"] = str(checkpoint_dir)
    report["data_roots"] = [root.strip() for root in args.data_roots.split(",") if root.strip()]
    report["seed"] = int(args.seed)
    report["val_subset_size"] = len(val_examples)
    runtime_metadata_path = checkpoint_dir / "runtime_metadata.json"
    if runtime_metadata_path.is_file():
        report["runtime_metadata"] = json.loads(runtime_metadata_path.read_text(encoding="utf-8"))

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = _evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
