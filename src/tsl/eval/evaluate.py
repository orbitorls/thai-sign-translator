"""Two-track evaluation driver: held-out ASL episodes + recorded Thai words."""
from __future__ import annotations

import argparse

from tsl.eval.metrics import accuracy, confusion_matrix_fig, topk_accuracy


def evaluate_track(store, clips, true_words, label_to_id, out_png=None) -> dict:
    if len(clips) != len(true_words):
        raise ValueError("clips and true_words must be the same length")
    label_names = sorted(label_to_id, key=lambda w: label_to_id[w])
    y_true, y_pred, topk_preds = [], [], []
    for clip, true_word in zip(clips, true_words):
        pred_word, _score = store.predict(clip)
        y_true.append(label_to_id[true_word])
        y_pred.append(label_to_id.get(pred_word, -1))
        topk_preds.append([label_to_id.get(pred_word, -1)])
    acc = accuracy(y_true, y_pred)
    top5 = topk_accuracy(y_true, topk_preds, k=5)
    if out_png is not None:
        confusion_matrix_fig(y_true, y_pred, label_names, out_png)
    return {"n": len(clips), "accuracy": acc, "top5_accuracy": top5}


def build_thai_track(encoder, thai_root: str):
    from tsl.data.thai import load_thai_clips
    from tsl.registry.prototype_store import PrototypeStore

    thai = load_thai_clips(thai_root)
    store = PrototypeStore(encoder)
    query, true_words = [], []
    for word, word_clips in thai.items():
        if len(word_clips) < 2:
            continue
        store.add_sign(word, word_clips[:-1])
        query.append(word_clips[-1])
        true_words.append(word)
    label_to_id = {w: i for i, w in enumerate(store.names())}
    return store, query, true_words, label_to_id


def build_asl_track(encoder, parquet_dir, csv_path, classes, k_shot=5, q_query=5):
    from tsl.data.episodic import EpisodicSampler
    from tsl.data.islr import ISLRDataset
    from tsl.registry.prototype_store import PrototypeStore

    dataset = ISLRDataset(parquet_dir=parquet_dir, csv_path=csv_path, classes=classes)
    n_way = dataset.num_classes
    sampler = EpisodicSampler(dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, episodes=1)
    episode = next(iter(sampler))
    store = PrototypeStore(encoder)
    label_to_id = {}
    support_x = episode["support_x"]
    support_y = episode["support_y"]
    for remapped in range(n_way):
        name = f"asl_{remapped}"
        label_to_id[name] = remapped
        clips = [support_x[j].numpy() for j in range(support_x.shape[0]) if int(support_y[j]) == remapped]
        store.add_sign(name, clips)
    query_x = episode["query_x"]
    query_y = episode["query_y"]
    query_clips = [query_x[j].numpy() for j in range(query_x.shape[0])]
    true_words = [f"asl_{int(query_y[j])}" for j in range(query_y.shape[0])]
    return store, query_clips, true_words, label_to_id


def run_two_track_eval(checkpoint, islr_parquet_dir, islr_csv, thai_root, held_out_classes=None, out_dir="eval_out") -> dict:
    import os

    import torch

    from tsl.features.normalize import SELECTED_LANDMARKS
    from tsl.models.encoder import LandmarkEncoder

    os.makedirs(out_dir, exist_ok=True)
    encoder = LandmarkEncoder(input_dim=len(SELECTED_LANDMARKS) * 3)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    encoder.eval()
    asl_store, asl_query, asl_true, asl_labels = build_asl_track(encoder, islr_parquet_dir, islr_csv, held_out_classes)
    asl_summary = evaluate_track(asl_store, asl_query, asl_true, asl_labels, out_png=os.path.join(out_dir, "asl_confusion.png"))
    thai_store, thai_query, thai_true, thai_labels = build_thai_track(encoder, thai_root)
    thai_summary = evaluate_track(thai_store, thai_query, thai_true, thai_labels, out_png=os.path.join(out_dir, "thai_confusion.png"))
    return {"asl_held_out_acc": asl_summary["accuracy"], "thai_acc": thai_summary["accuracy"], "n_episodes": 1, "asl": asl_summary, "thai": thai_summary}


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Two-track ProtoNet evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--islr-parquet-dir", required=True)
    parser.add_argument("--islr-csv", required=True)
    parser.add_argument("--thai-root", required=True)
    parser.add_argument("--held-out-classes", nargs="*", default=None)
    parser.add_argument("--out-dir", default="eval_out")
    args = parser.parse_args()
    summary = run_two_track_eval(
        checkpoint=args.checkpoint,
        islr_parquet_dir=args.islr_parquet_dir,
        islr_csv=args.islr_csv,
        thai_root=args.thai_root,
        held_out_classes=args.held_out_classes,
        out_dir=args.out_dir,
    )
    print("ASL held-out track:", summary["asl"])
    print("THAI track:", summary["thai"])


if __name__ == "__main__":  # pragma: no cover
    main()
