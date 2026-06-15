"""Training entrypoint for the SignToTextTransformer (Thai SLT).

Loads the TSL-51 sentence manifest, fits a char-level tokenizer on the
target texts, splits deterministically into 90/10 train/val, and runs a
short cross-entropy training loop with teacher forcing (the model's
``forward`` does not shift the decoder input, so we do it here).

Outputs in ``--out-dir``:
    - ``slt_model.pt``        : model ``state_dict``
    - ``tokenizer.json``      : serialized :class:`CharTokenizer`
    - ``model_config.json``   : constructor kwargs needed to rebuild the model
    - ``train_metrics.json``  : per-epoch losses + final train loss
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import random

import numpy as np

import torch
import torch.nn.functional as F

from tsl.data.slt_collate import slt_collate
from tsl.data.tsl51 import load_landmark_sequence, load_sentence_manifest
from tsl.models.slt import SignToTextTransformer
from tsl.text.tokenizer import CharTokenizer, WordTokenizer
from tsl.train.config import ModelSize, resolve_config
from tsl.train.runtime import resolve_device


_INPUT_DIM = 162


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _build_model(vocab_size: int, model_size: ModelSize = "small", input_dim: int = _INPUT_DIM) -> SignToTextTransformer:
    cfg = resolve_config(model_size, input_dim)
    return SignToTextTransformer(vocab_size=vocab_size, **asdict(cfg))


def _save_model_config(config_dict: dict, vocab_size: int, path: str) -> None:
    config = {**config_dict, "vocab_size": vocab_size}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _save_tokenizer(tok, path: str) -> None:
    """Serialize a CharTokenizer or WordTokenizer to ``path`` as JSON."""
    tok_type = "word" if isinstance(tok, WordTokenizer) else "char"
    vocab = list(tok._tok_to_id.keys())
    payload = {
        "tokenizer_type": tok_type,
        "pad": "<pad>",
        "bos": "<bos>",
        "eos": "<eos>",
        "unk": "<unk>",
        "vocab": vocab,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tokenizer(path: str):
    """Load a CharTokenizer or WordTokenizer from a JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"tokenizer file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("pad", "bos", "eos", "unk", "vocab"):
        if key not in data:
            raise ValueError(f"tokenizer file missing key: {key!r}")
    tok_type = data.get("tokenizer_type", "char")
    if tok_type == "word":
        tok = WordTokenizer()
    else:
        tok = CharTokenizer()
    tok._tok_to_id = {t: i for i, t in enumerate(data["vocab"])}
    return tok


def _save_metrics(metrics: dict, path: str) -> None:
    """Write the per-epoch metrics dict to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch.src.to(device),
        batch.src_lengths.to(device),
        batch.tgt.to(device),
    )


def _batch_loss(
    model: SignToTextTransformer,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt: torch.Tensor,
    pad_id: int,
    bos_id: int,
) -> torch.Tensor:
    """Compute next-token cross-entropy on a single batch.

    tgt = [BOS, t1, t2, ..., tN, EOS, PAD, ...]
    dec_input = tgt[:, :-1]  → [BOS, t1, t2, ..., tN, EOS, PAD]
    labels    = tgt[:, 1:]   → [t1, t2, ..., tN, EOS, PAD, PAD]
    """
    dec_input = tgt[:, :-1]
    labels = tgt[:, 1:]
    logits = model(src, src_lengths, dec_input)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def train_one_epoch(
    model: SignToTextTransformer,
    batches,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer,
) -> float:
    """Train ``model`` for one pass over ``batches``. Returns mean loss."""
    if not batches:
        return 0.0
    model.train()
    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id
    total = 0.0
    n = 0
    for batch in batches:
        src, src_lengths, tgt = _move_batch(batch, device)
        loss = _batch_loss(model, src, src_lengths, tgt, pad_id, bos_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def eval_loss(
    model: SignToTextTransformer,
    batches,
    device: torch.device,
    tokenizer,
) -> float:
    """Evaluate ``model`` on ``batches`` under teacher-forcing loss.

    Returns ``float('inf')`` when ``batches`` is empty.
    """
    if not batches:
        return float("inf")
    model.eval()
    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in batches:
            src, src_lengths, tgt = _move_batch(batch, device)
            loss = _batch_loss(model, src, src_lengths, tgt, pad_id, bos_id)
            total += float(loss.detach().cpu())
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SignToTextTransformer for Thai SLT (supports multi-stage)."
    )
    parser.add_argument(
        "--stage",
        default="tsl51",
        choices=["tsl51", "how2sign", "thaisignvis", "youtube_sl25", "combined", "finetune"],
        help=(
            "Training stage: tsl51 (default, 162-dim), how2sign (pretrain, 411-dim), "
            "thaisignvis (Thai sentence-level, 312-dim), "
            "youtube_sl25 (YouTube-SL-25 Thai, 162-dim, manifest.csv), "
            "combined (TSL-51 + YouTube-SL-25, 162-dim), "
            "finetune (from pretrained)."
        ),
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the dataset root (TSL-51, How2Sign, ThaiSignVis, or YouTube-SL-25 "
             "depending on --stage). For --stage combined, pass comma-separated paths: "
             "'data/tsl51,data/youtube_sl25_thai'.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="checkpoints/slt")
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device string (e.g. "auto", "cpu", "cuda").',
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless CUDA is available.",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["small", "base", "large"],
        help="Model architecture preset (default: small).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Feature dimension per frame. Auto-detected from stage if not set.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Path to a pretrained checkpoint dir (used with --stage finetune).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of training examples (deterministic: take first N).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer",
        default="word",
        choices=["char", "word"],
        help=(
            "Tokenizer granularity: 'word' (default, best for small datasets) "
            "or 'char' (character-level)."
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply random data augmentation during training (time stretch, jitter, dropout, mirror).",
    )
    parser.add_argument(
        "--eval-beam",
        type=int,
        default=4,
        help="Beam size used for chrF evaluation at the end of training (default: 4).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip chrF evaluation after training.",
    )
    return parser.parse_args(argv)


def _split_examples(
    examples: list, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = len(examples) // 10
    val_set = set(indices[:n_val])
    train_examples = [examples[i] for i in indices if i not in val_set]
    val_examples = [examples[i] for i in indices if i in val_set]
    return train_examples, val_examples


def _resolve_input_dim(stage: str, input_dim: int | None) -> int:
    if input_dim is not None:
        return input_dim
    if stage == "how2sign":
        return 411
    if stage == "thaisignvis":
        return 312
    return 162


def _load_data(stage: str, data_root: str, limit: int | None):
    if stage == "how2sign":
        from tsl.data.how2sign import load_how2sign_manifest, load_how2sign_keypoints
        train_ex = load_how2sign_manifest(data_root, split="train")
        val_ex = load_how2sign_manifest(data_root, split="val")
        load_fn = load_how2sign_keypoints
    elif stage == "thaisignvis":
        from tsl.data.thaisignvis import load_thaisignvis_manifest, load_thaisignvis_features
        train_ex = load_thaisignvis_manifest(data_root, split="train")
        val_ex = load_thaisignvis_manifest(data_root, split="val")
        load_fn = load_thaisignvis_features
    elif stage == "youtube_sl25":
        from tsl.data.thaisignvis import load_npy_manifest, load_npy_features
        train_ex = load_npy_manifest(data_root, split="train", source="youtube_sl25")
        val_ex = load_npy_manifest(data_root, split="val", source="youtube_sl25")
        load_fn = load_npy_features
    elif stage == "combined":
        from tsl.data.thaisignvis import load_npy_manifest, load_npy_features
        from tsl.data.tsl51 import load_landmark_sequence, load_sentence_manifest
        roots = [r.strip() for r in data_root.split(",")]
        tsl51_root = roots[0]
        ytsl25_root = roots[1] if len(roots) > 1 else None
        all_tsl51 = load_sentence_manifest(tsl51_root)
        tsl51_train, tsl51_val = _split_examples(all_tsl51, seed=42)
        ytsl25_train: list = []
        ytsl25_val: list = []
        if ytsl25_root:
            ytsl25_train = load_npy_manifest(ytsl25_root, split="train", source="youtube_sl25")
            ytsl25_val = load_npy_manifest(ytsl25_root, split="val", source="youtube_sl25")
        train_ex = tsl51_train + ytsl25_train
        val_ex = tsl51_val + ytsl25_val
        def load_fn(path: str) -> np.ndarray:
            if path.endswith(".npy"):
                return load_npy_features(path)
            return load_landmark_sequence(path)
    else:
        from tsl.data.tsl51 import load_landmark_sequence, load_sentence_manifest
        all_ex = load_sentence_manifest(data_root)
        train_ex, val_ex = _split_examples(all_ex, seed=42)
        load_fn = load_landmark_sequence
    if limit is not None:
        train_ex = train_ex[:limit]
    return train_ex, val_ex, load_fn


def _preload_features(examples: list, load_fn) -> dict:
    """Load every feature array into a dict keyed by features_path."""
    cache: dict = {}
    for ex in examples:
        if ex.features_path not in cache:
            cache[ex.features_path] = load_fn(ex.features_path)
    return cache


def _make_batches(
    examples: list,
    batch_size: int,
    tokenizer,
    load_features=None,
    cache: dict | None = None,
) -> list:
    if not examples:
        return []
    if cache is not None:
        def _cached_load(path: str):
            return cache[path]
        effective_load = _cached_load
    else:
        effective_load = load_features
    return [
        slt_collate(
            examples[i : i + batch_size],
            tokenizer,
            load_features=effective_load,
        )
        for i in range(0, len(examples), batch_size)
    ]


def _load_pretrained_weights(
    model: SignToTextTransformer,
    checkpoint_dir: str,
    device: torch.device,
    expected_config: dict | None = None,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
) -> SignToTextTransformer:
    state_path = os.path.join(checkpoint_dir, "slt_model.pt")
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {state_path!r}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"pretrained model config not found: {config_path!r}")

    with open(config_path, "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    if expected_config is not None:
        structural_keys = (
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
            "max_pos_len",
        )
        mismatched = {
            key: (saved_config.get(key), expected_config.get(key))
            for key in structural_keys
            if saved_config.get(key) != expected_config.get(key)
        }
        if mismatched:
            raise ValueError(
                "pretrained checkpoint architecture does not match current model: "
                f"{mismatched}"
            )

    pretrained = torch.load(state_path, map_location=device, weights_only=True)
    model_state = model.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, val in pretrained.items():
        if key in model_state and model_state[key].shape == val.shape:
            compatible[key] = val
        else:
            skipped.append(key)

    model_state.update(compatible)
    model.load_state_dict(model_state)
    if skipped:
        print(f"  skipped {len(skipped)} mismatched keys: {skipped}")

    if freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("input_proj") or name.startswith("encoder."):
                param.requires_grad = False

    if freeze_decoder:
        for name, param in model.named_parameters():
            if name.startswith("tgt_embed") or name.startswith("decoder.") or name.startswith("out_proj"):
                param.requires_grad = False

    return model


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device, args.require_gpu)

    input_dim = _resolve_input_dim(args.stage, args.input_dim)
    train_examples, val_examples, load_fn = _load_data(args.stage, args.data_root, args.limit)

    all_texts = [ex.target_text for ex in train_examples + val_examples]
    if args.tokenizer == "word":
        tokenizer = WordTokenizer(all_texts)
        print(f"Tokenizer: word-level (vocab_size={WordTokenizer(all_texts).vocab_size})")
    else:
        tokenizer = CharTokenizer(all_texts)
        print(f"Tokenizer: char-level (vocab_size={CharTokenizer(all_texts).vocab_size})")

    cfg = resolve_config(args.model_size, input_dim)
    config_dict = asdict(cfg)
    model = _build_model(tokenizer.vocab_size, args.model_size, input_dim).to(device)

    if args.stage == "finetune":
        if args.pretrained_checkpoint is None:
            raise ValueError("--pretrained-checkpoint is required for --stage finetune")
        print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
        model = _load_pretrained_weights(model, args.pretrained_checkpoint, device, expected_config=config_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    # Preload all features into RAM once (avoids per-epoch /mnt disk reads)
    print("Preloading features into memory … ", end="", flush=True)
    train_feat_cache = _preload_features(train_examples, load_fn)
    val_feat_cache = _preload_features(val_examples, load_fn)
    print(f"done ({len(train_feat_cache)+len(val_feat_cache)} arrays)")

    if args.augment:
        from tsl.data.augment import augment_sequence
        _aug_rng = np.random.default_rng(args.seed)

        def _augmented_load_fn(path: str):
            seq = train_feat_cache[path]
            return augment_sequence(seq, _aug_rng)

        train_load_fn = _augmented_load_fn
        print("Data augmentation enabled (time-stretch, jitter, frame-dropout, mirror).")
    else:
        train_load_fn = None  # will use cache

    metrics: dict = {"epochs": [], "final_train_loss": None}
    final_train_loss = 0.0
    for epoch in range(args.epochs):
        if args.augment:
            train_batches = _make_batches(
                train_examples, args.batch_size, tokenizer, load_features=train_load_fn
            )
        else:
            train_batches = _make_batches(
                train_examples, args.batch_size, tokenizer, cache=train_feat_cache
            )
        train_loss = train_one_epoch(
            model, train_batches, optimizer, device, tokenizer
        )
        val_batches = _make_batches(val_examples, args.batch_size, tokenizer, cache=val_feat_cache)
        val_loss = eval_loss(model, val_batches, device, tokenizer)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}"
        )
        metrics["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr}
        )
        final_train_loss = train_loss
    metrics["final_train_loss"] = final_train_loss

    torch.save(model.state_dict(), os.path.join(args.out_dir, "slt_model.pt"))
    _save_tokenizer(tokenizer, os.path.join(args.out_dir, "tokenizer.json"))
    _save_model_config(config_dict, tokenizer.vocab_size, os.path.join(args.out_dir, "model_config.json"))
    _save_metrics(metrics, os.path.join(args.out_dir, "train_metrics.json"))
    print(f"\nTraining complete. Output saved to {args.out_dir}")

    # ---- chrF evaluation on validation set ----------------------------------
    if not args.no_eval and val_examples:
        print("\nEvaluating on val set (beam_size={}) …".format(args.eval_beam))
        from tsl.eval.slt_metrics import evaluate_slt
        from tsl.inference.sentence_translator import SentenceTranslator

        translator = SentenceTranslator.from_model_and_tokenizer(
            model, tokenizer, device=str(device)
        )
        eval_results = evaluate_slt(
            translator, val_examples, load_fn,
            beam_size=args.eval_beam, verbose=True,
        )
        metrics["val_chrf"] = eval_results["chrf"]
        metrics["val_exact_match_pct"] = eval_results["exact_match_pct"]
        _save_metrics(metrics, os.path.join(args.out_dir, "train_metrics.json"))
        print(f"\n  chrF  : {eval_results['chrf']:.1f}")
        print(f"  exact : {eval_results['exact_match']}/{eval_results['n']} ({eval_results['exact_match_pct']:.1f}%)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
