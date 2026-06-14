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
import json
import os
import random

import torch
import torch.nn.functional as F

from tsl.data.slt_collate import slt_collate
from tsl.data.tsl51 import load_landmark_sequence, load_sentence_manifest
from tsl.models.slt import SignToTextTransformer
from tsl.text.tokenizer import CharTokenizer


_INPUT_DIM = 162


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _right_shift_target(tgt: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Right-shift ``tgt`` by one position and prepend ``bos_id``.

    Used for teacher forcing: if ``tgt`` is the gold ``[BOS, w1, w2, ...]``
    row, the returned tensor is ``[BOS, BOS, w1, w2, ...]`` so the
    decoder's output at position ``i`` predicts ``tgt[:, i+1]``.
    """
    if tgt.ndim != 2:
        raise ValueError(f"tgt must be 2-D (B, T), got shape {tuple(tgt.shape)}")
    B, T = tgt.shape
    shifted = torch.full((B, T), bos_id, dtype=tgt.dtype, device=tgt.device)
    if T > 0:
        shifted[:, 0] = bos_id
        if T > 1:
            shifted[:, 1:] = tgt[:, :-1]
    return shifted


_MODEL_KWARGS: dict = {
    "input_dim": _INPUT_DIM,
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "max_pos_len": 1024,
}


def _build_model(vocab_size: int) -> SignToTextTransformer:
    """Build a small :class:`SignToTextTransformer` for smoke + training."""
    return SignToTextTransformer(vocab_size=vocab_size, **_MODEL_KWARGS)


def _save_model_config(vocab_size: int, path: str) -> None:
    """Write the constructor kwargs (incl. ``vocab_size``) to ``path`` as JSON.

    The inference wrapper reads this file to rebuild the model from a
    checkpoint without having to re-derive the architecture defaults.
    """
    config = {**_MODEL_KWARGS, "vocab_size": vocab_size}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _save_tokenizer(tok: CharTokenizer, path: str) -> None:
    """Serialize a :class:`CharTokenizer` to ``path`` as JSON."""
    vocab = list(tok._char_to_id.keys())
    payload = {
        "pad": "<pad>",
        "bos": "<bos>",
        "eos": "<eos>",
        "unk": "<unk>",
        "vocab": vocab,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tokenizer(path: str) -> CharTokenizer:
    """Load a :class:`CharTokenizer` previously written by :func:`_save_tokenizer`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON is missing any of the required keys.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"tokenizer file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("pad", "bos", "eos", "unk", "vocab"):
        if key not in data:
            raise ValueError(f"tokenizer file missing key: {key!r}")
    tok = CharTokenizer()
    tok._char_to_id = {ch: i for i, ch in enumerate(data["vocab"])}
    return tok


def _save_metrics(metrics: dict, path: str) -> None:
    """Write the per-epoch metrics dict to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------


def _move_batch(batch, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """Compute next-token cross-entropy on a single batch."""
    shifted = _right_shift_target(tgt, bos_id)
    logits = model(src, src_lengths, shifted)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tgt[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )


def train_one_epoch(
    model: SignToTextTransformer,
    batches,
    optimizer: torch.optim.Optimizer,
    device: str,
    tokenizer: CharTokenizer,
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
    device: str,
    tokenizer: CharTokenizer,
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
        description="Train SignToTextTransformer on TSL-51 sentence data."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the TSL-51 dataset root (containing metadata/sentence_metadata.csv).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="checkpoints/slt")
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device string (e.g. "cpu", "cuda", "mps").',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of training examples (deterministic: take first N).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _split_examples(
    examples: list, seed: int
) -> tuple[list, list]:
    """Deterministic 90/10 train/val split."""
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = len(examples) // 10
    val_set = set(indices[:n_val])
    train_examples = [examples[i] for i in indices if i not in val_set]
    val_examples = [examples[i] for i in indices if i in val_set]
    return train_examples, val_examples


def _make_batches(
    examples: list, batch_size: int, tokenizer: CharTokenizer
) -> list:
    if not examples:
        return []
    return [
        slt_collate(
            examples[i : i + batch_size],
            tokenizer,
            load_features=load_landmark_sequence,
        )
        for i in range(0, len(examples), batch_size)
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_sentence_manifest(args.data_root)
    if args.limit is not None:
        examples = examples[: args.limit]

    train_examples, val_examples = _split_examples(examples, args.seed)
    tokenizer = CharTokenizer([ex.target_text for ex in examples])

    model = _build_model(tokenizer.vocab_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics: dict = {"epochs": [], "final_train_loss": None}
    final_train_loss = 0.0
    for epoch in range(args.epochs):
        train_batches = _make_batches(train_examples, args.batch_size, tokenizer)
        train_loss = train_one_epoch(
            model, train_batches, optimizer, args.device, tokenizer
        )
        val_batches = _make_batches(val_examples, args.batch_size, tokenizer)
        val_loss = eval_loss(model, val_batches, args.device, tokenizer)
        print(
            f"epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        metrics["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        final_train_loss = train_loss
    metrics["final_train_loss"] = final_train_loss

    torch.save(model.state_dict(), os.path.join(args.out_dir, "slt_model.pt"))
    _save_tokenizer(tokenizer, os.path.join(args.out_dir, "tokenizer.json"))
    _save_model_config(tokenizer.vocab_size, os.path.join(args.out_dir, "model_config.json"))
    _save_metrics(metrics, os.path.join(args.out_dir, "train_metrics.json"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
