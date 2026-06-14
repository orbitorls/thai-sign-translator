"""Episodic Prototypical-Network training for the landmark encoder.

Run locally for a smoke test, or on Kaggle/Colab free GPU for real training.
"""
from __future__ import annotations

import os

import torch

from tsl.data.episodic import EpisodicSampler, collate_pad
from tsl.models.encoder import LandmarkEncoder
from tsl.models.protonet import proto_loss


def _encode_split(encoder, x, device):
    nonpad = (x.abs().sum(dim=-1) > 0)
    lengths = nonpad.sum(dim=1).clamp(min=1).long()
    x = x.to(device)
    lengths = lengths.to(device)
    return encoder(x, lengths)


def _pad_to_tmax(x):
    seqs = [x[i] for i in range(x.shape[0])]
    padded, _ = collate_pad(seqs)
    return padded


def train(
    dataset,
    sampler: EpisodicSampler,
    epochs: int = 10,
    lr: float = 1e-3,
    emb_dim: int = 256,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    device: str = "cpu",
    checkpoint_path: str = "encoder_best.pt",
    export_path: str = "encoder_weights.pt",
    log_every: int = 50,
) -> list[dict]:
    sample_x, _ = dataset[0]
    input_dim = int(sample_x.shape[-1])
    encoder = LandmarkEncoder(
        input_dim=input_dim, emb_dim=emb_dim, d_model=d_model,
        nhead=nhead, num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    history: list[dict] = []
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_losses = []
        for ep_idx, episode in enumerate(sampler):
            support_x = _pad_to_tmax(episode["support_x"])
            query_x = _pad_to_tmax(episode["query_x"])
            support_y = episode["support_y"].to(device)
            query_y = episode["query_y"].to(device)
            encoder.train()
            optimizer.zero_grad()
            support_emb = _encode_split(encoder, support_x, device)
            query_emb = _encode_split(encoder, query_x, device)
            loss, acc = proto_loss(support_emb, support_y, query_emb, query_y, sampler.n_way)
            loss.backward()
            optimizer.step()
            loss_v = float(loss.detach().cpu())
            acc_v = float(acc.detach().cpu())
            epoch_losses.append(loss_v)
            history.append({"epoch": epoch, "episode": ep_idx, "loss": loss_v, "acc": acc_v})
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        torch.save(encoder.state_dict(), checkpoint_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(encoder.state_dict(), checkpoint_path)
    torch.save(encoder.state_dict(), export_path)
    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    islr_root = os.environ.get("ISLR_ROOT", "/kaggle/input/asl-signs")
    parquet_dir = os.environ.get("ISLR_PARQUET_DIR", os.path.join(islr_root, "train_landmark_files"))
    csv_path = os.environ.get("ISLR_CSV", os.path.join(islr_root, "train.csv"))
    out_dir = os.environ.get("OUT_DIR", "/kaggle/working")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "encoder_best.pt")
    export_path = os.path.join(out_dir, "encoder_weights.pt")
    from tsl.data.islr import ISLRDataset
    dataset = ISLRDataset(parquet_dir=parquet_dir, csv_path=csv_path, classes=None)
    sampler = EpisodicSampler(dataset, n_way=5, k_shot=5, q_query=5, episodes=200)
    train(dataset=dataset, sampler=sampler, epochs=30, lr=1e-3, device=device,
          checkpoint_path=checkpoint_path, export_path=export_path)
