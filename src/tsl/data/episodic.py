"""Episodic sampling and variable-length collation for ProtoNet training."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


def collate_pad(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length (T, D) tensors to (B, Tmax, D)."""
    if len(batch) > 0 and isinstance(batch[0], (tuple, list)):
        seqs = [item[0] for item in batch]
    else:
        seqs = list(batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    t_max = int(lengths.max().item())
    d = seqs[0].shape[1]
    x = torch.zeros(len(seqs), t_max, d, dtype=torch.float32)
    for i, s in enumerate(seqs):
        t = s.shape[0]
        x[i, :t] = s.to(torch.float32)
    return x, lengths


class EpisodicSampler:
    """Yield N-way K-shot (+Q query) episodes from a labelled dataset."""

    def __init__(self, dataset, n_way: int, k_shot: int, q_query: int, episodes: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        self._by_label: dict[int, list[int]] = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self._by_label.setdefault(int(label), []).append(idx)
        self._classes = [c for c, idxs in self._by_label.items() if len(idxs) >= k_shot + q_query]

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.default_rng()
        for _ in range(self.episodes):
            chosen = rng.choice(self._classes, size=self.n_way, replace=False)
            support_seqs, support_y = [], []
            query_seqs, query_y = [], []
            for remapped, orig in enumerate(chosen):
                idxs = self._by_label[int(orig)]
                picked = rng.choice(idxs, size=self.k_shot + self.q_query, replace=False)
                sup_idx = picked[: self.k_shot]
                qry_idx = picked[self.k_shot :]
                for j in sup_idx:
                    seq, _ = self.dataset[int(j)]
                    support_seqs.append(seq)
                    support_y.append(remapped)
                for j in qry_idx:
                    seq, _ = self.dataset[int(j)]
                    query_seqs.append(seq)
                    query_y.append(remapped)
            support_x, _ = collate_pad(support_seqs)
            query_x, _ = collate_pad(query_seqs)
            yield {
                "support_x": support_x,
                "support_y": torch.tensor(support_y, dtype=torch.long),
                "query_x": query_x,
                "query_y": torch.tensor(query_y, dtype=torch.long),
            }
