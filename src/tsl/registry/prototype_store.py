from __future__ import annotations

import numpy as np
import torch

from tsl.models.encoder import LandmarkEncoder
from tsl.models.protonet import euclidean_logits


class PrototypeStore:
    """Gradient-free few-shot registry of sign prototypes.

    A prototype is the mean encoder embedding over a sign's example clips.
    Adding a sign never calls backward()/optimizer; embeddings are computed
    under torch.no_grad() with the encoder in eval mode.
    """

    def __init__(self, encoder: LandmarkEncoder):
        self.encoder = encoder
        self.encoder.eval()
        self._prototypes: dict[str, torch.Tensor] = {}

    def _embed_clip(self, clip: np.ndarray) -> torch.Tensor:
        arr = np.asarray(clip, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"clip must be (T, D); got shape {arr.shape}")
        x = torch.from_numpy(arr).unsqueeze(0)
        lengths = torch.tensor([arr.shape[0]], dtype=torch.long)
        with torch.no_grad():
            emb = self.encoder(x, lengths)
        return emb.squeeze(0).detach()

    def add_sign(self, name: str, clips: list[np.ndarray]) -> None:
        if not clips:
            raise ValueError(f"add_sign requires at least one clip for '{name}'")
        embs = [self._embed_clip(c) for c in clips]
        proto = torch.stack(embs, dim=0).mean(dim=0).detach()
        self._prototypes[name] = proto

    def remove_sign(self, name: str) -> None:
        self._prototypes.pop(name, None)

    def names(self) -> list[str]:
        return list(self._prototypes.keys())

    def _stacked(self) -> tuple[list[str], torch.Tensor]:
        names = list(self._prototypes.keys())
        protos = torch.stack([self._prototypes[n] for n in names], dim=0)
        return names, protos

    def predict(self, seq: np.ndarray) -> tuple[str, float]:
        if not self._prototypes:
            raise ValueError("PrototypeStore is empty; add a sign before predict()")
        names, protos = self._stacked()
        query = self._embed_clip(seq).unsqueeze(0)
        logits = euclidean_logits(query, protos)
        idx = int(torch.argmax(logits, dim=1).item())
        return names[idx], float(logits[0, idx].item())

    def save(self, path: str) -> None:
        payload = {
            "names": list(self._prototypes.keys()),
            "prototypes": {n: p.cpu() for n, p in self._prototypes.items()},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, encoder: LandmarkEncoder) -> "PrototypeStore":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        store = cls(encoder)
        for name in payload["names"]:
            store._prototypes[name] = payload["prototypes"][name].detach()
        return store
