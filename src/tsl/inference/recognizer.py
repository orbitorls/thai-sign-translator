from __future__ import annotations

import numpy as np
import torch

from tsl.models.protonet import euclidean_logits
from tsl.registry.prototype_store import PrototypeStore


class Recognizer:
    """Encodes a normalized query sequence and ranks it against stored prototypes."""

    def __init__(self, store: PrototypeStore):
        self.store = store

    def recognize(self, seq_norm: np.ndarray) -> dict:
        if not self.store._prototypes:
            raise ValueError("PrototypeStore is empty; add a sign before recognize()")
        names, protos = self.store._stacked()
        query = self.store._embed_clip(seq_norm).unsqueeze(0)
        logits = euclidean_logits(query, protos)
        scores = logits.squeeze(0)
        order = torch.argsort(scores, descending=True)
        topk = [(names[int(i)], float(scores[int(i)].item())) for i in order]
        return {"word": topk[0][0], "score": topk[0][1], "topk": topk}
