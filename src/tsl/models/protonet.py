"""Prototypical-network operations: prototypes, euclidean logits, episodic loss."""
import torch
import torch.nn.functional as F


def compute_prototypes(emb: torch.Tensor, labels: torch.Tensor, n_way: int) -> torch.Tensor:
    protos = []
    for n in range(n_way):
        protos.append(emb[labels == n].mean(dim=0))
    return torch.stack(protos, dim=0)


def euclidean_logits(query_emb: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(query_emb, prototypes)
    return -dist.pow(2)


def proto_loss(
    support_emb: torch.Tensor,
    support_y: torch.Tensor,
    query_emb: torch.Tensor,
    query_y: torch.Tensor,
    n_way: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = compute_prototypes(support_emb, support_y, n_way)
    logits = euclidean_logits(query_emb, prototypes)
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=1) == query_y).float().mean()
    return loss, acc
