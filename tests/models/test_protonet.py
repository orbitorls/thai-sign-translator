import torch

from tsl.models.protonet import compute_prototypes


def test_compute_prototypes_shape():
    emb = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    protos = compute_prototypes(emb, labels, n_way=3)
    assert protos.shape == (3, 4)


def test_compute_prototypes_are_class_means_ordered_by_class():
    emb = torch.tensor([
        [0.0, 2.0], [2.0, 0.0],
        [4.0, 6.0], [6.0, 4.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    protos = compute_prototypes(emb, labels, n_way=2)
    assert torch.allclose(protos[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(protos[1], torch.tensor([5.0, 5.0]))


from tsl.models.protonet import euclidean_logits


def test_euclidean_logits_shape():
    query = torch.randn(5, 4)
    protos = torch.randn(3, 4)
    logits = euclidean_logits(query, protos)
    assert logits.shape == (5, 3)


def test_euclidean_logits_ranks_matching_prototype_highest():
    protos = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    query = torch.tensor([[0.1, 0.0], [9.9, 10.0]])
    logits = euclidean_logits(query, protos)
    assert logits.argmax(dim=1).tolist() == [0, 1]


def test_euclidean_logits_is_negative_squared_distance():
    query = torch.tensor([[0.0, 0.0]])
    protos = torch.tensor([[3.0, 4.0]])
    logits = euclidean_logits(query, protos)
    assert torch.allclose(logits, torch.tensor([[-25.0]]), atol=1e-4)


from tsl.models.protonet import proto_loss


def test_proto_loss_returns_finite_scalar_and_valid_acc():
    torch.manual_seed(0)
    emb_dim, n_way = 4, 3
    support_emb = torch.randn(n_way * 2, emb_dim)
    support_y = torch.tensor([0, 0, 1, 1, 2, 2])
    query_emb = torch.randn(n_way * 2, emb_dim)
    query_y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss, acc = proto_loss(support_emb, support_y, query_emb, query_y, n_way)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert 0.0 <= acc.item() <= 1.0


def test_proto_loss_perfect_accuracy_on_separable_case():
    n_way = 3
    support_emb = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    support_y = torch.tensor([0, 1, 2])
    query_emb = torch.tensor([[0.1, 0.0], [10.0, 9.9], [20.0, 20.1]])
    query_y = torch.tensor([0, 1, 2])
    loss, acc = proto_loss(support_emb, support_y, query_emb, query_y, n_way)
    assert acc.item() == 1.0
    assert torch.isfinite(loss)
