import torch
from tsl.data.episodic import collate_pad


def test_collate_pad_shapes_and_lengths():
    D = 6
    a = torch.ones(3, D)
    b = torch.ones(5, D) * 2.0
    c = torch.ones(2, D) * 3.0
    x, lengths = collate_pad([(a, 0), (b, 1), (c, 2)])
    assert x.shape == (3, 5, D)
    assert lengths.tolist() == [3, 5, 2]
    assert x.dtype == torch.float32
    assert torch.all(x[0, :3] == 1.0)
    assert torch.all(x[0, 3:] == 0.0)
    assert torch.all(x[2, :2] == 3.0)
    assert torch.all(x[2, 2:] == 0.0)


def test_collate_pad_accepts_plain_tensor_batch():
    x, lengths = collate_pad([torch.ones(2, 4), torch.ones(4, 4)])
    assert x.shape == (2, 4, 4)
    assert lengths.tolist() == [2, 4]
