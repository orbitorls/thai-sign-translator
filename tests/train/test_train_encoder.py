import math

import torch

from tsl.data.episodic import EpisodicSampler
from tsl.train.train_encoder import train


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, n_classes=4, per_class=5, d=12):
        self.d = d
        self.label_names = [f"sign_{c}" for c in range(n_classes)]
        self._items = []
        g = torch.Generator().manual_seed(0)
        for c in range(n_classes):
            for _ in range(per_class):
                t = int(torch.randint(6, 12, (1,), generator=g).item())
                x = torch.randn(t, d, generator=g)
                self._items.append((x, c))

    @property
    def num_classes(self):
        return len(self.label_names)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def test_train_smoke_writes_checkpoints_and_finite_loss(tmp_path):
    ds = FakeDataset(n_classes=4, per_class=5, d=12)
    sampler = EpisodicSampler(ds, n_way=3, k_shot=2, q_query=2, episodes=4)
    ckpt_path = tmp_path / "encoder_best.pt"
    export_path = tmp_path / "encoder_weights.pt"
    history = train(
        dataset=ds, sampler=sampler, epochs=2, lr=1e-3,
        emb_dim=16, d_model=16, nhead=2, num_layers=1, device="cpu",
        checkpoint_path=str(ckpt_path), export_path=str(export_path),
    )
    assert isinstance(history, list)
    assert len(history) > 0
    for rec in history:
        assert math.isfinite(rec["loss"])
        assert 0.0 <= rec["acc"] <= 1.0
    assert ckpt_path.exists()
    assert export_path.exists()
    from tsl.models.encoder import LandmarkEncoder
    enc = LandmarkEncoder(input_dim=12, emb_dim=16, d_model=16, nhead=2, num_layers=1)
    enc.load_state_dict(torch.load(str(export_path), map_location="cpu"))
