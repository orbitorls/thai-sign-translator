import torch
from tsl.data.episodic import EpisodicSampler


class _FakeDataset:
    def __init__(self, n_classes=4, per_class=6, T=3, D=6):
        self.D = D
        self._items = []
        self._labels = []
        for c in range(n_classes):
            for _ in range(per_class):
                self._items.append(torch.full((T, D), float(c)))
                self._labels.append(c)
        self.num_classes = n_classes
        self.label_names = [f"c{c}" for c in range(n_classes)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i], self._labels[i]


def test_episode_shapes_and_label_remap():
    n_way, k_shot, q_query, episodes = 3, 2, 2, 5
    ds = _FakeDataset(n_classes=4, per_class=6, T=3, D=6)
    sampler = EpisodicSampler(ds, n_way, k_shot, q_query, episodes)
    count = 0
    for ep in sampler:
        count += 1
        assert set(ep.keys()) == {"support_x", "support_y", "query_x", "query_y"}
        assert ep["support_x"].shape == (n_way * k_shot, 3, 6)
        assert ep["query_x"].shape == (n_way * q_query, 3, 6)
        assert ep["support_y"].shape == (n_way * k_shot,)
        assert ep["query_y"].shape == (n_way * q_query,)
        assert int(ep["support_y"].min()) >= 0
        assert int(ep["support_y"].max()) < n_way
        assert set(ep["support_y"].tolist()) == set(range(n_way))
        assert set(ep["query_y"].tolist()).issubset(set(range(n_way)))
    assert count == episodes


def test_support_and_query_are_disjoint_per_class():
    ds = _FakeDataset(n_classes=3, per_class=5, T=2, D=4)
    sampler = EpisodicSampler(ds, n_way=2, k_shot=2, q_query=2, episodes=1)
    ep = next(iter(sampler))
    for remapped in range(2):
        assert (ep["support_y"] == remapped).sum().item() == 2
        assert (ep["query_y"] == remapped).sum().item() == 2
