import os
from tsl.eval.evaluate import evaluate_track


class FakeStore:
    def __init__(self, mapping):
        self._mapping = mapping

    def names(self):
        return ["hello", "thanks", "yes"]

    def predict(self, seq):
        tag = int(seq[0])
        return self._mapping[tag]


def test_evaluate_track_computes_metrics(tmp_path):
    import numpy as np
    label_to_id = {"hello": 0, "thanks": 1, "yes": 2}
    mapping = {0: ("hello", 0.9), 1: ("thanks", 0.8), 2: ("yes", 0.7), 3: ("hello", 0.6)}
    store = FakeStore(mapping)
    clips = [np.array([float(t)]) for t in range(4)]
    true_words = ["hello", "thanks", "yes", "yes"]
    out_png = str(tmp_path / "track_cm.png")
    summary = evaluate_track(store, clips, true_words, label_to_id, out_png=out_png)
    assert summary["n"] == 4
    assert summary["accuracy"] == 0.75
    assert "top5_accuracy" in summary
    assert os.path.exists(out_png)


def test_evaluate_track_topk_uses_names_when_no_topk(tmp_path):
    import numpy as np
    label_to_id = {"hello": 0, "thanks": 1, "yes": 2}
    mapping = {0: ("hello", 0.9)}
    store = FakeStore(mapping)
    clips = [np.array([0.0])]
    summary = evaluate_track(store, clips, ["hello"], label_to_id, out_png=None)
    assert summary["accuracy"] == 1.0
    assert summary["top5_accuracy"] == 1.0
