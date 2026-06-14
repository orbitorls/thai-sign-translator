import json
import os
import sys
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.record_thai import save_clip

from tsl.data.thai import load_thai_clips


def test_save_clip_writes_npy_and_source_json(tmp_path):
    raw = np.ones((4, 543, 3), dtype=np.float32)
    npy_path, json_path = save_clip(str(tmp_path), word="sawasdee", raw_seq=raw, source="thai-dict-v1")
    assert os.path.isfile(npy_path)
    assert os.path.isfile(json_path)
    assert os.path.basename(os.path.dirname(npy_path)) == "sawasdee"
    assert npy_path.endswith(".npy")
    assert json_path.endswith(".json")
    loaded = np.load(npy_path)
    assert loaded.shape == (4, 543, 3)
    meta = json.loads(open(json_path, encoding="utf-8").read())
    assert meta["word"] == "sawasdee"
    assert meta["source"] == "thai-dict-v1"


def test_save_clip_autoincrements_index(tmp_path):
    raw = np.zeros((2, 543, 3), dtype=np.float32)
    p0, _ = save_clip(str(tmp_path), "khap", raw, "thai-dict-v1")
    p1, _ = save_clip(str(tmp_path), "khap", raw, "thai-dict-v1")
    assert os.path.basename(p0) == "0.npy"
    assert os.path.basename(p1) == "1.npy"


def test_output_round_trips_through_load_thai_clips(tmp_path):
    raw = np.ones((3, 543, 3), dtype=np.float32)
    save_clip(str(tmp_path), "sawasdee", raw, "thai-dict-v1")
    save_clip(str(tmp_path), "khap", raw, "thai-dict-v1")
    clips = load_thai_clips(str(tmp_path))
    assert set(clips.keys()) == {"sawasdee", "khap"}
    assert clips["khap"][0].ndim == 2
    assert clips["khap"][0].shape[0] == 3
    assert clips["khap"][0].dtype == np.float32
