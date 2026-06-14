import json
import numpy as np
from tsl.data.thai import load_thai_clips


def _write_clip(dir_path, name, T=3):
    raw = np.ones((T, 543, 3), dtype=np.float32)
    np.save(dir_path / f"{name}.npy", raw)
    (dir_path / f"{name}.json").write_text(
        json.dumps({"word": dir_path.name, "source": "thai-dict-v1"})
    )


def test_load_thai_clips_groups_by_word_and_normalizes(tmp_path):
    sawasdee = tmp_path / "sawasdee"
    khap = tmp_path / "khap"
    sawasdee.mkdir()
    khap.mkdir()
    _write_clip(sawasdee, "0", T=3)
    _write_clip(sawasdee, "1", T=4)
    _write_clip(khap, "0", T=2)
    clips = load_thai_clips(str(tmp_path))
    assert set(clips.keys()) == {"sawasdee", "khap"}
    assert len(clips["sawasdee"]) == 2
    assert len(clips["khap"]) == 1
    clip = clips["khap"][0]
    assert clip.ndim == 2
    assert clip.shape[0] == 2
    assert clip.dtype == np.float32


def test_load_thai_clips_empty_root(tmp_path):
    assert load_thai_clips(str(tmp_path)) == {}
