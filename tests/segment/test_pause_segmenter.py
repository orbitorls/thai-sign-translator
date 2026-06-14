import numpy as np
from tsl.segment.pause_segmenter import segment_stream


def _build_seq(D=12, left_wrist=0, right_wrist=3):
    T = 28
    seq = np.zeros((T, D), dtype=np.float32)
    pos = 0.0
    for t in range(T):
        if t < 10 or t >= 18:
            pos += 0.5
        seq[t, left_wrist:left_wrist + 3] = pos
        seq[t, right_wrist:right_wrist + 3] = pos
    return seq


def test_clear_pause_yields_two_segments():
    seq = _build_seq()
    spans = segment_stream(seq, wrist_idx_in_D=(0, 3), v_thresh=0.05, min_pause_frames=4)
    assert isinstance(spans, list)
    assert len(spans) == 2
    for start, end in spans:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert 0 <= start < end <= seq.shape[0]
    assert spans[0][1] <= 10
    assert spans[1][0] >= 18


def test_no_pause_yields_single_segment():
    T, D = 20, 12
    seq = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        seq[t, 0:3] = 0.5 * t
        seq[t, 3:6] = 0.5 * t
    spans = segment_stream(seq, wrist_idx_in_D=(0, 3), v_thresh=0.05, min_pause_frames=4)
    assert spans == [(0, T)]


def test_short_pause_below_min_frames_does_not_split():
    T, D = 16, 12
    seq = np.zeros((T, D), dtype=np.float32)
    pos = 0.0
    for t in range(T):
        if t not in (7, 8):
            pos += 0.5
        seq[t, 0:3] = pos
        seq[t, 3:6] = pos
    spans = segment_stream(seq, wrist_idx_in_D=(0, 3), v_thresh=0.05, min_pause_frames=4)
    assert spans == [(0, T)]


def test_empty_or_single_frame_returns_empty():
    assert segment_stream(np.zeros((0, 12), dtype=np.float32), (0, 3)) == []
    assert segment_stream(np.zeros((1, 12), dtype=np.float32), (0, 3)) == [(0, 1)]
