from __future__ import annotations

import os

import pandas as pd

from scripts.extract_thaisignvis_landmarks import (
    _build_segments,
    _find_video,
    _iter_transcript_files,
)


def test_iter_transcript_files_finds_nested_kaggle_layout(tmp_path):
    session_dir = tmp_path / "ThaiSignVis" / "process_videos" / "process_videos" / "session_a"
    session_dir.mkdir(parents=True)
    pd.DataFrame([{"start_ms": 0, "end_ms": 1000, "text": "alpha"}]).to_csv(
        session_dir / "transcript_window_0.csv",
        index=False,
    )

    rows = list(_iter_transcript_files(str(tmp_path)))

    assert len(rows) == 1
    session, transcript_dir, transcript_tag, df = rows[0]
    assert session == "session_a"
    assert transcript_dir == str(session_dir)
    assert transcript_tag == "transcript_window_0"
    assert list(df["text"]) == ["alpha"]


def test_build_segments_uses_transcript_tag_in_segment_id(tmp_path):
    session_dir = tmp_path / "process_videos" / "session_b"
    session_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"start_ms": 0, "end_ms": 1000, "text": "alpha"},
            {"start_ms": 1000, "end_ms": 2000, "text": "beta"},
        ]
    ).to_csv(session_dir / "transcript_window_3.csv", index=False)

    segments = _build_segments(str(tmp_path), limit=None)

    assert [segment["segment_id"] for segment in segments] == [
        "session_b__transcript_window_3__00000",
        "session_b__transcript_window_3__00001",
    ]
    assert all(segment["session_dir"] == str(session_dir) for segment in segments)


def test_build_segments_accepts_start_plus_duration_in_seconds(tmp_path):
    session_dir = tmp_path / "process_videos" / "session_d"
    session_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"start": 1.5, "duration": 2.25, "text": "alpha"},
        ]
    ).to_csv(session_dir / "transcript_window_0.csv", index=False)

    segments = _build_segments(str(tmp_path), limit=None)

    assert segments[0]["start_ms"] == 1500.0
    assert segments[0]["end_ms"] == 3750.0


def test_find_video_prefers_process_video_zero(tmp_path):
    session_dir = tmp_path / "process_videos" / "session_c"
    session_dir.mkdir(parents=True)
    (session_dir / "process_video_0.mp4").write_bytes(b"0")
    (session_dir / "process_video_1.mp4").write_bytes(b"1")

    video_path = _find_video(str(session_dir), "")

    assert video_path == os.path.join(str(session_dir), "process_video_0.mp4")
