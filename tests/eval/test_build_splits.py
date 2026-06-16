"""Tests for tsl.eval.build_splits."""
from __future__ import annotations

import os
import tempfile

import pytest

from tsl.data.manifest import SignTextExample
from tsl.eval.build_splits import (
    check_video_leakage,
    read_frozen_test_examples,
    split_by_video,
    write_splits_to_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(
    example_id: str,
    video_id: str | None = None,
    split: str = "train",
    source: str = "test_source",
) -> SignTextExample:
    metadata = {"video_id": video_id} if video_id is not None else None
    return SignTextExample(
        example_id=example_id,
        source=source,
        split=split,
        features_path=f"/data/{example_id}.npy",
        target_text="สวัสดี",
        metadata=metadata,
    )


def _make_dataset(n_videos: int = 10, clips_per_video: int = 3) -> list[SignTextExample]:
    """Create a dataset where each video has multiple clips."""
    examples = []
    for v in range(n_videos):
        for c in range(clips_per_video):
            ex = _make_example(
                example_id=f"vid{v:02d}_clip{c:02d}",
                video_id=f"vid{v:02d}",
            )
            examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# split_by_video
# ---------------------------------------------------------------------------

class TestSplitByVideo:
    def test_correct_proportions(self):
        examples = _make_dataset(n_videos=10, clips_per_video=3)
        fracs = {"train": 0.8, "val": 0.1, "test": 0.1}
        splits = split_by_video(examples, fracs, seed=42)

        assert set(splits.keys()) == {"train", "val", "test"}
        total = sum(len(v) for v in splits.values())
        assert total == len(examples)

        # With 10 videos at 0.8/0.1/0.1 we expect 8/1/1 videos in each split
        # Each video has 3 clips, so 24/3/3 examples
        train_vids = {ex.metadata["video_id"] for ex in splits["train"]}
        val_vids = {ex.metadata["video_id"] for ex in splits["val"]}
        test_vids = {ex.metadata["video_id"] for ex in splits["test"]}
        assert len(train_vids) == 8
        assert len(val_vids) == 1
        assert len(test_vids) == 1

    def test_all_examples_assigned(self):
        examples = _make_dataset(n_videos=20, clips_per_video=2)
        splits = split_by_video(examples, {"train": 0.7, "val": 0.15, "test": 0.15})
        total = sum(len(v) for v in splits.values())
        assert total == len(examples)

    def test_no_video_leakage_guaranteed(self):
        examples = _make_dataset(n_videos=30, clips_per_video=4)
        splits = split_by_video(examples, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)
        # check_video_leakage should not raise
        check_video_leakage(splits["train"], splits["val"], splits["test"])

    def test_deterministic_with_seed(self):
        examples = _make_dataset(n_videos=15, clips_per_video=2)
        fracs = {"train": 0.8, "val": 0.2}
        s1 = split_by_video(examples, fracs, seed=7)
        s2 = split_by_video(examples, fracs, seed=7)
        assert [ex.example_id for ex in s1["train"]] == [ex.example_id for ex in s2["train"]]
        assert [ex.example_id for ex in s1["val"]] == [ex.example_id for ex in s2["val"]]

    def test_different_seeds_give_different_results(self):
        examples = _make_dataset(n_videos=20, clips_per_video=2)
        fracs = {"train": 0.8, "val": 0.2}
        s1 = split_by_video(examples, fracs, seed=1)
        s2 = split_by_video(examples, fracs, seed=2)
        # With 20 videos the probability of identical ordering is negligible
        train_ids_1 = [ex.example_id for ex in s1["train"]]
        train_ids_2 = [ex.example_id for ex in s2["train"]]
        assert train_ids_1 != train_ids_2

    def test_examples_without_video_id_use_example_id(self):
        """Examples with metadata=None fall back to example_id as group key."""
        ex_no_meta = SignTextExample(
            example_id="standalone_001",
            source="src",
            split="train",
            features_path="/data/standalone.npy",
            target_text="ขอบคุณ",
            metadata=None,
        )
        ex_no_vid_key = SignTextExample(
            example_id="standalone_002",
            source="src",
            split="train",
            features_path="/data/standalone2.npy",
            target_text="ขอบคุณ",
            metadata={},  # metadata present but no video_id key
        )
        examples = [ex_no_meta, ex_no_vid_key]
        splits = split_by_video(examples, {"train": 1.0}, seed=42)
        total = sum(len(v) for v in splits.values())
        assert total == 2

    def test_two_split(self):
        examples = _make_dataset(n_videos=10, clips_per_video=1)
        splits = split_by_video(examples, {"train": 0.8, "val": 0.2}, seed=42)
        assert len(splits) == 2
        assert sum(len(v) for v in splits.values()) == 10


# ---------------------------------------------------------------------------
# check_video_leakage
# ---------------------------------------------------------------------------

class TestCheckVideoLeakage:
    def _examples_for_video(self, video_id: str, n: int = 2) -> list[SignTextExample]:
        return [_make_example(f"{video_id}_clip{i}", video_id=video_id) for i in range(n)]

    def test_passes_when_clean(self):
        train = self._examples_for_video("vid_a") + self._examples_for_video("vid_b")
        val = self._examples_for_video("vid_c")
        test = self._examples_for_video("vid_d")
        # Should not raise
        check_video_leakage(train, val, test)

    def test_raises_on_train_val_leakage(self):
        train = self._examples_for_video("vid_a") + self._examples_for_video("vid_b")
        val = self._examples_for_video("vid_a")  # leakage!
        with pytest.raises(ValueError, match="vid_a"):
            check_video_leakage(train, val)

    def test_raises_on_train_test_leakage(self):
        train = self._examples_for_video("vid_x")
        val = self._examples_for_video("vid_y")
        test = self._examples_for_video("vid_x")  # leakage!
        with pytest.raises(ValueError, match="vid_x"):
            check_video_leakage(train, val, test)

    def test_raises_on_val_test_leakage(self):
        train = self._examples_for_video("vid_a")
        val = self._examples_for_video("vid_b")
        test = self._examples_for_video("vid_b")  # leakage!
        with pytest.raises(ValueError, match="vid_b"):
            check_video_leakage(train, val, test)

    def test_no_test_arg_only_checks_train_val(self):
        train = self._examples_for_video("vid_a")
        val = self._examples_for_video("vid_b")
        # test=None should not raise even if we don't pass test
        check_video_leakage(train, val)

    def test_error_message_lists_offenders(self):
        train = (
            self._examples_for_video("bad_vid_1")
            + self._examples_for_video("bad_vid_2")
            + self._examples_for_video("good_vid")
        )
        val = self._examples_for_video("bad_vid_1") + self._examples_for_video("bad_vid_2")
        with pytest.raises(ValueError) as exc_info:
            check_video_leakage(train, val)
        msg = str(exc_info.value)
        assert "bad_vid_1" in msg
        assert "bad_vid_2" in msg


# ---------------------------------------------------------------------------
# write_splits_to_manifest / read_frozen_test_examples round-trip
# ---------------------------------------------------------------------------

class TestWriteReadRoundTrip:
    def test_round_trip_basic(self):
        examples = _make_dataset(n_videos=6, clips_per_video=2)
        splits = split_by_video(examples, {"train": 0.67, "test": 0.33}, seed=99)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "manifest.csv")
            write_splits_to_manifest(splits, manifest)
            assert os.path.exists(manifest)

            recovered = read_frozen_test_examples(manifest)
            # All recovered examples should have split="test"
            assert all(ex.split == "test" for ex in recovered)
            # Total count should match all examples written
            original_count = sum(len(v) for v in splits.values())
            assert len(recovered) == original_count

    def test_round_trip_preserves_fields(self):
        ex = _make_example("clip_001", video_id="myVid", source="tsl51")
        splits = {"test": [ex]}

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "frozen.csv")
            write_splits_to_manifest(splits, manifest)
            recovered = read_frozen_test_examples(manifest)

        assert len(recovered) == 1
        r = recovered[0]
        assert r.example_id == "clip_001"
        assert r.source == "tsl51"
        assert r.split == "test"
        assert r.features_path == ex.features_path
        assert r.target_text == ex.target_text
        assert r.metadata["video_id"] == "myVid"

    def test_overwrite_existing_file(self):
        ex1 = _make_example("clip_A", video_id="v1")
        ex2 = _make_example("clip_B", video_id="v2")

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "manifest.csv")

            write_splits_to_manifest({"test": [ex1]}, manifest)
            recovered_1 = read_frozen_test_examples(manifest)
            assert len(recovered_1) == 1

            # Overwrite with different content
            write_splits_to_manifest({"test": [ex2]}, manifest)
            recovered_2 = read_frozen_test_examples(manifest)
            assert len(recovered_2) == 1
            assert recovered_2[0].example_id == "clip_B"

    def test_data_root_joins_relative_path(self):
        ex = SignTextExample(
            example_id="rel_clip",
            source="src",
            split="test",
            features_path="subdir/rel_clip.npy",  # relative path
            target_text="ทดสอบ",
            metadata={"video_id": "vid_rel"},
        )
        splits = {"test": [ex]}

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "frozen.csv")
            write_splits_to_manifest(splits, manifest)
            data_root = "/mnt/datasets"
            recovered = read_frozen_test_examples(manifest, data_root=data_root)

        assert recovered[0].features_path == os.path.join(data_root, "subdir/rel_clip.npy")

    def test_data_root_does_not_alter_absolute_path(self):
        ex = SignTextExample(
            example_id="abs_clip",
            source="src",
            split="test",
            features_path="/absolute/path/clip.npy",
            target_text="ทดสอบ",
            metadata={"video_id": "vid_abs"},
        )
        splits = {"test": [ex]}

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "frozen.csv")
            write_splits_to_manifest(splits, manifest)
            recovered = read_frozen_test_examples(manifest, data_root="/some/root")

        assert recovered[0].features_path == "/absolute/path/clip.npy"

    def test_csv_has_correct_columns(self):
        import csv as csv_mod

        ex = _make_example("chk_001", video_id="chk_vid")
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "chk.csv")
            write_splits_to_manifest({"train": [ex]}, manifest)
            with open(manifest, newline="", encoding="utf-8") as fh:
                reader = csv_mod.DictReader(fh)
                fieldnames = reader.fieldnames
        expected = {"example_id", "video_id", "split", "source", "features_path", "target_text"}
        assert set(fieldnames) == expected
