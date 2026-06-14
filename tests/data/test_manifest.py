import dataclasses

import pytest

from tsl.data.manifest import SignTextExample


def _make(**overrides):
    base = dict(
        example_id="ex-001",
        source="tsl51",
        split="train",
        features_path="/data/tsl51/ex-001.npy",
        target_text="สวัสดีครับ",
    )
    base.update(overrides)
    return SignTextExample(**base)


def test_create_valid_example():
    ex = _make()
    assert ex.example_id == "ex-001"
    assert ex.source == "tsl51"
    assert ex.split == "train"
    assert ex.features_path == "/data/tsl51/ex-001.npy"
    assert ex.target_text == "สวัสดีครับ"
    assert ex.gloss is None
    assert ex.metadata is None


def test_create_valid_example_with_optional_fields():
    ex = _make(gloss="สวัสดี/ครับ", metadata={"speaker": "A", "fps": 30})
    assert ex.gloss == "สวัสดี/ครับ"
    assert ex.metadata == {"speaker": "A", "fps": 30}


def test_split_val_and_test_are_accepted():
    assert _make(split="val").split == "val"
    assert _make(split="test").split == "test"


def test_empty_target_text_raises():
    with pytest.raises(ValueError, match="target_text"):
        _make(target_text="")


def test_invalid_split_raises():
    with pytest.raises(ValueError, match="split"):
        _make(split="dev")


def test_empty_features_path_raises():
    with pytest.raises(ValueError, match="features_path"):
        _make(features_path="")


def test_empty_example_id_raises():
    with pytest.raises(ValueError, match="example_id"):
        _make(example_id="")


def test_empty_source_raises():
    with pytest.raises(ValueError, match="source"):
        _make(source="")


def test_frozen_assignment_raises():
    ex = _make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ex.target_text = "อื่น"  # type: ignore[misc]
