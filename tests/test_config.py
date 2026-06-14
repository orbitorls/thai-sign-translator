"""Assert the landmark slice constants are internally consistent.

The four canonical landmark groups must tile [0, 543) exactly with no gaps
or overlaps, and the named single-landmark indices must fall in their group.
"""

import config


def test_n_landmarks():
    assert config.N_LANDMARKS == 543


def test_slices_sum_to_n_landmarks():
    sizes = [
        config.FACE.stop - config.FACE.start,
        config.LEFT_HAND.stop - config.LEFT_HAND.start,
        config.POSE.stop - config.POSE.start,
        config.RIGHT_HAND.stop - config.RIGHT_HAND.start,
    ]
    assert sum(sizes) == config.N_LANDMARKS


def test_slice_group_sizes():
    assert config.FACE.stop - config.FACE.start == 468
    assert config.LEFT_HAND.stop - config.LEFT_HAND.start == 21
    assert config.POSE.stop - config.POSE.start == 33
    assert config.RIGHT_HAND.stop - config.RIGHT_HAND.start == 21


def test_slices_tile_without_gaps_or_overlap():
    covered = []
    for sl in (config.FACE, config.LEFT_HAND, config.POSE, config.RIGHT_HAND):
        covered.extend(range(sl.start, sl.stop))
    assert covered == list(range(config.N_LANDMARKS))


def test_canonical_order_boundaries():
    assert config.FACE == slice(0, 468)
    assert config.LEFT_HAND == slice(468, 489)
    assert config.POSE == slice(489, 522)
    assert config.RIGHT_HAND == slice(522, 543)


def test_named_indices_in_pose():
    assert config.POSE.start <= config.NOSE_IDX < config.POSE.stop
    assert config.POSE.start <= config.LSHOULDER_IDX < config.POSE.stop
    assert config.POSE.start <= config.RSHOULDER_IDX < config.POSE.stop


def test_named_index_values():
    assert config.NOSE_IDX == 489
    assert config.LSHOULDER_IDX == 500
    assert config.RSHOULDER_IDX == 501


def test_paths_are_strings():
    assert isinstance(config.DATA_DIR, str)
    assert isinstance(config.ISLR_PARQUET_DIR, str)
    assert isinstance(config.ISLR_CSV_PATH, str)
    assert isinstance(config.THAI_DATA_DIR, str)
    assert isinstance(config.CHECKPOINT_DIR, str)
    assert isinstance(config.ENCODER_WEIGHTS_PATH, str)
    assert isinstance(config.PROTOTYPE_STORE_PATH, str)
