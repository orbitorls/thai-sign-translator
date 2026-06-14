"""Every src/tsl subpackage must be importable.

This locks the authoritative module layout so downstream modules can import
their siblings (e.g. `from tsl.models.encoder import LandmarkEncoder`).
"""

import importlib

import pytest

SUBPACKAGES = [
    "tsl.features",
    "tsl.data",
    "tsl.models",
    "tsl.train",
    "tsl.registry",
    "tsl.inference",
    "tsl.segment",
    "tsl.grammar",
    "tsl.baseline",
    "tsl.eval",
    "tsl.api",
]


@pytest.mark.parametrize("modname", SUBPACKAGES)
def test_subpackage_importable(modname):
    mod = importlib.import_module(modname)
    assert mod is not None
