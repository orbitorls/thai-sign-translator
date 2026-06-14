"""Trivial smoke test: the package imports and exposes a version string.

This establishes the test harness before any real code exists.
"""


def test_package_imports():
    import tsl

    assert tsl is not None


def test_package_has_version():
    import tsl

    assert isinstance(tsl.__version__, str)
    assert tsl.__version__ != ""
