from __future__ import annotations

import importlib


def test_structured_script_packages_are_importable():
    modules = [
        "scripts.data.extract_dataset_keypoints",
        "scripts.maintenance.repo_inventory",
        "scripts.maintenance.repo_cleanup_plan",
        "scripts.maintenance.repo_hygiene_audit",
    ]

    for modname in modules:
        module = importlib.import_module(modname)
        assert module is not None


def test_legacy_wrappers_still_expose_entrypoints():
    import scripts.extract_dataset_keypoints as extract_dataset_keypoints
    import scripts.repo_cleanup_plan as repo_cleanup_plan
    import scripts.repo_inventory as repo_inventory

    assert callable(extract_dataset_keypoints.main)
    assert callable(repo_cleanup_plan.main)
    assert callable(repo_inventory.main)
