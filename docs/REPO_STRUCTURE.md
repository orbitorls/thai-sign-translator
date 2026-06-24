# Repository Structure

This repository mixes three very different concerns:

- product/source code under `src/`, `frontend/`, `web/`, and `tools/`
- operational entrypoints under `scripts/`
- large generated artifacts under `checkpoints/`, `data/`, `kaggle_upload/`, and `tmp/`

The codebase becomes hard to navigate when those concerns drift back into the
same layer. The authoritative structure going forward is:

## Source of truth

- `src/tsl/`
  - Python package for datasets, features, models, inference, API, and training
- `frontend/`
  - current React UI
- `web/`
  - legacy/static browser UI kept for fallback compatibility
- `tools/`
  - standalone local tools such as the Rust keypoint extractor GUI

## Scripts

`scripts/` remains the command entrypoint directory, but implementation should
live in themed subpackages instead of a flat pile of files.

- `scripts/data/`
  - dataset extraction, conversion, and packaging helpers
- `scripts/maintenance/`
  - repository hygiene, inventory, and cleanup planning

Compatibility wrappers may stay at `scripts/*.py` when an older command path is
already in use.

## Artifacts

These directories are not application source code and should be treated as
generated or externally managed assets:

- `checkpoints/`
- `data/`
- `kaggle_upload/`
- `tmp/`
- `output/`

They should not be used as a place to add new source modules.

## Rules for future changes

1. New reusable Python logic belongs in `src/tsl/` or a typed `scripts/<domain>/` package.
2. New one-off operational entrypoints can live in `scripts/`, but only as thin wrappers.
3. New reports or handoff notes belong in `docs/`.
4. New generated outputs should go under an artifact directory, never the repo root.
