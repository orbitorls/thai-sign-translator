# PoseT5 One-Shot Kaggle GPU Runbook

This runbook is the gate for mixed-all v6 PoseT5 training. It is designed to fail fast before long GPU spend and to prevent promotion of an unready export.

## Dataset Policy

- Expected manifest rows: `1938`
- Expected source counts: `tsl51=252,thaisignvis=60,youtube_sl25_thai=1626`
- Production-gated source: `tsl51`
- ThaiSignVis and YouTube-SL-25 are training/research signals only until they have source-level validation/test evidence.
- Runtime/product target is direct landmark-to-text PoseT5, not a gloss-first pipeline.

## GPU Policy

- Kaggle runs must use CUDA and pass `--require-gpu true`.
- No-GPU sessions fail before data/model loading.
- P100 / `sm_60` is rejected by default in the notebook. Set `TSL_ALLOW_LEGACY_SM60=1` only for an explicit non-default legacy experiment.
- Preferred accelerators: T4, V100, A100.

## Preflight

Run from the repo root on Windows PowerShell:

```powershell
git status --short

$env:PYTHONPATH='src'
python -B scripts\audit_pose_t5_dataset.py --data-roots data\mixed_all_train_v6 --expected-manifest-rows 1938 --expected-source-counts "tsl51=252,thaisignvis=60,youtube_sl25_thai=1626" --production-gated-sources tsl51 --json-out tmp\pose_t5_dataset_audit.json

python -B scripts\verify_pose_t5_cloud_preflight.py --data-roots data\mixed_all_train_v6 --expected-manifest-rows 1938 --expected-resolved-examples 1938 --expected-source-counts "tsl51=252,thaisignvis=60,youtube_sl25_thai=1626"
```

The audit should make train-only ThaiSignVis/YouTube status explicit. Do not treat those sources as production-ready gates.

## Package And Publish

```powershell
python scripts\prepare_kaggle_pose_t5_assets.py --archive-features --build-mixed-manifest true --mixed-allow-missing-roots false --mixed-required-sources tsl51 --mixed-manifest-quality-sources tsl51

python -B scripts\verify_pose_t5_cloud_preflight.py --data-roots kaggle_upload\thai-sign-mixed-all-v6-archived --required-files manifest.csv,manifest_quality.json,features.zip --expected-manifest-rows 1938 --expected-resolved-examples 1938 --expected-source-counts "tsl51=252,thaisignvis=60,youtube_sl25_thai=1626"

python scripts\publish_kaggle_dataset.py --dataset-dir kaggle_upload\thai-sign-code --message "Update PoseT5 one-shot code" --temp-dir tmp\kaggle_cli_temp
python scripts\publish_kaggle_dataset.py --dataset-dir kaggle_upload\thai-sign-mixed-all-v6-archived --message "Update mixed-all v6 archived" --temp-dir tmp\kaggle_cli_temp
.\scripts\run_kaggle_pose_t5.ps1 -KernelPath kaggle_upload\notebook -KernelSlug orbitorls/thai-sign-mixed-all-v6-train -Accelerator t4 -TimeoutSeconds 43200 -PollSeconds 60
```

`scripts/package_for_kaggle.py` and `scripts/kaggle_train.py` are legacy paths. Keep them for reference, but do not use them for mixed-all v6 PoseT5 one-shot training.

## Kaggle Wrapper Gates

The notebook calls `scripts/kaggle_train_pose_t5.py` with:

- `--require-gpu true`
- `--smoke-steps 20`
- `--balance-sources auto`
- `--focus-target-tokens ฉัน,คุณ,แม่,พี่,วันนี้,พรุ่งนี้`
- `--focus-target-max-multiplier 3.0`
- `--required-sources tsl51`
- `--manifest-quality-sources tsl51`
- `--eval-sources tsl51`
- `--min-val-chrf 80`
- `--min-val-exact-match-pct 80`

The wrapper must pass raw archived preflight, staged resolved-example preflight, smoke training with `stopped_reason=max_train_steps`, full training, and readiness evaluation before a run is considered usable.

## Promotion Rules

- `verified_eval.json` must contain `promotion_status.ready=true`.
- A v6 candidate must beat the trusted incumbent on comparable `tsl51` evaluation before promotion.
- If the better v5 export cannot be load-smoked or lacks compatible metadata, keep the current stable runtime export as the trusted incumbent and block long v6 promotion.
- `/translate-video` and `/translate` should use PoseT5 runtime exports. `/translate-sentence` remains legacy unless separately refactored.
