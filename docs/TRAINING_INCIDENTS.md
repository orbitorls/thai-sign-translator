# Training Incidents and Continuation Plan

Status snapshot: 2026-06-20

This file is the current operations note for training continuation. It records
the latest verified artifact, the incidents that affected training lanes, and
the next executable actions.

## Current truth sources

- Mixed local lane:
  - `checkpoints/pose_t5_rtx4060_resume_best/train_metrics.json`
  - `checkpoints/pose_t5_rtx4060_resume_best/local_train.stdout.log`
- Mixed verified export:
  - `checkpoints/pose_t5_rtx4060_best_export_verified_eval.json`
  - `checkpoints/pose_t5_rtx4060_best_export_verified_samples.json`
- TSL51-only verified export:
  - `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified_eval.json`
- Mixed manifest-quality evidence:
  - `checkpoints/pose_t5_rtx4060_balanced_beststate_lr5e6/manifest_quality.json`
- Colab launcher state:
  - `checkpoints/colab_sync/thai-sign-train-managed-r14/launcher.status.json`

## Verified current state

- The last mixed local run reached `step 3700`, but the best tracked metric in
  that lane remains the earlier window around `step 2900` with `val_chrf`
  about `15.48`.
- The current mixed verified export is still weak for open-vocab use:
  - `chrF 15.80`
  - `BLEU 13.87`
  - `exact_match_pct 6.0`
- The TSL51-only verified export is the strongest usable artifact currently
  present:
  - refreshed export from `best_model_state.pt step 3075`
  - `chrF 86.95`
  - `BLEU 89.38`
  - `exact_match_pct 64.0`
- Colab strong-GPU lanes are not currently available on this account:
  - `H100` rejected
  - `A100` rejected
  - `L4` rejected
  - `T4` still retrying on capacity errors

## Active incidents

### 1. Mixed local lane stopped uncleanly

Evidence:

- `Get-Process` found no active Python trainer process at the time of review.
- `local_train.stdout.log` stops at `step 3700`.
- `train_metrics.json` still says `"stopped_reason": "running"`.

Impact:

- The lane is no longer making progress.
- The metrics file is misleading for monitoring because it looks alive.

Working diagnosis:

- This was not a normal `early_stopping`, `completed`, or `max_runtime` exit.
- Treat the lane as interrupted and require explicit relaunch if it is to
  continue.

Immediate handling:

- Do not assume this lane is live based on `train_metrics.json` alone.
- Use process state plus log freshness when checking liveness.

### 2. Mixed open-vocab regime is still data-limited

Evidence:

- `checkpoints/pose_t5_rtx4060_balanced_beststate_lr5e6/manifest_quality.json`
  reports:
  - `youtube_sl25_thai: target_overlap_ratio = 0.0`
  - `youtube_sl25_thai: train_examples_per_target = 1.0`

Impact:

- Longer training on the same full mixed corpus is unlikely to produce a
  meaningful open-vocab gain.
- Improvements in overall metrics can still hide failure on the YouTube source.

Immediate handling:

- Do not start another long mixed full-corpus run until the data regime is
  changed or a research subset is defined.

### 3. Legacy mixed export evidence used old source labeling

Evidence:

- `checkpoints/pose_t5_rtx4060_best_export_verified_samples.json` still shows
  `source: "unified"` in old samples.

Impact:

- Older reports are weaker for source-aware diagnosis.
- Fresh evals should be preferred because current code now writes
  source-aware metrics and fails closed on missing source evidence.

Immediate handling:

- Re-run export/eval/promote only after the next training candidate is ready.

### 5. Legacy single-source eval JSON blocked promotion after gate hardening

Evidence:

- Refreshing `tsl51-only` export produced a better candidate immediately:
  - `checkpoint_step = 3075`
  - `chrF 86.95`
  - `BLEU 89.38`
  - `exact_match_pct 64.0`
- The first promotion attempt failed on `eval mismatch on seed` because the
  incumbent eval JSON predated the current source-aware evaluator.
- Re-evaluating the incumbent with the current evaluator backfilled the missing
  fields and removed the false mismatch.

Impact:

- Verified export refresh can fail even when the candidate is good, if the
  incumbent report is still on the legacy schema.

Immediate handling:

- Before comparing against an older incumbent, re-run
  `scripts/evaluate_pose_t5_export.py` on the incumbent export so the report has
  `data_roots`, `seed`, `source_metrics`, and `source_counts`.

### 4. Colab strong-GPU path is blocked by quota/entitlement

Evidence:

- `thai-sign-train-managed-r14/launcher.status.json` shows rejected
  `H100/A100/L4` and repeated `T4: Service Unavailable`.

Impact:

- Colab is not the primary execution path right now.

Immediate handling:

- Treat local RTX 4060 as the main training executor.
- Treat Colab as opportunistic fallback only.

## Continuation plan

### Track A: keep the usable artifact stable

Goal:

- Preserve the current usable model while research continues.

Action:

- Keep `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified/` as the stable
  benchmark artifact for immediate use and regression checks.
- `config.SLT_V3_CHECKPOINT_DIR` now points to this artifact so runtime uses the
  strongest verified model currently available.

### Track B: continue mixed-vocab as a research lane

Goal:

- Improve the mixed-vocab path without wasting long runs on the current failing
  regime.

Action order:

1. Audit and slice the YouTube source into a more learnable research subset.
2. Launch only a short-cycle local run on that subset.
3. Export and evaluate the resulting candidate with the source-aware verified
   flow.
4. Promote only if `youtube_sl25_thai` improves without regressing the other
   source.

## Launch policy from this point

- Primary executor: local RTX 4060
- Primary objective for the next run: mixed research subset, not the unchanged
  full mixed corpus
- Evaluation gate: always use
  `export_pose_t5_checkpoint.py -> evaluate_pose_t5_export.py -> promote_pose_t5_export.py`
- Monitoring rule:
  - process alive
  - stdout log still growing
  - `train_metrics.json` is only advisory, not sufficient proof of liveness

## Next concrete actions

1. Build the next mixed research subset and verify its manifest quality.
2. Relaunch a local training lane on that subset.
3. Record the new run path and metrics here after the first eval window.
