# Thai SLT Training Diagnosis

## Summary

The latest low-scoring training run is not a code-path failure. It is a data-regime mismatch:

- `checkpoints/slt_v2` is a closed-vocabulary TSL-51 model and is genuinely shipped.
- `checkpoints/slt_combined*` mixes TSL-51 with YouTube-SL-25 Thai and fails as an open-vocabulary problem.
- The open-vocab split is too sparse for sentence-level memorization: every YouTube-SL-25 target is unique, while the model is trying to learn a very high-entropy mapping from a relatively small number of examples.

## Evidence

- TSL-51:
  - 252 examples
  - 62 unique target sentences
  - validation and train share the same sentence inventory
  - word tokenizer works because the target vocabulary is tiny and repeated
- YouTube-SL-25 Thai:
  - 1,626 segments total
  - 1,464 train / 162 val in the current split
  - 1,626 unique target sentences
  - long targets, with many words unseen in train
  - no exact target overlap between train and val
- Combined training result:
  - `combined`: chrF 13.05, exact 0%
  - `combined_v2`: chrF 13.78, exact 0%
  - label smoothing reduced collapse but did not make the mapping learnable

## Root Cause

The model is not learning a reusable sentence mapping because the supervision is too sparse for the target space:

- the closed-vocab case is easy because the same target sentences repeat
- the open-vocab case is hard because the target distribution is effectively one example per sentence
- the current dataset scale is not enough to justify a direct sentence decoder for the Bible-like open-vocabulary corpus

## What To Keep

- Keep `slt_v2` as the shipped baseline.
- Keep `combined*` as research artifacts only.
- Keep the current 162-dim sentence pipeline for TSL-51 compatibility.

## What To Change Next

1. Add a data-quality gate before training:
   - split by `video_id`
   - reject runs with no repeated target structure or no meaningful train/val overlap in the target distribution
2. Build a larger Thai sentence corpus before expecting open-vocab gains.
3. If open-vocab work must continue now, run it as a research track with explicit thresholds and source-level evaluation, not as a production checkpoint.

## Next Training Policy

- Keep `checkpoints/slt_v2` as the only shipped sentence-level checkpoint for now.
- Treat `combined` and `combined_v2` as failed research runs, not as recoverable production checkpoints.
- For the next sentence-level experiment, keep the current 162-dim pipeline unless the entire corpus is moved to a new 312-dim contract.
- Do not run another long open-vocab training job until the manifest has:
  - a video-level split policy
  - explicit source-level metrics
  - a minimum target diversity threshold that proves the model is not learning one sentence per example
