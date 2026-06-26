# HANDOFF - Colab r17 and open problems

> Status captured on 2026-06-22, Asia/Bangkok. This file is the current handoff
> for the Colab `thai-sign-train-managed-r17` recovery/evaluation lane. Treat
> `docs/HANDOFF.md` as broader project history; start here for the latest cloud
> checkpoint issue.

---

## TL;DR

- A new Colab run was launched as `thai-sign-train-managed-r17`.
- Strong GPUs were rejected by Colab backend:
  - `H100`, `A100`, and `L4` are cached as `rejected` in `checkpoints/colab_sync/gpu_availability.json`.
  - The launcher fell back to `T4` and got a live session.
- Remote training did start and produced a new checkpoint:
  - remote checkpoint: `/content/checkpoints/pose_t5_mixed_all_v6_colab/ckpt_step00001800.pt`
  - remote size: `3,684,060,540` bytes
  - remote staging copy: `/content/kaggle_ckpt_publish/ckpt_step00001800.pt`
- Local mirror does **not** have `ckpt_step00001800.pt` yet.
- Kaggle dataset `orbitorls/thai-sign-ckpt` still does **not** expose `ckpt_step00001800.pt`; it only shows up to `ckpt_step00001700.pt`.
- Do **not** promote `r17` yet. It has not passed the repo's runtime eval/promotion gate.
- Best currently verified mixed/open-vocab runtime export remains:
  - `checkpoints/pose_t5_rtx4060_best_export_verified`
  - eval: `chrF 15.80`, `BLEU 13.87`, `exact_match_pct 6.0` on 50-sample mixed subset
- Best TSL-51-only export remains:
  - `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified`
  - eval: `chrF 86.95`, `BLEU 89.38`, `exact_match_pct 64.0` on 25 TSL-51 examples

---

## Current Colab state

Session:

```text
thai-sign-train-managed-r17
```

Last known `colab status`:

```text
[thai-sign-train-managed-r17] gpu-t4-s-kkb-usw4a2-1b73r4207wjd8 | Hardware: T4 | Variant: GPU | Status: BUSY
Last Execution: /mnt/c/Users/PANNAW~1/AppData/Local/Temp/colab-r17-publish-once-2f5ca6abea2d42e18dec87f71d033aeb.py at 2026-06-22 20:18:46
```

Important: a manual `publish --once` probe was still BUSY when this handoff was
written. Before doing anything destructive or launching a replacement session,
check whether that command has finished:

```powershell
wsl.exe bash -lc "/root/.venvs/colabcli/bin/colab status -s 'thai-sign-train-managed-r17'"
python -m kaggle datasets files -d orbitorls/thai-sign-ckpt
```

Launch command used for `r17`:

```powershell
.\scripts\colab_cli_pose_t5.ps1 `
  -SessionName thai-sign-train-managed-r17 `
  -GpuPriority H100,A100,L4,T4 `
  -MinGpu A100 `
  -AllowFallbackBelowMinGpuOnReject `
  -GpuRejectCooldownMinutes 360 `
  -GpuRetryMinutes 120 `
  -GpuRetryDelaySec 30 `
  -LearningRate 5e-5 `
  -Dropout 0.4 `
  -WeightDecay 0.1 `
  -EarlyStoppingPatience 6 `
  -EarlyStoppingMetric val_chrf `
  -CheckpointSteps 200 `
  -ResumeMode best `
  -ResetProgressHistory
```

Remote paths:

```text
out_dir: /content/checkpoints/pose_t5_mixed_all_v6_colab
staging_dir: /content/kaggle_ckpt_publish
dataset: orbitorls/thai-sign-mixed-all-v6-archived
checkpoint_dataset: orbitorls/thai-sign-ckpt
```

Local mirror:

```text
checkpoints/colab_sync/thai-sign-train-managed-r17/
```

Local mirror currently contains:

```text
ckpt_step00001000.pt
ckpt_step00001700.pt
launcher.status.json
launcher.stdout.log
launch.json
sync_state.json
train.log
train_metrics.json
```

Local mirror currently does **not** contain:

```text
ckpt_step00001800.pt
```

---

## Training evidence

Remote training log shows:

```text
[resume] Loading checkpoint: /content/checkpoints/pose_t5_mixed_all_v6_colab/ckpt_step00001700.pt
[resume] Resuming from step=1700, epoch=31
[resume] Resetting progress history at resumed step 1700.
[step 1800] epoch=31 | train_loss=3.1411 | val_loss=0.4221 | val_chrf=67.80
```

Local metrics file:

```text
checkpoints/colab_sync/thai-sign-train-managed-r17/train_metrics.json
```

Key metric:

```json
{
  "initial_step": 1700,
  "global_step": 1800,
  "final_step": 1800,
  "new_optimizer_steps": 100,
  "stopped_reason": "running",
  "history": [
    {"step": 1700, "epoch": 31, "val_loss": 3.836504611475714, "val_chrf": 12.701290002536384},
    {"step": 1800, "epoch": 31, "train_loss": 3.141131669282913, "val_loss": 0.42206851073673796, "val_chrf": 67.79977316570447}
  ]
}
```

Do not treat `val_chrf 67.80` as production proof. The validation split is small
and likely optimistic:

```text
[data] 1938 total | 1913 train | 25 val
[data] manifest quality | train_examples_per_target=1.0944 | target_overlap_ratio=1.0000 | video_overlap_count=0
```

The high `target_overlap_ratio` means train-side validation may be much easier
than the mixed/open-vocab runtime eval gate.

---

## Model comparison status

Use this as the current model ranking until `r17` is evaluated on the same gate.

| Candidate | Evidence type | Metric | Status |
|---|---:|---:|---|
| `pose_t5_rtx4060_tsl51_only_export_verified` | runtime eval, TSL-51 only, 25 examples | `chrF 86.95`, `BLEU 89.38`, `EM 64.0%` | Best closed-domain/TSL-51 result |
| `r17` / `ckpt_step00001800.pt` | train validation, 25 examples | `val_chrf 67.80` | Promising but not comparable yet |
| `pose_t5_rtx4060_best_export_verified` | runtime eval, mixed subset, 50 examples | `chrF 15.80`, `BLEU 13.87`, `EM 6.0%` | Best verified mixed/open-vocab runtime export |
| `thai-sign-train-managed-r4` final export | corrected runtime eval, 50 examples | `chrF 12.36`, `BLEU 9.17`, `EM 2.0%` | Older A100 baseline |
| `a100_step1500` | corrected runtime eval, 50 examples | `chrF 12.20`, `BLEU 8.89`, `EM 2.0%` | Older baseline |

The only fair next comparison is:

1. Get `ckpt_step00001800.pt` local.
2. Export it.
3. Evaluate it with `scripts/evaluate_pose_t5_export.py`.
4. Promote only if it beats `checkpoints/pose_t5_rtx4060_best_export_verified_eval.json`.

---

## Problems found

### 1. Colab strong GPU entitlement/quota problem

`H100`, `A100`, and `L4` were rejected by Colab backend:

```text
Backend rejected accelerator 'H100'
Backend rejected accelerator 'A100'
Backend rejected accelerator 'L4'
```

The run only proceeded because fallback to `T4` worked.

Evidence:

```text
checkpoints/colab_sync/gpu_availability.json
```

### 2. Local sync initially failed on Windows paths with spaces

The sync helper was launched with a raw argument list, so Python saw the script
path as `D:\New` instead of `D:\New folder\...`.

Observed error:

```text
C:\Users\Pannawat Khantong\AppData\Local\Microsoft\WindowsApps\python.exe: can't open file 'D:\New': [Errno 2] No such file or directory
```

Fix already made:

```text
scripts/colab_cli_pose_t5.ps1
```

The launcher now quotes `Start-Process` arguments before spawning
`scripts/colab_checkpoint_sync.py`.

### 3. Colab download endpoint returns 500 for this session

Direct `colab download` failed for the checkpoint and even small files under the
chunk directory.

Evidence:

```text
checkpoints/colab_sync/thai-sign-train-managed-r17/sync_state.json
```

Failures include:

```text
ckpt_step00001800.pt -> 500 Internal Server Error
ckpt_step00001500.pt -> 500 Internal Server Error
ckpt_step00001000.pt -> 500 Internal Server Error
special_tokens_map.json -> 500 Internal Server Error
spiece.model -> 500 Internal Server Error
tokenizer.json -> 500 Internal Server Error
tokenizer_config.json -> 500 Internal Server Error
```

This looks like a broken Colab contents/download proxy for the session, not only
a large-file issue.

### 4. Split-file workaround did not solve download

Remote split succeeded:

```text
/content/checkpoint_chunks/r17_step1800/ckpt_step00001800.pt.part-000 ... part-013
```

Remote manifest said:

```text
source_size = 3,684,060,540
14 parts, mostly 256 MiB each
```

But `colab download` still returned `500` even for:

```text
/content/checkpoint_chunks/r17_step1800/manifest.json
/content/checkpoint_chunks/r17_step1800/ckpt_step00001800.pt.part-000
```

### 5. Kaggle publish is staged but not visible

Remote staging has the new checkpoint:

```text
/content/kaggle_ckpt_publish/ckpt_step00001800.pt
size = 3,684,060,540
```

The publisher log says:

```text
[publisher] staging checkpoint: ckpt_step00001800.pt
[publisher] publishing checkpoint: ckpt_step00001800.pt
[publisher] checkpoint not visible yet; pending retry: ckpt_step00001800.pt
[publisher] verifying pending checkpoint: ckpt_step00001800.pt
```

But local and remote `kaggle datasets files -d orbitorls/thai-sign-ckpt` still
show only:

```text
ckpt_step00001000.pt
ckpt_step00001500.pt
ckpt_step00001700.pt
```

### 6. Manual publish-once command may still be running

A manual `colab exec` was started to force one more publish/poll pass. The local
tool call returned `-1073741510` without output, but Colab later reported that
the session was still BUSY running:

```text
colab-r17-publish-once-2f5ca6abea2d42e18dec87f71d033aeb.py
```

Check status before launching any new exec, because Colab allows only one active
exec path cleanly.

### 7. Train process is not proven to be still running

Earlier probe showed `train.pid=4752`, but `ps` returned no live train process.
`run_status.json` still said:

```json
{
  "phase": "evaluated",
  "global_step": 1800,
  "stopped_reason": "running"
}
```

That means the status file is stale or incomplete after step 1800. Do not infer
that training is still progressing from `stopped_reason=running`.

### 8. `r17` is not evaluated or promoted

There is no local runtime export for `ckpt_step00001800.pt` yet, and no
`evaluate_pose_t5_export.py` report. The checkpoint is promising, but not a
replacement for the verified RTX export.

### 9. Token leakage risk in sync error logs was fixed

`colab download` errors included `colab-runtime-proxy-token` in URLs. The local
sync state was sanitized, and the sync helper now redacts the token before
writing `sync_state.json` or `sync_error.log`.

Files changed:

```text
scripts/colab_checkpoint_sync.py
tests/scripts/test_colab_checkpoint_sync.py
```

Verification:

```text
python -m pytest tests\scripts\test_colab_checkpoint_sync.py
24 passed
```

### 10. Worktree has many untracked artifacts

`git status --short` contains many untracked directories under:

```text
checkpoints/
data/
kaggle_upload/
tmp_pytest/
.playwright-mcp/
```

Do not commit large checkpoint/data artifacts directly unless the user explicitly
asks. Treat most of these as generated artifacts or dataset/cache outputs.

---

## Next actions

### Step 1 - Check whether publish-once finished

```powershell
wsl.exe bash -lc "/root/.venvs/colabcli/bin/colab status -s 'thai-sign-train-managed-r17'"
python -m kaggle datasets files -d orbitorls/thai-sign-ckpt
```

If `ckpt_step00001800.pt` appears in Kaggle, download only that file if the
installed Kaggle CLI supports `-f`:

```powershell
python -m kaggle datasets download `
  -d orbitorls/thai-sign-ckpt `
  -f ckpt_step00001800.pt `
  -p checkpoints\colab_sync\thai-sign-train-managed-r17 `
  --unzip
```

If `-f` is not supported, use a scratch directory and avoid overwriting known
good local artifacts.

### Step 2 - If Kaggle still does not expose `01800`, inspect remote publisher

Use a small Colab exec probe; do not download files through `colab download`
until the proxy issue is understood.

Check:

```text
/content/kaggle_ckpt_publish/
/content/checkpoints/pose_t5_mixed_all_v6_colab/publisher.log
/content/checkpoints/pose_t5_mixed_all_v6_colab/publisher.pid
```

The staging file exists, so the likely problem is Kaggle dataset version
visibility/processing, not missing checkpoint creation.

### Step 3 - Export `r17` only after the checkpoint is local

After `checkpoints/colab_sync/thai-sign-train-managed-r17/ckpt_step00001800.pt`
exists locally:

```powershell
python scripts\export_pose_t5_checkpoint.py `
  --train-dir checkpoints\colab_sync\thai-sign-train-managed-r17 `
  --checkpoint ckpt_step00001800.pt `
  --export-dir checkpoints\pose_t5_colab_r17_step1800_export `
  --report-json checkpoints\pose_t5_colab_r17_step1800_export_report.json
```

### Step 4 - Evaluate on the same gate

```powershell
python scripts\evaluate_pose_t5_export.py `
  --export-dir checkpoints\pose_t5_colab_r17_step1800_export `
  --report-json checkpoints\pose_t5_colab_r17_step1800_export_eval.json `
  --samples-json checkpoints\pose_t5_colab_r17_step1800_export_samples.json
```

### Step 5 - Promote only if it beats the incumbent

```powershell
python scripts\promote_pose_t5_export.py `
  --candidate-export-dir checkpoints\pose_t5_colab_r17_step1800_export `
  --candidate-eval-json checkpoints\pose_t5_colab_r17_step1800_export_eval.json `
  --candidate-samples-json checkpoints\pose_t5_colab_r17_step1800_export_samples.json `
  --stable-export-dir checkpoints\pose_t5_rtx4060_best_export_verified `
  --stable-eval-json checkpoints\pose_t5_rtx4060_best_export_verified_eval.json
```

Only call the model "better" if this promotion/eval gate passes.

---

## Files changed in this session

Code/test fixes:

```text
scripts/colab_cli_pose_t5.ps1
scripts/colab_checkpoint_sync.py
tests/scripts/test_colab_checkpoint_sync.py
```

Documentation:

```text
docs/HANDOFF_COLAB_R17_ISSUES.md
docs/HANDOFF.md
```

Generated/local artifacts touched or created:

```text
checkpoints/colab_sync/thai-sign-train-managed-r17/
```

---

## Guardrails for the next agent

- Do not claim `r17` is ready until `ckpt_step00001800.pt` is local and runtime
  eval passes.
- Do not rely on `launcher.status.json` alone as proof of model readiness.
- Do not rely on `val_chrf=67.80` as comparable to the verified mixed runtime
  eval.
- Do not print or preserve Colab proxy tokens from failed download URLs.
- Do not overwrite `checkpoints/pose_t5_rtx4060_best_export_verified` unless
  `promote_pose_t5_export.py` says the candidate beats the incumbent.
- Do not delete untracked data/checkpoint artifacts casually; several are real
  training evidence even though they are not meant for git.
