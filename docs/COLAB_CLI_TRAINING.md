# Google Colab CLI Training

This repo now uses a managed Colab flow that avoids the two failure modes that
kept happening before: unstable multi-GB uploads into `/content` and losing
checkpoints when a Colab VM is reclaimed.

## Current approach

- Run `google-colab-cli` from WSL/Linux, not native PowerShell.
- Upload only a small code bundle plus a JSON launch config.
- Download datasets inside Colab from Kaggle using `~/.kaggle/access_token`.
- Persist checkpoints in two places:
  - remote Kaggle dataset `orbitorls/thai-sign-ckpt`
  - optional local mirror under `checkpoints/colab_sync/<session>`
- Prefer the strongest available GPU in this order:
  - `H100`
  - `A100`
  - `L4`
  - `T4`
- When a stronger tier is explicitly rejected by Colab for the current account,
  the launcher can now drop below `-MinGpu` if you opt in with
  `-AllowFallbackBelowMinGpuOnReject`.
- Backend-rejected tiers are cached much longer than capacity cooldowns, so the
  launcher does not keep re-requesting `H100/A100/L4` every few minutes when
  the account is not entitled to them.

Drive mount is no longer required.

## One-command launch

From the repo root in PowerShell:

```powershell
.\scripts\colab_cli_pose_t5.ps1 -SessionName thai-sign-train-managed
```

For "try the fastest first, but do not stall if this account cannot get that
tier":

```powershell
.\scripts\colab_cli_pose_t5.ps1 `
  -SessionName thai-sign-train-managed `
  -GpuPriority H100,A100,L4,T4 `
  -MinGpu A100 `
  -AllowFallbackBelowMinGpuOnReject `
  -GpuRejectCooldownMinutes 360 `
  -GpuRetryMinutes 120 `
  -GpuRetryDelaySec 30
```

If Google rejects new GPU assignment but you still have a live Colab VM, reuse
that session instead of asking for a fresh one:

```powershell
.\scripts\colab_cli_pose_t5.ps1 -SessionName thai-sign-train-managed -ReuseExistingSession
```

For a fresh run that should not resume from the latest remote checkpoint:

```powershell
.\scripts\colab_cli_pose_t5.ps1 `
  -SessionName thai-sign-train-managed `
  -ResumeMode none `
  -ResetRemoteOutDir `
  -LearningRate 5e-5 `
  -Dropout 0.4 `
  -WeightDecay 0.1 `
  -EarlyStoppingPatience 6 `
  -EarlyStoppingMetric val_chrf
```

What it does:

1. Builds a Linux-safe code zip with `scripts/package_colab_bundle.py`
2. Chooses the first available GPU from `H100,A100,L4,T4`
   - if stronger tiers are backend-rejected and fallback is enabled, it moves
     to the next lower tier instead of failing the whole launch
3. Uploads the code zip, launch config, and Kaggle access token to `/content`
4. Starts `scripts/colab_bootstrap_pose_t5.py` on the Colab VM
5. Downloads training datasets from Kaggle inside the VM
6. Restores the latest checkpoint from `orbitorls/thai-sign-ckpt` if present
7. Starts background training plus a remote checkpoint publisher
8. Starts a local sync helper for logs and checkpoints

## Default training config

The launcher currently uses these defaults for the next run:

- `lr = 1e-4`
- `dropout = 0.3`
- `weight_decay = 0.05`
- `early_stopping_patience = 10`
- `early_stopping_min_delta = 0.0`
- `early_stopping_metric = val_chrf`
- `batch_size = 8`
- `grad_accum = 4`
- `eval_steps = 100`
- `checkpoint_steps = 500`
- `keep_checkpoints = 3`
- `max_runtime_min = 690`
- `resume = auto`

These map to [train_pose_t5.py](/D:/New%20folder/thai-sign-translator/src/tsl/train/train_pose_t5.py).

## Why this is more reliable

The old flow broke for practical reasons:

- `colab drivemount` could hang
- direct upload of large datasets/checkpoints to `/content` was unstable
- `Compress-Archive` produced Windows-style paths that extracted incorrectly on Linux
- PowerShell JSON could include a UTF-8 BOM that broke strict readers
- checkpoint sync could stop making progress after a transient download failure

The current scripts address those directly:

- `scripts/package_colab_bundle.py` builds a Linux-safe zip
- JSON config readers use `utf-8-sig`
- `scripts/colab_checkpoint_sync.py` retries downloads, keeps partial progress,
  and mirrors the latest checkpoint first
- `scripts/colab_publish_checkpoints.py` versions the newest checkpoint to Kaggle
  and keeps the best `val_chrf` checkpoint alongside the latest one when it is
  available locally
- `scripts/colab_bootstrap_pose_t5.py` restores resumable state automatically
- `scripts/colab_cli_pose_t5.ps1` now distinguishes backend rejection from
  temporary capacity cooldown, and can record/activate lower-tier fallback in
  `launcher.status.json`
- `scripts/colab_cli_pose_t5.ps1` keeps backend-rejected GPU tiers blocked for
  `-GpuRejectCooldownMinutes` (default 360 minutes), while transient capacity
  errors still use a short retry cooldown

The training loop can now early-stop on either `val_loss` or `val_chrf`. For
the managed Colab flow, the launcher defaults to `val_chrf` so the stop signal
matches the translation-quality metric instead of a lower-level loss proxy.

One more behavior matters in practice: Kaggle dataset version creation is
asynchronous. A `kaggle datasets version` command can finish successfully
before `kaggle datasets files -d orbitorls/thai-sign-ckpt` shows the new
checkpoint. The publisher now polls the dataset files API for up to about
2 minutes before it accepts the publish as complete. If the checkpoint is
still not visible by then, the publisher keeps the checkpoint in a pending
state and retries on the next loop instead of crashing and abandoning later
checkpoints.

## Monitoring

Check the Colab session:

```powershell
wsl.exe bash -lc "/root/.venvs/colabcli/bin/colab status -s 'thai-sign-train-managed'"
```

Check local sync state:

```powershell
Get-Content .\checkpoints\colab_sync\thai-sign-train-managed\sync_state.json
Get-Content .\checkpoints\colab_sync\thai-sign-train-managed\sync_error.log
```

Check the remote checkpoint dataset:

```powershell
python -m kaggle datasets files -d orbitorls/thai-sign-ckpt
```

## Important notes

- Keep `C:\Users\Pannawat Khantong\.kaggle\access_token` available locally.
- Do not commit `data/`, `checkpoints/`, or Kaggle output artifacts to git.
- If a run already finished, the Colab session may show `IDLE`; that is expected.
- The strongest available GPU is chosen dynamically. The launcher does not force
  T4 anymore unless stronger GPUs are unavailable.
- `launcher.status.json` now includes the active candidate set and whether the
  launcher has fallen back below `-MinGpu` after backend rejection.
