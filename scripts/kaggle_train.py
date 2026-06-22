"""Kaggle training launcher for PoseToTextT5.

Paste this file into a Kaggle notebook code cell and run:
    !python /kaggle/input/thai-sign-code/scripts/kaggle_train.py

Or run cells step-by-step (see CELL MARKERS below).

Dataset inputs required in the notebook:
  - thai-sign-code   → attached as /kaggle/input/thai-sign-code/
  - thai-sign-data   → attached as /kaggle/input/thai-sign-data/
  - thai-sign-ckpt   → (optional) previous checkpoint dataset for resume

Output:
  - Checkpoints saved to /kaggle/working/pose_t5_v3/
  - Download best checkpoint after session ends, upload as thai-sign-ckpt dataset
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

# ============================================================
# CELL 1: Install dependencies
# ============================================================
def install_deps():
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.40",
        "sentencepiece>=0.2.0",
        "sacrebleu>=2.4.0",
    ], check=True)
    print("Dependencies installed.")

# ============================================================
# CELL 2: Set up paths
# ============================================================
CODE_DIR  = "/kaggle/input/thai-sign-code"
DATA_DIR  = "/kaggle/input/thai-sign-data"
CKPT_DIR  = "/kaggle/working/pose_t5_v3"
PREV_CKPT = "/kaggle/input/thai-sign-ckpt"  # optional resume dataset

def setup_paths():
    src_path = os.path.join(CODE_DIR, "src")
    for p in (src_path, CODE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"sys.path has src: {src_path}")

    # Copy previous checkpoint if available
    if os.path.isdir(PREV_CKPT):
        for f in os.listdir(PREV_CKPT):
            src = os.path.join(PREV_CKPT, f)
            dst = os.path.join(CKPT_DIR, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Restored checkpoint: {f}")

# ============================================================
# CELL 3: Verify GPU + data
# ============================================================
def verify():
    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    data_roots = _resolve_data_roots()
    print(f"Data roots: {data_roots}")
    for root in data_roots.split(","):
        manifest = os.path.join(root.strip(), "manifest.csv")
        if os.path.isfile(manifest):
            import pandas as pd
            df = pd.read_csv(manifest)
            print(f"  {root.strip()}: {len(df)} segments")
        else:
            print(f"  WARNING: {manifest} not found!")

def _resolve_data_roots() -> str:
    roots = []
    for sub in ("tsl51_v3", "youtube_sl25_thai_v3"):
        p = os.path.join(DATA_DIR, sub)
        if os.path.isdir(p):
            roots.append(p)
    return ",".join(roots)

# ============================================================
# CELL 4: Train
# ============================================================
def train(epochs: int = 150, batch_size: int = 8, grad_accum: int = 4):
    data_roots = _resolve_data_roots()
    if not data_roots:
        raise RuntimeError(f"No data found under {DATA_DIR}")

    train_script = os.path.join(CODE_DIR, "scripts", "train_local_gpu.py")

    cmd = [
        sys.executable, "-u", train_script,
        "--data-roots", data_roots,
        "--out-dir", CKPT_DIR,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--amp", "auto",
        "--max-runtime-min", "690",   # stop before Kaggle 12h kill
        "--resume", "auto",
        "--eval-steps", "100",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ============================================================
# CELL 5: Show checkpoint files (after training)
# ============================================================
def list_checkpoints():
    if not os.path.isdir(CKPT_DIR):
        print("No checkpoints yet.")
        return
    files = sorted(os.listdir(CKPT_DIR))
    total_mb = sum(
        os.path.getsize(os.path.join(CKPT_DIR, f))
        for f in files if os.path.isfile(os.path.join(CKPT_DIR, f))
    ) / 1024**2
    print(f"\nCheckpoints in {CKPT_DIR}  (total {total_mb:.0f} MB):")
    for f in files:
        print(f"  {f}")
    print("\nTo resume next session:")
    print("  1. Download /kaggle/working/pose_t5_v3/ as a zip")
    print("  2. Upload as Kaggle dataset 'thai-sign-ckpt'")
    print("  3. Attach it in the next notebook session")


if __name__ == "__main__":
    install_deps()
    setup_paths()
    verify()
    train(epochs=150)
    list_checkpoints()
