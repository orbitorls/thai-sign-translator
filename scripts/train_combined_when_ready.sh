#!/usr/bin/env bash
# Waits for the YouTube-SL-25 manifest.csv to appear, then trains combined model.
# Usage: bash scripts/train_combined_when_ready.sh
set -e

MANIFEST="data/youtube_sl25_thai/manifest.csv"
OUT_DIR="checkpoints/slt_combined"

echo "Waiting for $MANIFEST …"
while [ ! -f "$MANIFEST" ]; do
    sleep 30
    echo "  still waiting… ($(ls data/youtube_sl25_thai/landmarks/*.npy 2>/dev/null | wc -l) segments so far)"
done

echo "Manifest found! Starting combined training …"
PYTHONPATH=src python3.12 -u -m tsl.train.train_slt \
    --stage combined \
    --data-root "data/tsl51,data/youtube_sl25_thai" \
    --tokenizer char \
    --model-size base \
    --epochs 100 \
    --batch-size 16 \
    --lr 3e-4 \
    --augment \
    --out-dir "$OUT_DIR" \
    --eval-beam 4

echo "Training complete. Checkpoint: $OUT_DIR"
