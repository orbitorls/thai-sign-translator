#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec python3.12 -u -m tsl.train.train_slt \
    --stage combined \
    --data-root "data/tsl51,data/youtube_sl25_thai" \
    --tokenizer char \
    --model-size base \
    --epochs 100 \
    --batch-size 16 \
    --lr 3e-4 \
    --augment \
    --out-dir checkpoints/slt_combined \
    --eval-beam 4
