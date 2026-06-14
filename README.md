# Thai Sign Language Few-Shot Translator

Pretrain a temporal landmark encoder on the Google ISLR (asl-signs) dataset with
Prototypical Networks (episodic training); add new signs few-shot by storing
prototypes; recognize signs from a webcam via MediaPipe landmarks; (stretch)
segment a continuous stream at hand-pauses and rule-reorder into a Thai sentence.

## Tech stack

Python 3.11, PyTorch, MediaPipe, FastAPI + uvicorn, NumPy, pandas + pyarrow
(parquet), pytest, fastdtw, scikit-learn, matplotlib. Training runs on free
Kaggle/Colab GPU; inference runs on a laptop CPU with a webcam.

## Landmark format

543 landmarks per frame, canonical concat order:
`face(468) | left_hand(21) | pose(33) | right_hand(21)`. A raw frame is a
`(543, 3)` float32 array; missing landmarks (e.g. an absent hand) are `NaN`.
All layout constants live in `config.py`.

## Repository layout

```
config.py                 # global constants + landmark layout + paths
src/tsl/features/         # MediaPipe extraction + normalization
src/tsl/data/             # ISLR loader, episodic sampler, Thai clips
src/tsl/models/           # LandmarkEncoder + ProtoNet
src/tsl/train/            # episodic training loop + checkpoint export
src/tsl/registry/         # few-shot prototype store
src/tsl/inference/        # webcam recognizer
src/tsl/segment/          # (stretch) micro-pause segmentation
src/tsl/grammar/          # (stretch) rule-based Thai reorder
src/tsl/baseline/         # (stretch) DTW baseline
src/tsl/eval/             # metrics + two-track evaluation
src/tsl/api/              # FastAPI app + pydantic schemas
web/                      # webcam UI
scripts/                  # data-collection helpers
tests/                    # mirror src/tsl structure
```

## Setup

```bash
python -m venv .venv
. .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the tests

```bash
pytest
```

Tests use tiny synthetic tensors and a 2-3 sample fake parquet fixture; they
never require the full ISLR download.
