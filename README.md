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
src/tsl/data/manifest.py    # SignTextExample dataclass
src/tsl/data/tsl51.py       # TSL-51 sentence manifest + landmark loader
src/tsl/data/eaf.py         # NSTDA ELAN .eaf parser
src/tsl/data/slt_collate.py # batch collate for SLT
src/tsl/text/tokenizer.py   # Thai char-level tokenizer
src/tsl/models/slt.py       # SignToTextTransformer (encoder-decoder)
src/tsl/train/train_slt.py  # training entrypoint
src/tsl/inference/sentence_translator.py  # checkpoint-based inference
src/tsl/eval/sentence_metrics.py          # CER, TER, exact match
src/tsl/api/app.py          # POST /translate-sentence
web/                      # webcam UI
scripts/                  # data-collection helpers
tests/                    # mirror src/tsl structure
```

## Sentence-level translation (Thai Sign -> Thai sentence)

A separate, additive pipeline that maps a full landmark sequence directly to a
Thai sentence. Independent of the word recognizer except for the shared
MediaPipe 543-landmark normalizer.

### How it differs from the word recognizer

The existing `PredictResponse` returns a single Thai word via a Prototype store
+ Transformer encoder. The new `POST /translate-sentence` endpoint takes a full
landmark sequence and returns a Thai SENTENCE via a `SignToTextTransformer`
encoder-decoder. The two share infrastructure (the MediaPipe 543-landmark
normalizer) but are otherwise independent.

### Datasets used

| Dataset | Used for | Why |
|---|---|---|
| `Namonpas/thai-sign-language-tsl51` (Hugging Face) | Main training data | 252 continuous sentence videos + landmark CSVs, CC BY-NC-SA 4.0 |
| NSTDA Thai Sign Language Multi-tier Annotation | Auxiliary annotation | ELAN .eaf files with CC + Gloss tiers; access via AI for Thai |
| TSL-ONE-S (skeletal npy) | Optional word-level pretrain | 4152 samples, request videos separately |
| (Foreign datasets like How2Sign/OpenASL) | Not used in v1 | Different language; not Thai Sign Language |

ThaiSignVis is NOT used in the initial release because the spec required
ready-to-use landmarks only (no raw video download). The new pipeline is
`landmarks -> text`, not `video -> text`.

### Training

```bash
python -m tsl.train.train_slt \
  --data-root /path/to/tsl51-local \
  --epochs 5 \
  --batch-size 4 \
  --out-dir checkpoints/slt
```

Outputs in `checkpoints/slt/`:
- `slt_model.pt` — state dict
- `tokenizer.json` — char-level vocab
- `model_config.json` — constructor kwargs (for inference reload)
- `train_metrics.json` — per-epoch loss

### Inference via API

Request:
```json
{
  "frames": [[0.0, 0.0, 0.0, ...], [0.0, 0.0, 0.0, ...]],
  "feature_dim": 162,
  "max_len": 128
}
```

Response:
```json
{
  "sentence": "ฉันกินข้าว",
  "tokens": [1, 5, 6, 7, 8, 9, 2],
  "score": 0.83
}
```

Status codes:
- 200: success
- 400: malformed input (wrong `feature_dim`, etc.)
- 503: no checkpoint trained yet

### Evaluation

```bash
python -c "
from tsl.eval.sentence_metrics import evaluate_sentences
refs = ['ฉันกินข้าว', 'สวัสดี']
hyps = ['ฉันกินข้าว', 'สวัสดี']
print(evaluate_sentences(refs, hyps))
# {'exact_match': 1.0, 'cer': 0.0, 'ter': 0.0, 'n': 2}
"
```

### Limitations

- This pipeline is `landmarks -> text`, not `video -> text`. The webcam flow
  still goes through the existing word recognizer.
- TSL-51 has only 252 sentence examples, so the model is a baseline, not
  production-grade.
- The decoder uses greedy search (no beam search yet).
- Foreign sign-language datasets (How2Sign/OpenASL/PHOENIX) are NOT used; they
  were considered for pretraining but excluded to keep the v1 purely Thai.

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
