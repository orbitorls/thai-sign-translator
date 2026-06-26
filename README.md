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
scripts/                  # thin operational entrypoints
scripts/data/             # dataset extraction / conversion implementations
scripts/maintenance/      # repo hygiene / cleanup implementations
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
| `Namonpas/thai-sign-language-tsl51` (Hugging Face) | Fine-tuning | 252 continuous sentence videos + landmark CSVs, CC BY-NC-SA 4.0 |
| How2Sign B-F-H keypoints (Google Drive) | Pretraining (ASL-English) | ~80k sentences, ~80 hr, ready-to-use OpenPose keypoints (137 per frame) |
| NSTDA Thai Sign Language Multi-tier Annotation | Auxiliary annotation | ELAN .eaf files with CC + Gloss tiers; access via AI for Thai |
| TSL-ONE-S (skeletal npy) | Optional word-level pretrain | 4152 samples, 29 signers, 184 glosses |

ThaiSignVis is NOT used because the spec required ready-to-use landmarks only
(no raw video download). The pipeline is `landmarks -> text`, not `video -> text`.

### Model-size presets

Three architecture presets are available via `--model-size`:

| Preset | d_model | nhead | Layers (enc/dec) | dim_feedforward | max_pos_len |
|---|---|---|---|---|---|
| `small` | 64 | 4 | 2/2 | 128 | 1024 |
| `base` | 256 | 8 | 4/4 | 1024 | 1024 |
| `large` | 512 | 8 | 6/6 | 2048 | 2048 |

### Two-stage training

**Stage 1 — Pretrain on How2Sign** (ASL keypoints → English text):
```bash
python -m tsl.train.train_slt \
  --stage how2sign \
  --data-root /path/to/how2sign \
  --model-size base \
  --epochs 10 \
  --batch-size 8 \
  --lr 5e-4 \
  --out-dir checkpoints/slt_pretrain
```

**Stage 2 — Fine-tune on TSL-51** (Thai landmarks → Thai text):
```bash
python -m tsl.train.train_slt \
  --stage finetune \
  --data-root /path/to/tsl51-local \
  --pretrained-checkpoint checkpoints/slt_pretrain \
  --model-size base \
  --epochs 20 \
  --batch-size 4 \
  --lr 2e-4 \
  --out-dir checkpoints/slt_finetune
```

**Direct TSL-51 training** (no pretraining):
```bash
python -m tsl.train.train_slt \
  --data-root /path/to/tsl51-local \
  --model-size base \
  --epochs 5 \
  --batch-size 4 \
  --out-dir checkpoints/slt
```

### WSL GPU training

Use WSL2 with the NVIDIA WSL driver and a CUDA-enabled PyTorch install. From
WSL, verify GPU access with:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Recommended training command:

```bash
python -m tsl.train.train_slt \
  --data-root /path/to/tsl51-local \
  --model-size base \
  --device auto \
  --require-gpu \
  --epochs 5 \
  --batch-size 4 \
  --out-dir checkpoints/slt
```

`--device auto` uses CUDA when available and falls back to CPU otherwise.
`--require-gpu` fails fast if no GPU is detected.

Outputs in `<out-dir>/`:
- `slt_model.pt` — state dict
- `tokenizer.json` — char-level vocab
- `model_config.json` — constructor kwargs (for inference reload)
- `train_metrics.json` — per-epoch loss

### Download How2Sign for pretraining

```bash
bash scripts/download_how2sign_keypoints.sh /path/to/how2sign
```

This downloads ~23 GB of frontal-view B-F-H 2D keypoints + re-aligned English
text CSVs and arranges them in the expected directory structure.

### Decoding

The model supports two decoding strategies:

- **Greedy** (default): fast, deterministic.
- **Beam search**: higher quality, configurable beam width. Pass `beam_size` to
  `SentenceTranslator.translate()` or use the CLI in evaluation scripts.

```python
from tsl.inference.sentence_translator import SentenceTranslator
translator = SentenceTranslator("checkpoints/slt_v2")
pred = translator.translate(features, beam_size=5, length_penalty=1.0)
print(pred.sentence)  # better quality than greedy
```

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
- TSL-51 has only 252 sentence examples, so the model benefits substantially
  from How2Sign pretraining.
- How2Sign keypoints are OpenPose format (137 × 3), different from MediaPipe;
  the input projection layer handles the dimensionality difference.

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
