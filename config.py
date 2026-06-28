"""Global constants and filesystem paths for the Thai Sign Language translator.

Landmark layout matches the Google ISLR (asl-signs) format exactly:
a frame is (N_LANDMARKS, 3) with the canonical concat order
face(468) | left_hand(21) | pose(33) | right_hand(21).
"""

import os

# --- Landmark layout (matches ISLR exactly) ---
N_LANDMARKS = 543

# Canonical concat order. Global indices into the (543, 3) frame.
FACE = slice(0, 468)          # face mesh, idx 0..467
LEFT_HAND = slice(468, 489)   # left hand, idx 468..488 (21 pts)
POSE = slice(489, 522)        # pose, idx 489..521 (33 pts)
RIGHT_HAND = slice(522, 543)  # right hand, idx 522..542 (21 pts)

# Named single-landmark global indices (live inside the POSE group).
NOSE_IDX = 489        # pose nose -> reference / centering point
LSHOULDER_IDX = 500   # pose left shoulder (pose 11)
RSHOULDER_IDX = 501   # pose right shoulder (pose 12)
# Scale factor = inter-shoulder distance between LSHOULDER_IDX and RSHOULDER_IDX.

# --- Feature layout versions ---
RAW_MEDIAPIPE_COORD_DIM_V3 = 3
RAW_MEDIAPIPE_COORD_DIM_V4 = 4
FEATURE_LAYOUT_VERSION_V3 = "v3-312"
FEATURE_LAYOUT_VERSION_V4 = "v4-312-masked"

# --- Filesystem paths ---
# Repo root = directory containing this file.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
ISLR_PARQUET_DIR = os.path.join(DATA_DIR, "islr", "train_landmark_files")
ISLR_CSV_PATH = os.path.join(DATA_DIR, "islr", "train.csv")
THAI_DATA_DIR = os.path.join(DATA_DIR, "thai")

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
ENCODER_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "encoder.pt")
# Saved/loaded with torch.save/torch.load (see PrototypeStore.save), so use .pt.
PROTOTYPE_STORE_PATH = os.path.join(CHECKPOINT_DIR, "prototypes.pt")
# Sign-to-text (SLT) Transformer checkpoint directory. The shipped
# deliverable is stored under ``checkpoints/slt_v2``; the inference
# wrapper reads ``slt_model.pt``, ``tokenizer.json`` and
# ``model_config.json`` lazily from there.
SLT_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "slt_v2")
SLT_COMBINED_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "slt_combined_v2")
# Canonical PoseT5 runtime config. Public runtime paths should resolve through
# this bundle config first so API/catalog/CLI share the same source of truth.
SLT_V3_MODEL_CONFIG = os.environ.get(
    "SLT_V3_MODEL_CONFIG",
    os.path.join(ROOT_DIR, "configs", "model", "pose_t5_tsl51_tune3.json"),
).strip()
# Current best verified PoseT5 artifact as of 2026-06-20.
# The mixed-vocab export remains a research lane; the strongest verified
# runtime artifact currently usable in practice is the TSL51-only export.
SLT_V3_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "pose_t5_rtx4060_tsl51_only_export_verified")

USER_CONTRIBUTIONS_DIR = os.path.join(DATA_DIR, "user_contributions")
CONSENT_REGISTRY_DIR = os.path.join(DATA_DIR, "consent")

# PDPA / privacy (set in production via environment variables).
TSL_CONSENT_HMAC_KEY = os.environ.get("TSL_CONSENT_HMAC_KEY", "").strip()
TSL_DATA_ENCRYPTION_KEY = os.environ.get("TSL_DATA_ENCRYPTION_KEY", "").strip()
VIDEO_RETENTION_DAYS = int(os.environ.get("VIDEO_RETENTION_DAYS", "365"))
MAX_FEEDBACK_VIDEO_BYTES = 10 * 1024 * 1024
