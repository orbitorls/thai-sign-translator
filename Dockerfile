# ─── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend

# Install deps first (layer-cached until package-lock.json changes)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
RUN npm run build
# Output: /app/frontend/dist/


# ─── Stage 2: Python runtime + model weights ──────────────────────────────────
FROM python:3.11-slim AS runtime

# System deps needed by mediapipe and torch (no GPU — CPU inference only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (pinned in requirements.txt)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY config.py ./
COPY src/ ./src/

# Pre-baked frontend bundle
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# ── Checkpoint provisioning ───────────────────────────────────────────────────
# The verified PoseT5 checkpoint (~GB) is NOT baked into this image.
# Mount it at runtime or pull at startup. Two options:
#
# Option A – Docker volume / bind-mount (recommended for large files):
#   docker run -v /host/path/to/checkpoints:/app/checkpoints ...
#
# Option B – Pull from Kaggle at boot (set KAGGLE_USERNAME + KAGGLE_KEY env vars):
#   Add a startup script that calls:
#     kaggle datasets download orbitorls/thai-sign-ckpt -p /app/checkpoints --unzip
#
# The server returns HTTP 503 for /translate until the checkpoint is present;
# the frontend shows "โมเดลนี้ยังไม่พร้อมใช้งาน" in that case.
# ─────────────────────────────────────────────────────────────────────────────

# HuggingFace model cache — pre-warm mt5-small weights so the first request
# doesn't trigger a cold download. Omit the RUN line if network is unavailable
# at build time; the first /translate call will download on-the-fly instead.
# RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
#     AutoTokenizer.from_pretrained('google/mt5-small'); \
#     AutoModelForSeq2SeqLM.from_pretrained('google/mt5-small')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# FastAPI + uvicorn, bind all interfaces for container networking
# HTTPS termination is handled by the host's reverse proxy (e.g. Caddy, nginx, Fly.io edge)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/models')"

CMD ["uvicorn", "tsl.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
