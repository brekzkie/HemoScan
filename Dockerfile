# ================================================================
# Dockerfile — HemoScan FastAPI Application
# Base: Python 3.10-slim (ringan, cocok untuk TensorFlow + FastAPI)
# ================================================================

FROM python:3.10-slim

# ── Environment variables ────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

# ── Install system dependencies ──────────────────────────────────
# libgl1 & libglib2.0 dibutuhkan oleh OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements dan install dulu (cache layer) ────────────
COPY requirements.docker.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ────────────────────────────────────────────
COPY server.py ./
COPY inference.py ./
COPY init_db.py ./
COPY postgres_schema.sql ./
COPY database.py ./
COPY index.html ./

# ── Copy static assets ──────────────────────────────────────────
COPY logo1.png ./logo1.png
COPY logo2.png ./logo2.png
COPY Backgorund.png ./
COPY loading.mp4 ./
COPY loading.svg ./

# ── Copy model files ────────────────────────────────────────────
COPY model_output/ ./model_output/

# ── Create uploads directory ────────────────────────────────────
RUN mkdir -p uploads

# ── Expose port ──────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint script ───────────────────────────────────────────
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
