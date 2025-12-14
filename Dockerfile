FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache

WORKDIR /app

# System deps: ffmpeg for audio decode, git for some pip deps (rare), build tools for wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch/torchaudio CPU wheels explicitly (stable + avoids CUDA/MPS weirdness in containers)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchaudio==2.3.1 \
 && pip install -r requirements.txt

# Prefetch faster-whisper tiny model into the image to avoid first-run downloads
RUN python - <<'PY'
from faster_whisper.utils import download_model

# Downloads model weights into cache (HF_HOME/XDG_CACHE_HOME).
download_model("tiny")
PY

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
