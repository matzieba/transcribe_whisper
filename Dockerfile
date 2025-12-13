FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
