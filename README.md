# Local Streamlit Whisper

## Default setup (Docker Compose)
- Ensure Docker + docker-compose are installed and running.
- Build and start the stack: `docker-compose up --build`
- Access the Streamlit app at the URL printed in logs (typically http://localhost:8501).
- Logs show ffmpeg/whisper/pyannote output; stop with `Ctrl+C`.
- Models: faster-whisper `tiny` is baked into the image for quicker first run. Caches are stored in `_cache/` via a volume.

## Configure credentials (no secrets committed)
- Hugging Face token is required for diarization (pyannote). Create one at https://huggingface.co/settings/tokens with access to `pyannote/speaker-diarization-3.1`.
- Add it to a local `.env` (loaded by the app) or export before `docker-compose up`:
  - `.env` example: `HF_TOKEN=hf_your_token_here`
  - Optional: `ENABLE_DIARIZATION=false` to skip diarization (no HF token needed).
- Speed/compute toggles (override in `.env`):
  - `FAST_WHISPER_DEVICE`: `auto` (default), `cpu`, `cuda`, or `mps`. Use `cuda` with `docker run --gpus all`/compose GPU support.
  - `FAST_WHISPER_MODEL`: whisper size; defaults to `tiny` for speed (`base`/`small` for quality).

## Run the transcription script (CLI)
If you prefer local (non-docker) execution:
- Install system deps: `ffmpeg` and Python 3.10+.
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run: `python transcribe.py` (downloads sample audio to `audio.wav`, transcribes, prints transcript).

## Run the Streamlit demo locally (optional)
- With the venv active and `HF_TOKEN` set, run: `streamlit run streamlit_app.py`
- Open the provided local URL, set “Max duration” and other options, then click **Process**.
