import os
import time
from functools import lru_cache

import requests
from faster_whisper import WhisperModel

AUDIO_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"
LOCAL_AUDIO_FILE = "audio.wav"


def download_audio_file(url: str = AUDIO_URL, local_path: str = LOCAL_AUDIO_FILE) -> str:
    """
    Downloads an audio file from the specified URL and saves it locally.
    Returns the path to the downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    """
    Load faster-whisper once to avoid re-initializing the model.
    """
    device_env = os.environ.get("FAST_WHISPER_DEVICE") or os.environ.get("WHISPER_DEVICE")
    device = (device_env or "auto").strip().lower()
    if device == "auto":
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, ""} else "cpu"

    model_size = (os.environ.get("FAST_WHISPER_MODEL") or "tiny").strip()
    compute_type = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(local_path: str) -> str:
    """
    Uses faster-whisper to transcribe the audio file at local_path.
    Returns the transcript text.
    """
    model = _load_model()

    segments, _info = model.transcribe(
        local_path,
        vad_filter=True,
        beam_size=1,
    )

    parts = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            parts.append(text)

    return " ".join(parts).strip()


def main():
    t0 = time.perf_counter()

    # 1) Download the file
    audio_path = download_audio_file()
    t_dl = time.perf_counter()

    # 2) Transcribe the audio
    transcript_text = transcribe_audio(audio_path)
    t_tx = time.perf_counter()

    # 3) Print or save the transcript
    print("TRANSCRIPT:\n", transcript_text, flush=True)

    print(
        f"[timing] download={t_dl - t0:.2f}s transcribe={t_tx - t_dl:.2f}s total={t_tx - t0:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
