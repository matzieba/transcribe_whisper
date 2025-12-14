import time

import requests
import whisper

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


def transcribe_audio(local_path: str) -> str:
    """
    Uses the Whisper model to transcribe the audio file at local_path.
    Returns the transcript text.
    """
    # Load a Whisper model - choose "tiny", "base", "small", "medium", "large"
    # or e.g. "medium.en" for English-only transcription:
    model = whisper.load_model("small")

    # Transcribe the audio
    result = model.transcribe(local_path)
    transcript = result["text"].strip()
    return transcript


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
