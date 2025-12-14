from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import requests
import torchaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

AUDIO_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

HF_TOKEN = os.environ.get("HF_TOKEN")
DIAR_MODEL = "pyannote/speaker-diarization-3.1"
ENABLE_DIARIZATION = (
    os.environ.get("ENABLE_DIARIZATION", "true").strip().lower() in {"1", "true", "yes", "on"}
)
DEFAULT_WORKDIR = Path("data")
DEFAULT_WORKDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data models (clean + typed)
# -----------------------------
@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class LabeledSegment:
    start: float
    end: float
    speaker: str
    text: str


# -----------------------------
# Utilities
# -----------------------------
def _run(cmd: List[str]) -> None:
    """Run external command safely (no shell=True)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")


def download_file(url: str, dst: Path, timeout_sec: int = 60) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # simple cache: if already present and non-trivial size, reuse
    if dst.exists() and dst.stat().st_size > 50_000:
        return dst

    with requests.get(url, stream=True, timeout=timeout_sec) as response:
        response.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dst)
    return dst


def ensure_wav_16k_mono(
    src: Path,
    dst: Path,
    max_duration_sec: Optional[float] = None,
) -> Path:
    """
    Convert audio to 16kHz mono WAV once using ffmpeg (fast + reliable for mp3).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 50_000:
        return dst

    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000"]
    if max_duration_sec is not None:
        cmd += ["-t", str(float(max_duration_sec))]
    cmd += [str(dst)]
    _run(cmd)
    return dst


# -----------------------------
# Model loaders (cached)
# -----------------------------
@lru_cache(maxsize=1)
def get_diarization_pipeline():
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. Export HF_TOKEN to use pyannote diarization."
        )

    try:
        pipeline = Pipeline.from_pretrained(
            DIAR_MODEL,
            use_auth_token=HF_TOKEN,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pyannote diarization pipeline. "
            "Ensure your HF_TOKEN is valid and that you accepted access at "
            "https://hf.co/pyannote/speaker-diarization-3.1."
        ) from exc

    if pipeline is None:
        raise RuntimeError("pyannote Pipeline.from_pretrained returned None (unexpected).")
    return pipeline


@lru_cache(maxsize=1)
def get_transcriber():
    device = "cpu"
    model_size = (os.environ.get("FAST_WHISPER_MODEL") or "tiny").strip()
    compute_type = "int8"

    return WhisperModel(model_size, device=device, compute_type=compute_type)


# -----------------------------
# Core pipeline steps
# -----------------------------
def diarize_audio(wav_16k_mono: Path) -> List[DiarizationSegment]:
    waveform, sample_rate = torchaudio.load(str(wav_16k_mono))
    pipeline = get_diarization_pipeline()
    diarization_out = pipeline(
        {"waveform": waveform, "sample_rate": int(sample_rate), "uri": str(wav_16k_mono)}
    )

    annotation = (
        diarization_out.speaker_diarization
        if hasattr(diarization_out, "speaker_diarization")
        else diarization_out
    )

    segments: List[DiarizationSegment] = []
    for segment, unused_track, speaker_label in annotation.itertracks(yield_label=True):
        segments.append(DiarizationSegment(segment.start, segment.end, str(speaker_label)))

    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def transcribe_audio(wav_16k_mono: Path) -> List[TranscriptionSegment]:
    model = get_transcriber()
    raw_segments, unused_model_info = model.transcribe(
        str(wav_16k_mono),
        vad_filter=True,  # skip silences => faster
        beam_size=1,      # faster
    )
    transcription_segments: List[TranscriptionSegment] = []
    for segment in raw_segments:
        text = (segment.text or "").strip()
        if text:
            transcription_segments.append(TranscriptionSegment(segment.start, segment.end, text))
    transcription_segments.sort(key=lambda s: (s.start, s.end))
    return transcription_segments


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_diarization_and_transcript(
    diarization_segments: List[DiarizationSegment],
    transcription_segments: List[TranscriptionSegment],
) -> List[LabeledSegment]:
    """
    Linear-time sweep merge (fast). Assumes inputs sorted by start time.
    """
    out: List[LabeledSegment] = []
    diar_index = 0

    for transcript in transcription_segments:
        while diar_index < len(diarization_segments) and diarization_segments[diar_index].end <= transcript.start:
            diar_index += 1

        best_speaker = "Unknown"
        best_ov = 0.0

        overlap_index = diar_index
        while overlap_index < len(diarization_segments) and diarization_segments[overlap_index].start < transcript.end:
            ov = _overlap(
                transcript.start,
                transcript.end,
                diarization_segments[overlap_index].start,
                diarization_segments[overlap_index].end,
            )
            if ov > best_ov:
                best_ov = ov
                best_speaker = diarization_segments[overlap_index].speaker
            overlap_index += 1

        out.append(LabeledSegment(transcript.start, transcript.end, best_speaker, transcript.text))
    return out


def run_pipeline(
    url: str = AUDIO_URL,
    workdir: Path = DEFAULT_WORKDIR,
    max_duration_sec: Optional[float] = None,
    enable_diarization: Optional[bool] = None,
) -> List[LabeledSegment]:
    start_time = time.perf_counter()
    downloaded_audio = download_file(url, workdir / "input.mp3")
    duration_tag = "full" if max_duration_sec is None else str(max_duration_sec).replace(".", "p")
    wav_path = ensure_wav_16k_mono(
        downloaded_audio,
        workdir / f"audio_16k_mono_{duration_tag}.wav",
        max_duration_sec=max_duration_sec,
    )

    do_diarization = ENABLE_DIARIZATION if enable_diarization is None else bool(enable_diarization)

    diarization_segments: List[DiarizationSegment]
    if do_diarization:
        diarization_segments = diarize_audio(wav_path)
    else:
        diarization_segments = []

    transcription_segments = transcribe_audio(wav_path)
    labeled_segments = merge_diarization_and_transcript(diarization_segments, transcription_segments)

    elapsed = time.perf_counter() - start_time
    print(
        f"[timing] pipeline completed in {elapsed:.2f}s "
        f"(max_duration_sec={max_duration_sec}, diarization={do_diarization}, faster_whisper=True)"
    )
    return labeled_segments


def main():
    labeled_segments = run_pipeline(max_duration_sec=60)
    for segment in labeled_segments:
        print(f"[{segment.start:6.2f} → {segment.end:6.2f}] {segment.speaker}: {segment.text}")


if __name__ == "__main__":
    main()
