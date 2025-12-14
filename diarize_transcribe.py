
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import requests

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

    with requests.get(url, stream=True, timeout=timeout_sec) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
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

    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(
            DIAR_MODEL,
            use_auth_token=HF_TOKEN,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load pyannote diarization pipeline. "
            "Ensure your HF_TOKEN is valid and that you accepted access at "
            "https://hf.co/pyannote/speaker-diarization-3.1."
        ) from e

    if pipeline is None:
        raise RuntimeError("pyannote Pipeline.from_pretrained returned None (unexpected).")
    return pipeline


@lru_cache(maxsize=1)
def get_transcriber(prefer_faster_whisper: bool = True):
    """
    Returns a callable that transcribes wav_path -> List[TranscriptionSegment].
    Prefers faster-whisper if installed (free, local, significantly faster).
    """
    if prefer_faster_whisper:
        try:
            from faster_whisper import WhisperModel

            device = "cpu"  # faster-whisper uses ctranslate2; MPS not used here
            compute_type = "int8"  # speed + lower RAM; good on laptops

            model = WhisperModel("small", device=device, compute_type=compute_type)

            def _fw_transcribe(wav_path: Path) -> List[TranscriptionSegment]:
                segments, _info = model.transcribe(
                    str(wav_path),
                    vad_filter=True,          # skip silences => faster
                    beam_size=1,              # faster
                )
                out: List[TranscriptionSegment] = []
                for s in segments:
                    text = (s.text or "").strip()
                    if text:
                        out.append(TranscriptionSegment(s.start, s.end, text))
                return out

            return _fw_transcribe
        except ImportError:
            pass  # fall back to openai-whisper

    import whisper

    device = "cpu"
    model = whisper.load_model("small", device=device)

    def _whisper_transcribe(wav_path: Path) -> List[TranscriptionSegment]:
        result = model.transcribe(str(wav_path), verbose=False)
        out: List[TranscriptionSegment] = []
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if text:
                out.append(TranscriptionSegment(float(seg["start"]), float(seg["end"]), text))
        return out

    return _whisper_transcribe


# -----------------------------
# Core pipeline steps
# -----------------------------
def diarize_audio(wav_16k_mono: Path) -> List[DiarizationSegment]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(wav_16k_mono))
    # waveform should already be mono 16k, but keep this robust:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(sample_rate) != 16000:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), 16000)
        sample_rate = 16000

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
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        segments.append(DiarizationSegment(segment.start, segment.end, str(speaker)))

    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def transcribe_audio(wav_16k_mono: Path, prefer_faster_whisper: bool = True) -> List[TranscriptionSegment]:
    transcriber = get_transcriber(prefer_faster_whisper=prefer_faster_whisper)
    segs = transcriber(wav_16k_mono)
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_diarization_and_transcript(
    diar: List[DiarizationSegment],
    tr: List[TranscriptionSegment],
) -> List[LabeledSegment]:
    """
    Linear-time sweep merge (fast). Assumes inputs sorted by start time.
    """
    out: List[LabeledSegment] = []
    j = 0  # diar index

    for t in tr:
        # advance diar segments that end before this transcript starts
        while j < len(diar) and diar[j].end <= t.start:
            j += 1

        best_speaker = "Unknown"
        best_ov = 0.0

        k = j
        # check diar segments that might overlap this transcript segment
        while k < len(diar) and diar[k].start < t.end:
            ov = _overlap(t.start, t.end, diar[k].start, diar[k].end)
            if ov > best_ov:
                best_ov = ov
                best_speaker = diar[k].speaker
            k += 1

        out.append(LabeledSegment(t.start, t.end, best_speaker, t.text))
    return out


def run_pipeline(
    url: str = AUDIO_URL,
    workdir: Path = DEFAULT_WORKDIR,
    max_duration_sec: Optional[float] = None,
    prefer_faster_whisper: bool = True,
    enable_diarization: Optional[bool] = None,
) -> List[LabeledSegment]:
    t0 = time.perf_counter()
    raw = download_file(url, workdir / "input.mp3")
    # Use a duration-specific cache key so changing max_duration actually regenerates audio.
    duration_tag = "full" if max_duration_sec is None else str(max_duration_sec).replace(".", "p")
    wav = ensure_wav_16k_mono(
        raw,
        workdir / f"audio_16k_mono_{duration_tag}.wav",
        max_duration_sec=max_duration_sec,
    )

    do_diarization = ENABLE_DIARIZATION if enable_diarization is None else bool(enable_diarization)

    diar: List[DiarizationSegment]
    if do_diarization:
        diar = diarize_audio(wav)
    else:
        diar = []

    tr = transcribe_audio(wav, prefer_faster_whisper=prefer_faster_whisper)
    merged = merge_diarization_and_transcript(diar, tr)

    elapsed = time.perf_counter() - t0
    print(
        f"[timing] pipeline completed in {elapsed:.2f}s "
        f"(max_duration_sec={max_duration_sec}, diarization={do_diarization}, faster_whisper={prefer_faster_whisper})",
        flush=True,
    )
    return merged


def main():
    merged = run_pipeline(max_duration_sec=60, prefer_faster_whisper=True)
    for s in merged:
        print(f"[{s.start:6.2f} → {s.end:6.2f}] {s.speaker}: {s.text}")


if __name__ == "__main__":
    main()
