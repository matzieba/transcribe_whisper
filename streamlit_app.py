from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from diarize_transcribe import run_pipeline, AUDIO_URL
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

st.set_page_config(page_title="Transcription + Diarization", layout="wide")


@st.cache_data(show_spinner=False)
def cached_run(url: str, max_duration: float | None, prefer_faster_whisper: bool):
    return run_pipeline(
        url=url,
        max_duration_sec=max_duration,
        prefer_faster_whisper=prefer_faster_whisper,
    )


def main():
    st.title("Whisper + PyAnnote Speaker Diarization Optimized")

    url = st.text_input("Audio URL", value=AUDIO_URL)
    max_duration = st.number_input("Max duration (seconds, 0 = full)", min_value=0, value=60)
    prefer_faster = st.checkbox("Prefer faster-whisper (recommended if installed)", value=True)

    if st.button("Process"):
        with st.spinner("Running pipeline..."):
            md = None if max_duration == 0 else float(max_duration)
            merged = cached_run(url, md, prefer_faster)

        st.subheader("Speaker-Labeled Transcript")
        for seg in merged:
            st.markdown(
                f"[{seg.start:.2f} - {seg.end:.2f}] **{seg.speaker}:** {seg.text}"
            )


if __name__ == "__main__":
    main()
