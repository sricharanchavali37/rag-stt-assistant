"""
services/transcriber.py — Stage 1
"""

import os
import uuid
import whisper

from models import RAGContext


def transcribe(audio_bytes: bytes, filename: str, whisper_model) -> RAGContext:
    ctx = RAGContext()

    ext = os.path.splitext(filename)[-1].lower()
    if ext not in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
        ext = ".wav"

    # Write to project root — avoids spaces/permission issues with system temp
    tmp_path = f"tmp_audio_{uuid.uuid4().hex}{ext}"

    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()

        if not transcript:
            ctx.error = "Whisper returned an empty transcription."
            return ctx

        ctx.query = transcript
        ctx.transcription = transcript

    except Exception as e:
        ctx.error = f"Transcription failed: {str(e)}"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return ctx