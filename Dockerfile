# ── Stage 1: build the React frontend ────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /build
COPY src/ui/frontend/package.json src/ui/frontend/package-lock.json ./
RUN npm ci
COPY src/ui/frontend/ ./
RUN npm run build


# ── Stage 2: Python runtime ──────────────────────────────────────────────
FROM python:3.11-slim

# ffmpeg: Whisper STT decodes uploads with it.
# libasound2*: runtime dependency of the Azure Speech SDK (name differs
# between Debian bookworm and trixie, hence the fallback).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && (apt-get install -y --no-install-recommends libasound2t64 \
        || apt-get install -y --no-install-recommends libasound2) \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000 with no writable $HOME by
# default; create a real user so model caches have somewhere to live.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VOICE_BACKEND=real \
    VOICE_TTS=auto

WORKDIR /app
RUN chown user:user /app
USER user

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
ENV PATH="/home/user/.local/bin:${PATH}"

# Pre-fetch the voice models at build time so the first request isn't a
# 1.6 GB download repeated on every Space restart (ephemeral disk).
RUN python -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; \
    AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3-turbo'); \
    AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo')"
RUN python -c "from pathlib import Path; from piper.download_voices import download_voice; \
    d = Path.home() / '.cache' / 'piper-voices'; d.mkdir(parents=True, exist_ok=True); \
    download_voice('ar_JO-kareem-medium', d)"

# App source, knowledge base, and the built frontend bundle.
# Secrets (GITHUB_TOKEN, GROQ_API_KEY, HF_TOKEN, AZURE_SPEECH_KEY) are NOT
# baked in — they come from the environment at runtime.
COPY --chown=user:user src/ ./src/
COPY --chown=user:user data/chroma_db/ ./data/chroma_db/
COPY --from=frontend --chown=user:user /build/dist/ ./src/ui/frontend/dist/

EXPOSE 7860
CMD ["uvicorn", "src.ui.main:app", "--host", "0.0.0.0", "--port", "7860"]
