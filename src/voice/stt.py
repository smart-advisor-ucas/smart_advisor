"""Speech-to-Text. Public surface:  transcribe_audio  /  transcribe_audio_async.

    from voice import transcribe_audio
    result = transcribe_audio(audio)        # audio: path | bytes | numpy array
    print(result.text)                      # MSA transcript

Runs in mock mode by default (no model, no GPU). Set VOICE_BACKEND=real to use
Whisper Large-v3-turbo (override the model via the WHISPER_MODEL_ID env var).
"""
from __future__ import annotations
import asyncio
import logging
import os
import time
import wave
from pathlib import Path

from .types import (
    TranscriptionResult,
    STTError,
    AudioFormatError,
    AudioTooLongError,
    EmptyTranscriptError,
)
from .config import resolve_backend

logger = logging.getLogger(__name__)

# --- Tunables --------------------------------------------------------------- #
MAX_AUDIO_SEC = 60.0
NO_SPEECH_THRESHOLD = 0.6
# Large-v3-turbo: ~1.6GB vs large-v3's ~3GB, much faster on CPU, small accuracy
# tradeoff. Override with the WHISPER_MODEL_ID env var to swap models without
# editing code (e.g. back to "openai/whisper-large-v3" for max accuracy).
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-large-v3-turbo")
TARGET_SR = 16000

# The transcript the mock returns, so Fatema's RAG gets deterministic input.
_MOCK_TRANSCRIPT = (
    "┘à╪د ┘ç┘è ╪د┘┘à┘ç╪د╪▒╪د╪ز ╪د┘╪ز┘è ╪│╪ث┘â╪ز╪│╪ذ┘ç╪د ╪╣┘╪» ╪»╪▒╪د╪│╪ر ╪ز╪«╪╡╪╡ ╪د┘╪░┘â╪د╪ة ╪د┘╪د╪╡╪╖┘╪د╪╣┘è ┘ê╪╣┘┘à ╪د┘╪ذ┘è╪د┘╪د╪ز╪ا"
)


def set_mock_transcript(text: str) -> None:
    """Change what the mock backend returns (handy for testing different queries)."""
    global _MOCK_TRANSCRIPT
    _MOCK_TRANSCRIPT = text


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def transcribe_audio(
    audio,
    *,
    language: str = "ar",
    normalize_to_msa: bool = True,
    backend: str | None = None,
) -> TranscriptionResult:
    """Transcribe Arabic speech to MSA text.

    Args:
        audio: a file path (str/Path), raw bytes (e.g. a browser upload), or a
               decoded mono waveform (numpy float32 array).
        language: language hint for the model (default "ar").
        normalize_to_msa: whether to flag the transcript as MSA-normalized.
        backend: "mock" | "real" (defaults to VOICE_BACKEND env, then "mock").

    Returns:
        TranscriptionResult: text, language, backend, latency_sec,
        audio_duration_sec, sample_rate, no_speech_prob, normalized_to_msa.

    Raises:
        AudioFormatError     - input type/format not understood.
        AudioTooLongError    - audio longer than MAX_AUDIO_SEC.
        EmptyTranscriptError - silence / noise / no speech detected.
        STTError             - real-mode infrastructure failure (ffmpeg
                                missing, torch/transformers missing, model
                                load or inference failure). `except VoiceError`
                                (or `except STTError`) catches this too.
    """
    be = resolve_backend(backend)
    if be == "mock":
        result = _mock_transcribe(audio, language, normalize_to_msa)
    else:
        result = _whisper_transcribe(audio, language, normalize_to_msa)
    logger.info(
        "STT engine used: %s (%.3fs, %.2fs audio)",
        result.backend, result.latency_sec, result.audio_duration_sec,
    )
    return result


async def transcribe_audio_async(audio, **kw) -> TranscriptionResult:
    """Async wrapper. Whisper is compute-bound, so we offload to a thread rather
    than fake async ظ¤ this keeps a FastAPI event loop responsive."""
    return await asyncio.to_thread(transcribe_audio, audio, **kw)


# --------------------------------------------------------------------------- #
# Shared input validation                                                      #
# --------------------------------------------------------------------------- #
def _validate_input(audio) -> None:
    """Reject obviously-wrong inputs early, with a clear error, in any backend."""
    if isinstance(audio, (str, Path)):
        if not Path(audio).exists():
            raise AudioFormatError(f"file not found: {audio}")
        return
    if isinstance(audio, (bytes, bytearray)):
        if len(audio) == 0:
            raise AudioFormatError("empty audio bytes")
        return
    # numpy array (duck-typed so we don't import numpy in mock mode)
    if hasattr(audio, "shape") and hasattr(audio, "dtype"):
        return
    raise AudioFormatError(
        f"unsupported audio type {type(audio).__name__}; "
        "expected path, bytes, or a numpy waveform"
    )


# --------------------------------------------------------------------------- #
# Mock backend                                                                 #
# --------------------------------------------------------------------------- #
def _mock_transcribe(audio, language, normalize_to_msa) -> TranscriptionResult:
    t0 = time.perf_counter()
    _validate_input(audio)
    return TranscriptionResult(
        text=_MOCK_TRANSCRIPT,
        language=language,
        backend="mock-stt",
        latency_sec=round(time.perf_counter() - t0, 4),
        audio_duration_sec=0.0,
        sample_rate=TARGET_SR,
        no_speech_prob=0.01,
        normalized_to_msa=normalize_to_msa,
    )


# --------------------------------------------------------------------------- #
# Real backend (Whisper, model id set by WHISPER_MODEL_ID) ظ¤ lazy-loaded so
# importing this module is cheap
# --------------------------------------------------------------------------- #
_model = None
_processor = None


def _load_whisper():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID, dtype=dtype
        ).to(device)
        _processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    return _model, _processor


def _coerce_to_clean_wav(audio) -> Path:
    """Normalize any input to a 16 kHz mono WAV on disk via FFmpeg.

    We never trust the file extension ظ¤ we always re-encode path/bytes inputs
    through FFmpeg so container-format mismatches (e.g. m4a named .wav) are
    caught early.  Numpy arrays are assumed to be 16 kHz mono float32 already
    and are written directly via soundfile without FFmpeg.

    The returned path is always a freshly-created temp file; the caller is
    responsible for deleting it when done.
    """
    import subprocess
    import tempfile

    # numpy array: write directly ظ¤ no FFmpeg needed.
    # ASSUMPTION: the array is already 16 kHz mono float32.  If your capture
    # device records at a different rate, resample before calling this function.
    if hasattr(audio, "shape") and hasattr(audio, "dtype"):
        import numpy as np
        import soundfile as sf
        fd, out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(out, np.asarray(audio).squeeze(), TARGET_SR)
        return Path(out)

    # bytes: write to a temp file so FFmpeg can read it.
    src_is_temp = isinstance(audio, (bytes, bytearray))
    if src_is_temp:
        fd, src = tempfile.mkstemp(suffix=".in")
        os.close(fd)
        Path(src).write_bytes(audio)
    else:
        src = str(audio)

    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", str(TARGET_SR), "-ac", "1", out],
            check=True, capture_output=True,
        )
    except FileNotFoundError as e:
        # ffmpeg itself isn't installed/on PATH - an environment problem, not
        # a bad input, so don't call it an "audio format" error.
        Path(out).unlink(missing_ok=True)
        raise STTError(
            "ffmpeg not found on PATH - install ffmpeg to use real-mode STT"
        ) from e
    except subprocess.CalledProcessError as e:
        Path(out).unlink(missing_ok=True)
        raise AudioFormatError(f"could not decode audio: {e}") from e
    finally:
        if src_is_temp:
            Path(src).unlink(missing_ok=True)
    return Path(out)


def _read_wav_mono16k(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
    return audio, sr


def _whisper_transcribe(audio, language, normalize_to_msa) -> TranscriptionResult:
    _validate_input(audio)
    t0 = time.perf_counter()

    wav_path = _coerce_to_clean_wav(audio)
    try:
        with wave.open(str(wav_path), "rb") as w:
            duration = w.getnframes() / float(w.getframerate())
        if duration > MAX_AUDIO_SEC:
            raise AudioTooLongError(f"{duration:.1f}s exceeds {MAX_AUDIO_SEC}s limit")

        try:
            model, processor = _load_whisper()
            array, _ = _read_wav_mono16k(wav_path)

            inputs = processor(array, sampling_rate=TARGET_SR, return_tensors="pt")
            # Match the input tensor to the model's device and precision.
            features = inputs.input_features.to(model.device, dtype=model.dtype)
            gen = model.generate(features, language=language, task="transcribe")
            text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        except Exception as e:
            # Covers missing torch/transformers, model download failures, and
            # inference errors (e.g. CUDA OOM) - none of these are the caller's
            # fault, but they must still surface as VoiceError so Fatema's
            # `except VoiceError` catches them.
            raise STTError(f"Whisper transcription failed: {e}") from e

        if not text:
            raise EmptyTranscriptError("model returned no speech")
    finally:
        # Always remove the temp WAV ظ¤ _coerce_to_clean_wav always returns a
        # freshly-created file, so deleting it here is always correct.
        wav_path.unlink(missing_ok=True)

    return TranscriptionResult(
        text=text,
        language=language,
        backend=WHISPER_MODEL_ID.rsplit("/", 1)[-1],
        latency_sec=round(time.perf_counter() - t0, 3),
        audio_duration_sec=round(duration, 2),
        sample_rate=TARGET_SR,
        no_speech_prob=None,  # raw generate() doesn't expose this; gate on empty text
        normalized_to_msa=normalize_to_msa,
    )
