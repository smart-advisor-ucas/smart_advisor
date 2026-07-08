"""Configuration: backend mode, TTS engine selection, and credentials.

Backend mode (VOICE_BACKEND):
  - "mock" (default): no models, no GPU, no API key.
  - "real": Whisper Large-v3 for STT and a real TTS engine (see below).

TTS engine (VOICE_TTS) — only used when backend == "real":
  - "azure": Azure Neural only (raises if it fails).
  - "piper": Piper offline only (tashkeel -> ar_JO-kareem, no network).
  - "auto"  (default): try Azure, fall back to Piper if Azure fails
            (timeout, missing key). Quality online, guaranteed audio offline.

Priority (each): explicit function arg, else env var, else default.

Azure (real + azure/auto):  AZURE_SPEECH_KEY, AZURE_SPEECH_REGION (default "eastus")
Piper (real + piper/auto):  PIPER_MODEL_PATH (optional local .onnx),
                            PIPER_LENGTH_SCALE (speaking rate, <1.0 faster, default "0.95")
"""
from __future__ import annotations
import importlib.util
import os
import shutil
from pathlib import Path

try:
    # Optional convenience: load a local .env file into os.environ so
    # VOICE_BACKEND/AZURE_SPEECH_KEY/etc. can live outside the shell. A no-op
    # if python-dotenv isn't installed, so mock mode stays dependency-free.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_BACKEND = "mock"
VALID_BACKENDS = {"mock", "real"}

DEFAULT_TTS_ENGINE = "auto"
VALID_TTS_ENGINES = {"azure", "piper", "auto"}


def resolve_backend(explicit: str | None = None) -> str:
    be = (explicit or os.getenv("VOICE_BACKEND") or DEFAULT_BACKEND).lower()
    if be not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {be!r}")
    return be


def resolve_tts_engine(explicit: str | None = None) -> str:
    eng = (explicit or os.getenv("VOICE_TTS") or DEFAULT_TTS_ENGINE).lower()
    if eng not in VALID_TTS_ENGINES:
        raise ValueError(f"tts engine must be one of {VALID_TTS_ENGINES}, got {eng!r}")
    return eng


def azure_credentials() -> tuple[str | None, str]:
    return os.getenv("AZURE_SPEECH_KEY"), os.getenv("AZURE_SPEECH_REGION", "eastus")


def piper_settings() -> dict:
    return {
        "model_path": os.getenv("PIPER_MODEL_PATH"),
        "length_scale": os.getenv("PIPER_LENGTH_SCALE", "0.95"),
    }


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def health_check() -> dict:
    """Report which real-mode capabilities are actually usable right now.

    Cheap and side-effect-free: no model loading, no network calls, no
    huggingface downloads. Safe to call anytime - including in mock mode -
    as a pre-flight check before setting VOICE_BACKEND=real.
    """
    key, region = azure_credentials()
    model_override = piper_settings()["model_path"]

    return {
        "stt": {
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "whisper_importable": _importable("torch") and _importable("transformers"),
        },
        "tts": {
            "azure_key_set": bool(key),
            "azure_region": region,
            "piper_cli_available": shutil.which("piper") is not None,
            "mishkal_importable": _importable("mishkal"),
            "piper_model_path_override": model_override,
            "piper_model_override_exists": bool(
                model_override and Path(model_override).exists()
            ),
        },
    }