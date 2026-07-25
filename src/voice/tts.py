"""Text-to-Speech. Public surface:  synthesize_speech  /  synthesize_speech_async.

    from voice import synthesize_speech
    result = synthesize_speech("╪د┘┘╪╡ ╪د┘╪╣╪▒╪ذ┘è ┘ç┘╪د")
    play(result.audio_path)                 # a .wav file on disk

Mock by default (a placeholder tone, no setup). In real mode the engine is chosen
by VOICE_TTS (or the `engine=` arg):
  - "azure": Azure Neural (needs key + network)
  - "piper": offline ظ¤ Mishkal tashkeel -> Piper (ar_JO-kareem)
  - "auto"  (default): try Azure, fall back to Piper if Azure fails.
"""
from __future__ import annotations
import array
import asyncio
import logging
import math
import os
import tempfile
import time
import wave
from pathlib import Path

from .types import SynthesisResult, TextValidationError, TTSBackendError
from .config import (
    resolve_backend,
    resolve_tts_engine,
    azure_credentials,
    piper_settings,
)

logger = logging.getLogger(__name__)

# --- Tunables --------------------------------------------------------------- #
MAX_CHARS = 3000  # Azure F0 free-tier per-request character cap
DEFAULT_VOICE = "ar-SA-HamedNeural"
SR = 24000

AZURE_VOICES = {
    "female": "ar-SA-ZariyahNeural",
    "male": "ar-SA-HamedNeural",
}

# Offline Piper voice we settled on.
PIPER_VOICE_NAME = "ar_JO-kareem-medium"  # <lang>_<region>-<name>-<quality>
PIPER_CACHE_DIR = Path.home() / ".cache" / "piper-voices"


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def synthesize_speech(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    output_path: str | Path | None = None,
    backend: str | None = None,
    engine: str | None = None,
) -> SynthesisResult:
    """Synthesize MSA Arabic text to a WAV file.

    Args:
        text: the text to speak (1..MAX_CHARS characters).
        voice: an Azure voice name or an AZURE_VOICES key ("female"/"male").
               Ignored by the Piper engine (it uses ar_JO-kareem).
        output_path: where to write the .wav (a temp file if omitted).
        backend: "mock" | "real" (defaults to VOICE_BACKEND env, then "mock").
        engine: "azure" | "piper" | "auto" (defaults to VOICE_TTS env, then "auto").
                Only applies in real mode.

    Returns:
        SynthesisResult: audio_path, backend (records which engine actually
        ran, e.g. "piper:ar_JO-kareem" or "azure:ar-SA-ZariyahNeural"),
        latency_sec, audio_duration_sec, sample_rate, num_chars.

    Raises:
        TextValidationError - empty or over-long text.
        TTSBackendError     - the chosen engine failed (in "auto", only if BOTH fail).
    """
    text = (text or "").strip()
    if not text:
        raise TextValidationError("empty text")
    if len(text) > MAX_CHARS:
        raise TextValidationError(f"{len(text)} chars exceeds {MAX_CHARS} limit")

    voice = AZURE_VOICES.get(voice, voice)
    auto_generated_path = output_path is None
    out = Path(output_path) if output_path else Path(
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        if resolve_backend(backend) == "mock":
            result = _mock_synth(text, voice, out)
        else:
            eng = resolve_tts_engine(engine)
            if eng == "piper":
                result = _piper_synth(text, out)
            elif eng == "azure":
                result = _azure_synth(text, voice, out)
            else:
                # "auto": try Azure, fall back to Piper on ANY Azure failure.
                try:
                    result = _azure_synth(text, voice, out)
                except Exception as e:
                    logger.warning("Azure TTS failed (%s); falling back to Piper", e)
                    result = _piper_synth(text, out, fallback_from=str(e)[:80])
    except Exception:
        # Only remove a file we created ourselves - a caller-supplied
        # output_path is theirs to manage, even if synthesis failed into it.
        if auto_generated_path:
            out.unlink(missing_ok=True)
        raise

    logger.info(
        "TTS engine used: %s (%.3fs, %d chars)",
        result.backend, result.latency_sec, result.num_chars,
    )
    return result


async def synthesize_speech_async(text: str, **kw) -> SynthesisResult:
    """Async wrapper. One thread-offload covers every engine and keeps one signature."""
    return await asyncio.to_thread(synthesize_speech, text, **kw)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        return sr, w.getnframes() / float(sr)


# --------------------------------------------------------------------------- #
# Mock backend                                                                 #
# --------------------------------------------------------------------------- #
def _mock_synth(text, voice, out: Path) -> SynthesisResult:
    t0 = time.perf_counter()
    duration = min(max(len(text) / 15.0, 0.5), 20.0)
    n = int(SR * duration)
    samples = array.array(
        "h", (int(3000 * math.sin(2 * math.pi * 220 * i / SR)) for i in range(n))
    )
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(samples.tobytes())
    return SynthesisResult(
        audio_path=out, backend="mock-tts",
        latency_sec=round(time.perf_counter() - t0, 4),
        audio_duration_sec=round(duration, 2), sample_rate=SR, num_chars=len(text),
    )


# --------------------------------------------------------------------------- #
# Real backend 1 ظ¤ Azure Neural (cloud)                                        #
# --------------------------------------------------------------------------- #
def _azure_synth(text, voice, out: Path) -> SynthesisResult:
    key, region = azure_credentials()
    if not key:
        raise TTSBackendError("AZURE_SPEECH_KEY is not set")

    t0 = time.perf_counter()
    try:
        import azure.cognitiveservices.speech as speechsdk

        cfg = speechsdk.SpeechConfig(subscription=key, region=region)
        cfg.speech_synthesis_voice_name = voice
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        audio_cfg = speechsdk.audio.AudioOutputConfig(filename=str(out))
        synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)
        result = synth.speak_text_async(text).get()
    except Exception as e:
        # Covers a missing azure-cognitiveservices-speech install as well as
        # any SDK setup/network failure - all must surface as TTSBackendError
        # so Fatema's `except VoiceError` catches them.
        raise TTSBackendError(f"Azure TTS setup/call failed: {e}") from e

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        try:
            d = result.cancellation_details
            reason, details = str(d.reason), str(d.error_details)
        except Exception:
            reason, details = "?", "no details"
        raise TTSBackendError(f"Azure synthesis failed: {reason} / {details}")

    sr, dur = _wav_info(out)
    return SynthesisResult(
        audio_path=out, backend=f"azure:{voice}",
        latency_sec=round(time.perf_counter() - t0, 3),
        audio_duration_sec=round(dur, 2), sample_rate=sr, num_chars=len(text),
    )


# --------------------------------------------------------------------------- #
# Real backend 2 ظ¤ Piper (offline):  Mishkal tashkeel -> ar_JO-kareem          #
# --------------------------------------------------------------------------- #
_mishkal = None
_piper_model_path = None


def _diacritize(text: str) -> str:
    """Add tashkeel (Mishkal). espeak-ng needs diacritics for correct Arabic vowels.
    Degrades to plain text if Mishkal isn't installed, rather than failing."""
    global _mishkal
    try:
        if _mishkal is None:
            from mishkal.tashkeel import TashkeelClass
            _mishkal = TashkeelClass()
        return _mishkal.tashkeel(text)
    except Exception:
        return text


def _ensure_piper_model() -> str:
    """Return a path to the ar_JO-kareem .onnx, downloading it once if needed.

    Uses Piper's own `piper.download_voices.download_voice()` rather than
    hf_hub_download. hf_hub_download revalidates each file against the HF
    Hub API on every call (even when already cached) and writes into its
    blob/symlink cache layout; two separate calls (one for .onnx, one for
    .onnx.json) could intermittently resolve to different snapshot revisions,
    which is what caused the CLI's "Unable to find voice" errors - piper's
    downloader instead writes both files as plain, literal, colocated files
    in one local directory, which is exactly the layout PiperVoice.load()
    expects (config defaults to f"{model_path}.json").
    """
    global _piper_model_path
    if _piper_model_path:
        return _piper_model_path

    override = piper_settings()["model_path"]
    if override and Path(override).exists():
        _piper_model_path = override
        return _piper_model_path

    try:
        from piper.download_voices import download_voice

        PIPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        download_voice(PIPER_VOICE_NAME, PIPER_CACHE_DIR)
        onnx = PIPER_CACHE_DIR / f"{PIPER_VOICE_NAME}.onnx"
        config = PIPER_CACHE_DIR / f"{PIPER_VOICE_NAME}.onnx.json"
        if not (onnx.exists() and config.exists()):
            raise FileNotFoundError(
                f"voice download did not produce both files: {onnx}, {config}"
            )
    except Exception as e:
        raise TTSBackendError(f"Piper voice unavailable: {e}") from e

    _piper_model_path = str(onnx)
    return _piper_model_path


def _piper_synth(text, out: Path, fallback_from: str | None = None) -> SynthesisResult:
    import subprocess

    t0 = time.perf_counter()
    model = _ensure_piper_model()
    spoken = _diacritize(text)
    length_scale = piper_settings()["length_scale"]
    try:
        subprocess.run(
            ["piper", "-m", model, "-f", str(out), "--length-scale", length_scale],
            input=spoken.encode("utf-8"), check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # capture_output means CalledProcessError carries the real stderr -
        # surface it instead of the useless default "exit status 1" message.
        stderr = getattr(e, "stderr", None)
        detail = stderr.decode("utf-8", errors="replace") if stderr else str(e)
        raise TTSBackendError(f"Piper synthesis failed: {detail}") from e

    try:
        sr, dur = _wav_info(out)
    except Exception as e:
        raise TTSBackendError(f"Piper produced an unreadable WAV file: {e}") from e

    backend = "piper:ar_JO-kareem"
    if fallback_from:
        backend += f" (fell back from azure: {fallback_from})"
    return SynthesisResult(
        audio_path=out, backend=backend,
        latency_sec=round(time.perf_counter() - t0, 3),
        audio_duration_sec=round(dur, 2), sample_rate=sr, num_chars=len(text),
    )
