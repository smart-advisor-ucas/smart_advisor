"""Shared types for the voice layer: result objects and the exception hierarchy.

Fatema only ever needs to know two things from this file:
  1. The result objects she gets back (TranscriptionResult, SynthesisResult).
  2. The exceptions she should catch (everything inherits from VoiceError).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# Exceptions — the voice layer RAISES on failure, it never returns error text. #
# --------------------------------------------------------------------------- #
class VoiceError(Exception):
    """Base class for every error the voice layer can raise.

    Catch this if you just want 'did the voice layer fail?' without caring why.
    """


class STTError(VoiceError):
    """Any speech-to-text failure."""


class TTSError(VoiceError):
    """Any text-to-speech failure."""


class AudioFormatError(STTError):
    """The input audio could not be decoded / normalized to 16 kHz mono WAV."""


class AudioTooLongError(STTError):
    """The input audio is longer than the configured maximum duration."""


class EmptyTranscriptError(STTError):
    """No usable speech was found (silence, noise, or the model returned nothing)."""


class TextValidationError(TTSError):
    """The text to synthesize is empty or too long."""


class TTSBackendError(TTSError):
    """The TTS backend (Azure, etc.) failed to produce audio."""


# --------------------------------------------------------------------------- #
# Result objects — immutable, typed, with useful metadata alongside the data.  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TranscriptionResult:
    """What `transcribe_audio` returns on success."""
    text: str                              # the transcript (MSA-normalized)
    language: str                          # detected/forced language, e.g. "ar"
    backend: str                           # "whisper-large-v3" or "mock-stt"
    latency_sec: float                     # how long transcription took
    audio_duration_sec: float              # length of the input audio
    sample_rate: int                       # always 16000 after normalization
    no_speech_prob: float | None = None    # confidence the clip was silence (None if N/A)
    normalized_to_msa: bool = True


@dataclass(frozen=True)
class SynthesisResult:
    """What `synthesize_speech` returns on success."""
    audio_path: Path                       # the written .wav file
    backend: str                           # "azure:ar-SA-ZariyahNeural" or "mock-tts"
    latency_sec: float                     # how long synthesis took
    audio_duration_sec: float              # length of the produced audio
    sample_rate: int                       # 24000 for Azure / mock
    num_chars: int                         # characters synthesized (for Azure quota tracking)