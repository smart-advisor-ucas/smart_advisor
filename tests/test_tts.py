"""Tests for synthesize_speech (text-to-speech)."""
import asyncio
import wave

import pytest

from voice import (
    synthesize_speech,
    synthesize_speech_async,
    SynthesisResult,
    TextValidationError,
    TTSError,
    VoiceError,
)


# --- happy path ------------------------------------------------------------- #
def test_returns_synthesis_result():
    text = "نص عربي للاختبار"
    r = synthesize_speech(text)
    assert isinstance(r, SynthesisResult)
    assert r.backend == "mock-tts"
    assert r.num_chars == len(text)
    assert r.sample_rate == 24000


def test_writes_a_valid_wav_file():
    r = synthesize_speech("اختبار الصوت")
    assert r.audio_path.exists()
    with wave.open(str(r.audio_path), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnframes() > 0          # the file actually has audio in it


def test_output_path_is_respected(tmp_path):
    out = tmp_path / "my_audio.wav"
    r = synthesize_speech("مرحبا", output_path=str(out))
    assert r.audio_path == out
    assert out.exists()


def test_voice_alias_resolves():
    # "female"/"male" are friendly aliases; passing one must not error.
    r = synthesize_speech("نص", voice="female")
    assert isinstance(r, SynthesisResult)


# --- error contract --------------------------------------------------------- #
def test_empty_text_raises():
    with pytest.raises(TextValidationError):
        synthesize_speech("   ")


def test_over_long_text_raises():
    with pytest.raises(TextValidationError):
        synthesize_speech("x" * 5000)


def test_exception_hierarchy():
    assert issubclass(TextValidationError, TTSError)
    assert issubclass(TTSError, VoiceError)


# --- async wrapper ---------------------------------------------------------- #
def test_async_wrapper_returns_same_type():
    r = asyncio.run(synthesize_speech_async("مرحبا"))
    assert isinstance(r, SynthesisResult)