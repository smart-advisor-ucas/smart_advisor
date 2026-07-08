"""Tests for transcribe_audio (speech-to-text)."""
import asyncio

import pytest

from voice import (
    transcribe_audio,
    transcribe_audio_async,
    set_mock_transcript,
    TranscriptionResult,
    AudioFormatError,
    STTError,
    VoiceError,
)


# --- happy path ------------------------------------------------------------- #
def test_returns_transcription_result():
    r = transcribe_audio(b"some-audio-bytes")
    assert isinstance(r, TranscriptionResult)
    assert isinstance(r.text, str) and r.text          # non-empty transcript
    assert r.backend == "mock-stt"
    assert r.sample_rate == 16000
    assert r.language == "ar"


def test_accepts_a_file_path(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"pretend-audio")
    r = transcribe_audio(str(p))
    assert isinstance(r, TranscriptionResult)


def test_language_argument_is_passed_through():
    r = transcribe_audio(b"x", language="en")
    assert r.language == "en"


# --- error contract --------------------------------------------------------- #
def test_unknown_input_type_raises_audioformaterror():
    with pytest.raises(AudioFormatError):
        transcribe_audio(12345)


def test_missing_file_raises_audioformaterror():
    with pytest.raises(AudioFormatError):
        transcribe_audio("this/file/does/not/exist.wav")


def test_empty_bytes_raises_audioformaterror():
    with pytest.raises(AudioFormatError):
        transcribe_audio(b"")


def test_exception_hierarchy():
    assert issubclass(AudioFormatError, STTError)
    assert issubclass(STTError, VoiceError)


# --- mock controllability --------------------------------------------------- #
def test_set_mock_transcript_changes_output():
    set_mock_transcript("جملة اختبار")
    try:
        assert transcribe_audio(b"x").text == "جملة اختبار"
    finally:
        # restore the default so test order never matters
        set_mock_transcript(
            "ما هي المهارات التي سأكتسبها عند دراسة تخصص الذكاء الاصطناعي وعلم البيانات؟"
        )


# --- async wrapper ---------------------------------------------------------- #
def test_async_wrapper_returns_same_type():
    r = asyncio.run(transcribe_audio_async(b"x"))
    assert isinstance(r, TranscriptionResult)