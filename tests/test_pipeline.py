"""Tests for the public API and the STT -> RAG -> TTS loop."""
import pytest

import voice
from voice import transcribe_audio, synthesize_speech, VoiceError


def test_public_api_is_exported():
    expected = [
        "transcribe_audio",
        "transcribe_audio_async",
        "synthesize_speech",
        "synthesize_speech_async",
        "TranscriptionResult",
        "SynthesisResult",
        "VoiceError",
    ]
    for name in expected:
        assert hasattr(voice, name), f"voice.{name} is missing from the public API"


def test_full_loop_runs():
    # 1. speech -> text
    transcript = transcribe_audio(b"student-audio-bytes")
    assert transcript.text

    # 2. (Fatema's RAG would turn transcript.text into this answer)
    answer = "يتطلب التخصص إتمام 126 ساعة معتمدة للتخرج."

    # 3. text -> speech
    speech = synthesize_speech(answer)
    assert speech.audio_path.exists()


def test_any_failure_is_catchable_as_voiceerror():
    # A caller that only wants "did the voice layer fail?" can catch the base class.
    with pytest.raises(VoiceError):
        synthesize_speech("")        # empty text -> TextValidationError -> VoiceError


def test_health_check_is_side_effect_free_and_shaped_correctly():
    # Callable in mock mode with zero heavy deps installed - must not raise,
    # download, or load anything.
    report = voice.health_check()
    assert set(report.keys()) == {"stt", "tts"}
    assert isinstance(report["stt"]["ffmpeg_available"], bool)
    assert isinstance(report["stt"]["whisper_importable"], bool)
    assert isinstance(report["tts"]["azure_key_set"], bool)
    assert isinstance(report["tts"]["piper_cli_available"], bool)