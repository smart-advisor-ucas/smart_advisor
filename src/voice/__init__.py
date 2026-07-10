"""voice — the Smart Advisor voice layer.

Public API (this is the whole contract Fatema depends on):

    from voice import transcribe_audio, synthesize_speech
    from voice import VoiceError            # base error to catch

    transcript = transcribe_audio(audio).text
    audio_path = synthesize_speech(answer_text).audio_path

Async variants (for FastAPI):
    from voice import transcribe_audio_async, synthesize_speech_async

By default everything runs in MOCK mode — no models, no GPU, no API key — so you
can build and test against it immediately. Set the env var VOICE_BACKEND=real
(plus AZURE_SPEECH_KEY / AZURE_SPEECH_REGION) to use Whisper + Azure for real.
"""
from .types import (
    TranscriptionResult,
    SynthesisResult,
    VoiceError,
    STTError,
    TTSError,
    AudioFormatError,
    AudioTooLongError,
    EmptyTranscriptError,
    TextValidationError,
    TTSBackendError,
)
from .stt import transcribe_audio, transcribe_audio_async, set_mock_transcript
from .tts import synthesize_speech, synthesize_speech_async, AZURE_VOICES
from .config import health_check

__all__ = [
    "transcribe_audio",
    "transcribe_audio_async",
    "synthesize_speech",
    "synthesize_speech_async",
    "TranscriptionResult",
    "SynthesisResult",
    "AZURE_VOICES",
    "set_mock_transcript",
    "health_check",
    "VoiceError",
    "STTError",
    "TTSError",
    "AudioFormatError",
    "AudioTooLongError",
    "EmptyTranscriptError",
    "TextValidationError",
    "TTSBackendError",
]