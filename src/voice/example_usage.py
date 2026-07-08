"""Runnable demo of the voice layer in mock mode.

    python example_usage.py

Shows exactly how Fatema calls STT and TTS, including the error handling.
Works with zero setup (mock mode). Set VOICE_BACKEND=real to exercise the real models.
"""
from voice import (
    transcribe_audio,
    synthesize_speech,
    set_mock_transcript,
    EmptyTranscriptError,
    AudioFormatError,
    TextValidationError,
    VoiceError,
)


def demo_happy_path():
    print("\n--- STT -> RAG -> TTS (the real loop) ---")

    # 1. Student audio comes in (here: fake bytes; in prod: a real upload).
    audio_bytes = b"...student audio from the UI..."
    transcript = transcribe_audio(audio_bytes)
    print(f"  heard: {transcript.text}   [{transcript.backend}]")

    # 2. >>> Fatema's RAG pipeline turns transcript.text into an answer <<<
    answer_text = (
        "يتطلب تخصص الذكاء الاصطناعي وعلم البيانات إتمام مئة وستة وعشرين ساعة معتمدة."
    )

    # 3. Speak the answer back.
    speech = synthesize_speech(answer_text)
    print(f"  spoke: {speech.audio_path.name}  ({speech.audio_duration_sec}s)  [{speech.backend}]")


def demo_error_handling():
    print("\n--- error contract ---")

    try:
        transcribe_audio(42)  # wrong type
    except AudioFormatError:
        print("  AudioFormatError raised for bad input  ✓")

    try:
        synthesize_speech("")  # empty
    except TextValidationError:
        print("  TextValidationError raised for empty text  ✓")

    try:
        synthesize_speech("x" * 9999)  # too long
    except VoiceError as e:  # base class catches everything
        print(f"  VoiceError base caught over-long text  ✓")


def demo_changing_mock():
    print("\n--- swapping the mock transcript while testing ---")
    set_mock_transcript("كم ساعة معتمدة يحتاجها التخصص؟")
    print("  now returns:", transcribe_audio(b"xx").text)


if __name__ == "__main__":
    demo_happy_path()
    demo_error_handling()
    demo_changing_mock()
    print("\nAll good. Set VOICE_BACKEND=real to run Whisper + Azure.\n")