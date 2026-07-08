# Voice Layer — Integration Contract 

This is everything you need to call the voice layer from the RAG pipeline. It works
**today, in mock mode** — no GPU, no Whisper download, no Azure key. Build against it
now; Saja swaps the real Whisper/Azure models in behind the exact same functions later,
and **your code won't change**.

## Install / location

Drop the `voice/` folder into `src/` so you can:

```python
from voice import transcribe_audio, synthesize_speech
```

## The two functions

### Speech → text

```python
result = transcribe_audio(audio)        # audio: file path, bytes, or numpy waveform
result.text                             # -> "ما هي المهارات ..."  (MSA, ready for retrieval)
```

`transcribe_audio(audio, *, language="ar", normalize_to_msa=True) -> TranscriptionResult`

`TranscriptionResult` fields: `text`, `language`, `backend`, `latency_sec`,
`audio_duration_sec`, `sample_rate`, `no_speech_prob`, `normalized_to_msa`.

### Text → speech

```python
result = synthesize_speech(answer_text)  # answer_text: the RAG answer (MSA)
result.audio_path                        # -> Path to a .wav you can stream/serve
```

`synthesize_speech(text, *, voice="ar-SA-ZariyahNeural", output_path=None, backend=None, engine=None) -> SynthesisResult`

`SynthesisResult` fields: `audio_path`, `backend`, `latency_sec`, `audio_duration_sec`,
`sample_rate`, `num_chars`.

### Async (if your pipeline is async / FastAPI)

```python
from voice import transcribe_audio_async, synthesize_speech_async
result = await transcribe_audio_async(audio)
result = await synthesize_speech_async(answer_text)
```

## Error handling — we RAISE, we never return error text

Every failure is an exception under `VoiceError`. Wrap calls like this:

```python
from voice import (transcribe_audio, EmptyTranscriptError,
                   AudioFormatError, VoiceError)

try:
    text = transcribe_audio(audio).text
except EmptyTranscriptError:
    # silence / no speech — show "I didn't catch that, please try again"
    ...
except AudioFormatError:
    # bad/undecodable upload — show "couldn't read that audio file"
    ...
except VoiceError:
    # any other voice failure — generic fallback
    ...
```

Exception types: `AudioFormatError`, `AudioTooLongError`, `EmptyTranscriptError`
(STT side); `TextValidationError`, `TTSBackendError` (TTS side). All inherit from
`VoiceError`, so `except VoiceError` catches everything if you don't want to be specific.

## Mock vs real

| | Mock (default) | Real |
|---|---|---|
| How to enable | nothing — it's the default | `export VOICE_BACKEND=real` |
| STT returns | a fixed MSA transcript | Whisper Large-v3 output |
| TTS returns | a short placeholder tone (valid .wav) | Azure Neural *or* offline Piper speech |
| Needs GPU / key | no | GPU for STT; TTS depends on `VOICE_TTS` (see below) |

Real-mode TTS has three engines, chosen by `VOICE_TTS` (or the `engine=` arg):
`azure` (cloud, needs `AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION`), `piper` (fully offline,
no key/network, needs `ffmpeg`/`piper` on PATH), or `auto` (default — tries Azure, falls
back to Piper on any failure). Either way you still just catch `TTSBackendError`/`VoiceError`.

Not sure your machine has what real mode needs? `voice.health_check()` reports (with no
model loading or network calls) whether ffmpeg/torch/piper/mishkal are importable and
whether Azure/Piper env vars are set — a quick way to check before flipping the switch.

You can change the mock transcript while testing:

```python
from voice import set_mock_transcript
set_mock_transcript("كم ساعة معتمدة يحتاجها التخصص؟")
```

## Things we agreed (please hold these)

1. **Audio format into `transcribe_audio`:** we settle on one of *bytes* or *file path*
   from the UI (Shahd) → you → voice layer. Whatever the UI emits (WebM/Opus, M4A, WAV),
   the voice layer re-encodes it internally, so you don't have to convert — just pass it
   through. Let's confirm the single type you'll pass.
2. **Errors are raised, not returned** — wrap calls in try/except as above.
3. **The interface is frozen.** If a signature ever needs to change, it's a breaking
   change: Saja announces it, bumps it, and updates this doc. Treat it like a published API.
4. **Bugs** go in a GitHub issue against `voice/` with the failing input attached
   (the clip or the exact text) + expected vs actual — not Slack. Each becomes a test.