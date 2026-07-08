"""Tests for real-backend code paths (Whisper STT, Azure TTS) via mocking.

These tests exercise the real-backend wiring logic without needing a GPU or
Azure API key.  Heavy dependencies (torch, transformers, azure-sdk, ffmpeg)
are patched out so the suite stays fast and dependency-free.

What is tested:
  - _whisper_transcribe: happy path, empty-transcript error, too-long error,
    temp-WAV cleanup on both success and failure.
  - _coerce_to_clean_wav: FFmpeg invocation for bytes/path inputs, FFmpeg
    failure → AudioFormatError.
  - _azure_synth: missing-key guard, happy path, SDK failure → TTSBackendError.
"""
from __future__ import annotations

import array
import math
import os
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── shared helpers ───────────────────────────────────────────────────────── #

def _write_wav(path: Path, duration: float = 1.0, sr: int = 16_000) -> None:
    """Write a minimal valid WAV file for test fixtures."""
    n = int(sr * duration)
    samples = array.array(
        "h", (int(1000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n))
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def _make_azure_sys_modules(sdk: MagicMock) -> dict:
    """Build a sys.modules patch dict so that `import azure.cognitiveservices.speech`
    resolves to *sdk* via attribute traversal.

    Python 3 compiles `import a.b.c as x` as:
      IMPORT_NAME 'a.b.c'  →  returns sys.modules['a']   (top-level only)
      LOAD_ATTR b
      LOAD_ATTR c
      STORE_NAME x
    So the attribute chain on the top-level mock must be wired explicitly;
    just putting sdk in sys.modules["a.b.c"] isn't enough.
    """
    azure_mock = MagicMock()
    cog_mock = MagicMock()
    azure_mock.cognitiveservices = cog_mock
    cog_mock.speech = sdk
    return {
        "azure": azure_mock,
        "azure.cognitiveservices": cog_mock,
        "azure.cognitiveservices.speech": sdk,
    }


def _make_whisper_mocks(transcript: str = "مرحبا"):
    """Return (model_mock, processor_mock) that simulate a Whisper inference."""
    model = MagicMock()
    model.device = "cpu"
    model.dtype = None
    model.generate.return_value = MagicMock()

    proc = MagicMock()
    proc.return_value = MagicMock(
        input_features=MagicMock(to=MagicMock(return_value=MagicMock()))
    )
    proc.batch_decode.return_value = [transcript]
    return model, proc


# ─── STT: _whisper_transcribe ─────────────────────────────────────────────── #

@pytest.fixture
def wav16k(tmp_path) -> Path:
    p = tmp_path / "test.wav"
    _write_wav(p)
    return p


class TestWhisperTranscribe:
    def test_happy_path(self, tmp_path, wav16k):
        import voice.stt as stt

        fresh = tmp_path / "coerced.wav"
        _write_wav(fresh)
        model, proc = _make_whisper_mocks("مرحبا بالعالم")

        with patch.object(stt, "_load_whisper", return_value=(model, proc)), \
             patch.object(stt, "_coerce_to_clean_wav", return_value=fresh):
            r = stt._whisper_transcribe(wav16k, "ar", True)

        assert r.text == "مرحبا بالعالم"
        assert r.backend == "whisper-large-v3"
        assert r.language == "ar"
        assert r.audio_duration_sec > 0
        assert r.sample_rate == 16_000
        assert r.normalized_to_msa is True

    def test_model_load_failure_raises_stterror(self, tmp_path, wav16k):
        """Missing torch/transformers, a download failure, or any other
        _load_whisper() blowup must surface as STTError, not a raw exception."""
        import voice.stt as stt
        from voice import STTError

        fresh = tmp_path / "coerced.wav"
        _write_wav(fresh)

        with patch.object(stt, "_load_whisper", side_effect=ModuleNotFoundError("no torch")), \
             patch.object(stt, "_coerce_to_clean_wav", return_value=fresh):
            with pytest.raises(STTError, match="Whisper transcription failed"):
                stt._whisper_transcribe(wav16k, "ar", True)

        assert not fresh.exists()  # temp WAV still cleaned up on this failure path

    def test_empty_transcript_raises(self, tmp_path, wav16k):
        import voice.stt as stt
        from voice import EmptyTranscriptError

        fresh = tmp_path / "coerced.wav"
        _write_wav(fresh)
        model, proc = _make_whisper_mocks("")  # empty → EmptyTranscriptError

        with patch.object(stt, "_load_whisper", return_value=(model, proc)), \
             patch.object(stt, "_coerce_to_clean_wav", return_value=fresh):
            with pytest.raises(EmptyTranscriptError):
                stt._whisper_transcribe(wav16k, "ar", True)

    def test_too_long_raises(self, tmp_path):
        import voice.stt as stt
        from voice import AudioTooLongError

        long_wav = tmp_path / "long.wav"
        _write_wav(long_wav, duration=stt.MAX_AUDIO_SEC + 5)

        # _load_whisper should never be reached for over-long audio.
        with patch.object(stt, "_coerce_to_clean_wav", return_value=long_wav):
            with pytest.raises(AudioTooLongError):
                stt._whisper_transcribe(b"audio", "ar", True)

    def test_temp_wav_deleted_on_success(self, tmp_path):
        """The temp WAV returned by _coerce_to_clean_wav is removed after use."""
        import voice.stt as stt

        fresh = tmp_path / "tracked.wav"
        _write_wav(fresh)
        model, proc = _make_whisper_mocks("text")

        with patch.object(stt, "_load_whisper", return_value=(model, proc)), \
             patch.object(stt, "_coerce_to_clean_wav", return_value=fresh):
            stt._whisper_transcribe(b"audio", "ar", True)

        assert not fresh.exists()

    def test_temp_wav_deleted_on_error(self, tmp_path):
        """The temp WAV is still cleaned up when transcription raises."""
        import voice.stt as stt
        from voice import EmptyTranscriptError

        fresh = tmp_path / "tracked.wav"
        _write_wav(fresh)
        model, proc = _make_whisper_mocks("")  # empty → EmptyTranscriptError

        with patch.object(stt, "_load_whisper", return_value=(model, proc)), \
             patch.object(stt, "_coerce_to_clean_wav", return_value=fresh):
            with pytest.raises(EmptyTranscriptError):
                stt._whisper_transcribe(b"audio", "ar", True)

        assert not fresh.exists()

    def test_too_long_temp_wav_deleted(self, tmp_path):
        """Temp WAV is cleaned up even when AudioTooLongError is raised."""
        import voice.stt as stt
        from voice import AudioTooLongError

        long_wav = tmp_path / "long.wav"
        _write_wav(long_wav, duration=stt.MAX_AUDIO_SEC + 5)

        with patch.object(stt, "_coerce_to_clean_wav", return_value=long_wav):
            with pytest.raises(AudioTooLongError):
                stt._whisper_transcribe(b"audio", "ar", True)

        assert not long_wav.exists()


# ─── STT: _coerce_to_clean_wav ────────────────────────────────────────────── #

class TestCoerceToCleanWav:
    def test_bytes_calls_ffmpeg(self, tmp_path):
        """Bytes input causes FFmpeg to be invoked."""
        import voice.stt as stt

        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            _write_wav(Path(cmd[-1]))  # create a valid output so the path is returned
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            result = stt._coerce_to_clean_wav(b"fake-audio")

        result.unlink(missing_ok=True)
        assert calls and calls[0][0] == "ffmpeg"

    def test_ffmpeg_missing_raises_stterror(self):
        """A missing ffmpeg binary is an environment problem, not a bad
        input, so it's a plain STTError rather than AudioFormatError."""
        import voice.stt as stt
        from voice import STTError, AudioFormatError

        with patch("subprocess.run", side_effect=FileNotFoundError("no ffmpeg")):
            with pytest.raises(STTError, match="ffmpeg not found") as exc_info:
                stt._coerce_to_clean_wav(b"audio")

        assert not isinstance(exc_info.value, AudioFormatError)

    def test_ffmpeg_nonzero_exit_raises_audioformaterror(self):
        import voice.stt as stt
        from voice import AudioFormatError
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
        ):
            with pytest.raises(AudioFormatError):
                stt._coerce_to_clean_wav(b"audio")

    def test_path_input_produces_new_file(self, tmp_path, wav16k):
        """Path input is re-encoded to a fresh temp file; the original is kept."""
        import voice.stt as stt

        def fake_run(cmd, **kw):
            _write_wav(Path(cmd[-1]))
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            result = stt._coerce_to_clean_wav(wav16k)

        assert result != wav16k     # a fresh file, not the original
        assert wav16k.exists()      # original untouched
        result.unlink(missing_ok=True)

    def test_bytes_src_temp_cleaned_on_ffmpeg_failure(self):
        """The temp src file written from bytes is deleted even when FFmpeg fails."""
        import voice.stt as stt, tempfile
        from voice import AudioFormatError
        import subprocess

        created_temps: list[str] = []
        real_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(**kw):
            fd, path = real_mkstemp(**kw)
            created_temps.append(path)
            return fd, path

        with patch("tempfile.mkstemp", side_effect=tracking_mkstemp), \
             patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            with pytest.raises(AudioFormatError):
                stt._coerce_to_clean_wav(b"audio")

        # Every temp file created during the failed call must be gone.
        for p in created_temps:
            assert not Path(p).exists(), f"temp file leaked: {p}"


# ─── TTS: _azure_synth ───────────────────────────────────────────────────── #

class TestAzureSynth:
    def test_missing_sdk_raises_ttsbackenderror(self, tmp_path, monkeypatch):
        """If azure-cognitiveservices-speech isn't installed, the import
        failure must surface as TTSBackendError, not a raw ImportError."""
        import voice.tts as tts
        from voice import TTSBackendError

        monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
        # sys.modules[name] = None forces `import name` to raise ImportError,
        # regardless of whether the real package happens to be installed.
        with patch.dict("sys.modules", {
            "azure": None,
            "azure.cognitiveservices": None,
            "azure.cognitiveservices.speech": None,
        }):
            with pytest.raises(TTSBackendError, match="Azure TTS setup/call failed"):
                tts._azure_synth("نص", "ar-SA-ZariyahNeural", tmp_path / "out.wav")

    def test_missing_key_raises_without_sdk(self, tmp_path, monkeypatch):
        """Key check happens before SDK import — no Azure package needed."""
        import voice.tts as tts
        from voice import TTSBackendError

        monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
        with pytest.raises(TTSBackendError, match="AZURE_SPEECH_KEY"):
            tts._azure_synth("نص", "ar-SA-ZariyahNeural", tmp_path / "out.wav")

    def test_success(self, tmp_path, monkeypatch):
        import voice.tts as tts
        from voice import SynthesisResult

        monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")

        out_wav = tmp_path / "out.wav"
        _write_wav(out_wav)  # pre-write so _wav_duration can read it

        sdk = MagicMock()
        sdk.ResultReason.SynthesizingAudioCompleted = "done"
        synth_result = MagicMock(reason="done")
        (sdk.SpeechSynthesizer.return_value
             .speak_text_async.return_value
             .get.return_value) = synth_result

        with patch.dict("sys.modules", _make_azure_sys_modules(sdk)):
            result = tts._azure_synth("مرحبا", "ar-SA-ZariyahNeural", out_wav)

        assert isinstance(result, SynthesisResult)
        assert result.audio_path == out_wav
        assert result.backend == "azure:ar-SA-ZariyahNeural"
        assert result.num_chars == len("مرحبا")

    def test_sdk_failure_raises_tts_backend_error(self, tmp_path, monkeypatch):
        import voice.tts as tts
        from voice import TTSBackendError

        monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")

        sdk = MagicMock()
        sdk.ResultReason.SynthesizingAudioCompleted = "done"
        cancel = MagicMock(reason="AuthenticationFailure", error_details="bad key")
        synth_result = MagicMock(reason="canceled", cancellation_details=cancel)
        (sdk.SpeechSynthesizer.return_value
             .speak_text_async.return_value
             .get.return_value) = synth_result

        with patch.dict("sys.modules", _make_azure_sys_modules(sdk)):
            with pytest.raises(TTSBackendError, match="Azure synthesis failed"):
                tts._azure_synth("نص", "ar-SA-ZariyahNeural", tmp_path / "out.wav")

    def test_sdk_failure_with_bad_cancellation_details(self, tmp_path, monkeypatch):
        """If cancellation_details itself raises, we still get a clean TTSBackendError."""
        import voice.tts as tts
        from voice import TTSBackendError

        monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")

        sdk = MagicMock()
        sdk.ResultReason.SynthesizingAudioCompleted = "done"

        broken_result = MagicMock(reason="canceled")
        type(broken_result).cancellation_details = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("SDK internal error"))
        )
        (sdk.SpeechSynthesizer.return_value
             .speak_text_async.return_value
             .get.return_value) = broken_result

        with patch.dict("sys.modules", _make_azure_sys_modules(sdk)):
            with pytest.raises(TTSBackendError, match="Azure synthesis failed"):
                tts._azure_synth("نص", "ar-SA-ZariyahNeural", tmp_path / "out.wav")


# ─── TTS: Piper (_ensure_piper_model / _diacritize / _piper_synth) ────────── #

@pytest.fixture(autouse=True)
def _reset_piper_module_caches():
    """_ensure_piper_model/_diacritize memoize into module globals; reset
    around every test in this file so one test's cache can't leak into another."""
    import voice.tts as tts
    tts._piper_model_path = None
    tts._mishkal = None
    yield
    tts._piper_model_path = None
    tts._mishkal = None


class TestEnsurePiperModel:
    def test_uses_override_path_if_it_exists(self, tmp_path, monkeypatch):
        import voice.tts as tts

        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake-onnx")
        monkeypatch.setenv("PIPER_MODEL_PATH", str(model_file))

        assert tts._ensure_piper_model() == str(model_file)

    def test_downloads_when_no_override(self, tmp_path, monkeypatch):
        """_ensure_piper_model uses piper's own download_voice() - it writes
        the .onnx and .onnx.json as plain colocated files, which is what
        fixed the "Unable to find voice" / hf_hub_download snapshot mismatch."""
        import voice.tts as tts

        monkeypatch.delenv("PIPER_MODEL_PATH", raising=False)
        requested = []

        def fake_download_voice(voice_name, download_dir, **kw):
            requested.append(voice_name)
            (download_dir / f"{voice_name}.onnx").write_bytes(b"fake-onnx")
            (download_dir / f"{voice_name}.onnx.json").write_text("{}")

        with patch.object(tts, "PIPER_CACHE_DIR", tmp_path), \
             patch("piper.download_voices.download_voice", side_effect=fake_download_voice):
            result = tts._ensure_piper_model()

        assert result == str(tmp_path / f"{tts.PIPER_VOICE_NAME}.onnx")
        assert requested == [tts.PIPER_VOICE_NAME]

    def test_download_failure_raises_ttsbackenderror(self, tmp_path, monkeypatch):
        import voice.tts as tts
        from voice import TTSBackendError

        monkeypatch.delenv("PIPER_MODEL_PATH", raising=False)
        with patch.object(tts, "PIPER_CACHE_DIR", tmp_path), \
             patch("piper.download_voices.download_voice", side_effect=OSError("network down")):
            with pytest.raises(TTSBackendError, match="Piper voice unavailable"):
                tts._ensure_piper_model()

    def test_download_incomplete_raises_ttsbackenderror(self, tmp_path, monkeypatch):
        """download_voice() can return without error yet not produce both
        files - that must still surface as TTSBackendError, not a confusing
        FileNotFoundError from deep inside the piper CLI subprocess later."""
        import voice.tts as tts
        from voice import TTSBackendError

        monkeypatch.delenv("PIPER_MODEL_PATH", raising=False)

        def fake_download_voice(voice_name, download_dir, **kw):
            (download_dir / f"{voice_name}.onnx").write_bytes(b"fake-onnx")
            # .onnx.json deliberately not written

        with patch.object(tts, "PIPER_CACHE_DIR", tmp_path), \
             patch("piper.download_voices.download_voice", side_effect=fake_download_voice):
            with pytest.raises(TTSBackendError, match="Piper voice unavailable"):
                tts._ensure_piper_model()

    def test_result_is_cached_across_calls(self, tmp_path, monkeypatch):
        import voice.tts as tts

        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake-onnx")
        monkeypatch.setenv("PIPER_MODEL_PATH", str(model_file))

        first = tts._ensure_piper_model()
        monkeypatch.delenv("PIPER_MODEL_PATH", raising=False)
        second = tts._ensure_piper_model()  # override gone, but cache still hits
        assert first == second == str(model_file)


class TestDiacritize:
    def test_uses_mishkal_when_available(self):
        import voice.tts as tts

        fake_mishkal = MagicMock()
        fake_mishkal.tashkeel.return_value = "مُرَحَّبًا"
        with patch("mishkal.tashkeel.TashkeelClass", return_value=fake_mishkal):
            assert tts._diacritize("مرحبا") == "مُرَحَّبًا"

    def test_degrades_to_plain_text_when_mishkal_missing(self):
        """Import failure must not raise - the docstring promises graceful
        degradation to plain (non-diacritized) text."""
        import voice.tts as tts

        with patch.dict("sys.modules", {"mishkal": None, "mishkal.tashkeel": None}):
            assert tts._diacritize("مرحبا") == "مرحبا"

    def test_caches_the_tashkeel_instance(self):
        import voice.tts as tts

        fake_mishkal = MagicMock()
        fake_mishkal.tashkeel.side_effect = lambda t: t
        with patch("mishkal.tashkeel.TashkeelClass", return_value=fake_mishkal) as ctor:
            tts._diacritize("a")
            tts._diacritize("b")

        ctor.assert_called_once()


class TestPiperSynth:
    def test_success(self, tmp_path):
        import voice.tts as tts

        out = tmp_path / "out.wav"
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake")

        def fake_run(cmd, **kw):
            _write_wav(Path(cmd[cmd.index("-f") + 1]), sr=22_050)
            return MagicMock(returncode=0)

        with patch.object(tts, "_ensure_piper_model", return_value=str(model_file)), \
             patch.object(tts, "_diacritize", side_effect=lambda t: t), \
             patch("subprocess.run", side_effect=fake_run) as run_mock:
            result = tts._piper_synth("نص عربي", out)

        assert result.audio_path == out
        assert result.backend == "piper:ar_JO-kareem"
        assert result.num_chars == len("نص عربي")
        assert result.sample_rate == 22_050
        cmd = run_mock.call_args[0][0]
        assert cmd[0] == "piper"
        assert str(model_file) in cmd

    def test_records_fallback_reason_in_backend(self, tmp_path):
        import voice.tts as tts

        out = tmp_path / "out.wav"

        def fake_run(cmd, **kw):
            _write_wav(Path(cmd[cmd.index("-f") + 1]))
            return MagicMock(returncode=0)

        with patch.object(tts, "_ensure_piper_model", return_value="/fake/model.onnx"), \
             patch.object(tts, "_diacritize", side_effect=lambda t: t), \
             patch("subprocess.run", side_effect=fake_run):
            result = tts._piper_synth("نص", out, fallback_from="AZURE_SPEECH_KEY is not set")

        assert "fell back from azure" in result.backend

    def test_model_unavailable_raises_ttsbackenderror(self, tmp_path):
        import voice.tts as tts
        from voice import TTSBackendError

        with patch.object(
            tts, "_ensure_piper_model",
            side_effect=TTSBackendError("Piper voice unavailable: no network"),
        ):
            with pytest.raises(TTSBackendError, match="unavailable"):
                tts._piper_synth("نص", tmp_path / "out.wav")

    def test_piper_cli_missing_raises_ttsbackenderror(self, tmp_path):
        import voice.tts as tts
        from voice import TTSBackendError

        with patch.object(tts, "_ensure_piper_model", return_value="/fake/model.onnx"), \
             patch.object(tts, "_diacritize", side_effect=lambda t: t), \
             patch("subprocess.run", side_effect=FileNotFoundError("no piper")):
            with pytest.raises(TTSBackendError, match="Piper synthesis failed"):
                tts._piper_synth("نص", tmp_path / "out.wav")

    def test_piper_nonzero_exit_raises_ttsbackenderror(self, tmp_path):
        import voice.tts as tts
        from voice import TTSBackendError
        import subprocess as sp

        with patch.object(tts, "_ensure_piper_model", return_value="/fake/model.onnx"), \
             patch.object(tts, "_diacritize", side_effect=lambda t: t), \
             patch("subprocess.run", side_effect=sp.CalledProcessError(1, "piper")):
            with pytest.raises(TTSBackendError, match="Piper synthesis failed"):
                tts._piper_synth("نص", tmp_path / "out.wav")


# ─── TTS: synthesize_speech "auto" engine dispatch ────────────────────────── #

class TestAutoFallback:
    def test_azure_success_does_not_call_piper(self, tmp_path):
        import voice.tts as tts

        out = tmp_path / "out.wav"
        _write_wav(out)
        fake_result = tts.SynthesisResult(
            audio_path=out, backend="azure:ar-SA-ZariyahNeural",
            latency_sec=0.1, audio_duration_sec=1.0, sample_rate=24000, num_chars=3,
        )
        with patch.object(tts, "_azure_synth", return_value=fake_result) as azure_mock, \
             patch.object(tts, "_piper_synth") as piper_mock:
            result = tts.synthesize_speech("نص", backend="real", engine="auto")

        azure_mock.assert_called_once()
        piper_mock.assert_not_called()
        assert result.backend == "azure:ar-SA-ZariyahNeural"

    def test_azure_failure_falls_back_to_piper(self, tmp_path):
        import voice.tts as tts

        out = tmp_path / "out.wav"
        fallback_result = tts.SynthesisResult(
            audio_path=out, backend="piper:ar_JO-kareem (fell back from azure: boom)",
            latency_sec=0.1, audio_duration_sec=1.0, sample_rate=22050, num_chars=3,
        )
        with patch.object(tts, "_azure_synth", side_effect=RuntimeError("boom")), \
             patch.object(tts, "_piper_synth", return_value=fallback_result) as piper_mock:
            result = tts.synthesize_speech("نص", backend="real", engine="auto")

        piper_mock.assert_called_once()
        assert "fell back from azure" in result.backend

    def test_both_engines_fail_raises_ttsbackenderror(self, tmp_path):
        import voice.tts as tts
        from voice import TTSBackendError

        with patch.object(tts, "_azure_synth", side_effect=RuntimeError("azure down")), \
             patch.object(tts, "_piper_synth", side_effect=TTSBackendError("piper down too")):
            with pytest.raises(TTSBackendError, match="piper down too"):
                tts.synthesize_speech("نص", backend="real", engine="auto")

    def test_explicit_azure_engine_does_not_fall_back(self, tmp_path):
        """engine="azure" must propagate the failure as-is, never call Piper."""
        import voice.tts as tts

        with patch.object(tts, "_azure_synth", side_effect=RuntimeError("azure down")), \
             patch.object(tts, "_piper_synth") as piper_mock:
            with pytest.raises(RuntimeError):
                tts.synthesize_speech("نص", backend="real", engine="azure")

        piper_mock.assert_not_called()

    def test_explicit_piper_engine_never_calls_azure(self, tmp_path):
        import voice.tts as tts

        out = tmp_path / "out.wav"
        _write_wav(out)
        fake_result = tts.SynthesisResult(
            audio_path=out, backend="piper:ar_JO-kareem",
            latency_sec=0.1, audio_duration_sec=1.0, sample_rate=22050, num_chars=3,
        )
        with patch.object(tts, "_piper_synth", return_value=fake_result), \
             patch.object(tts, "_azure_synth") as azure_mock:
            tts.synthesize_speech("نص", backend="real", engine="piper")

        azure_mock.assert_not_called()


# ─── TTS: temp-file cleanup on failure ────────────────────────────────────── #

class TestTempFileCleanupOnFailure:
    def test_auto_generated_temp_file_removed_when_both_engines_fail(self):
        import voice.tts as tts
        from voice import TTSBackendError

        created_paths = []
        real_azure = tts._azure_synth

        def spy_and_fail(text, voice, out):
            created_paths.append(out)
            raise RuntimeError("azure down")

        with patch.object(tts, "_azure_synth", side_effect=spy_and_fail), \
             patch.object(tts, "_piper_synth", side_effect=TTSBackendError("piper down too")):
            with pytest.raises(TTSBackendError):
                tts.synthesize_speech("نص", backend="real", engine="auto")

        assert created_paths and not created_paths[0].exists()

    def test_explicit_engine_failure_also_cleans_up(self):
        import voice.tts as tts

        created_paths = []

        def spy_and_fail(text, out):
            created_paths.append(out)
            raise RuntimeError("boom")

        with patch.object(tts, "_piper_synth", side_effect=spy_and_fail):
            with pytest.raises(RuntimeError):
                tts.synthesize_speech("نص", backend="real", engine="piper")

        assert created_paths and not created_paths[0].exists()

    def test_caller_supplied_output_path_is_not_deleted_on_failure(self, tmp_path):
        """We only clean up files we created ourselves - a caller-supplied
        output_path is the caller's responsibility even if synthesis failed."""
        import voice.tts as tts

        out = tmp_path / "mine.wav"
        out.write_bytes(b"pre-existing-caller-data")

        with patch.object(tts, "_piper_synth", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                tts.synthesize_speech("نص", backend="real", engine="piper", output_path=str(out))

        assert out.exists()
        assert out.read_bytes() == b"pre-existing-caller-data"
