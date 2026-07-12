"""
/voice endpoints — speech-to-text and text-to-speech for the frontend.
"""
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from src.voice import (
    transcribe_audio,
    synthesize_speech,
    VoiceError,
    EmptyTranscriptError,
    AudioFormatError,
    AudioTooLongError,
    TextValidationError,
    TTSBackendError,
)
from src.utils.schemas import ChatMessage  # موجود أصلاً عندك لو حابة توسعي لاحقاً

router = APIRouter(prefix="/voice", tags=["voice"])

import re

def clean_for_speech(text: str) -> str:
    """ينظف النص من رموز الماركداون قبل تحويله لصوت، عشان القراءة تطلع طبيعية."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)      # **بولد**
    text = re.sub(r'\*(.*?)\*', r'\1', text)          # *مائل*
    text = re.sub(r'#{1,6}\s*', '', text)              # # عناوين
    text = re.sub(r'`(.*?)`', r'\1', text)             # `كود`
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)  # نقاط القوائم
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE) # 1. 2. 3.
    text = re.sub(r'\|', '، ', text)                   # فواصل الجداول
    text = re.sub(r'\n{2,}', '. ', text)                # فقرات متعددة → توقف طبيعي
    text = re.sub(r'\n', '. ', text)                   # سطر جديد → توقف
    text = re.sub(r'\s{2,}', ' ', text)                 # مسافات زايدة
    return text.strip()

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Receives an audio file (webm/wav/m4a...) from the frontend,
    transcribes it to Arabic text using Whisper.
    """
    try:
        audio_bytes = await audio.read()
        result = transcribe_audio(audio_bytes)
        return {"text": result.text}
    except EmptyTranscriptError:
        return {"text": "", "error": "لم أتمكن من سماع أي كلام، حاول مرة أخرى."}
    except AudioFormatError:
        raise HTTPException(status_code=400, detail="تعذّر قراءة الملف الصوتي.")
    except AudioTooLongError:
        raise HTTPException(status_code=400, detail="التسجيل طويل جداً.")
    except VoiceError as e:
        raise HTTPException(status_code=500, detail=f"خطأ في تحويل الصوت إلى نص: {e}")


@router.post("/synthesize")
async def synthesize(payload: dict):
    """
    Receives {"text": "..."} from the frontend, returns a .wav audio file
    of the text spoken aloud.
    """
    text = payload.get("text", "")
    try:
        result = synthesize_speech(clean_for_speech(text))
        return FileResponse(
            path=str(result.audio_path),
            media_type="audio/wav",
            filename="response.wav",
        )
    except TextValidationError:
        raise HTTPException(status_code=400, detail="النص فارغ أو طويل جداً.")
    except TTSBackendError as e:
        raise HTTPException(status_code=500, detail=f"تعذّر توليد الصوت: {e}")
    except VoiceError as e:
        raise HTTPException(status_code=500, detail=f"خطأ في تحويل النص إلى صوت: {e}")