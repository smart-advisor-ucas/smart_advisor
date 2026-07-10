from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from src.voice import transcribe_audio, synthesize_speech, VoiceError, EmptyTranscriptError, AudioFormatError

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    يستقبل ملف صوتي ويرجع النص العربي.
    الـ Frontend يبعث الصوت كـ multipart/form-data
    """
    try:
        audio_bytes = await audio.read()
        result = transcribe_audio(audio_bytes)
        return {"text": result.text, "language": result.language}
    except EmptyTranscriptError:
        return {"text": "", "error": "لم يتم التقاط أي كلام، يرجى المحاولة مرة أخرى"}
    except AudioFormatError:
        return {"text": "", "error": "تعذّر قراءة الملف الصوتي"}
    except VoiceError as e:
        return {"text": "", "error": str(e)}


@router.post("/synthesize")
async def synthesize(data: dict):
    """
    يستقبل نص عربي ويرجع ملف صوتي .wav
    الـ Frontend يبعث {"text": "..."}
    """
    try:
        text = data.get("text", "")
        if not text:
            return {"error": "النص فارغ"}
        result = synthesize_speech(text)
        return FileResponse(
            str(result.audio_path),
            media_type="audio/wav",
            filename="response.wav"
        )
    except VoiceError as e:
        return {"error": str(e)}