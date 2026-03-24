from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.pipeline.audio.stt import transcribe_audio
from app.pipeline.audio.tts import generate_speech

router = APIRouter(tags=["audio"])

class TTSRequest(BaseModel):
    text: str
    voice: str = "hm_omega"  # Hindi male voice ('Rohan' equivalent)
    speed: float = 1.0

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        text = transcribe_audio(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")

@router.post("/speak")
async def speak(request: TTSRequest):
    try:
        wav_bytes = generate_speech(request.text, request.voice, request.speed)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")
