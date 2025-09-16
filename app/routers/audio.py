from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from ..models.schemas import AudioResult
from ..services.audio import AudioService


router = APIRouter()
service = AudioService()


@router.post("/infer", response_model=AudioResult)
async def infer_audio(
    clip: UploadFile = File(...),
    prompt_variant: str = Form("fewshot"),
) -> AudioResult:
    if not clip.content_type or not clip.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")
    audio_bytes = await clip.read()
    result = await service.analyze_audio_bytes(audio_bytes=audio_bytes, prompt_variant=prompt_variant)
    return result

