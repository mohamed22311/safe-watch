from fastapi import APIRouter, UploadFile, File, Form
from fastapi import HTTPException
from typing import Optional
from ..models.schemas import VisionResult
from ..services.vision import VisionService


router = APIRouter()
service = VisionService()


@router.post("/infer", response_model=VisionResult)
async def infer_image(
    image: UploadFile = File(...),
    prompt_variant: str = Form("fewshot"),
) -> VisionResult:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    image_bytes = await image.read()
    result = await service.analyze_image_bytes(image_bytes=image_bytes, prompt_variant=prompt_variant)
    return result

