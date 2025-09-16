import io
import cv2
import numpy as np
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..config import settings
from ..models.schemas import VisionResult
from ..utils.parsers import parse_vision_text
from ..utils.storage import get_storage
from ..prompts.vision import get_prompt, build_messages


class VisionModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Collect PIL images from messages for processor
        images = []
        for m in messages:
            for c in m.get("content", []) if isinstance(m.get("content"), list) else []:
                if isinstance(c, dict) and c.get("type") == "image":
                    images.append(c.get("image"))
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        out_texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return out_texts[0].strip()


_vision_singleton: Optional[VisionModel] = None


def get_vision_model() -> VisionModel:
    global _vision_singleton
    if _vision_singleton is None:
        _vision_singleton = VisionModel(settings.qwen_vl_model)
    return _vision_singleton


class VisionService:
    async def analyze_image_bytes(self, image_bytes: bytes, prompt_variant: str = "default") -> VisionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model = get_vision_model()
        messages = build_messages(prompt_variant, image)
        text = model.generate(messages=messages)
        parsed = parse_vision_text(text)
        return VisionResult(**parsed, raw_text=text)


class VisionSampler:
    def __init__(self, source: str, interval_seconds: int) -> None:
        self.source = source
        self.interval_seconds = interval_seconds
        self.storage = get_storage()

    async def run(self) -> None:
        cap = cv2.VideoCapture(0 if self.source == "0" else self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")
        try:
            while True:
                # Grab current frame for inference
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(self.interval_seconds)
                    continue

                # Encode as JPEG
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    await asyncio.sleep(self.interval_seconds)
                    continue
                result = await VisionService().analyze_image_bytes(buf.tobytes())

                # If violent, extract 60s clip centered on now
                if result.violence_detected:
                    clip_url = await self._clip_and_store(cap)
                    result.incident_clip_url = clip_url
                await asyncio.sleep(self.interval_seconds)
        finally:
            cap.release()

    async def _clip_and_store(self, cap: cv2.VideoCapture) -> Optional[str]:
        # Attempt to compute current timestamp position and clip +/- 30s around now
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps if current_frame else 0.0
        half = settings.incident_clip_seconds / 2
        start_time = max(0.0, current_time - half)
        duration = settings.incident_clip_seconds
        src = self.source
        # Build ffmpeg command
        with tempfile.TemporaryDirectory() as td:
            out_path = str(Path(td) / "incident.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.2f}",
                "-i", src,
                "-t", f"{duration}",
                "-c:v", "libx264", "-c:a", "aac",
                out_path,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(out_path, "rb") as f:
                    data = f.read()
                url = await self.storage.store_bytes(data, suffix=".mp4", subdir="incidents/video")
                return url
            except Exception:
                return None

