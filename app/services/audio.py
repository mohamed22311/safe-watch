import io
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import librosa
import numpy as np
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from ..config import settings
from ..models.schemas import AudioResult, AudioAnalysis
from ..utils.storage import get_storage
from ..prompts.audio import get_prompt, build_messages


class AudioModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, conversation: list[dict], audio: np.ndarray) -> str:
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        generate_ids = self.model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


_audio_singleton: Optional[AudioModel] = None


def get_audio_model() -> AudioModel:
    global _audio_singleton
    if _audio_singleton is None:
        _audio_singleton = AudioModel(settings.qwen_audio_model)
    return _audio_singleton


class AudioService:
    async def analyze_audio_bytes(self, audio_bytes: bytes, prompt_variant: str = "default") -> AudioResult:
        # Load bytes into numpy at required sampling rate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=get_audio_model().processor.feature_extractor.sampling_rate)
        model = get_audio_model()
        conversation = build_messages(prompt_variant)
        # Append the actual audio message placeholder expected by the template
        conversation.append({"role": "user", "content": [{"type": "audio", "audio_url": "local.wav"}]})
        text = model.generate(conversation=conversation, audio=y)
        # Split JSON and summary best-effort
        json_part, summary_part = _split_json_and_summary(text)
        analysis = AudioAnalysis.model_validate(json_part) if isinstance(json_part, dict) and json_part else AudioAnalysis.model_validate({"audio_metadata": {"duration_s": 0.0}})
        return AudioResult(analysis=analysis, post_json_summary=summary_part, raw_text=text)


def _split_json_and_summary(text: str) -> tuple[Dict[str, Any], Optional[str]]:
    import json
    text = text.strip()
    # naive split: first { ... }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        json_str = text[first:last + 1]
        try:
            data = json.loads(json_str)
            summary = text[last + 1:].strip()
            return data, summary if summary else None
        except Exception:
            return {}, text
    return {}, text


class AudioSampler:
    def __init__(self, source: str, interval_seconds: int, sample_window_seconds: int) -> None:
        self.source = source
        self.interval_seconds = interval_seconds
        self.sample_window_seconds = sample_window_seconds
        self.storage = get_storage()

    async def run(self) -> None:
        # For simplicity, if source is a file, we re-sample segments over time
        while True:
            clip_url = None
            try:
                with tempfile.TemporaryDirectory() as td:
                    # Extract 15s window from the end or from random offset if file
                    in_path = self.source
                    out_path = str(Path(td) / "sample.wav")
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", in_path,
                        "-t", str(settings.audio_sample_seconds),
                        "-ac", "1", "-ar", "16000",
                        out_path,
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    with open(out_path, "rb") as f:
                        sample_bytes = f.read()
                result = await self.analyze_audio_bytes(sample_bytes)
                if _audio_indicates_incident(result):
                    clip_url = await self._clip_and_store()
                    result.incident_clip_url = clip_url
            except Exception:
                pass
            await asyncio.sleep(self.interval_seconds)

    async def _clip_and_store(self) -> Optional[str]:
        with tempfile.TemporaryDirectory() as td:
            out_path = str(Path(td) / "incident.wav")
            cmd = [
                "ffmpeg", "-y",
                "-i", self.source,
                "-t", str(settings.incident_clip_seconds),
                "-ac", "1", "-ar", "16000",
                out_path,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(out_path, "rb") as f:
                    data = f.read()
                url = await self.storage.store_bytes(data, suffix=".wav", subdir="incidents/audio")
                return url
            except Exception:
                return None


def _audio_indicates_incident(result: AudioResult) -> bool:
    # Minimal heuristic: severity >= 4 if present
    try:
        emergencies = result.json.get("emergencies", [])
        for item in emergencies:
            if float(item.get("severity", 0)) >= 4:
                return True
    except Exception:
        pass
    return False

