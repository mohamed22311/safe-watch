import io
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import librosa
import numpy as np
import torch
import soundfile as sf
import whisper  # openai-whisper

from ..config import settings
from ..models.schemas import AudioResult, AudioAnalysis, AudioMetadata, DetectedLanguage
from ..utils.storage import get_storage
from ..prompts.audio import build_messages_from_transcript
from .vision import get_vision_model


class WhisperTranscriber:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Device selection handled internally by whisper; torch.cuda.is_available influences speed
        self.model = whisper.load_model(model_name)

    def transcribe_path(self, audio_path: str) -> Dict[str, Any]:
        # Use high-level API that leverages ffmpeg to decode
        result: Dict[str, Any] = self.model.transcribe(audio_path)
        return result


_whisper_singleton: Optional[WhisperTranscriber] = None


def get_transcriber() -> WhisperTranscriber:
    global _whisper_singleton
    if _whisper_singleton is None:
        _whisper_singleton = WhisperTranscriber(settings.whisper_model)
    return _whisper_singleton


class AudioService:
    async def analyze_audio_bytes(self, audio_bytes: bytes, prompt_variant: str = "default") -> AudioResult:
        # Persist to temp file for whisper and metadata tools
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            # Metadata via soundfile
            try:
                info = sf.info(tmp.name)
                duration_s = float(info.duration) if hasattr(info, "duration") and info.duration else (info.frames / info.samplerate if info.samplerate else 0.0)
                sample_rate_hz = int(info.samplerate) if info.samplerate else None
                channels = int(info.channels) if info.channels else None
            except Exception:
                # Fallback via librosa
                try:
                    y, sr = librosa.load(tmp.name, sr=None, mono=False)
                    duration_s = float(len(y) / sr) if isinstance(y, np.ndarray) and y.size > 0 and sr else 0.0
                    sample_rate_hz = int(sr) if sr else None
                    channels = int(1 if y.ndim == 1 else y.shape[0]) if isinstance(y, np.ndarray) else None
                except Exception:
                    duration_s = 0.0
                    sample_rate_hz = None
                    channels = None

            transcriber = get_transcriber()
            wres = transcriber.transcribe_path(tmp.name)
            transcript_text: str = wres.get("text", "").strip()
            language: Optional[str] = wres.get("language")

        detected_languages: list[DetectedLanguage] = []
        if language:
            try:
                detected_languages.append(DetectedLanguage(language=language, confidence=1.0))
            except Exception:
                pass

        metadata = AudioMetadata(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            detected_languages=detected_languages,
            notes=None,
        )

        # Build VLM messages from transcript and metadata
        messages = build_messages_from_transcript(
            transcript=transcript_text or "",
            variant=prompt_variant,
            metadata={
                "duration_s": metadata.duration_s,
                "sample_rate_hz": metadata.sample_rate_hz,
                "channels": metadata.channels,
                "language": language,
            },
        )

        vlm = get_vision_model()
        vlm_text = vlm.generate(messages=messages, max_new_tokens=512)

        # Split JSON and summary best-effort
        json_part, summary_part = _split_json_and_summary(vlm_text)
        # If JSON lacks audio_metadata, inject ours
        if isinstance(json_part, dict) and "audio_metadata" not in json_part:
            json_part["audio_metadata"] = {
                "duration_s": metadata.duration_s,
                "sample_rate_hz": metadata.sample_rate_hz,
                "channels": metadata.channels,
                "detected_languages": [dl.model_dump() for dl in metadata.detected_languages],
                "notes": metadata.notes,
            }
        analysis = AudioAnalysis.model_validate(json_part) if isinstance(json_part, dict) and json_part else AudioAnalysis.model_validate({"audio_metadata": metadata.model_dump()})
        return AudioResult(analysis=analysis, post_json_summary=summary_part, raw_text=vlm_text)


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
        for item in result.analysis.emergencies:
            if float(item.severity) >= 4:
                return True
    except Exception:
        pass
    return False

