from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# Vision structured fields as per prompt
class VisionResult(BaseModel):
    violence_detected: bool
    severity_score: Optional[int] = None  # 0–100
    weapons_observed: List[str] | None = None
    fights_or_aggression: Optional[str] = None
    blood_or_injuries: Optional[str] = None
    scene_description: Optional[str] = None
    raw_text: str
    incident_clip_url: Optional[str] = None


# Audio structured models as per JSON schema
class DetectedLanguage(BaseModel):
    language: str
    confidence: float


class AudioMetadata(BaseModel):
    duration_s: float
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    detected_languages: List[DetectedLanguage] = []
    notes: Optional[str] = None


class SoundEvent(BaseModel):
    type: str
    start: str
    end: str
    confidence: Optional[float] = None


class Emergency(BaseModel):
    kind: str
    severity: float | int
    rationale: Optional[str] = None
    confidence: Optional[float] = None


class AudioAnalysis(BaseModel):
    audio_metadata: AudioMetadata
    sounds_detected: List[SoundEvent] = []
    speech_detected: List[Dict[str, Any]] = []
    emergencies: List[Emergency] = []
    summary: Optional[str] = None
    recommendations: List[str] = []
    warnings: List[str] = []


class AudioResult(BaseModel):
    analysis: AudioAnalysis
    post_json_summary: Optional[str] = None  # free-form text after the JSON block
    raw_text: str
    incident_clip_url: Optional[str] = None

