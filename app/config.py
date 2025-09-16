from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Models
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_audio_model: str = "Qwen/Qwen2-Audio-7B-Instruct"

    # Sources (use absolute/relative file paths or device indices like 0 for webcam)
    video_source: Optional[str] = None  # e.g., "./trash/Sample1.mp4" or "0" for webcam
    audio_source: Optional[str] = None  # e.g., "./angry_01.wav"

    # Few-shot example paths (comma-separated)
    vision_fewshot_images: Optional[str] = None  # e.g., "./img1.jpg,./img2.jpg"
    audio_fewshot_clips: Optional[str] = None    # e.g., "./ex1.wav,./ex2.wav"

    # Few-shot example definition files (JSON)
    fewshot_vision_file: Optional[str] = None     # JSON list: {"image": path, "output": text}
    fewshot_audio_file: Optional[str] = None      # JSON list: {"output": text}

    # Scheduling
    vision_interval_seconds: int = 10
    audio_interval_seconds: int = 60
    audio_sample_seconds: int = 15

    # Clipping
    incident_clip_seconds: int = 60

    # Storage
    storage_backend: str = "local"  # local | s3
    storage_base_dir: str = "./storage"
    external_base_url: Optional[str] = None  # if behind a CDN or static server

    # Feature flags
    auto_run: bool = False  # start background samplers on startup

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

