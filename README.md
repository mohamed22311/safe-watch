SafeWatch API

FastAPI service for periodic surveillance analysis using Qwen vision and audio models.

Endpoints
- POST `/vision/infer` form-data: `image` (file), `prompt_variant` (optional)
- POST `/audio/infer` form-data: `clip` (file), `prompt_variant` (optional)

Run locally
```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docker
```
docker build -t safewatch .
docker run --gpus all -p 8000:8000 --env AUTO_RUN=false safewatch
```

Auto run samplers
Configure `.env` or environment variables:
- AUTO_RUN=true
- VIDEO_SOURCE=0 or path to video
- AUDIO_SOURCE=path/to/audio.wav

Artifacts are stored under `storage/` or served via `EXTERNAL_BASE_URL` if provided.

