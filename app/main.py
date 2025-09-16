import asyncio
import logging
from fastapi import FastAPI
from .config import settings
from .routers.vision import router as vision_router
from .routers.audio import router as audio_router
from .services.vision import VisionSampler
from .services.audio import AudioSampler


logger = logging.getLogger(__name__)

app = FastAPI(title="SafeWatch API", version="0.1.0")
app.include_router(vision_router, prefix="/vision", tags=["vision"])
app.include_router(audio_router, prefix="/audio", tags=["audio"])


background_tasks: list[asyncio.Task] = []


@app.on_event("startup")
async def on_startup() -> None:
    if settings.auto_run:
        logger.info("Starting background samplers...")
        if settings.video_source:
            vision_sampler = VisionSampler(
                source=settings.video_source,
                interval_seconds=settings.vision_interval_seconds,
            )
            background_tasks.append(asyncio.create_task(vision_sampler.run()))

        if settings.audio_source:
            audio_sampler = AudioSampler(
                source=settings.audio_source,
                interval_seconds=settings.audio_interval_seconds,
                sample_window_seconds=settings.audio_sample_seconds,
            )
            background_tasks.append(asyncio.create_task(audio_sampler.run()))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

