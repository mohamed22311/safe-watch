import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.audio import get_transcriber
from app.prompts.audio import build_messages_from_transcript
from app.services.vision import get_vision_model


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/audio_infer.py <audio_path> [fewshot|default]")
        sys.exit(1)
    audio_path = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "fewshot"

    transcriber = get_transcriber()
    wres = transcriber.transcribe_path(audio_path)
    transcript = wres.get("text", "").strip()
    messages = build_messages_from_transcript(transcript=transcript, variant=variant)
    vlm = get_vision_model()
    text = vlm.generate(messages=messages, max_new_tokens=512)
    print(text)


if __name__ == "__main__":
    asyncio.run(main())

