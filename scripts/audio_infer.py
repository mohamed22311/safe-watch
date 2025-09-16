import sys
from pathlib import Path
import asyncio
import librosa

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.audio import get_audio_model
from app.prompts.audio import build_messages


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/audio_infer.py <audio_path> [fewshot|default]")
        sys.exit(1)
    audio_path = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "fewshot"

    model = get_audio_model()
    sr = model.processor.feature_extractor.sampling_rate
    y, _ = librosa.load(audio_path, sr=sr)

    conversation = build_messages(variant)
    conversation.append({"role": "user", "content": [{"type": "audio", "audio_url": "local.wav"}]})
    text = model.generate(conversation=conversation, audio=y)
    print(text)


if __name__ == "__main__":
    asyncio.run(main())

