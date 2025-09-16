import sys
from pathlib import Path
from PIL import Image
import asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.vision import get_vision_model
from app.prompts.vision import build_messages


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/vision_infer.py <image_path> [fewshot|default]")
        sys.exit(1)
    image_path = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "fewshot"
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(variant, image)
    model = get_vision_model()
    text = model.generate(messages=messages)
    print(text)


if __name__ == "__main__":
    asyncio.run(main())

