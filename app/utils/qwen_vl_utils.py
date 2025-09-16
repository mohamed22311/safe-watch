from typing import Any, List, Tuple
from PIL import Image


def process_vision_info(messages: List[dict]) -> Tuple[list[Image.Image], list[Any]]:
    images: list[Image.Image] = []
    videos: list[Any] = []
    for m in messages:
        content = m.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                img = item.get("image")
                if isinstance(img, Image.Image):
                    images.append(img)
    return images, videos

