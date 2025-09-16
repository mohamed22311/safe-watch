from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from ..config import settings


def base_instruction() -> str:
    return (
        "You are an AI security monitoring assistant responsible for automatic violence detection "
        "and detailed scene reporting for on-site security guards. "
        "Your mission is to help human operators rapidly assess potential threats captured in surveillance images. "
        "Carefully examine the provided image and strictly follow these instructions:\n\n"
        "ROLE & BEHAVIOR:\n"
        "- Act as a professional safety and security analyst.\n"
        "- Remain objective, neutral, and factual—avoid speculation beyond what is visually evident.\n"
        "- Focus on detecting any signs of violence, weapons, fights, or blood.\n"
        "- Provide concise, structured, and actionable information suitable for security incident reports.\n\n"
        "ANALYSIS STEPS:\n"
        "1. Observe the entire image for people, objects, and ongoing activities.\n"
        "2. Identify any indicators of violence such as physical aggression, fighting, or assault.\n"
        "3. Detect the presence of weapons (e.g., knives, guns, blunt objects) if visible.\n"
        "4. Look for signs of injuries, visible blood, or damage related to violent acts.\n"
        "5. If no violence is detected, clearly state this but still briefly describe the general scene.\n\n"
        "REQUIRED OUTPUT FORMAT:\n"
        "Provide your findings in the following exact structure (use clear, concise sentences):\n"
        "- Violence Detected: Yes/No\n"
        "- Violence Severity Score: [0–100] (0 = no violence, 100 = extreme violence)\n"
        "- Weapons Observed: [List of weapons or 'None']\n"
        "- Signs of Fights or Physical Aggression: [Describe or 'None']\n"
        "- Signs of Blood or Injuries: [Describe or 'None']\n"
        "- Scene Description: [Brief factual summary of the overall scene]\n\n"
        "IMPORTANT:\n"
        "- Only describe what is clearly visible.\n"
        "- Do not include personal opinions or guesses.\n"
        "- Follow the output structure exactly so it can be automatically parsed."
    )


def get_examples() -> List[Dict[str, Any]]:
    # Prefer JSON file defining exact pairs
    if settings.fewshot_vision_file:
        fp = Path(settings.fewshot_vision_file)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                examples: List[Dict[str, Any]] = []
                for item in data:
                    examples.append({
                        "image": item.get("image"),
                        "prompt": item.get("prompt", base_instruction()),
                        "output": item.get("output", "")
                    })
                return examples
            except Exception:
                pass
    # Fallback: Use env-configured images if provided, else no examples
    examples: List[Dict[str, Any]] = []
    paths = []
    if settings.vision_fewshot_images:
        paths = [p.strip() for p in settings.vision_fewshot_images.split(",") if p.strip()]
    for p in paths:
        img_path = Path(p)
        if not img_path.exists():
            continue
        # Create a placeholder target output. User should replace as needed.
        examples.append({
            "image": str(img_path),
            "prompt": base_instruction(),
            "output": (
                "- Violence Detected: No\n"
                "- Violence Severity Score: 0\n"
                "- Weapons Observed: None\n"
                "- Signs of Fights or Physical Aggression: None\n"
                "- Signs of Blood or Injuries: None\n"
                "- Scene Description: Placeholder example. Replace with your expected output."
            ),
        })
    return examples


def build_messages(variant: str, image_obj) -> list[dict]:
    messages: list[dict] = []
    # Default to few-shot unless explicitly "default"
    if variant != "default":
        # Add text-only few-shot if no images available; add image shots when present
        for ex in get_examples():
            # Your exact style: each example defines its own "prompt" and "output"
            messages.append({
                "role": "user",
                "content": [
                    *([{ "type": "image", "image": ex["image"] }] if ex.get("image") else []),
                    {"type": "text", "text": ex.get("prompt", base_instruction())}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex.get("output", "")}]
            })

    # Query (your style: image + the text prompt)
    user_content = [
        {"type": "image", "image": image_obj},
        {"type": "text", "text": base_instruction()}
    ]
    messages.append({"role": "user", "content": user_content})
    return messages


def get_prompt(variant: str = "default") -> str:
    # Kept for backward compatibility when only text is needed
    return base_instruction()

