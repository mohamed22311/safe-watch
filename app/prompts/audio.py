from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
from ..config import settings


def base_instruction() -> str:
    return (
        "You are an expert CCTV Incident Analyst. You will be given an automatic transcript of a surveillance audio clip (from Whisper). Use ONLY this transcript and reasonable world knowledge to judge for violent, threatening, or emergency events. Produce a strict JSON first, then a short human summary.\n\n"
        "RULES\n"
        "- Be conservative; do not invent acoustic-only events.\n"
        "- If the transcript suggests urgency or threat, report it even without explicit sounds.\n"
        "- If transcript is empty or vague, return empty lists and explain uncertainty in warnings.\n\n"
        "OUTPUT JSON SCHEMA (first): {\n"
        "  \"audio_metadata\": {\n"
        "    \"duration_s\": float,\n"
        "    \"sample_rate_hz\": int|null,\n"
        "    \"channels\": int|null,\n"
        "    \"detected_languages\": [{\"language\": string, \"confidence\": float}],\n"
        "    \"notes\": string|null\n"
        "  },\n"
        "  \"sounds_detected\": [],\n"
        "  \"speech_detected\": [],\n"
        "  \"emergencies\": [{\"kind\": string, \"severity\": number, \"rationale\": string, \"confidence\": number}],\n"
        "  \"summary\": string|null,\n"
        "  \"recommendations\": [string],\n"
        "  \"warnings\": [string]\n"
        "}\n\n"
        "Then provide a short human summary."
    )


def get_examples() -> List[Dict[str, Any]]:
    # Prefer JSON file defining example audio paths and outputs
    if settings.fewshot_audio_file:
        fp = Path(settings.fewshot_audio_file)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                examples: List[Dict[str, Any]] = []
                for item in data:
                    audio_path = item.get("audio")
                    if not audio_path:
                        continue
                    examples.append({
                        "audio": audio_path,
                        "output": item.get("output", "")
                    })
                return examples
            except Exception:
                pass
    # Fallback: Build examples from env-configured audio paths; default placeholders
    examples: List[Dict[str, Any]] = []
    paths: List[str] = []
    if settings.audio_fewshot_clips:
        paths = [p.strip() for p in settings.audio_fewshot_clips.split(",") if p.strip()]
    for p in paths:
        examples.append({
            "audio": p,
            "output": (
                '{"audio_metadata":{"duration_s":15.0,"sample_rate_hz":16000,"channels":1,"detected_languages":[],"notes":null},'
                '"sounds_detected":[],"speech_detected":[],"emergencies":[],'
                '"summary":"Placeholder example, replace with your expected output.","recommendations":[],"warnings":[]}'
                "\n\nNo incident in this placeholder example."
            ),
        })
    return examples


def build_messages_from_transcript(transcript: str, variant: str, metadata: Dict[str, Any] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": base_instruction()}]
    if variant != "default":
        # Include one or two few-shot hints if available
        for ex in get_examples()[:2]:
            messages.append({"role": "user", "content": [{"type": "text", "text": f"TRANSCRIPT:\n{ex.get('output','')}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ex.get("output", "")}]} )
    meta_text = "" if not metadata else f"\n\nMETADATA: {json.dumps(metadata)}"
    messages.append({"role": "user", "content": [{"type": "text", "text": f"TRANSCRIPT:\n{transcript}{meta_text}"}]})
    return messages


def get_prompt(variant: str = "default") -> str:
    return base_instruction()

