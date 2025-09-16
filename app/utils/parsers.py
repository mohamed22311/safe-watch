import re
from typing import Dict, Any


def parse_vision_text(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out: Dict[str, Any] = {
        "violence_detected": False,
        "severity_score": None,
        "weapons_observed": None,
        "fights_or_aggression": None,
        "blood_or_injuries": None,
        "scene_description": None,
    }

    for line in lines:
        key, val = None, None
        if line.lower().startswith("- violence detected:"):
            val = line.split(":", 1)[1].strip()
            out["violence_detected"] = val.lower().startswith("y")
        elif line.lower().startswith("- violence severity score:"):
            val = line.split(":", 1)[1]
            m = re.search(r"(\d+)", val)
            if m:
                out["severity_score"] = int(m.group(1))
        elif line.lower().startswith("- weapons observed:"):
            val = line.split(":", 1)[1].strip()
            if val and val.lower() != "none":
                out["weapons_observed"] = [s.strip() for s in re.split(r",|/|;", val) if s.strip()]
            else:
                out["weapons_observed"] = []
        elif line.lower().startswith("- signs of fights or physical aggression:"):
            out["fights_or_aggression"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("- signs of blood or injuries:"):
            out["blood_or_injuries"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("- scene description:"):
            out["scene_description"] = line.split(":", 1)[1].strip()

    return out

