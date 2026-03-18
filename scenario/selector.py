"""
Final scenario selection.

Picks the single best candidate after all filtering and scoring.
Saves to history.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path(__file__).parent.parent / "data" / "history.json"


def load_history(path: Path | None = None) -> list[dict]:
    """Load generation history."""
    p = path or DEFAULT_HISTORY_PATH
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def save_history(history: list[dict], path: Path | None = None) -> None:
    """Save generation history."""
    p = path or DEFAULT_HISTORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def select_best(candidates: list[dict]) -> dict | None:
    """Select the highest adjusted_score candidate."""
    if not candidates:
        logger.warning("No candidates remaining after filtering.")
        return None

    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("adjusted_score", 0),
        reverse=True,
    )

    winner = sorted_candidates[0]
    logger.info(
        "Selected: %s (adjusted_score=%.1f, buzz_total=%d, bonus=%.1f)",
        winner.get("scenario_summary", "")[:60],
        winner.get("adjusted_score", 0),
        winner.get("buzz_total", 0),
        winner.get("category_bonus", 0),
    )

    return winner


def record_selection(
    scenario: dict,
    history: list[dict],
    history_path: Path | None = None,
    max_history: int = 500,
) -> list[dict]:
    """Record the selected scenario to history."""
    now = datetime.now(timezone.utc)
    entry = {
        "id": f"vid_{now.strftime('%Y%m%d_%H%M%S')}",
        "created_at": now.isoformat(),
        "scenario": {
            "category": scenario.get("category", ""),
            "event_type": scenario.get("event_type", ""),
            "scenario_summary": scenario.get("scenario_summary", ""),
            "location_style": scenario.get("location_style", ""),
            "time_of_day": scenario.get("time_of_day", ""),
            "weather_atmosphere": scenario.get("weather_atmosphere", ""),
            "camera_pov": scenario.get("camera_pov", ""),
            "camera_movement": scenario.get("camera_movement", ""),
            "opening_hook_type": scenario.get("opening_hook_type", ""),
            "opening_hook_description": scenario.get("opening_hook_description", ""),
            "peak_moment": scenario.get("peak_moment", ""),
            "aftermath": scenario.get("aftermath", ""),
            "visual_tags": scenario.get("visual_tags", []),
            "tone_tags": scenario.get("tone_tags", []),
            "dominant_colors": scenario.get("dominant_colors", []),
            "sound_atmosphere": scenario.get("sound_atmosphere", ""),
        },
        "buzz_score": scenario.get("buzz_score", {}),
        "buzz_total": scenario.get("buzz_total", 0),
        "adjusted_score": scenario.get("adjusted_score", 0),
        "similarity_info": {
            "max_similarity": scenario.get("_max_similarity", 0),
            "most_similar_id": scenario.get("_most_similar_id", ""),
        },
        "generation_model": None,
        "generation_cost_usd": 0.0,
        "output_file": None,
        "status": "planned",
    }

    history.append(entry)

    # Trim to max history
    if len(history) > max_history:
        history = history[-max_history:]

    save_history(history, history_path)
    return history
