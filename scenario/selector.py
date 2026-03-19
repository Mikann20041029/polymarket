"""
Final scenario selection for construction timelapse system.

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
        "Selected: %s (adjusted=%.1f, total=%d, bonus=%.1f)",
        winner.get("one_line_concept", "")[:60],
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
        "id": f"build_{now.strftime('%Y%m%d_%H%M%S')}",
        "created_at": now.isoformat(),
        "scenario": {
            "one_line_concept": scenario.get("one_line_concept", ""),
            "category": scenario.get("category", ""),
            "construction_type": scenario.get("construction_type", ""),
            "before_space": scenario.get("before_space", {}),
            "construction_process": scenario.get("construction_process", {}),
            "after_space": scenario.get("after_space", {}),
            "reveal_type": scenario.get("reveal_type", ""),
            "time_structure": scenario.get("time_structure", {}),
            "camera_style": scenario.get("camera_style", ""),
            "location_feel": scenario.get("location_feel", ""),
            "similarity_tags": scenario.get("similarity_tags", []),
            "_concept_template": scenario.get("_concept_template", ""),
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

    if len(history) > max_history:
        history = history[-max_history:]

    save_history(history, history_path)
    return history
