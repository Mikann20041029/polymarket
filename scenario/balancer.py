"""
Category balance adjustment.

Prevents over-representation of any single category by applying
score bonuses/penalties based on historical frequency.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = Path(__file__).parent.parent / "data" / "category_stats.json"


def load_category_stats(path: Path | None = None) -> dict:
    """Load category usage statistics."""
    p = path or DEFAULT_STATS_PATH
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_category_stats(stats: dict, path: Path | None = None) -> None:
    """Save category usage statistics."""
    p = path or DEFAULT_STATS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)


def compute_category_bonus(
    category: str,
    stats: dict,
    categories_config: dict,
) -> float:
    """
    Compute score adjustment for a category.

    Over-represented categories get negative bonus (penalty).
    Under-represented categories get positive bonus.

    Returns: float from -5.0 to +5.0
    """
    if not stats:
        return 2.0  # Slight bonus when no history (encourage all categories)

    total_count = sum(s.get("count", 0) for s in stats.values())
    if total_count == 0:
        return 2.0

    cat_stats = stats.get(category, {})
    cat_count = cat_stats.get("count", 0)
    actual_ratio = cat_count / total_count

    cat_config = categories_config.get(category, {})
    target_ratio = cat_config.get("target_ratio", 0.11)

    # How far off from target?
    deviation = actual_ratio - target_ratio

    # Convert to bonus: -5 to +5 range
    # deviation of +0.1 (10% over target) => -5 penalty
    # deviation of -0.1 (10% under target) => +5 bonus
    bonus = -deviation * 50.0
    return max(-5.0, min(5.0, bonus))


def adjust_scores(
    candidates: list[dict],
    stats: dict,
    categories_config: dict,
) -> list[dict]:
    """
    Adjust buzz scores based on category balance.

    Modifies candidates in-place, adding 'adjusted_score' field.
    """
    for c in candidates:
        category = c.get("category", "")
        bonus = compute_category_bonus(category, stats, categories_config)
        c["category_bonus"] = round(bonus, 2)
        c["adjusted_score"] = c.get("buzz_total", 0) + bonus

    return candidates


def update_stats_after_selection(
    selected_scenario: dict,
    stats: dict,
    categories_config: dict,
) -> dict:
    """Update category stats after a scenario is selected."""
    category = selected_scenario.get("category", "unknown")

    if category not in stats:
        cat_config = categories_config.get(category, {})
        stats[category] = {
            "count": 0,
            "last_used": None,
            "target_ratio": cat_config.get("target_ratio", 0.11),
        }

    stats[category]["count"] = stats[category].get("count", 0) + 1

    from datetime import datetime, timezone
    stats[category]["last_used"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return stats
