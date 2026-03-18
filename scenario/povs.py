"""
Camera POV management.

POV is fixed BEFORE scenario generation. Only compatible event categories
are generated for each POV.
"""
import random


def load_povs(config: dict) -> dict[str, dict]:
    """Load POV definitions from config."""
    return config.get("camera_povs", {})


def select_pov_for_hook(
    povs: dict[str, dict],
    hook_template: dict,
    history: list[dict],
    category_candidates: list[str] | None = None,
) -> tuple[str, dict]:
    """
    Select a camera POV that is compatible with the hook and not recently used.

    Returns (pov_id, pov_config).
    """
    recent_povs = []
    for entry in history[-5:]:
        sc = entry.get("scenario", {})
        recent_povs.append(sc.get("camera_pov", ""))

    # All POVs
    candidates = list(povs.items())
    random.shuffle(candidates)

    # Prefer POVs not used recently
    fresh = [(pid, cfg) for pid, cfg in candidates if pid not in recent_povs]
    stale = [(pid, cfg) for pid, cfg in candidates if pid in recent_povs]
    ordered = fresh + stale

    # If category_candidates specified, filter by compatibility
    if category_candidates:
        compatible = [
            (pid, cfg) for pid, cfg in ordered
            if any(cat in cfg.get("compatible_categories", []) for cat in category_candidates)
        ]
        if compatible:
            return compatible[0]

    return ordered[0] if ordered else candidates[0]


def validate_pov_event_compatibility(
    pov_id: str,
    category: str,
    povs: dict[str, dict],
) -> bool:
    """Check that a POV is compatible with the event category."""
    pov_cfg = povs.get(pov_id, {})
    compatible = pov_cfg.get("compatible_categories", [])
    return category in compatible


def get_pov_traits(pov_id: str, povs: dict[str, dict]) -> str:
    """Get camera traits string for prompt generation."""
    pov_cfg = povs.get(pov_id, {})
    return pov_cfg.get("camera_traits", "handheld phone camera")
