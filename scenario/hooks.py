"""
Hook template management.

Hooks are the FIRST thing decided before any scenario is generated.
A hook defines the structure of the opening 1-second that stops viewers.
"""
import random
from typing import Optional


def load_hook_templates(config: dict) -> list[dict]:
    """Load hook templates from config."""
    return config.get("hook_templates", [])


def select_hooks_for_run(
    templates: list[dict],
    history: list[dict],
    count: int = 10,
) -> list[dict]:
    """
    Select hook templates for this generation run.

    Prioritizes hooks not used recently. Returns `count` templates,
    ensuring maximum diversity.
    """
    recent_hooks = set()
    for entry in history[-10:]:
        sc = entry.get("scenario", {})
        recent_hooks.add(sc.get("opening_hook_type", ""))

    # Split into not-recently-used and recently-used
    fresh = [t for t in templates if t["id"] not in recent_hooks]
    stale = [t for t in templates if t["id"] in recent_hooks]

    random.shuffle(fresh)
    random.shuffle(stale)

    # Fresh first, then fill with stale if needed
    selected = fresh[:count]
    if len(selected) < count:
        selected.extend(stale[: count - len(selected)])

    # If still not enough, duplicate fresh ones (different scenarios will vary)
    while len(selected) < count:
        selected.append(random.choice(templates))

    return selected[:count]


def validate_hook_strength(
    scenario: dict, min_hook_score: int = 7
) -> bool:
    """
    Quick pre-filter: does the scenario's hook description sound strong?

    Enforces the 0.5-second rule: the anomaly MUST be visible immediately.
    Rejects scenarios where the opening is too vague, delayed, or requires buildup.
    """
    hook_desc = scenario.get("opening_hook_description", "")
    if not hook_desc or len(hook_desc) < 20:
        return False

    hook_lower = hook_desc.lower()

    # Weak indicators - phrases that suggest delayed/slow opening
    weak_phrases = [
        "slowly", "gradually", "begins to", "starts to",
        "something seems", "might be", "could be", "appears to be",
        "in the distance a small", "barely visible",
        "nothing unusual", "normal day", "quiet", "peaceful",
        "will appear", "about to", "is going to", "soon",
        "building up", "growing", "emerging",
    ]
    for phrase in weak_phrases:
        if phrase in hook_lower:
            return False

    # Strong indicators - anomaly already present
    strong_phrases = [
        "already", "fills", "visible", "dominate", "covering",
        "cracking", "falling", "collapsing", "surging", "flooding",
        "impossibly close", "too close", "overhead", "directly",
    ]
    has_strong = any(p in hook_lower for p in strong_phrases)

    if not has_strong and len(hook_desc) < 40:
        return False

    return True
