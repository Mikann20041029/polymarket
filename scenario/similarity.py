"""
Similarity filter for candidate scenarios.

Compares candidates against history using weighted multi-axis similarity.
Both soft (weighted score) and hard (absolute rule) filters.
"""


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _keyword_similarity(text_a: str, text_b: str) -> float:
    """Simple keyword-based similarity (no LLM needed)."""
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    # Remove common stop words
    stop = {"the", "a", "an", "in", "on", "at", "of", "to", "and", "is",
            "are", "was", "were", "with", "from", "for", "by", "as", "it"}
    words_a -= stop
    words_b -= stop
    return _jaccard(words_a, words_b)


def _exact_match(val_a: str, val_b: str) -> float:
    """1.0 if exact match, 0.0 otherwise."""
    if not val_a or not val_b:
        return 0.0
    return 1.0 if val_a.strip().lower() == val_b.strip().lower() else 0.0


def _partial_match(val_a: str, val_b: str) -> float:
    """1.0 if exact, 0.5 if one contains the other, 0.0 otherwise."""
    if not val_a or not val_b:
        return 0.0
    a, b = val_a.strip().lower(), val_b.strip().lower()
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.5
    return 0.0


def compute_similarity(candidate: dict, existing: dict, weights: dict) -> float:
    """
    Compute weighted similarity score between a candidate and an existing scenario.

    Returns float 0.0 (totally different) to 1.0 (identical).
    """
    score = 0.0

    # event_type: partial match
    score += weights.get("event_type", 0.15) * _partial_match(
        candidate.get("event_type", ""), existing.get("event_type", "")
    )

    # scenario_summary: keyword similarity
    score += weights.get("scenario_summary", 0.15) * _keyword_similarity(
        candidate.get("scenario_summary", ""), existing.get("scenario_summary", "")
    )

    # location_style: keyword similarity
    score += weights.get("location_style", 0.08) * _keyword_similarity(
        candidate.get("location_style", ""), existing.get("location_style", "")
    )

    # camera_pov: exact match
    score += weights.get("camera_pov", 0.12) * _exact_match(
        candidate.get("camera_pov", ""), existing.get("camera_pov", "")
    )

    # opening_hook_type: exact match
    score += weights.get("opening_hook_type", 0.15) * _exact_match(
        candidate.get("opening_hook_type", ""), existing.get("opening_hook_type", "")
    )

    # peak_moment: keyword similarity (replaces escalation_pattern + climax_type)
    score += weights.get("escalation_pattern", 0.10) * _keyword_similarity(
        candidate.get("peak_moment", ""), existing.get("peak_moment", "")
    )

    # climax_type: kept for backward compat with old history entries
    score += weights.get("climax_type", 0.10) * _keyword_similarity(
        candidate.get("peak_moment", candidate.get("climax_type", "")),
        existing.get("peak_moment", existing.get("climax_type", "")),
    )

    # aftermath: keyword similarity
    score += weights.get("aftermath_type", 0.05) * _keyword_similarity(
        candidate.get("aftermath", candidate.get("aftermath_type", "")),
        existing.get("aftermath", existing.get("aftermath_type", "")),
    )

    # visual_tags: jaccard
    vis_a = set(candidate.get("visual_tags", []))
    vis_b = set(existing.get("visual_tags", []))
    score += weights.get("visual_tags", 0.05) * _jaccard(vis_a, vis_b)

    # tone_tags: jaccard
    tone_a = set(candidate.get("tone_tags", []))
    tone_b = set(existing.get("tone_tags", []))
    score += weights.get("tone_tags", 0.05) * _jaccard(tone_a, tone_b)

    return score


def check_hard_filters(
    candidate: dict,
    history: list[dict],
    hard_filter_config: dict,
) -> tuple[bool, str]:
    """
    Apply hard exclusion rules. Returns (passed, reason).

    If passed=False, the candidate must be rejected regardless of score.
    """
    c_event = candidate.get("event_type", "").lower()
    c_pov = candidate.get("camera_pov", "").lower()
    c_hook = candidate.get("opening_hook_type", "").lower()
    c_cat = candidate.get("category", "").lower()

    # Rule 1: No same event_type in last N
    n = hard_filter_config.get("same_event_type_last_n", 5)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        if sc.get("event_type", "").lower() == c_event and c_event:
            return False, f"event_type '{c_event}' used in last {n}"

    # Rule 2: No same POV in last N
    n = hard_filter_config.get("same_pov_last_n", 3)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        if sc.get("camera_pov", "").lower() == c_pov and c_pov:
            return False, f"camera_pov '{c_pov}' used in last {n}"

    # Rule 3: No same hook in last N
    n = hard_filter_config.get("same_hook_last_n", 3)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        if sc.get("opening_hook_type", "").lower() == c_hook and c_hook:
            return False, f"opening_hook '{c_hook}' used in last {n}"

    # Rule 4: Category frequency cap
    max_count = hard_filter_config.get("same_category_max_in_last_n", 3)
    window = hard_filter_config.get("same_category_window", 10)
    cat_count = 0
    for entry in history[-window:]:
        sc = entry.get("scenario", {})
        if sc.get("category", "").lower() == c_cat:
            cat_count += 1
    if cat_count >= max_count and c_cat:
        return False, f"category '{c_cat}' appeared {cat_count} times in last {window}"

    return True, ""


def filter_candidates(
    candidates: list[dict],
    history: list[dict],
    config: dict,
) -> list[dict]:
    """
    Filter candidates through both hard rules and soft similarity threshold.

    Returns list of candidates that passed all checks.
    """
    sim_config = config.get("similarity", {})
    weights = sim_config.get("weights", {})
    threshold = sim_config.get("threshold", 0.45)
    lookback = sim_config.get("history_lookback", 50)
    hard_filters = sim_config.get("hard_filters", {})

    recent_history = history[-lookback:]
    passed = []

    for candidate in candidates:
        # Hard filter first
        ok, reason = check_hard_filters(candidate, recent_history, hard_filters)
        if not ok:
            candidate["_rejected"] = True
            candidate["_reject_reason"] = f"hard_filter: {reason}"
            continue

        # Soft similarity check against all recent
        max_sim = 0.0
        most_similar_id = ""
        for entry in recent_history:
            sc = entry.get("scenario", {})
            sim = compute_similarity(candidate, sc, weights)
            if sim > max_sim:
                max_sim = sim
                most_similar_id = entry.get("id", "")

        candidate["_max_similarity"] = max_sim
        candidate["_most_similar_id"] = most_similar_id

        if max_sim > threshold:
            candidate["_rejected"] = True
            candidate["_reject_reason"] = (
                f"similarity {max_sim:.2f} > {threshold} (vs {most_similar_id})"
            )
            continue

        passed.append(candidate)

    return passed
