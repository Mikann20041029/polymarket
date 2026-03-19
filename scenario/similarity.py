"""
Similarity filter for construction timelapse scenarios.

3-axis weighted similarity: concept (0.35), process (0.35), completion (0.30).
Both soft (weighted score) and hard (absolute rule) filters.
"""
import logging

logger = logging.getLogger(__name__)


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def _keyword_similarity(text_a: str, text_b: str) -> float:
    """Simple keyword-based similarity."""
    if not text_a or not text_b:
        return 0.0
    stop = {"the", "a", "an", "in", "on", "at", "of", "to", "and", "is",
            "are", "was", "were", "with", "from", "for", "by", "as", "it"}
    words_a = set(text_a.lower().split()) - stop
    words_b = set(text_b.lower().split()) - stop
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


def _concept_similarity(candidate: dict, existing: dict) -> float:
    """
    Concept axis: how similar is the idea/hook?
    Compares one_line_concept, category, and concept template.
    """
    score = 0.0
    # one_line_concept keyword overlap (most important)
    score += 0.50 * _keyword_similarity(
        candidate.get("one_line_concept", ""),
        existing.get("one_line_concept", ""),
    )
    # category exact match
    score += 0.30 * _exact_match(
        candidate.get("category", ""),
        existing.get("category", ""),
    )
    # concept template exact match
    score += 0.20 * _exact_match(
        candidate.get("_concept_template", ""),
        existing.get("_concept_template", ""),
    )
    return score


def _process_similarity(candidate: dict, existing: dict) -> float:
    """
    Process axis: how similar is the construction process?
    Compares construction_type, machinery, materials, excavation.
    """
    score = 0.0
    # construction_type
    score += 0.35 * _partial_match(
        candidate.get("construction_type", ""),
        existing.get("construction_type", ""),
    )
    # heavy_machinery overlap
    c_proc = candidate.get("construction_process", {})
    e_proc = existing.get("construction_process", {})
    c_machines = set(c_proc.get("heavy_machinery", []))
    e_machines = set(e_proc.get("heavy_machinery", []))
    score += 0.25 * _jaccard(c_machines, e_machines)
    # key_materials overlap
    c_mats = set(c_proc.get("key_materials", []))
    e_mats = set(e_proc.get("key_materials", []))
    score += 0.25 * _jaccard(c_mats, e_mats)
    # excavation_required match
    c_exc = c_proc.get("excavation_required", False)
    e_exc = e_proc.get("excavation_required", False)
    score += 0.15 * (1.0 if c_exc == e_exc else 0.0)
    return score


def _completion_similarity(candidate: dict, existing: dict) -> float:
    """
    Completion axis: how similar is the final reveal/space?
    Compares after_space, reveal_type, similarity_tags.
    """
    score = 0.0
    # after_space type
    c_after = candidate.get("after_space", {})
    e_after = existing.get("after_space", {})
    score += 0.30 * _partial_match(
        c_after.get("type", ""),
        e_after.get("type", ""),
    )
    # reveal_type
    score += 0.25 * _exact_match(
        candidate.get("reveal_type", ""),
        existing.get("reveal_type", ""),
    )
    # water_element match
    c_water = c_after.get("water_element", False)
    e_water = e_after.get("water_element", False)
    score += 0.15 * (1.0 if c_water == e_water else 0.0)
    # similarity_tags jaccard
    c_tags = set(candidate.get("similarity_tags", []))
    e_tags = set(existing.get("similarity_tags", []))
    score += 0.30 * _jaccard(c_tags, e_tags)
    return score


def compute_similarity(candidate: dict, existing: dict, weights: dict) -> float:
    """
    Compute 3-axis weighted similarity between candidate and existing scenario.
    Returns float 0.0 (totally different) to 1.0 (identical).
    """
    w_concept = weights.get("concept", 0.35)
    w_process = weights.get("process", 0.35)
    w_completion = weights.get("completion", 0.30)

    return (
        w_concept * _concept_similarity(candidate, existing)
        + w_process * _process_similarity(candidate, existing)
        + w_completion * _completion_similarity(candidate, existing)
    )


def check_hard_filters(
    candidate: dict,
    history: list[dict],
    hard_filter_config: dict,
) -> tuple[bool, str]:
    """
    Apply hard exclusion rules. Returns (passed, reason).
    If passed=False, the candidate must be rejected.
    """
    c_cat = candidate.get("category", "").lower()
    c_ctype = candidate.get("construction_type", "").lower()
    c_reveal = candidate.get("reveal_type", "").lower()
    c_after = candidate.get("after_space", {}).get("type", "").lower()
    c_proc = candidate.get("construction_process", {})
    c_excavation = c_proc.get("excavation_required", False)
    c_water = candidate.get("after_space", {}).get("water_element", False)

    # Rule 1: No same category in last N
    n = hard_filter_config.get("same_category_last_n", 3)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        if sc.get("category", "").lower() == c_cat and c_cat:
            return False, f"category '{c_cat}' used in last {n}"

    # Rule 2: No same construction_type in last N
    n = hard_filter_config.get("same_construction_type_last_n", 5)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        if sc.get("construction_type", "").lower() == c_ctype and c_ctype:
            return False, f"construction_type '{c_ctype}' used in last {n}"

    # Rule 3: No same reveal_type in last N
    n = hard_filter_config.get("same_reveal_type_last_n", 3)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        if sc.get("reveal_type", "").lower() == c_reveal and c_reveal:
            return False, f"reveal_type '{c_reveal}' used in last {n}"

    # Rule 4: No same after_space type in last N
    n = hard_filter_config.get("same_after_space_last_n", 5)
    for entry in history[-n:]:
        sc = entry.get("scenario", {})
        sc_after = sc.get("after_space", {})
        if isinstance(sc_after, dict):
            after_type = sc_after.get("type", "").lower()
        else:
            after_type = ""
        if after_type == c_after and c_after:
            return False, f"after_space '{c_after}' used in last {n}"

    # Rule 5: No more than N consecutive excavation videos
    max_exc = hard_filter_config.get("consecutive_excavation_max", 3)
    if c_excavation:
        consecutive = 0
        for entry in reversed(history):
            sc = entry.get("scenario", {})
            proc = sc.get("construction_process", {})
            if isinstance(proc, dict) and proc.get("excavation_required", False):
                consecutive += 1
            else:
                break
        if consecutive >= max_exc:
            return False, f"excavation consecutive {consecutive} >= {max_exc}"

    # Rule 6: No more than N consecutive water element videos
    max_water = hard_filter_config.get("consecutive_water_element_max", 3)
    if c_water:
        consecutive = 0
        for entry in reversed(history):
            sc = entry.get("scenario", {})
            after = sc.get("after_space", {})
            if isinstance(after, dict) and after.get("water_element", False):
                consecutive += 1
            else:
                break
        if consecutive >= max_water:
            return False, f"water_element consecutive {consecutive} >= {max_water}"

    return True, ""


def filter_candidates(
    candidates: list[dict],
    history: list[dict],
    config: dict,
) -> list[dict]:
    """
    Filter candidates through hard rules and soft similarity threshold.
    Returns list of candidates that passed all checks.
    """
    sim_config = config.get("similarity", {})
    weights = sim_config.get("weights", {})
    threshold = sim_config.get("threshold", 0.40)
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

        # Soft similarity check
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
