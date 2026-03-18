"""
Buzz score calculation for scenario candidates.

Uses LLM to evaluate each candidate against 7 criteria.
Falls back to heuristic scoring if LLM is unavailable.
"""
import json
import logging

logger = logging.getLogger(__name__)

SCORING_SYSTEM_PROMPT = """You are a viral short-form video scoring expert.
You evaluate scenario concepts for their potential to go viral on YouTube Shorts / TikTok.

Target audience: global viewers who stop scrolling for shocking, realistic-looking witness footage.

Score each scenario on these 7 criteria (1-10 scale):

1. first_second_shock: Does the opening frame make viewers STOP scrolling instantly?
   10 = impossible to scroll past  |  5 = interesting but not stopping  |  1 = boring opening

2. instant_clarity: Can viewers understand what's happening within 1 second, no text needed?
   10 = crystal clear instantly  |  5 = takes 2-3 seconds  |  1 = confusing

3. scale_impact: Is the scale dramatic? Giant objects, vast phenomena, human-vs-nature contrast?
   10 = jaw-dropping scale  |  5 = moderate  |  1 = small/mundane

4. realism_potential: Can current AI video generation (5-second clips, 480-720p) render this convincingly?
   10 = mostly distant/atmospheric, easy to fake  |  5 = some tricky elements  |  1 = requires perfect human faces/physics

5. curiosity_gap: Will viewers wonder "Where is this?" or "Is this real?"
   10 = absolutely  |  5 = somewhat  |  1 = obviously fake/uninteresting

6. uniqueness: Is this unlike content commonly seen on YouTube/TikTok?
   10 = never seen before  |  5 = somewhat rare  |  1 = overdone

7. replay_potential: Will viewers rewatch or share?
   10 = must share  |  5 = might rewatch  |  1 = one and done

IMPORTANT: Be strict. Most scenarios should score 5-7. Only truly exceptional concepts get 9-10.
A score of 8+ on first_second_shock means the opening is genuinely shocking."""

SCORING_USER_PROMPT = """Score these scenario candidates. Return ONLY a JSON array.

Candidates:
{candidates_json}

Return format (JSON array, one object per candidate):
[
  {{
    "candidate_index": 0,
    "scores": {{
      "first_second_shock": <1-10>,
      "instant_clarity": <1-10>,
      "scale_impact": <1-10>,
      "realism_potential": <1-10>,
      "curiosity_gap": <1-10>,
      "uniqueness": <1-10>,
      "replay_potential": <1-10>
    }},
    "total": <sum>,
    "brief_note": "<one sentence on strongest/weakest aspect>"
  }},
  ...
]"""


def score_candidates_llm(
    candidates: list[dict],
    llm_client,
    config: dict,
) -> list[dict]:
    """
    Score candidates using LLM. Attaches buzz_score to each candidate.

    Args:
        candidates: list of scenario dicts
        llm_client: OpenAI-compatible client
        config: full config dict

    Returns:
        candidates with buzz_score attached
    """
    if not candidates:
        return candidates

    llm_config = config.get("llm", {})
    scoring_config = config.get("scoring", {})

    # Prepare summaries for scoring (don't send full metadata)
    summaries = []
    for i, c in enumerate(candidates):
        summaries.append({
            "index": i,
            "category": c.get("category", ""),
            "event_type": c.get("event_type", ""),
            "camera_pov": c.get("camera_pov", ""),
            "opening_hook_type": c.get("opening_hook_type", ""),
            "opening_hook_description": c.get("opening_hook_description", ""),
            "scenario_summary": c.get("scenario_summary", ""),
            "escalation_pattern": c.get("escalation_pattern", ""),
            "climax_type": c.get("climax_type", ""),
            "aftermath_type": c.get("aftermath_type", ""),
        })

    user_prompt = SCORING_USER_PROMPT.format(
        candidates_json=json.dumps(summaries, indent=2)
    )

    try:
        response = llm_client.chat.completions.create(
            model=llm_config.get("model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=llm_config.get("max_tokens", 8192),
            temperature=llm_config.get("scoring_temperature", 0.3),
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON from response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        scores_list = json.loads(raw)

        for score_entry in scores_list:
            idx = score_entry.get("candidate_index", -1)
            if 0 <= idx < len(candidates):
                candidates[idx]["buzz_score"] = score_entry.get("scores", {})
                candidates[idx]["buzz_total"] = score_entry.get("total", 0)
                candidates[idx]["buzz_note"] = score_entry.get("brief_note", "")

        # Validate totals
        for c in candidates:
            if "buzz_score" in c:
                actual_total = sum(c["buzz_score"].values())
                c["buzz_total"] = actual_total

    except Exception as e:
        logger.error("LLM scoring failed: %s. Using heuristic fallback.", e)
        candidates = score_candidates_heuristic(candidates)

    # Mark candidates without scores
    for c in candidates:
        if "buzz_score" not in c:
            c["buzz_score"] = {}
            c["buzz_total"] = 0
            c["buzz_note"] = "scoring_failed"

    return candidates


def score_candidates_heuristic(candidates: list[dict]) -> list[dict]:
    """
    Fallback heuristic scoring when LLM is unavailable.

    Uses simple rules based on metadata to estimate buzz potential.
    Less accurate but costs nothing.
    """
    for c in candidates:
        scores = {}

        # first_second_shock: based on hook type
        high_shock_hooks = {
            "massive_object_too_close", "collapse_already_started",
            "animal_proximity", "scale_reveal",
        }
        hook = c.get("opening_hook_type", "")
        scores["first_second_shock"] = 8 if hook in high_shock_hooks else 6

        # instant_clarity: penalize abstract scenarios
        summary = c.get("scenario_summary", "").lower()
        if any(w in summary for w in ["mysterious", "strange", "unknown", "unexplained"]):
            scores["instant_clarity"] = 5
        else:
            scores["instant_clarity"] = 7

        # scale_impact: based on visual tags
        big_tags = {"giant", "massive", "enormous", "huge", "towering", "colossal"}
        vtags = set(t.lower() for t in c.get("visual_tags", []))
        scores["scale_impact"] = 8 if vtags & big_tags else 6

        # realism_potential: penalize close-up human action
        pov = c.get("camera_pov", "")
        if pov in ("drone", "balcony", "interior"):
            scores["realism_potential"] = 8  # distant = easier to render
        elif pov in ("street", "tourist"):
            scores["realism_potential"] = 6
        else:
            scores["realism_potential"] = 7

        # curiosity_gap: location specificity helps
        loc = c.get("location_style", "")
        if loc and len(loc) > 10:
            scores["curiosity_gap"] = 7
        else:
            scores["curiosity_gap"] = 5

        # uniqueness: assume moderate by default
        scores["uniqueness"] = 6

        # replay_potential: assume moderate
        scores["replay_potential"] = 6

        c["buzz_score"] = scores
        c["buzz_total"] = sum(scores.values())
        c["buzz_note"] = "heuristic_scoring"

    return candidates


def filter_by_score(
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """Remove candidates below minimum score thresholds."""
    scoring_config = config.get("scoring", {})
    min_total = scoring_config.get("min_total_score", 45)
    min_hook = scoring_config.get("min_hook_score", 7)

    passed = []
    for c in candidates:
        total = c.get("buzz_total", 0)
        hook_score = c.get("buzz_score", {}).get("first_second_shock", 0)

        if total < min_total:
            c["_rejected"] = True
            c["_reject_reason"] = f"buzz_total {total} < {min_total}"
            continue
        if hook_score < min_hook:
            c["_rejected"] = True
            c["_reject_reason"] = f"first_second_shock {hook_score} < {min_hook}"
            continue

        passed.append(c)

    return passed
