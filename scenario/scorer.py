"""
Construction timelapse scenario scoring.

8 criteria, 80 points max. Uses LLM or heuristic fallback.
"""
import json
import logging

logger = logging.getLogger(__name__)

SCORING_SYSTEM_PROMPT = """You are a viral construction timelapse YouTube Shorts scoring expert.
You evaluate scenario concepts for rebornspacestv-style videos.

WHAT THESE VIDEOS ARE:
- 15-second construction timelapse showing a space being built/transformed
- Workers, heavy machinery, tools, materials visible
- High-speed time-lapse: digging, framing, pouring, tiling, finishing
- Ends with luxury/hidden/amazing completed space

Score each scenario on these 8 criteria (1-10 scale):

1. one_line_concept_strength: Does the one-line concept make you NEED to watch?
   10 = impossible to scroll past | 5 = interesting but not compelling | 1 = boring

2. first_second_clarity: Is the before-state instantly clear in 1 second?
   10 = immediate understanding | 5 = takes a moment | 1 = confusing

3. construction_process_satisfaction: Is the build process satisfying to watch in fast-forward?
   10 = incredibly satisfying | 5 = okay | 1 = boring/unclear process

4. reveal_satisfaction: Does the final reveal deliver a wow moment?
   10 = jaw-dropping | 5 = nice but expected | 1 = underwhelming

5. realism_believability: Could this actually be built? Does it look real?
   10 = totally real | 5 = stretch but possible | 1 = impossible fantasy

6. luxury_desire: Do viewers want this space for themselves?
   10 = dream space | 5 = cool but not personally desired | 1 = no appeal

7. uniqueness_vs_history: Is this unlike previous videos?
   10 = never seen before | 5 = somewhat fresh | 1 = overdone

8. loop_rewatch_potential: Will viewers watch again or share?
   10 = must share/rewatch | 5 = might rewatch | 1 = one and done

IMPORTANT: Be strict. Most scenarios should score 5-7. Only truly exceptional get 9-10."""

SCORING_USER_PROMPT = """Score these construction timelapse scenario candidates. Return ONLY a JSON array.

Candidates:
{candidates_json}

Return format:
[
  {{
    "candidate_index": 0,
    "scores": {{
      "one_line_concept_strength": <1-10>,
      "first_second_clarity": <1-10>,
      "construction_process_satisfaction": <1-10>,
      "reveal_satisfaction": <1-10>,
      "realism_believability": <1-10>,
      "luxury_desire": <1-10>,
      "uniqueness_vs_history": <1-10>,
      "loop_rewatch_potential": <1-10>
    }},
    "total": <sum>,
    "brief_note": "<one sentence>"
  }}
]"""


def score_candidates_llm(
    candidates: list[dict],
    llm_client,
    config: dict,
) -> list[dict]:
    """Score candidates using LLM. Attaches scores to each candidate."""
    if not candidates:
        return candidates

    llm_config = config.get("llm", {})

    summaries = []
    for i, c in enumerate(candidates):
        summaries.append({
            "index": i,
            "one_line_concept": c.get("one_line_concept", ""),
            "category": c.get("category", ""),
            "construction_type": c.get("construction_type", ""),
            "before_space": c.get("before_space", {}),
            "construction_process": c.get("construction_process", {}),
            "after_space": c.get("after_space", {}),
            "reveal_type": c.get("reveal_type", ""),
            "location_feel": c.get("location_feel", ""),
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

        # Recalculate totals for accuracy
        for c in candidates:
            if "buzz_score" in c:
                c["buzz_total"] = sum(c["buzz_score"].values())

    except Exception as e:
        logger.error("LLM scoring failed: %s. Using heuristic fallback.", e)
        candidates = score_candidates_heuristic(candidates)

    for c in candidates:
        if "buzz_score" not in c:
            c["buzz_score"] = {}
            c["buzz_total"] = 0
            c["buzz_note"] = "scoring_failed"

    return candidates


def score_candidates_heuristic(candidates: list[dict]) -> list[dict]:
    """Fallback heuristic scoring for offline/dry mode."""
    for c in candidates:
        scores = {}

        # one_line_concept_strength: longer, more specific = better
        concept = c.get("one_line_concept", "")
        scores["one_line_concept_strength"] = 7 if len(concept) > 40 else 6

        # first_second_clarity: based on before_space description
        before = c.get("before_space", {})
        if before.get("description", "") and before.get("visual", ""):
            scores["first_second_clarity"] = 7
        else:
            scores["first_second_clarity"] = 5

        # construction_process_satisfaction: more stages + machinery = better
        proc = c.get("construction_process", {})
        stages = len(proc.get("stages", []))
        machines = len(proc.get("heavy_machinery", []))
        if stages >= 5 and machines >= 2:
            scores["construction_process_satisfaction"] = 8
        elif stages >= 4 and machines >= 1:
            scores["construction_process_satisfaction"] = 7
        else:
            scores["construction_process_satisfaction"] = 6

        # reveal_satisfaction: based on reveal type and luxury level
        after = c.get("after_space", {})
        luxury = after.get("luxury_level", "medium")
        if luxury == "ultra":
            scores["reveal_satisfaction"] = 8
        elif luxury == "high":
            scores["reveal_satisfaction"] = 7
        else:
            scores["reveal_satisfaction"] = 6

        # realism_believability
        scores["realism_believability"] = 7

        # luxury_desire: ultra/high luxury + water = desirable
        if luxury in ("ultra", "high") and after.get("water_element", False):
            scores["luxury_desire"] = 8
        elif luxury in ("ultra", "high"):
            scores["luxury_desire"] = 7
        else:
            scores["luxury_desire"] = 6

        # uniqueness_vs_history: default moderate
        scores["uniqueness_vs_history"] = 6

        # loop_rewatch_potential
        reveal = c.get("reveal_type", "")
        high_rewatch = {"mechanical_reveal", "hidden_entrance_reveal", "water_fill_reveal"}
        scores["loop_rewatch_potential"] = 7 if reveal in high_rewatch else 6

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
    min_total = scoring_config.get("min_total_score", 50)
    min_concept = scoring_config.get("min_concept_score", 7)
    min_process = scoring_config.get("min_process_score", 6)

    passed = []
    for c in candidates:
        total = c.get("buzz_total", 0)
        scores = c.get("buzz_score", {})
        concept_score = scores.get("one_line_concept_strength", 0)
        process_score = scores.get("construction_process_satisfaction", 0)

        if total < min_total:
            c["_rejected"] = True
            c["_reject_reason"] = f"buzz_total {total} < {min_total}"
            continue
        if concept_score < min_concept:
            c["_rejected"] = True
            c["_reject_reason"] = f"concept_strength {concept_score} < {min_concept}"
            continue
        if process_score < min_process:
            c["_rejected"] = True
            c["_reject_reason"] = f"process_satisfaction {process_score} < {min_process}"
            continue

        passed.append(c)

    return passed
