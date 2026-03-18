"""
Two-stage scenario candidate generator.

Stage 1: Select hook templates + POVs (structure-first)
Stage 2: LLM generates concrete scenarios per (hook, POV) pair

Generation order (strict):
  1. POV is fixed
  2. Opening hook is fixed
  3. Scenario is generated to fit both
  4. Escalation/climax/aftermath are generated last
"""
import json
import logging
import random

from .hooks import load_hook_templates, select_hooks_for_run, validate_hook_strength
from .povs import load_povs, select_pov_for_hook, validate_pov_event_compatibility

logger = logging.getLogger(__name__)

CANDIDATE_SYSTEM_PROMPT = """You are a creative director for a viral short-form video channel.
You create concepts for "witness-type shock footage" - vertical videos (9:16) that look like
real footage accidentally captured by ordinary people.

RULES:
- Videos must feel like real phone footage of extraordinary events
- The opening 1 second MUST contain something visually shocking or abnormal
- No text, narration, or dialogue in the video
- Audio: natural environment sounds + subtle SFX + quiet BGM only
- Target audience: global viewers (not Japan-specific)
- If signs/text appear in the scene, use language appropriate to the location
- NO anime, NO historical recreation - this is about "what if this happened right now"
- Prioritize events that make viewers think "Wait, where did this happen? Is this real?"

QUALITY STANDARDS:
- The scenario must be filmable as 5-second clips (6 clips = 30 seconds)
- Prefer distant/atmospheric shots over close-up human faces (AI renders these better)
- Include natural motion: water, wind, debris, smoke, crowds in distance
- Camera behavior must match the POV (shaky handheld, stable dashcam, etc.)"""

CANDIDATE_USER_PROMPT = """Generate exactly {count} scenario candidates.

CONSTRAINTS FOR THIS BATCH:
- Camera POV: {pov_id} ({pov_description})
- Camera traits: {pov_traits}
- Opening hook structure: {hook_label} - {hook_description}
- First frame must match: {hook_first_frame}
- Categories to consider: {compatible_categories}
- Categories to AVOID (overused recently): {avoid_categories}

Each candidate MUST:
1. Have the opening hook ALREADY happening in the first frame (not building up to it)
2. Work naturally from the given camera POV
3. Have a distinct event_type, location, escalation, and climax from other candidates

Return ONLY a JSON array of {count} objects with this exact structure:
[
  {{
    "category": "<one of: natural_phenomenon, urban_anomaly, animal_encounter, maritime_anomaly, aviation_anomaly, traffic_transport, infrastructure_failure, space_sky_anomaly, realistic_whatif>",
    "event_type": "<specific event, e.g. 'glacier_calving_tsunami'>",
    "scenario_summary": "<2-3 sentences describing the full scenario>",
    "location_style": "<specific location feel, e.g. 'Norwegian fjord, summer afternoon'>",
    "time_of_day": "<dawn/morning/midday/afternoon/dusk/night>",
    "weather_atmosphere": "<e.g. 'overcast, humid, haze'>",
    "camera_pov": "{pov_id}",
    "camera_movement": "<specific movement: e.g. 'handheld slight tremor, quick zoom in, then pan right'>",
    "opening_hook_type": "{hook_id}",
    "opening_hook_description": "<exactly what is visible/audible in the first 1 second>",
    "escalation_pattern": "<how tension builds across clips 2-4>",
    "climax_type": "<what happens at peak moment>",
    "climax_description": "<specific visual description of the climax>",
    "aftermath_type": "<how it ends: stunned_silence / chaos / eerie_calm / flee / damage_reveal>",
    "aftermath_description": "<what the final clip shows>",
    "visual_tags": ["<5-8 specific visual elements>"],
    "tone_tags": ["<2-4 emotional tones: dread, awe, panic, helplessness, wonder, etc.>"],
    "dominant_colors": ["<3-4 main colors in the scene>"],
    "sound_atmosphere": "<ambient sound description for SFX generation>"
  }}
]

Make each candidate MAXIMALLY different from the others in this batch.
Different locations, different events, different scales, different moods."""


def generate_candidates(
    llm_client,
    config: dict,
    history: list[dict],
    category_stats: dict,
) -> list[dict]:
    """
    Two-stage candidate generation.

    Stage 1: Select 10 hook templates, pair each with a compatible POV
    Stage 2: For each (hook, POV) pair, generate 3 concrete scenarios

    Returns: list of 30+ candidate scenario dicts
    """
    gen_config = config.get("generation", {})
    llm_config = config.get("llm", {})
    categories_config = config.get("categories", {})

    hook_templates = load_hook_templates(config)
    povs = load_povs(config)

    n_templates = gen_config.get("min_hook_templates", 10)
    n_per_template = gen_config.get("candidates_per_template", 3)

    # Stage 1: Select hooks and pair with POVs
    selected_hooks = select_hooks_for_run(hook_templates, history, count=n_templates)

    # Determine over-represented categories to avoid
    avoid_categories = _get_overused_categories(category_stats, categories_config)

    # Stage 2: Generate scenarios per (hook, POV) pair
    all_candidates = []

    for hook in selected_hooks:
        # Select a POV compatible with this hook
        pov_id, pov_cfg = select_pov_for_hook(povs, hook, history)
        compatible_cats = pov_cfg.get("compatible_categories", [])

        prompt = CANDIDATE_USER_PROMPT.format(
            count=n_per_template,
            pov_id=pov_id,
            pov_description=pov_cfg.get("description", ""),
            pov_traits=pov_cfg.get("camera_traits", ""),
            hook_label=hook.get("label", ""),
            hook_description=hook.get("description", ""),
            hook_first_frame=hook.get("first_frame", ""),
            hook_id=hook.get("id", ""),
            compatible_categories=", ".join(compatible_cats),
            avoid_categories=", ".join(avoid_categories) if avoid_categories else "none",
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=llm_config.get("max_tokens", 8192),
                temperature=llm_config.get("temperature", 0.9),
            )
            raw = response.choices[0].message.content.strip()
            candidates = _parse_candidates_json(raw)

            # Validate each candidate
            for c in candidates:
                # Enforce POV and hook from this batch
                c["camera_pov"] = pov_id
                c["opening_hook_type"] = hook["id"]

                # Validate POV-event compatibility
                cat = c.get("category", "")
                if not validate_pov_event_compatibility(pov_id, cat, povs):
                    logger.debug(
                        "Rejected: POV '%s' incompatible with category '%s'",
                        pov_id, cat,
                    )
                    continue

                # Validate hook strength
                if not validate_hook_strength(c, hook.get("min_shock_score", 7)):
                    logger.debug(
                        "Rejected: weak hook for '%s'",
                        c.get("scenario_summary", "")[:40],
                    )
                    continue

                all_candidates.append(c)

            logger.info(
                "Hook '%s' + POV '%s': generated %d valid candidates",
                hook["id"], pov_id, len(candidates),
            )

        except Exception as e:
            logger.error(
                "Generation failed for hook '%s': %s", hook["id"], e
            )
            continue

    logger.info("Total candidates generated: %d", len(all_candidates))
    return all_candidates


def generate_candidates_dry(config: dict) -> list[dict]:
    """
    Generate candidates without any API call (for testing structure).

    Returns a small set of hardcoded example candidates.
    """
    return [
        {
            "category": "maritime_anomaly",
            "event_type": "cargo_ship_near_miss",
            "scenario_summary": "A massive container ship drifts dangerously close to a small fishing pier in a Southeast Asian port.",
            "location_style": "Southeast Asian commercial port",
            "time_of_day": "afternoon",
            "weather_atmosphere": "hazy, humid, overcast",
            "camera_pov": "tourist",
            "camera_movement": "handheld panic zoom",
            "opening_hook_type": "massive_object_too_close",
            "opening_hook_description": "Towering ship hull fills 80% of frame, impossibly close to the pier",
            "escalation_pattern": "ship drifts closer, ropes snap, small boats pushed aside",
            "climax_type": "near_impact",
            "climax_description": "Ship hull scrapes pier, wood splinters",
            "aftermath_type": "stunned_silence",
            "aftermath_description": "Damage trail visible, distant shouting",
            "visual_tags": ["giant_ship", "port", "water_displacement", "pier", "scrambling_people"],
            "tone_tags": ["dread", "helplessness", "scale_shock"],
            "dominant_colors": ["steel_gray", "ocean_blue", "rust_orange"],
            "sound_atmosphere": "harbor ambience, metal groaning, water slapping",
        },
        {
            "category": "natural_phenomenon",
            "event_type": "highway_sinkhole",
            "scenario_summary": "A massive sinkhole opens under a highway, swallowing vehicles in real time from a dashcam perspective.",
            "location_style": "American suburban highway",
            "time_of_day": "morning",
            "weather_atmosphere": "clear, bright",
            "camera_pov": "dashcam",
            "camera_movement": "fixed dashcam, slight vibration",
            "opening_hook_type": "collapse_already_started",
            "opening_hook_description": "Road surface ahead is cracking and sinking, front car's rear wheels are dropping",
            "escalation_pattern": "hole expands, second car tilts, cracks spread laterally",
            "climax_type": "full_collapse",
            "climax_description": "Road section drops, car half-falls into hole",
            "aftermath_type": "chaos",
            "aftermath_description": "Emergency stop, hazard lights, honking chain",
            "visual_tags": ["sinkhole", "highway", "cracking_asphalt", "tilting_car", "dust_cloud"],
            "tone_tags": ["shock", "disbelief"],
            "dominant_colors": ["asphalt_gray", "dust_brown", "sky_blue"],
            "sound_atmosphere": "engine hum, cracking concrete, car alarms",
        },
    ]


def _parse_candidates_json(raw: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse candidates JSON: %s", e)
        return []


def _get_overused_categories(
    stats: dict,
    categories_config: dict,
) -> list[str]:
    """Find categories that are over their target ratio."""
    if not stats:
        return []

    total = sum(s.get("count", 0) for s in stats.values())
    if total == 0:
        return []

    overused = []
    for cat_id, cat_stats in stats.items():
        actual_ratio = cat_stats.get("count", 0) / total
        target = categories_config.get(cat_id, {}).get("target_ratio", 0.11)
        if actual_ratio > target + 0.05:  # 5% tolerance
            overused.append(cat_id)

    return overused
