"""
Two-stage scenario candidate generator.

Stage 1: Select hook templates + POVs (structure-first)
Stage 2: LLM generates concrete scenarios per (hook, POV) pair

Generation order (strict):
  1. POV is fixed
  2. Opening hook is fixed
  3. Scenario is generated to fit both
  4. Peak/aftermath generated last

Single-scene model:
  1 video = 1 continuous clip = 10-15 seconds
  No multi-clip. No story arc. No cuts.
  Camera is fixed or near-fixed.
  Anomaly visible within 0.5 seconds.
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

CRITICAL FORMAT: 1 video = 1 single continuous shot = 10-15 seconds. NO cuts, NO multi-clip, NO story.

RULES:
- Each video is ONE continuous 10-15 second clip from a FIXED or near-fixed camera
- Camera does NOT move significantly. It stays in place. Natural tremor only.
- The anomaly MUST be visible within the first 0.5 seconds of the video
- The viewer MUST understand what's happening within 3 seconds WITHOUT thinking
- No text, narration, dialogue, or story structure
- Audio: natural environment sounds only
- Target audience: global viewers (not Japan-specific)
- If signs/text appear in the scene, use language appropriate to the location
- NO anime, NO historical recreation
- Prioritize "is this real?" reactions

TIME STRUCTURE (within the single 10-15s clip):
  0.0-0.5s: Anomaly ALREADY visible in frame
  0.5-3.0s: Viewer fully understands what's happening
  3.0-10.0s: Event intensifies / reaches peak
  10.0-15.0s: Aftermath or lingering moment

WHAT MAKES IT WORK:
- ONE event, ONE viewpoint, ONE continuous moment
- No buildup needed - we're dropping into the middle of something
- Fixed camera = more realistic (security cam, phone propped up, dashcam, mounted GoPro)
- Simplicity = clarity = instant understanding = viral"""

CANDIDATE_USER_PROMPT = """Generate exactly {count} scenario candidates.

CONSTRAINTS FOR THIS BATCH:
- Camera POV: {pov_id} ({pov_description})
- Camera traits: {pov_traits}
- Opening hook structure: {hook_label} - {hook_description}
- First frame must match: {hook_first_frame}
- Categories to consider: {compatible_categories}
- Categories to AVOID (overused recently): {avoid_categories}

REMEMBER: 1 video = 1 single continuous clip of 10-15 seconds.
Camera is FIXED or near-fixed. No cuts, no story, no multi-scene.

Each candidate MUST:
1. Have the anomaly ALREADY visible at 0.5 seconds (not building up to it)
2. Be instantly understandable without thinking
3. Work as a SINGLE continuous fixed-camera shot
4. Have a distinct event_type, location, and mood from other candidates

Return ONLY a JSON array of {count} objects with this exact structure:
[
  {{
    "category": "<one of: natural_phenomenon, urban_anomaly, animal_encounter, maritime_anomaly, aviation_anomaly, traffic_transport, infrastructure_failure, space_sky_anomaly, realistic_whatif>",
    "event_type": "<specific event, e.g. 'glacier_calving'>",
    "scenario_summary": "<1-2 sentences: what is happening in this single continuous shot>",
    "location_style": "<specific location feel, e.g. 'Norwegian fjord, summer afternoon'>",
    "time_of_day": "<dawn/morning/midday/afternoon/dusk/night>",
    "weather_atmosphere": "<e.g. 'overcast, humid, haze'>",
    "camera_pov": "{pov_id}",
    "camera_movement": "<fixed / near-fixed with slight tremor / mounted stable>",
    "opening_hook_type": "{hook_id}",
    "opening_hook_description": "<exactly what is visible at 0.5 seconds>",
    "peak_moment": "<what happens at maximum intensity (5-10s mark)>",
    "aftermath": "<what the last few seconds show (10-15s)>",
    "visual_tags": ["<5-8 specific visual elements>"],
    "tone_tags": ["<2-4 emotional tones>"],
    "dominant_colors": ["<3-4 main colors>"],
    "sound_atmosphere": "<ambient sound for this single continuous moment>"
  }}
]

Make each candidate MAXIMALLY different from the others."""


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
    Single-scene format: 1 video = 1 continuous 10-15s clip.
    """
    return [
        {
            "category": "maritime_anomaly",
            "event_type": "cargo_ship_near_miss",
            "scenario_summary": "A massive container ship hull fills the frame, drifting dangerously close to a small pier. Single continuous shot from dock level.",
            "location_style": "Southeast Asian commercial port",
            "time_of_day": "afternoon",
            "weather_atmosphere": "hazy, humid, overcast",
            "camera_pov": "tourist",
            "camera_movement": "near-fixed, slight hand tremor",
            "opening_hook_type": "massive_object_too_close",
            "opening_hook_description": "Towering ship hull fills 80% of frame at 0.5s, impossibly close to the pier",
            "peak_moment": "Ship hull scrapes pier edge, wood splinters fly, water surges between hull and dock",
            "aftermath": "Ship slowly drifts past, wake rocks small boats, debris floats",
            "visual_tags": ["giant_ship_hull", "pier_wood", "water_surge", "rust_streaks", "scale_contrast"],
            "tone_tags": ["dread", "helplessness", "scale_shock"],
            "dominant_colors": ["steel_gray", "ocean_blue", "rust_orange"],
            "sound_atmosphere": "deep metal groaning, water churning, wood cracking",
        },
        {
            "category": "natural_phenomenon",
            "event_type": "highway_sinkhole",
            "scenario_summary": "Road surface ahead cracks and sinks in real time, swallowing the front car. Single continuous dashcam shot.",
            "location_style": "American suburban highway",
            "time_of_day": "morning",
            "weather_atmosphere": "clear, bright",
            "camera_pov": "dashcam",
            "camera_movement": "fixed dashcam angle",
            "opening_hook_type": "collapse_already_started",
            "opening_hook_description": "Road surface visibly cracked and sinking at 0.5s, front car tilting into depression",
            "peak_moment": "Road section drops fully, car slides into hole, dust cloud erupts",
            "aftermath": "Dust settles, hole edge visible, car alarms in distance",
            "visual_tags": ["cracking_asphalt", "sinking_road", "tilting_car", "dust_cloud", "cracks_spreading"],
            "tone_tags": ["shock", "disbelief"],
            "dominant_colors": ["asphalt_gray", "dust_brown", "sky_blue"],
            "sound_atmosphere": "cracking concrete, grinding earth, distant car alarms",
        },
        {
            "category": "aviation_anomaly",
            "event_type": "low_flyover_residential",
            "scenario_summary": "Massive cargo plane belly fills the sky directly overhead a residential street. Single continuous shot looking up.",
            "location_style": "South American residential street near airport",
            "time_of_day": "afternoon",
            "weather_atmosphere": "clear, sunny, light wind",
            "camera_pov": "street",
            "camera_movement": "fixed vertical phone, looking up",
            "opening_hook_type": "massive_object_too_close",
            "opening_hook_description": "Giant aircraft underside already fills upper 70% of frame at 0.5s, shadow covers everything",
            "peak_moment": "Plane passes directly overhead, deafening roar, tree branches whip violently",
            "aftermath": "Plane recedes, shadow lifts, leaves and debris settle from the air",
            "visual_tags": ["plane_belly", "shadow_coverage", "tree_whipping", "residential_street", "scale_shock"],
            "tone_tags": ["awe", "fear", "disbelief"],
            "dominant_colors": ["aircraft_white", "shadow_dark", "sky_blue"],
            "sound_atmosphere": "overwhelming jet roar, wind blast, rattling windows",
        },
    ]


def _parse_candidates_json(raw: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

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
        if actual_ratio > target + 0.05:
            overused.append(cat_id)

    return overused
