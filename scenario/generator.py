"""
Two-stage construction timelapse scenario generator.

Stage 1: Select concept templates + camera styles
Stage 2: LLM generates concrete build scenarios per template

Generation order:
  1. Concept template is fixed (what makes people click)
  2. Construction process is designed (what's satisfying to watch)
  3. Before/after spaces are defined
  4. 15-second timelapse timeline is structured
"""
import json
import logging
import random

logger = logging.getLogger(__name__)

CANDIDATE_SYSTEM_PROMPT = """You are a creative director for a viral construction timelapse YouTube Shorts channel.
You create concepts for "luxury space transformation" videos in the style of rebornspacestv.

WHAT THESE VIDEOS ARE:
- 15-second construction timelapse showing a space being built/transformed
- Workers, heavy machinery, tools, materials are ALL visible
- High-speed time-lapse of real construction: digging, framing, pouring, tiling, finishing
- Ends with a luxury/hidden/amazing completed space
- "Could this really exist?" feeling - not pure fantasy, but aspirational

WHAT THESE VIDEOS ARE NOT:
- NOT disasters, accidents, or shock footage
- NOT magic/instant transformations (the BUILD PROCESS is the content)
- NOT just before/after photos
- NOT pure fantasy or sci-fi

CONSTRUCTION PROCESS IS KEY:
- Excavators digging, cranes lifting, trucks hauling
- Workers framing, welding, pouring concrete, laying tile
- Materials: wood, steel, glass, stone, concrete, soil
- The satisfaction comes from WATCHING things get built fast

TIME STRUCTURE (15 seconds):
  0-1s:   Before state clearly visible (boring/empty/ugly space)
  1-4s:   Construction begins (demolition, excavation, first work)
  4-10s:  Major build (structure, installation, big changes)
  10-15s: Finishing touches + completed reveal (lights on, water fills, door opens)

TARGET: Global audience. Locations should feel international."""

CANDIDATE_USER_PROMPT = """Generate exactly {count} construction timelapse scenario candidates.

CONCEPT TEMPLATE FOR THIS BATCH:
- Template: {template_label}
- Core idea: {template_description}
- Key process: {template_key_process}
- Satisfaction driver: {template_satisfaction}
- Camera style: {camera_label} - {camera_description}

CATEGORIES TO CONSIDER: {compatible_categories}
CATEGORIES TO AVOID (overused): {avoid_categories}

Each candidate MUST:
1. Have workers and/or machinery visible in the construction phase
2. Have a one-line concept that makes someone NEED to watch
3. Have a clear before → build process → after progression
4. End with something luxurious/hidden/amazing

Return ONLY a JSON array of {count} objects:
[
  {{
    "one_line_concept": "<one sentence that makes you click, e.g. 'Burying a shipping container to build an underground cinema'>",
    "category": "<one of: backyard_excavation, underground_container, hidden_under_pool, pool_conversion, garage_warehouse, rooftop_construction, pond_water_feature, container_luxury, narrow_space_build, exterior_normal_interior_unreal>",
    "construction_type": "<e.g. excavation_and_burial, surface_renovation, structural_framing, etc.>",
    "before_space": {{
      "type": "<e.g. flat_grass_backyard, empty_pool, rusty_garage>",
      "description": "<what it looks like at 0 seconds>",
      "visual": "<specific colors, textures, mood>"
    }},
    "construction_process": {{
      "stages": ["<stage 1: what happens first>", "<stage 2>", "<stage 3>", "<stage 4>", "<stage 5>"],
      "heavy_machinery": ["<list machines or empty>"],
      "worker_presence": "<high/medium/low>",
      "key_materials": ["<main materials used>"],
      "excavation_required": <true/false>
    }},
    "after_space": {{
      "type": "<e.g. underground_cinema, infinity_pool, secret_bar>",
      "description": "<what the finished space looks like>",
      "luxury_level": "<ultra/high/medium>",
      "water_element": <true/false>,
      "final_visual_hook": "<the single most impressive visual in the last 3 seconds>"
    }},
    "reveal_type": "<hidden_entrance_reveal / water_fill_reveal / lights_on_reveal / mechanical_reveal / walkthrough_reveal / aerial_reveal>",
    "time_structure": {{
      "0_1s": "<before state>",
      "1_4s": "<construction start>",
      "4_10s": "<main build>",
      "10_15s": "<finishing + reveal>"
    }},
    "camera_style": "{camera_id}",
    "location_feel": "<e.g. American suburban, Mediterranean coastal, Scandinavian forest>",
    "similarity_tags": ["<8-12 tags for dedup>"]
  }}
]

Make each candidate MAXIMALLY different: different spaces, different processes, different reveals."""


def generate_candidates(
    llm_client,
    config: dict,
    history: list[dict],
    category_stats: dict,
) -> list[dict]:
    """
    Two-stage candidate generation.

    Stage 1: Select concept templates, pair with camera styles
    Stage 2: LLM generates concrete scenarios per pair
    """
    gen_config = config.get("generation", {})
    llm_config = config.get("llm", {})
    categories_config = config.get("categories", {})

    templates = config.get("concept_templates", [])
    cameras = config.get("camera_styles", {})

    n_templates = gen_config.get("min_concept_templates", 10)
    n_per_template = gen_config.get("candidates_per_template", 3)

    # Stage 1: Select templates (prefer not recently used)
    selected = _select_templates(templates, history, count=n_templates)
    avoid_categories = _get_overused_categories(category_stats, categories_config)

    # Stage 2: Generate per template
    all_candidates = []

    for template in selected:
        # Pick a camera style
        camera_id, camera_cfg = _select_camera(cameras, template, history)

        prompt = CANDIDATE_USER_PROMPT.format(
            count=n_per_template,
            template_label=template.get("label", ""),
            template_description=template.get("description", ""),
            template_key_process=template.get("key_process", ""),
            template_satisfaction=template.get("satisfaction_driver", ""),
            camera_id=camera_id,
            camera_label=camera_cfg.get("label", ""),
            camera_description=camera_cfg.get("description", ""),
            compatible_categories=", ".join(categories_config.keys()),
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
            candidates = _parse_json(raw)

            for c in candidates:
                c["_concept_template"] = template["id"]
                c["camera_style"] = camera_id
                all_candidates.append(c)

            logger.info(
                "Template '%s' + camera '%s': %d candidates",
                template["id"], camera_id, len(candidates),
            )
        except Exception as e:
            logger.error("Generation failed for '%s': %s", template["id"], e)

    logger.info("Total candidates: %d", len(all_candidates))
    return all_candidates


def generate_candidates_dry(config: dict) -> list[dict]:
    """Hardcoded examples for offline testing."""
    return [
        {
            "one_line_concept": "Burying two shipping containers in the backyard to build an underground gym with a skylight",
            "category": "underground_container",
            "construction_type": "excavation_and_burial",
            "before_space": {
                "type": "suburban_backyard",
                "description": "Flat grass yard with wooden fence",
                "visual": "green grass, wood fence, boring suburban"
            },
            "construction_process": {
                "stages": [
                    "Excavator digs deep rectangular pit",
                    "Crane lowers two containers into pit",
                    "Workers weld containers together",
                    "Waterproofing and backfill",
                    "Interior: rubber floor, mirrors, LED strips, skylight"
                ],
                "heavy_machinery": ["excavator", "crane"],
                "worker_presence": "high",
                "key_materials": ["shipping_container", "soil", "rubber_mat", "LED_strips"],
                "excavation_required": True
            },
            "after_space": {
                "type": "underground_gym",
                "description": "Underground gym with mirrors, LEDs, skylight showing grass above",
                "luxury_level": "high",
                "water_element": False,
                "final_visual_hook": "Skylight view from inside gym showing normal yard above"
            },
            "reveal_type": "lights_on_reveal",
            "time_structure": {
                "0_1s": "Flat boring backyard",
                "1_4s": "Excavator rips into yard, huge pit",
                "4_10s": "Containers lowered, welded, backfilled, interior built",
                "10_15s": "Lights on: gym with mirrors, LEDs, skylight to grass above"
            },
            "camera_style": "crane_descend",
            "location_feel": "American suburban",
            "similarity_tags": ["container", "underground", "gym", "skylight", "excavation", "backyard", "mirror", "LED"],
            "_concept_template": "dig_and_bury",
        },
        {
            "one_line_concept": "Building a secret whiskey bar behind a rotating bookshelf in the basement",
            "category": "exterior_normal_interior_unreal",
            "construction_type": "hidden_mechanism_build",
            "before_space": {
                "type": "messy_basement",
                "description": "Cluttered basement with old shelving",
                "visual": "dim lighting, concrete walls, dusty shelves"
            },
            "construction_process": {
                "stages": [
                    "Clear out basement, demolish old wall",
                    "Build rotating door mechanism frame",
                    "Construct bookshelf facade with real books",
                    "Behind wall: brick, bar counter, whiskey shelves",
                    "Leather stools, ambient lighting, finishing"
                ],
                "heavy_machinery": [],
                "worker_presence": "medium",
                "key_materials": ["brick", "wood", "steel_bearings", "leather", "glass"],
                "excavation_required": False
            },
            "after_space": {
                "type": "secret_whiskey_bar",
                "description": "Brick-walled speakeasy bar with leather stools, warm lighting",
                "luxury_level": "ultra",
                "water_element": False,
                "final_visual_hook": "Push bookshelf, it rotates to reveal glowing bar behind"
            },
            "reveal_type": "mechanical_reveal",
            "time_structure": {
                "0_1s": "Normal basement with bookshelf",
                "1_4s": "Demolition, mechanism frame build",
                "4_10s": "Brick wall, bar counter, shelving, bookshelf facade",
                "10_15s": "Push bookshelf → rotates → warm glow of whiskey bar"
            },
            "camera_style": "walkthrough_reveal",
            "location_feel": "American East Coast brownstone",
            "similarity_tags": ["bookshelf", "secret_door", "whiskey", "bar", "basement", "brick", "rotating", "speakeasy"],
            "_concept_template": "hidden_behind",
        },
        {
            "one_line_concept": "Digging a natural swimming pond with underwater viewing window in the backyard",
            "category": "pond_water_feature",
            "construction_type": "excavation_and_burial",
            "before_space": {
                "type": "empty_yard",
                "description": "Wide empty backyard, just grass",
                "visual": "flat green, nothing special"
            },
            "construction_process": {
                "stages": [
                    "Excavator shapes pond basin",
                    "Lay waterproof liner, pile rocks for edges",
                    "Build wooden dock extending over water",
                    "Install acrylic panel in side wall for underwater view",
                    "Fill with water, add aquatic plants"
                ],
                "heavy_machinery": ["excavator"],
                "worker_presence": "high",
                "key_materials": ["pond_liner", "stone", "wood", "acrylic_panel", "aquatic_plants"],
                "excavation_required": True
            },
            "after_space": {
                "type": "swimming_pond_with_window",
                "description": "Natural pond, wooden dock, underground viewing window shows fish",
                "luxury_level": "high",
                "water_element": True,
                "final_visual_hook": "View from underwater window: fish swimming, sunlight filtering through"
            },
            "reveal_type": "water_fill_reveal",
            "time_structure": {
                "0_1s": "Empty grass yard",
                "1_4s": "Excavator carves pond shape",
                "4_10s": "Liner, rocks, dock, viewing window installed",
                "10_15s": "Water fills up, plants placed, underwater window view"
            },
            "camera_style": "drone_overhead",
            "location_feel": "Northern European countryside",
            "similarity_tags": ["pond", "swimming", "underwater_window", "dock", "excavation", "natural", "fish", "plants"],
            "_concept_template": "water_creation",
        },
    ]


def _select_templates(templates, history, count=10):
    """Select concept templates, preferring recently unused ones."""
    recent_ids = set()
    for entry in history[-10:]:
        sc = entry.get("scenario", {})
        recent_ids.add(sc.get("_concept_template", ""))

    fresh = [t for t in templates if t["id"] not in recent_ids]
    stale = [t for t in templates if t["id"] in recent_ids]
    random.shuffle(fresh)
    random.shuffle(stale)

    selected = fresh + stale
    while len(selected) < count:
        selected.append(random.choice(templates))
    return selected[:count]


def _select_camera(cameras, template, history):
    """Select camera style based on template and recent usage."""
    best_for_map = {}
    for cam_id, cam_cfg in cameras.items():
        best_for_map[cam_id] = cam_cfg

    # Simple selection: pick a camera not used in last 3
    recent_cams = []
    for entry in history[-3:]:
        sc = entry.get("scenario", {})
        recent_cams.append(sc.get("camera_style", ""))

    candidates = [(k, v) for k, v in cameras.items() if k not in recent_cams]
    if not candidates:
        candidates = list(cameras.items())

    random.shuffle(candidates)
    return candidates[0]


def _get_overused_categories(stats, categories_config):
    """Find over-represented categories."""
    if not stats:
        return []
    total = sum(s.get("count", 0) for s in stats.values())
    if total == 0:
        return []
    overused = []
    for cat_id, cat_stats in stats.items():
        actual = cat_stats.get("count", 0) / total
        target = categories_config.get(cat_id, {}).get("target_ratio", 0.10)
        if actual > target + 0.05:
            overused.append(cat_id)
    return overused


def _parse_json(raw):
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
        logger.error("Failed to parse JSON: %s", e)
        return []
