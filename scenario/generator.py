"""
Two-stage construction scenario generator with deep rebornspacestv visual DNA.

Stage 1: Select concept templates + camera styles (deterministic)
Stage 2: LLM generates vivid, specific build scenarios per template

The LLM prompt is heavily engineered to produce scenarios that:
- Have VISUAL SPECIFICITY (not generic descriptions)
- Have NARRATIVE COHERENCE (before → process → reveal flows logically)
- Are MAXIMALLY DIVERSE (no two scenarios feel similar)
- Are BUILDABLE but ASPIRATIONAL (not fantasy, not boring)
"""
import json
import logging
import random

from prompts.style_guide import CONTENT_PILLARS, ANTI_PATTERNS

logger = logging.getLogger(__name__)

# ── SYSTEM PROMPT ────────────────────────────────────────────────
# This is the DNA of the entire channel baked into one prompt.

CANDIDATE_SYSTEM_PROMPT = """You are the creative mastermind behind a viral construction transformation YouTube Shorts channel.
Your videos get millions of views because you understand ONE thing better than anyone:
People cannot stop watching when they see an UGLY/BORING space become LUXURY through VISIBLE construction.

YOUR CHANNEL'S VISUAL IDENTITY (rebornspacestv style):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FIXED CAMERA: One camera angle, never moves. The transformation happens IN FRONT of you.
2. REAL CONSTRUCTION: Workers in vests, excavators, concrete trucks, welding sparks, dust clouds.
   The BUILD PROCESS is the content — not magic, not instant transformation.
3. TIME COMPRESSION: Shadows sweep across the ground. Hours pass in seconds. Workers blur with speed.
4. DRAMATIC REVEAL: At the end, lighting shifts (golden hour / interior lights turn on), and the
   finished space is BREATHTAKING. Viewers want to LIVE there.
5. SOUND DESIGN: Rhythmic construction sounds (hammering, drilling, pouring) → fade to atmospheric
   reveal ambience (water, warm hum, silence).

WHAT YOUR VIDEOS ARE:
- 15-second vertical clips (9:16)
- Before: boring/ugly/empty space
- Middle: intense construction timelapse with visible workers and machinery
- After: luxury space that makes viewers say "I NEED this"
- Camera: FIXED throughout. Same angle from start to finish.

WHAT YOUR VIDEOS ARE NOT:
- NOT magic transformations (the work must be visible)
- NOT before/after slideshows (the PROCESS is the content)
- NOT fantasy/sci-fi (must feel buildable even if ambitious)
- NOT generic (each video has a UNIQUE, specific concept)

YOUR SECRET SAUCE — WHAT GOES VIRAL:
1. "Wait... they're building WHAT?" — The concept must be surprising
2. "Woah, look at that excavator!" — Heavy machinery is visually impressive
3. "I want to live there" — The reveal must trigger desire
4. "How is that possible?" — Aspirational but feels real
5. "I have to show someone" — Shareability through uniqueness

CONTENT PILLARS (rotate between these):
""" + "\n".join(
    f"• {p['name']}: {p['description']}"
    for p in CONTENT_PILLARS
) + """

ANTI-PATTERNS (your videos NEVER do this):
""" + "\n".join(f"✗ {ap}" for ap in ANTI_PATTERNS[:6])

# ── USER PROMPT ──────────────────────────────────────────────────

CANDIDATE_USER_PROMPT = """Generate exactly {count} construction transformation scenario candidates.

CONCEPT DIRECTION FOR THIS BATCH:
- Template: {template_label}
- Core idea: {template_description}
- Key process: {template_key_process}
- Satisfaction driver: {template_satisfaction}
- Camera style: {camera_label} — {camera_description}

CONTENT PILLAR TO DRAW FROM: {pillar_name}
Pillar examples for INSPIRATION (do NOT copy these, invent NEW ones):
{pillar_examples}

CATEGORIES TO CONSIDER: {compatible_categories}
CATEGORIES TO AVOID (overused recently): {avoid_categories}

PREVIOUS CONCEPTS (do NOT repeat or closely resemble):
{recent_concepts}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each candidate MUST have:
1. A one_line_concept that makes someone STOP SCROLLING
   Bad: "Building a pool in the backyard" (boring, generic)
   Good: "Cutting a hole in the rooftop to install a glass-bottom pool with a living room view below"

2. A visually SPECIFIC before_space — not "empty yard" but exact colors, textures, objects
   Bad: "empty room" | Good: "cracked beige tile floor, water-stained drop ceiling, flickering fluorescent tube"

3. A construction_process with 5 stages that feel VISUALLY SATISFYING in timelapse
   Each stage must describe a VISIBLE action (what the viewer SEES), not abstract work

4. An after_space with SPECIFIC luxury details — not "luxury room" but exact materials and atmosphere
   Bad: "beautiful pool" | Good: "infinity-edge pool with dark basalt coping, turquoise water catching sunset"

5. A time_structure describing what EXACTLY is visible at each time mark

Return ONLY a JSON array of {count} objects:
[
  {{
    "one_line_concept": "<one clickbait sentence — specific action + surprising result>",
    "category": "<one of the categories listed above>",
    "construction_type": "<excavation_and_burial|surface_renovation|structural_framing|container_modification|mechanical_installation|waterproofing_and_tiling|interior_finishing|landscape_and_planting|demolition_and_rebuild|hidden_mechanism_build>",
    "before_space": {{
      "type": "<specific space identifier>",
      "description": "<what it looks like at second 0 — be SPECIFIC: colors, textures, objects, state of decay>",
      "visual": "<exact color palette, mood, lighting — think cinematography>"
    }},
    "construction_process": {{
      "stages": [
        "<stage 1: what the viewer SEES — workers doing what with what tool?>",
        "<stage 2: what visible change happens? what machinery is operating?>",
        "<stage 3: what structure/material is being installed?>",
        "<stage 4: what finishing work is visible?>",
        "<stage 5: what is the LAST visible action before reveal?>"
      ],
      "heavy_machinery": ["<specific machines — excavator, mini crane, concrete mixer, etc.>"],
      "worker_presence": "<high|medium|low>",
      "key_materials": ["<specific materials with color — 'honey-toned cedar planks' not just 'wood'>"],
      "excavation_required": true/false
    }},
    "after_space": {{
      "type": "<specific finished space identifier>",
      "description": "<what the finished space looks like — SPECIFIC luxury details, colors, textures>",
      "luxury_level": "<ultra|high|medium>",
      "water_element": true/false,
      "final_visual_hook": "<the SINGLE most impressive visual in the last 3 seconds — be cinematic>"
    }},
    "reveal_type": "<hidden_entrance_reveal|water_fill_reveal|lights_on_reveal|mechanical_reveal|walkthrough_reveal|aerial_reveal>",
    "time_structure": {{
      "0_1s": "<what is visible — specific visual description>",
      "1_4s": "<what construction action starts — specific>",
      "4_10s": "<what major changes are visible — specific>",
      "10_15s": "<what reveal moment happens — specific>"
    }},
    "camera_style": "{camera_id}",
    "location_feel": "<specific location vibe — 'Los Angeles hillside with palm trees' not just 'suburban'>",
    "similarity_tags": ["<8-12 specific tags for deduplication>"]
  }}
]

Make each candidate WILDLY different from each other:
- Different locations (different countries/climates)
- Different before-states (don't repeat the same "empty backyard")
- Different construction methods (not all excavation)
- Different reveal moments (not all "lights turn on")
- Different luxury aesthetics (not all "modern minimalist")"""


def generate_candidates(
    llm_client,
    config: dict,
    history: list[dict],
    category_stats: dict,
) -> list[dict]:
    """
    Two-stage candidate generation with content pillar rotation.

    Stage 1: Select concept templates, pair with camera styles and content pillars
    Stage 2: LLM generates vivid concrete scenarios per combination
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

    # Get recent concepts to avoid repetition
    recent_concepts = _get_recent_concepts(history, count=15)

    # Track which content pillars we've used this run
    used_pillars = []

    # Stage 2: Generate per template
    all_candidates = []

    for template in selected:
        # Pick a camera style
        camera_id, camera_cfg = _select_camera(cameras, template, history)

        # Pick a content pillar (rotate through them)
        pillar = _select_pillar(used_pillars)
        used_pillars.append(pillar["id"])

        prompt = CANDIDATE_USER_PROMPT.format(
            count=n_per_template,
            template_label=template.get("label", ""),
            template_description=template.get("description", ""),
            template_key_process=template.get("key_process", ""),
            template_satisfaction=template.get("satisfaction_driver", ""),
            camera_id=camera_id,
            camera_label=camera_cfg.get("label", ""),
            camera_description=camera_cfg.get("description", ""),
            pillar_name=pillar["name"],
            pillar_examples="\n".join(f"  - {ex}" for ex in pillar["examples"]),
            compatible_categories=", ".join(categories_config.keys()),
            avoid_categories=", ".join(avoid_categories) if avoid_categories else "none",
            recent_concepts="\n".join(f"  • {c}" for c in recent_concepts) if recent_concepts else "(none yet)",
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
                c["_content_pillar"] = pillar["id"]
                c["camera_style"] = camera_id
                all_candidates.append(c)

            logger.info(
                "Template '%s' + pillar '%s' + camera '%s': %d candidates",
                template["id"], pillar["id"], camera_id, len(candidates),
            )
        except Exception as e:
            logger.error("Generation failed for '%s': %s", template["id"], e)

    logger.info("Total candidates: %d", len(all_candidates))
    return all_candidates


def generate_candidates_dry(config: dict) -> list[dict]:
    """Hardcoded examples with rich visual detail for offline testing."""
    return [
        {
            "one_line_concept": "Cutting a rectangular hole in a suburban rooftop to lower in a glass-bottom infinity pool visible from the living room below",
            "category": "rooftop_construction",
            "construction_type": "structural_framing",
            "before_space": {
                "type": "flat_gravel_rooftop",
                "description": "Grey gravel-covered flat rooftop with rusted HVAC units, pigeon droppings, and cracked tar patches. Chain-link safety railing around edges.",
                "visual": "desaturated grey and brown, harsh midday sun, industrial ugliness, city skyline in background hazy with smog"
            },
            "construction_process": {
                "stages": [
                    "Workers in orange vests chalk outline on rooftop, diamond-blade saw cuts through concrete deck sending dust clouds skyward",
                    "Mini crane hoists pre-fabricated steel pool frame through the new opening, workers guide it with ropes",
                    "Welders attach frame to building structure, sparks cascade down through the opening, waterproofing membrane rolled across edges",
                    "Glass-bottom panels carefully lowered and sealed, dark basalt coping stones mortared around perimeter",
                    "Teak decking laid around pool, brushed-steel railing installed, water slowly fills — turquoise against grey city"
                ],
                "heavy_machinery": ["mini crane", "diamond concrete saw", "welding rig"],
                "worker_presence": "high",
                "key_materials": ["reinforced steel frame", "tempered glass panels", "dark basalt coping", "teak decking", "waterproofing membrane"],
                "excavation_required": False
            },
            "after_space": {
                "type": "rooftop_glass_bottom_pool",
                "description": "Infinity-edge rooftop pool with dark basalt surround, turquoise water, teak deck with loungers, city skyline panorama. Glass bottom shows living room chandelier below.",
                "luxury_level": "ultra",
                "water_element": True,
                "final_visual_hook": "Camera holds on pool as sunset paints water gold, then subtle tilt down reveals living room chandelier glowing through glass bottom"
            },
            "reveal_type": "water_fill_reveal",
            "time_structure": {
                "0_1s": "Ugly gravel rooftop with rusted HVAC, harsh flat light",
                "1_4s": "Concrete saw cuts rectangle, dust billows, crane arrives",
                "4_10s": "Steel frame lowered, welding sparks, glass panels sealed, basalt placed",
                "10_15s": "Water fills to turquoise, sunset hits, glass bottom reveals living room below"
            },
            "camera_style": "drone_overhead",
            "location_feel": "Manhattan-style dense urban skyline, hazy afternoon light",
            "similarity_tags": ["rooftop", "glass_bottom", "pool", "infinity_edge", "urban", "skyline", "crane", "basalt", "teak", "sunset"],
            "_concept_template": "stack_and_build_up",
        },
        {
            "one_line_concept": "Excavating beneath a crumbling garden shed to build a hidden Japanese onsen bath with natural volcanic stone",
            "category": "underground_container",
            "construction_type": "excavation_and_burial",
            "before_space": {
                "type": "old_garden_shed",
                "description": "Weathered wooden garden shed with peeling green paint, cobwebbed windows, rusty tools hanging on walls. Overgrown ivy climbing one side.",
                "visual": "muted greens and browns, dappled forest light through overhead maple leaves, nostalgic but neglected"
            },
            "construction_process": {
                "stages": [
                    "Workers pull up shed floorboards, compact excavator squeezes through doorway and begins digging downward, soil hauled out in buckets",
                    "Concrete foundation walls poured in underground chamber, waterproofing painted on, drainage pipes laid in gravel bed",
                    "Natural volcanic basalt stones carefully placed to form soaking tub basin, copper hot water pipes threaded through walls",
                    "Cedar plank walls and ceiling installed, recessed warm LED strips hidden in joints, bamboo privacy screen at entrance",
                    "Tub filled with steaming water, river stones placed around edge, single bonsai on cedar shelf, steam rises through shed above"
                ],
                "heavy_machinery": ["compact excavator"],
                "worker_presence": "medium",
                "key_materials": ["volcanic basalt stone", "western red cedar planks", "copper piping", "warm LED strips", "river stones", "bamboo"],
                "excavation_required": True
            },
            "after_space": {
                "type": "hidden_underground_onsen",
                "description": "Traditional Japanese onsen bath beneath garden shed — volcanic stone soaking tub, cedar walls with warm LED glow, bamboo accents, steam rising. Accessed through trapdoor in shed floor.",
                "luxury_level": "ultra",
                "water_element": True,
                "final_visual_hook": "Steam rises through the shed floorboards above, camera descends through trapdoor to reveal warm cedar-and-stone onsen glowing amber below"
            },
            "reveal_type": "hidden_entrance_reveal",
            "time_structure": {
                "0_1s": "Old garden shed, peeling paint, cobwebs, dappled light through leaves",
                "1_4s": "Floorboards ripped up, excavator digs down, soil hauled up in buckets",
                "4_10s": "Underground chamber formed, basalt tub built, cedar walls installed",
                "10_15s": "Hot water fills tub, steam rises, warm amber LED glow, bonsai placed on shelf"
            },
            "camera_style": "crane_descend",
            "location_feel": "Pacific Northwest forest clearing, moss and ferns, overcast filtered light",
            "similarity_tags": ["onsen", "underground", "japanese", "cedar", "volcanic_stone", "steam", "hidden", "garden_shed", "trapdoor", "bonsai"],
            "_concept_template": "dig_and_bury",
        },
        {
            "one_line_concept": "Converting a rusted 1960s Airstream trailer into a mirror-clad minimalist recording studio in the Nevada desert",
            "category": "container_luxury",
            "construction_type": "container_modification",
            "before_space": {
                "type": "abandoned_airstream",
                "description": "Dented silver Airstream trailer with oxidized aluminum skin, flat tires, broken windows stuffed with newspaper. Parked on cracked desert hardpan.",
                "visual": "bleached desert palette — pale sand, oxidized silver, washed-out blue sky, heat shimmer on horizon"
            },
            "construction_process": {
                "stages": [
                    "Workers strip interior to bare shell, angle grinder sparks fly cutting out rusted panels, sandblaster strips oxidation from exterior",
                    "New structural ribs welded inside, spray foam insulation fills cavities, acoustic isolation panels bolted to frame",
                    "Exterior wrapped in mirror-polished stainless steel panels, reflecting desert landscape in distorted curves",
                    "Interior: charcoal acoustic fabric walls, floating birch plywood desk, studio monitors mounted, cable management routed through floor",
                    "Final: amber pendant lights hung, vocal booth glass installed, exterior mirrors catch sunset making trailer glow like liquid gold"
                ],
                "heavy_machinery": ["angle grinder", "sandblaster", "welding rig"],
                "worker_presence": "medium",
                "key_materials": ["mirror-polished stainless steel", "acoustic foam panels", "charcoal fabric", "birch plywood", "amber glass pendants"],
                "excavation_required": False
            },
            "after_space": {
                "type": "desert_mirror_studio",
                "description": "Mirror-clad Airstream reflecting infinite desert. Inside: intimate recording studio with charcoal acoustic walls, birch desk, warm amber lighting. Outside disappears in reflections.",
                "luxury_level": "high",
                "water_element": False,
                "final_visual_hook": "Sunset hits mirror exterior — entire trailer becomes a glowing gold sculpture in the desert, then door opens revealing warm amber studio interior"
            },
            "reveal_type": "lights_on_reveal",
            "time_structure": {
                "0_1s": "Rusted Airstream on cracked desert floor, heat shimmer, bleached landscape",
                "1_4s": "Grinder sparks, sandblasting clouds, interior stripped bare",
                "4_10s": "Mirror panels mounted, acoustic interior built, equipment installed",
                "10_15s": "Sunset turns mirrors to gold, door opens, warm amber studio interior revealed"
            },
            "camera_style": "fixed_wide",
            "location_feel": "Nevada high desert, Joshua trees, vast empty sky, dramatic golden hour light",
            "similarity_tags": ["airstream", "desert", "mirror", "recording_studio", "acoustic", "renovation", "stainless_steel", "sunset", "minimalist"],
            "_concept_template": "convert_vehicle",
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


def _select_pillar(used_pillars: list[str]) -> dict:
    """Select a content pillar, rotating through them evenly."""
    # Count usage
    usage = {}
    for pid in used_pillars:
        usage[pid] = usage.get(pid, 0) + 1

    # Find least-used pillars
    min_usage = min((usage.get(p["id"], 0) for p in CONTENT_PILLARS), default=0)
    candidates = [p for p in CONTENT_PILLARS if usage.get(p["id"], 0) == min_usage]

    return random.choice(candidates)


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


def _get_recent_concepts(history: list[dict], count: int = 15) -> list[str]:
    """Extract recent one_line_concepts from history for dedup."""
    concepts = []
    for entry in history[-count:]:
        sc = entry.get("scenario", {})
        concept = sc.get("one_line_concept", "")
        if concept:
            concepts.append(concept)
    return concepts


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
