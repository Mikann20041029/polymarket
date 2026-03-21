"""
Multi-stage video prompt builder for rebornspacestv-style construction transformations.

Generates 3 separate prompts (before → construction → reveal), each for a 5-second clip.
Clips are stitched together in post-production for a coherent 15-second video.

KEY PRINCIPLE: Each prompt describes ONE continuous 5-second shot from a FIXED camera.
No cuts, no angle changes. The AI model only needs to animate one clear scene.
"""
import json
import logging

from prompts.style_guide import (
    CAMERA_RULES,
    LIGHTING_RULES,
    TIMELAPSE_PHYSICS,
    STAGE_STRUCTURE,
    PROMPT_RULES,
    SFX_STYLE,
    validate_prompt,
)

logger = logging.getLogger(__name__)

# ── SYSTEM PROMPT ────────────────────────────────────────────────
# This teaches the LLM how to write prompts that AI video models understand.

PROMPT_SYSTEM = """You write text-to-video prompts for AI video generation models (Wan 2.1, Kling).
You produce prompts that generate COHERENT, PHOTOREALISTIC construction transformation clips.

CRITICAL RULES FOR AI VIDEO MODELS:
1. Each prompt = ONE continuous 5-second shot. NO cuts, NO scene changes.
2. Camera is FIXED. Describe the angle once. It NEVER moves (except subtle push-in for reveal).
3. NEVER use words: "timelapse", "fast-forward", "transition", "cut to", "next scene", "suddenly".
   Instead describe the SPEED: "workers move at rapid pace", "shadows sweep across ground".
4. Be SPECIFIC about what is VISIBLE: materials, colors, textures, people, tools.
5. Be SPECIFIC about ATMOSPHERE: light direction, dust particles, shadow angle, weather.
6. Do NOT use vague praise words: "beautiful", "stunning", "amazing". Describe WHAT MAKES IT SO.
7. Each prompt must be 60-120 words. More = incoherent. Less = generic.
8. Describe a SINGLE coherent visual scene, not a list of features.

WHAT MAKES GOOD PROMPTS:
- "Fixed overhead shot, warm afternoon sun casting long shadows across suburban backyard.
   Two workers in orange vests rapidly dig a rectangular pit with shovels, dark rich soil
   piling up around edges. Yellow excavator in background swings bucket. Dust particles
   float in golden sunlight. Fresh timber beams stacked nearby."

WHAT MAKES BAD PROMPTS:
- "Construction timelapse video showing workers building a pool with materials including
   concrete, wood, glass. The pool is luxury and has water. Beautiful final result."
"""

PROMPT_USER_TEMPLATE = """Generate 3 video prompts for a construction transformation video.
Each prompt is for ONE 5-second clip. They will be played back-to-back to form a 15-second video.

SCENARIO:
{scenario_json}

CAMERA SETUP (must be IDENTICAL across all 3 prompts):
- Angle: {camera_angle}
- Position: {camera_position}

Return ONLY a JSON object with this structure:
{{
  "camera_description": "<one sentence describing the exact camera angle/position used for ALL clips>",
  "stage_1_before": {{
    "video_prompt": "<60-120 word prompt for the BEFORE state + first construction activity>",
    "sfx_prompt": "<15-25 word ambient sound description>"
  }},
  "stage_2_construction": {{
    "video_prompt": "<60-120 word prompt for MAIN construction activity — same camera>",
    "sfx_prompt": "<15-25 word construction sound description>"
  }},
  "stage_3_reveal": {{
    "video_prompt": "<60-120 word prompt for FINISHING + luxury reveal — same camera, lighting shift>",
    "sfx_prompt": "<15-25 word reveal atmosphere sound description>"
  }},
  "description": "<one sentence human summary>"
}}

RULES:
- Camera angle in all 3 prompts MUST match camera_description exactly
- Stage 1: Show the ugly/empty before-state, then first signs of work beginning
- Stage 2: Active construction — workers, tools, materials, dust, shadows moving
- Stage 3: Final touches complete, dramatic lighting shift, the luxury space is revealed
- NEVER use: "timelapse", "fast-forward", "transition", "cut to", "suddenly", "beautiful", "stunning"
- Workers MUST be visible in stages 1 and 2
- Describe specific materials by name and color
- Include atmosphere: dust, sunlight, shadows, particles
- Each prompt is a SINGLE continuous 5-second shot — no cuts within a prompt"""


def build_video_prompt(
    scenario: dict,
    llm_client,
    config: dict,
) -> dict:
    """Generate 3-stage video prompts from a construction scenario."""
    gen_config = config.get("generation", {})
    llm_config = config.get("llm", {})

    cameras = config.get("camera_styles", {})
    cam_id = scenario.get("camera_style", "fixed_wide")
    cam_cfg = cameras.get(cam_id, {})

    # Build camera description for prompt
    camera_angle = cam_cfg.get("description", "fixed wide angle shot")
    camera_position = cam_cfg.get("traits", "stable, everything visible")

    scenario_json = json.dumps({
        k: v for k, v in scenario.items()
        if not k.startswith("_") and k not in (
            "buzz_score", "buzz_total", "adjusted_score",
            "category_bonus", "buzz_note",
        )
    }, indent=2)

    prompt = PROMPT_USER_TEMPLATE.format(
        scenario_json=scenario_json,
        camera_angle=camera_angle,
        camera_position=camera_position,
    )

    try:
        response = llm_client.chat.completions.create(
            model=llm_config.get("model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=llm_config.get("max_tokens", 4096),
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        result = _parse_prompt_json(raw)

        if result and "stage_1_before" in result:
            # Validate each stage prompt
            for stage_key in ["stage_1_before", "stage_2_construction", "stage_3_reveal"]:
                stage = result.get(stage_key, {})
                vp = stage.get("video_prompt", "")
                violations = validate_prompt(vp)
                if violations:
                    logger.warning("Prompt violations in %s: %s", stage_key, violations)

            # Convert to pipeline-compatible format
            return _format_multi_stage_result(result, scenario, config)

    except Exception as e:
        logger.error("Prompt generation failed: %s", e)

    return build_video_prompt_fallback(scenario, config)


def build_video_prompt_fallback(scenario: dict, config: dict) -> dict:
    """Generate high-quality multi-stage prompts WITHOUT LLM."""
    cam_id = scenario.get("camera_style", "fixed_wide")
    cameras = config.get("camera_styles", {})
    cam_cfg = cameras.get(cam_id, {})
    cam_traits = cam_cfg.get("traits", "fixed wide angle")
    cam_desc = cam_cfg.get("description", "fixed wide angle shot")

    location = scenario.get("location_feel", "suburban area")
    before = scenario.get("before_space", {})
    after = scenario.get("after_space", {})
    proc = scenario.get("construction_process", {})
    time_struct = scenario.get("time_structure", {})

    stages = proc.get("stages", ["construction proceeds"])
    machines = proc.get("heavy_machinery", [])
    materials = proc.get("key_materials", [])
    workers = proc.get("worker_presence", "medium")

    # Map worker presence to visual description
    worker_desc = {
        "high": "team of four workers in safety vests moving at rapid pace",
        "medium": "two workers in work clothes moving quickly",
        "low": "single worker methodically working",
    }.get(workers, "workers visible")

    machine_desc = ", ".join(machines) if machines else "hand tools and power drills"
    material_colors = _infer_material_colors(materials)

    # Camera description consistent across all stages
    camera_line = f"Fixed {cam_desc}, {cam_traits}, 9:16 vertical format, {location}"

    # STAGE 1: Before + first work
    before_desc = before.get("description", "empty unremarkable space")
    before_visual = before.get("visual", "dull colors, nothing noteworthy")
    first_work = stages[0] if stages else "construction begins"

    stage_1_prompt = (
        f"{camera_line}. "
        f"Warm afternoon sunlight, long shadows on ground. "
        f"{before_desc}, {before_visual}. "
        f"Space is still and empty for a moment. "
        f"Then {worker_desc} arrive with {machine_desc}. "
        f"{first_work}. "
        f"Dust rises from first impact, golden light catches floating particles."
    )

    # STAGE 2: Main construction
    mid_stages = stages[1:4] if len(stages) > 1 else ["structure takes shape"]
    mid_text = ". ".join(mid_stages)

    stage_2_prompt = (
        f"{camera_line}. "
        f"Same angle, construction well underway. "
        f"{worker_desc} in constant rapid motion. "
        f"{mid_text}. "
        f"{material_colors} visible across the workspace. "
        f"Shadows sweep across the ground indicating hours passing. "
        f"Dust and activity fill the frame, {machine_desc} operating."
    )

    # STAGE 3: Reveal
    after_desc = after.get("description", "luxury finished space")
    hook = after.get("final_visual_hook", "completed space glows with warm light")
    luxury = after.get("luxury_level", "high")
    has_water = after.get("water_element", False)

    lighting_shift = "golden sunset light floods the completed space"
    if has_water:
        lighting_shift = "water catches golden reflections as the sun lowers"
    if luxury == "ultra":
        lighting_shift = "warm interior lighting contrasts with blue-hour sky outside"

    final_stage = stages[-1] if stages else "finishing touches complete"

    stage_3_prompt = (
        f"{camera_line}. "
        f"Same angle, {final_stage}. "
        f"Last worker places final element and steps away. "
        f"{after_desc}. "
        f"{lighting_shift}. "
        f"{hook}. "
        f"Every surface catches light, space feels alive and inviting."
    )

    # SFX per stage
    sfx_1 = (
        "gentle outdoor ambience, birdsong, light wind, then first metallic clang "
        "of tools, shovel piercing earth, distant truck engine"
    )
    sfx_2 = (
        "rapid rhythmic construction sounds layered, power drill whirring, "
        "hammering in quick succession, concrete pouring, metallic clangs"
    )
    if has_water:
        sfx_3 = (
            "construction fading, replaced by water rushing and filling, "
            "gentle splashing, warm ambient hum, serene atmosphere"
        )
    else:
        sfx_3 = (
            "construction sounds fading to silence, then warm atmospheric hum, "
            "soft ambient light buzz, distant evening sounds, satisfying click"
        )

    return {
        "stages": [
            {
                "stage": 1,
                "name": "before",
                "video_prompt": stage_1_prompt,
                "sfx_prompt": sfx_1,
                "duration_seconds": 5,
            },
            {
                "stage": 2,
                "name": "construction",
                "video_prompt": stage_2_prompt,
                "sfx_prompt": sfx_2,
                "duration_seconds": 5,
            },
            {
                "stage": 3,
                "name": "reveal",
                "video_prompt": stage_3_prompt,
                "sfx_prompt": sfx_3,
                "duration_seconds": 5,
            },
        ],
        "camera_description": camera_line,
        "total_duration_seconds": 15,
        "description": scenario.get("one_line_concept", ""),
        # Legacy compatibility
        "video_prompt": stage_1_prompt,
        "sfx_prompt": sfx_1,
        "duration_seconds": 15,
    }


def build_video_prompt_dry(scenario: dict, config: dict) -> dict:
    """Dry-run version: no API call, uses fallback."""
    return build_video_prompt_fallback(scenario, config)


def _format_multi_stage_result(llm_result: dict, scenario: dict, config: dict) -> dict:
    """Convert LLM output to pipeline-compatible multi-stage format."""
    stages = []
    for i, key in enumerate(["stage_1_before", "stage_2_construction", "stage_3_reveal"], 1):
        stage_data = llm_result.get(key, {})
        stages.append({
            "stage": i,
            "name": ["before", "construction", "reveal"][i - 1],
            "video_prompt": stage_data.get("video_prompt", ""),
            "sfx_prompt": stage_data.get("sfx_prompt", ""),
            "duration_seconds": 5,
        })

    first_prompt = stages[0]["video_prompt"] if stages else ""
    first_sfx = stages[0]["sfx_prompt"] if stages else ""

    return {
        "stages": stages,
        "camera_description": llm_result.get("camera_description", ""),
        "total_duration_seconds": 15,
        "description": llm_result.get("description", scenario.get("one_line_concept", "")),
        # Legacy compatibility
        "video_prompt": first_prompt,
        "sfx_prompt": first_sfx,
        "duration_seconds": 15,
    }


def _infer_material_colors(materials: list[str]) -> str:
    """Convert material names to visual color descriptions."""
    color_map = {
        "concrete": "grey concrete",
        "wood": "warm honey-toned timber",
        "steel": "silver steel beams",
        "brick": "red-brown brick",
        "glass": "clear glass panels",
        "stone": "natural grey stone",
        "marble": "white marble slabs",
        "copper": "warm copper fixtures",
        "tile": "ceramic tile",
        "soil": "dark rich soil",
        "sand": "golden sand",
        "gravel": "pale gravel",
        "rubber_mat": "black rubber matting",
        "LED_strips": "LED strip lighting",
        "pond_liner": "dark pond liner",
        "acrylic_panel": "clear acrylic panels",
        "shipping_container": "corrugated steel container",
        "leather": "dark leather upholstery",
        "bamboo": "green bamboo stalks",
    }

    descriptions = []
    for mat in materials[:4]:  # Max 4 for brevity
        clean = mat.lower().strip()
        desc = color_map.get(clean, clean)
        descriptions.append(desc)

    return ", ".join(descriptions) if descriptions else "various construction materials"


def _parse_prompt_json(raw: str) -> dict | None:
    """Extract JSON from LLM response."""
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
        logger.error("Failed to parse prompt JSON: %s", e)
        return None
