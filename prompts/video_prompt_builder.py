"""
Convert a selected construction timelapse scenario into video generation prompts.

1 video = 1 continuous 15-second timelapse clip.
Construction process visible: workers, machinery, materials.
"""
import json
import logging

logger = logging.getLogger(__name__)

PROMPT_SYSTEM = """You are an expert at writing text-to-video prompts for AI video generation models.
You convert construction timelapse scenario descriptions into a SINGLE precise video prompt.

RULES:
- ONE continuous 15-second timelapse clip, NO cuts
- Format: 9:16 vertical
- Prompt MUST start with "Construction timelapse video..."
- This is a TIME-LAPSE: everything moves at high speed
- Workers, machinery, and materials MUST be described as visible
- Include specific construction actions: digging, lifting, pouring, hammering, welding
- Describe the FULL progression: empty/boring space → active construction → luxury reveal
- Use specific sensory details: colors, textures, materials, lighting changes
- NO text overlays, NO narration, NO dialogue
- Prompt must be 80-150 words
- Include atmosphere keywords: dust, sparks, sunlight, shadows moving
- Camera style must match the specified camera type
- The REVEAL in the last 3-5 seconds should be visually stunning"""

PROMPT_USER = """Convert this construction timelapse scenario into ONE video prompt (15-second continuous timelapse).

SCENARIO:
{scenario_json}

Return ONLY a JSON object:
{{
  "video_prompt": "Construction timelapse video...",
  "sfx_prompt": "<ambient construction/reveal sound, 15-25 words>",
  "duration_seconds": {duration},
  "description": "<brief human-readable summary>"
}}

CRITICAL:
- ONE continuous timelapse, NO cuts, NO scene changes
- Camera style: {camera_style} - {camera_description}
- 0-1s:   Before state: {before_desc}
- 1-4s:   Construction start: {time_1_4s}
- 4-10s:  Main build: {time_4_10s}
- 10-15s: Finishing + reveal: {time_10_15s}
- Workers and/or machinery MUST be visible during construction phases
- The prompt must describe the ENTIRE 15-second timelapse progression in one paragraph"""


def build_video_prompt(
    scenario: dict,
    llm_client,
    config: dict,
) -> dict:
    """Generate a video prompt from a construction timelapse scenario."""
    gen_config = config.get("generation", {})
    llm_config = config.get("llm", {})
    duration = gen_config.get("duration_seconds", 15)

    cameras = config.get("camera_styles", {})
    cam_id = scenario.get("camera_style", "fixed_wide")
    cam_cfg = cameras.get(cam_id, {})

    time_struct = scenario.get("time_structure", {})

    scenario_json = json.dumps({
        k: v for k, v in scenario.items()
        if not k.startswith("_") and k not in (
            "buzz_score", "buzz_total", "adjusted_score",
            "category_bonus", "buzz_note",
        )
    }, indent=2)

    prompt = PROMPT_USER.format(
        scenario_json=scenario_json,
        duration=duration,
        camera_style=cam_id,
        camera_description=cam_cfg.get("description", "fixed camera"),
        before_desc=scenario.get("before_space", {}).get("description", ""),
        time_1_4s=time_struct.get("1_4s", "construction begins"),
        time_4_10s=time_struct.get("4_10s", "major build"),
        time_10_15s=time_struct.get("10_15s", "finishing + reveal"),
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

        if result and "video_prompt" in result:
            result.setdefault("duration_seconds", duration)
            return result

    except Exception as e:
        logger.error("Prompt generation failed: %s", e)

    return build_video_prompt_fallback(scenario, config)


def build_video_prompt_fallback(scenario: dict, config: dict) -> dict:
    """Generate prompt without LLM (fallback)."""
    gen_config = config.get("generation", {})
    duration = gen_config.get("duration_seconds", 15)

    cam_id = scenario.get("camera_style", "fixed_wide")
    cameras = config.get("camera_styles", {})
    cam_traits = cameras.get(cam_id, {}).get("traits", "fixed wide angle")
    location = scenario.get("location_feel", "suburban area")
    before = scenario.get("before_space", {})
    after = scenario.get("after_space", {})
    proc = scenario.get("construction_process", {})
    time_struct = scenario.get("time_structure", {})

    stages_text = ", ".join(proc.get("stages", ["construction proceeds"])[:3])
    machines = ", ".join(proc.get("heavy_machinery", [])) or "hand tools"
    materials = ", ".join(proc.get("key_materials", []))

    video_prompt = (
        f"Construction timelapse video, {cam_traits}, "
        f"set in {location}, 9:16 vertical format. "
        f"Single continuous {duration}-second timelapse shot. "
        f"Opens on {before.get('description', 'empty space')}, "
        f"{before.get('visual', 'dull and unremarkable')}. "
        f"Construction begins rapidly: {stages_text}. "
        f"Machinery visible: {machines}. Materials: {materials}. "
        f"Workers moving at high speed throughout. "
        f"Final reveal: {after.get('description', 'luxury finished space')}. "
        f"{after.get('final_visual_hook', 'Stunning completed space')}. "
        f"Dramatic lighting change at completion, photorealistic quality."
    )

    sfx_prompt = (
        f"Construction timelapse sounds: machinery, hammering, "
        f"power tools at high speed, then ambient reveal atmosphere, "
        f"{duration} seconds"
    )

    return {
        "video_prompt": video_prompt,
        "sfx_prompt": sfx_prompt,
        "duration_seconds": duration,
        "description": scenario.get("one_line_concept", ""),
    }


def build_video_prompt_dry(scenario: dict, config: dict) -> dict:
    """Dry-run version: no API call."""
    return build_video_prompt_fallback(scenario, config)


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
