"""
Convert a selected scenario into a SINGLE video generation prompt.

1 video = 1 continuous clip = 10-15 seconds.
No multi-clip. No story. Fixed camera.
Also generates a single SFX prompt.
"""
import json
import logging

logger = logging.getLogger(__name__)

PROMPT_SYSTEM = """You are an expert at writing text-to-video prompts for AI video generation models.
You convert scenario descriptions into a SINGLE precise video prompt.

RULES:
- ONE continuous clip, 10-15 seconds, NO cuts
- Format: 9:16 vertical (phone recording)
- Prompt MUST start with "Photorealistic video footage of..."
- Camera is FIXED or near-fixed (natural tremor only, no panning/tracking)
- The anomaly MUST be visible from the very first frame
- Include: camera angle, lighting, weather, motion elements, atmospheric effects
- Describe what is VISIBLE and MOVING throughout the entire 10-15 seconds
- Use specific sensory details: colors, textures, materials
- NO text overlays, NO narration, NO dialogue
- Prompt must be 80-150 words for maximum quality
- Include motion keywords: "moving", "flowing", "shaking", "falling", "rushing"
- Include atmosphere: smoke, dust, mist, spray, debris particles
- Describe the PROGRESSION within the single shot:
  * What's visible immediately (frame 1)
  * How it intensifies over 5-10 seconds
  * What the final seconds look like"""

PROMPT_USER = """Convert this scenario into ONE single video prompt (10-15 second continuous shot).

SCENARIO:
{scenario_json}

Return ONLY a JSON object with this exact structure:
{{
  "video_prompt": "Photorealistic video footage of...",
  "sfx_prompt": "<ambient/environmental sound for this scene, 15-25 words>",
  "duration_seconds": {duration},
  "description": "<brief human-readable summary of the shot>"
}}

CRITICAL:
- ONE continuous shot, NO cuts, NO scene changes
- Camera POV: {pov} - {pov_traits}
- Camera is FIXED or near-fixed (minimal movement)
- Anomaly visible from FIRST FRAME: {hook_description}
- Peak moment at 5-10s: {peak_moment}
- Final seconds: {aftermath}
- The prompt must describe the ENTIRE 10-15 second progression in one paragraph"""


def build_video_prompt(
    scenario: dict,
    llm_client,
    config: dict,
) -> dict:
    """
    Generate a single video prompt from a scenario.

    Returns dict with video_prompt, sfx_prompt, duration_seconds, description.
    """
    gen_config = config.get("generation", {})
    llm_config = config.get("llm", {})
    duration = gen_config.get("duration_seconds", 14)

    povs = config.get("camera_povs", {})
    pov_id = scenario.get("camera_pov", "tourist")
    pov_cfg = povs.get(pov_id, {})

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
        pov=pov_id,
        pov_traits=pov_cfg.get("camera_traits", "fixed camera"),
        hook_description=scenario.get("opening_hook_description", ""),
        peak_moment=scenario.get("peak_moment", ""),
        aftermath=scenario.get("aftermath", ""),
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
    """Generate a single prompt without LLM (emergency fallback)."""
    gen_config = config.get("generation", {})
    duration = gen_config.get("duration_seconds", 14)

    pov = scenario.get("camera_pov", "tourist")
    povs = config.get("camera_povs", {})
    pov_traits = povs.get(pov, {}).get("camera_traits", "fixed camera")
    location = scenario.get("location_style", "urban area")
    weather = scenario.get("weather_atmosphere", "clear day")
    hook = scenario.get("opening_hook_description", "anomaly visible")
    peak = scenario.get("peak_moment", "event intensifies")
    aftermath = scenario.get("aftermath", "aftermath visible")
    sound = scenario.get("sound_atmosphere", "ambient atmosphere")

    video_prompt = (
        f"Photorealistic video footage, {pov_traits}, "
        f"shot in {location}, {weather}, 9:16 vertical phone recording. "
        f"Fixed camera position, single continuous {duration}-second shot. "
        f"From the very first frame: {hook}. "
        f"Over the next several seconds the event intensifies: {peak}. "
        f"In the final moments: {aftermath}. "
        f"Natural lighting, atmospheric particles visible, realistic motion throughout."
    )

    return {
        "video_prompt": video_prompt,
        "sfx_prompt": f"{sound}, continuous natural ambience, {duration} seconds",
        "duration_seconds": duration,
        "description": scenario.get("scenario_summary", ""),
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
