"""
Convert a selected scenario into per-clip video generation prompts.

Each clip prompt is optimized for text-to-video models (Wan 2.1, Runway, etc.).
Also generates SFX prompts for each clip.
"""
import json
import logging

logger = logging.getLogger(__name__)

PROMPT_SYSTEM = """You are an expert at writing text-to-video prompts for AI video generation models.
You convert scenario descriptions into precise, per-clip prompts.

RULES:
- Each clip is exactly 5 seconds of footage
- Format: 9:16 vertical (phone recording)
- Every prompt MUST start with "Photorealistic video footage of..."
- Include: camera angle, camera movement, lighting, weather, motion elements
- Describe what is VISIBLE and MOVING, not abstract concepts
- Use specific sensory details: colors, textures, materials, atmospheric effects
- Camera behavior must match the POV style (handheld shake, dashcam stability, etc.)
- NO text overlays, NO narration, NO dialogue in prompts
- Clip 1 = THE HOOK (most visually shocking moment, the anomaly is ALREADY visible)
- Clips 2-4 = ESCALATION (situation worsens progressively)
- Clips 5-6 = CLIMAX + AFTERMATH
- Each prompt must be 60-100 words for maximum quality
- Include motion keywords: "moving", "flowing", "shaking", "falling", "rushing"
- Include atmosphere: smoke, dust, mist, spray, embers, debris particles"""

PROMPT_USER = """Convert this scenario into {clip_count} clip prompts (5 seconds each).

SCENARIO:
{scenario_json}

Return ONLY a JSON object with this exact structure:
{{
  "clips": [
    {{
      "clip_number": 1,
      "video_prompt": "Photorealistic video footage of...",
      "sfx_prompt": "<ambient sound for this specific clip, 10-20 words>",
      "description": "<brief human-readable description of what happens>"
    }},
    ...
  ],
  "total_duration_seconds": {total_duration}
}}

CRITICAL:
- Clip 1 MUST open with the hook already in progress: {hook_description}
- Camera POV: {pov} with traits: {pov_traits}
- Maintain visual continuity between clips (same location, lighting, weather)
- Each prompt 60-100 words minimum"""


def build_video_prompts(
    scenario: dict,
    llm_client,
    config: dict,
) -> dict:
    """
    Generate per-clip video and SFX prompts from a scenario.

    Args:
        scenario: selected scenario dict
        llm_client: OpenAI-compatible client
        config: full config dict

    Returns:
        dict with 'clips' list and metadata
    """
    gen_config = config.get("generation", {})
    llm_config = config.get("llm", {})

    clip_count = gen_config.get("clips_per_video", 6)
    clip_duration = gen_config.get("clip_duration_seconds", 5)
    total_duration = clip_count * clip_duration

    # Load POV traits
    povs = config.get("camera_povs", {})
    pov_id = scenario.get("camera_pov", "tourist")
    pov_cfg = povs.get(pov_id, {})

    scenario_json = json.dumps({
        k: v for k, v in scenario.items()
        if not k.startswith("_") and k not in ("buzz_score", "buzz_total", "adjusted_score", "category_bonus", "buzz_note")
    }, indent=2)

    prompt = PROMPT_USER.format(
        clip_count=clip_count,
        scenario_json=scenario_json,
        total_duration=total_duration,
        hook_description=scenario.get("opening_hook_description", ""),
        pov=pov_id,
        pov_traits=pov_cfg.get("camera_traits", "handheld phone camera"),
    )

    try:
        response = llm_client.chat.completions.create(
            model=llm_config.get("model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=llm_config.get("max_tokens", 8192),
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        result = _parse_prompt_json(raw)

        if result and "clips" in result:
            # Validate clip count
            if len(result["clips"]) != clip_count:
                logger.warning(
                    "Expected %d clips, got %d", clip_count, len(result["clips"])
                )
            return result

    except Exception as e:
        logger.error("Prompt generation failed: %s", e)

    # Fallback: generate simple prompts without LLM
    return build_video_prompts_fallback(scenario, config)


def build_video_prompts_fallback(scenario: dict, config: dict) -> dict:
    """Generate basic prompts without LLM (emergency fallback)."""
    gen_config = config.get("generation", {})
    clip_count = gen_config.get("clips_per_video", 6)

    pov = scenario.get("camera_pov", "handheld")
    povs = config.get("camera_povs", {})
    pov_traits = povs.get(pov, {}).get("camera_traits", "handheld phone camera")
    location = scenario.get("location_style", "urban area")
    weather = scenario.get("weather_atmosphere", "clear day")
    hook = scenario.get("opening_hook_description", "anomaly visible")
    escalation = scenario.get("escalation_pattern", "situation worsens")
    climax = scenario.get("climax_description", "peak moment")
    aftermath = scenario.get("aftermath_description", "aftermath visible")
    sound = scenario.get("sound_atmosphere", "ambient atmosphere")

    base = (
        f"Photorealistic video footage, {pov_traits}, "
        f"shot in {location}, {weather}, 9:16 vertical phone recording"
    )

    clips = []
    for i in range(clip_count):
        if i == 0:
            desc = f"HOOK: {hook}"
            vprompt = f"{base}. {hook}. Camera reacts with slight shake. Dramatic natural lighting."
        elif i < clip_count - 2:
            desc = f"ESCALATION {i}: {escalation}"
            vprompt = f"{base}. {escalation}. Continuous motion, debris/particles in air. Tension building."
        elif i == clip_count - 2:
            desc = f"CLIMAX: {climax}"
            vprompt = f"{base}. {climax}. Maximum intensity, dramatic lighting change."
        else:
            desc = f"AFTERMATH: {aftermath}"
            vprompt = f"{base}. {aftermath}. Camera slowly stabilizes. Eerie atmosphere."

        clips.append({
            "clip_number": i + 1,
            "video_prompt": vprompt,
            "sfx_prompt": f"{sound}, {'intense' if i < clip_count - 1 else 'fading'} atmosphere",
            "description": desc,
        })

    return {
        "clips": clips,
        "total_duration_seconds": clip_count * 5,
    }


def build_video_prompts_dry(scenario: dict, config: dict) -> dict:
    """Dry-run version: generate prompts without any API call."""
    return build_video_prompts_fallback(scenario, config)


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
