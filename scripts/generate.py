"""
Concept generation using DeepSeek API (OpenAI-compatible).
Generates "Impossible Satisfying" video concepts — surreal, physics-defying,
visually mesmerizing scenes with matching ASMR sound descriptions.
"""
import json
import re
import logging
import argparse
from pathlib import Path
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a creative director for "Impossible Satisfying" — a viral TikTok/YouTube Shorts channel.

WHAT MAKES SATISFYING VIDEOS GO VIRAL:
1. HOOK IN 0.3 SECONDS — first frame must show something impossible already happening
2. CONTINUOUS ESCALATION — each moment more satisfying than the last
3. PERFECT LOOP — end connects seamlessly to start (3-5x more views)
4. HIGH CONTRAST — dark/black background + vivid glowing subject
5. SLOW MOTION feel — smooth, deliberate movements
6. ASMR SOUND — every visual must have a matching crisp, satisfying sound

RULES:
- No words, no characters, no narration — pure visual + sound
- Dark/black backgrounds ONLY
- Cinematic rim lighting, volumetric light, dramatic shadows
- 9:16 vertical, 5 seconds per clip
- Family-friendly

AVOID:
- Static scenes, slow reveals, realistic physics, cluttered backgrounds, flat lighting

OUTPUT: Return {"concepts": [...]} as JSON. No markdown fences."""

CONCEPT_TEMPLATE = """Generate {num_clips} video concepts. Theme: {theme}

DO NOT REPEAT these previous concepts:
{used_concepts}

Return this exact JSON structure:
{{"concepts": [
  {{
    "clip_number": 1,
    "title": "2-4 word title",
    "visual_prompt": "Single flowing paragraph, 80+ words. Must include: the specific object and its material/texture, dark background, cinematic lighting setup, exact camera movement, the IMPOSSIBLE physics frame by frame (what happens at second 0, 1, 2, 3, 4), particle effects. First frame must already be mid-action. Be hyper-specific about motion speed, direction, deformation.",
    "sound_prompt": "Single paragraph, 40+ words. Describe: primary satisfying sound, layered textures, spatial quality (reverb, stereo), ASMR crispness. Sound must perfectly match the visual action.",
    "text_overlay": null,
    "hook_type": "impossible_physics",
    "color_palette": "dark bg + 2 vivid colors",
    "loop_friendly": true
  }}
]}}

QUALITY REQUIREMENTS:
- visual_prompt: 80+ words, hyper-detailed physics description
- sound_prompt: 40+ words, layered audio design
- hook_type: one of impossible_physics, satisfying_transformation, surreal_beauty, unexpected_reveal, perfect_loop
- loop_friendly: strongly prefer true
- text_overlay: null unless truly needed (rare — only "???" or "HOW")
- Vary materials across concepts: glass, metal, liquid, crystal, organic, magnetic
- Vary physics: reverse gravity, phase change, impossible geometry, scale shift"""


def _load_used_concepts() -> list[str]:
    """Load previously used concept titles to avoid repetition."""
    path = config.USED_CONCEPTS_FILE
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if len(data) > 200:
            data = data[-200:]
            _save_used_concepts(data)
        return data
    return []


def _save_used_concepts(used: list[str]):
    """Save updated used concept titles."""
    config.USED_CONCEPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.USED_CONCEPTS_FILE, "w") as f:
        json.dump(used, f, indent=2)


def _validate_concept(concept: dict, index: int) -> dict:
    """
    Validate a single concept. Raises ValueError if it would produce bad output.
    Catches problems HERE so we never waste API money on garbage input.
    """
    required = ["visual_prompt", "sound_prompt", "title", "hook_type"]
    for field in required:
        if field not in concept or not concept[field]:
            raise ValueError(f"Concept {index}: missing '{field}'")

    vp = concept["visual_prompt"]
    sp = concept["sound_prompt"]

    vp_words = len(vp.split())
    sp_words = len(sp.split())

    if vp_words < 50:
        raise ValueError(
            f"Concept {index} '{concept['title']}': visual_prompt only {vp_words} words "
            f"(need 50+). Short prompts produce generic, boring videos."
        )

    if sp_words < 20:
        raise ValueError(
            f"Concept {index} '{concept['title']}': sound_prompt only {sp_words} words "
            f"(need 20+). Short prompts produce generic sounds."
        )

    # Defaults
    concept.setdefault("clip_number", index + 1)
    concept.setdefault("text_overlay", None)
    concept.setdefault("color_palette", "white, black")
    concept.setdefault("loop_friendly", True)

    if concept["text_overlay"] in ("null", "none", "", "None"):
        concept["text_overlay"] = None

    return concept


def _parse_json_response(raw: str) -> dict | list:
    """
    Parse LLM JSON response with multiple fallback strategies.
    Handles: clean JSON, markdown-fenced JSON, wrapper objects.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object {...}
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try extracting JSON array [...]
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try stripping markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1::2]:  # Odd-indexed parts are inside fences
            clean = part.lstrip("json").strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM response: {raw[:300]}")


def generate_concepts(
    theme: str = "surreal physics",
    num_clips: int = None,
) -> list[dict]:
    """
    Generate video concepts using DeepSeek.
    Validates every concept before returning — no garbage goes to paid APIs.
    """
    num_clips = num_clips or config.CLIPS_PER_VIDEO
    num_clips = max(1, min(num_clips, 10))

    used_concepts = _load_used_concepts()

    # Sanitize theme
    theme = re.sub(r'[^\w\s,.\-]', '', theme)[:200]

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    model = config.DEEPSEEK_MODEL
    logger.info(f"Generating {num_clips} concepts, theme='{theme}'")

    used_str = "\n".join(f"- {c}" for c in used_concepts[-50:]) if used_concepts else "(none)"
    user_msg = CONCEPT_TEMPLATE.format(
        num_clips=num_clips,
        theme=theme,
        used_concepts=used_str,
    )

    resp = client.chat.completions.create(
        model=model,
        max_tokens=config.SCRIPT_MAX_TOKENS,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    logger.debug(f"Raw LLM response: {raw[:500]}")

    parsed = _parse_json_response(raw)

    # Extract concepts list from various wrapper formats
    if isinstance(parsed, dict) and "concepts" in parsed:
        concepts = parsed["concepts"]
    elif isinstance(parsed, list):
        concepts = parsed
    elif isinstance(parsed, dict):
        # Try any key that has a list value
        concepts = None
        for v in parsed.values():
            if isinstance(v, list):
                concepts = v
                break
        if concepts is None:
            raise ValueError(f"Unexpected JSON structure: {list(parsed.keys())}")
    else:
        raise ValueError(f"Unexpected response type: {type(parsed)}")

    if not concepts:
        raise ValueError("Empty concepts list from DeepSeek")

    # Validate EVERY concept before saving or returning
    validated = []
    for i, concept in enumerate(concepts):
        validated.append(_validate_concept(concept, i))

    # Update de-duplication list
    new_titles = [c["title"] for c in validated]
    used_concepts.extend(new_titles)
    _save_used_concepts(used_concepts)

    logger.info(f"Generated {len(validated)} validated concepts: {new_titles}")
    return validated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--theme", default="surreal physics")
    p.add_argument("--num-clips", type=int, default=None)
    args = p.parse_args()
    result = generate_concepts(args.theme, args.num_clips)
    print(json.dumps(result, indent=2, ensure_ascii=False))
