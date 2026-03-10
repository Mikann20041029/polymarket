"""
Script generation using DeepSeek API (OpenAI-compatible).
Generates English life-hack scripts with anthropomorphic object characters.
Includes de-duplication to avoid repeating previously used hacks.
"""
import json
import re
import logging
import argparse
from pathlib import Path
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a viral short-video script writer for "Crazy Bird News" — a channel featuring
stylized 3D animated household objects that come alive and explain life hacks.

RULES:
- Each life hack features a DIFFERENT object as the speaking character (broccoli, mug, sponge, knife, pan, etc.)
- The object IS the star. It has a face, arms, hands, and a big personality.
- Write in English only.
- Voice style: HIGH-ENERGY, manic, comedic, exaggerated, emotionally chaotic.
  Think: panic → smugness → excitement → mock seriousness → relief. Never flat or calm.
- Each hack is 15-20 seconds when spoken aloud (roughly 40-60 words).
- Start each hack with a dramatic emotional hook (frustration, panic, outrage).
- End each hack with a punchline, payoff, or smug satisfaction.
- Hacks should be genuinely useful, visually demonstrable, and kitchen/household-focused.
- AVOID: dangerous, medical, misleading, or abstract-advice hacks.

OUTPUT FORMAT: Return a JSON object with a single key "hacks" containing the array. No markdown. Example: {"hacks": [...]}"""

HACK_TEMPLATE = """Generate {num_hacks} unique life-hack segments for a vertical short video.

Topic preference: {topic}

PREVIOUSLY USED HACKS (DO NOT REPEAT THESE):
{used_hacks}

For each hack, output this JSON structure:
[
  {{
    "hack_number": 1,
    "title": "Short catchy title (3-5 words)",
    "object_character": "the main object that speaks (e.g., broccoli, mug, sponge)",
    "character_gender": "male or female — pick whichever fits the character's personality and voice better. Mix it up across hacks.",
    "narration": "The EXACT dialogue the object says. 40-60 words. Extremely expressive, comedic, emotional. Include [PAUSE], [EXCITED], [FRUSTRATED], [SMUG] emotion tags inline.",
    "scene_description": "Detailed visual: what the character looks like, its expression, what it's holding, the background (kitchen/household), lighting (warm, shallow DOF), camera angle (close-up or medium close-up)",
    "motion_prompt": "How the character moves: big arm gestures, exaggerated facial expressions, body bouncing, leaning forward, pointing, spinning, celebrating. Be VERY specific about physical movement.",
    "sfx_cues": ["list", "of", "sound", "effects"],
    "emotional_arc": "e.g., panic → realization → smug satisfaction"
  }}
]

Make each hack's character, tone, and emotional arc DIFFERENT from the others.
The hacks should flow well when stitched together in sequence."""


def _load_used_hacks() -> list[str]:
    """Load previously used hack titles to avoid repetition."""
    path = config.USED_HACKS_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_used_hacks(used: list[str]):
    """Save updated used hack titles."""
    with open(config.USED_HACKS_FILE, "w") as f:
        json.dump(used, f, indent=2)


def generate_script(
    topic: str = "kitchen and household",
    num_hacks: int = None,
) -> list[dict]:
    """
    Generate life-hack script segments using DeepSeek.

    Returns list of hack dicts with narration, scene_description, motion_prompt, etc.
    """
    num_hacks = num_hacks or config.HACKS_PER_VIDEO
    used_hacks = _load_used_hacks()

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    model = config.DEEPSEEK_MODEL
    logger.info(f"Using DeepSeek API: {model}")

    used_str = "\n".join(f"- {h}" for h in used_hacks[-100:]) if used_hacks else "(none yet)"
    user_msg = HACK_TEMPLATE.format(
        num_hacks=num_hacks,
        topic=topic,
        used_hacks=used_str,
    )

    logger.info(f"Generating {num_hacks} hacks, topic='{topic}', {len(used_hacks)} previously used")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=config.SCRIPT_MAX_TOKENS,
        temperature=0.9,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    logger.debug(f"Raw LLM response (first 300 chars): {raw[:300]}")

    # Extract JSON array from response, handling markdown fences and surrounding text
    # Method 1: Find JSON array with regex
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        json_str = match.group(0)
    else:
        # Method 2: Strip code fences manually
        json_str = raw
        if "```" in json_str:
            # Remove everything before first ``` and after last ```
            parts = json_str.split("```")
            # Take the content between first pair of ```
            if len(parts) >= 3:
                json_str = parts[1]
            elif len(parts) >= 2:
                json_str = parts[1]
            # Remove language identifier like "json"
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        logger.error(f"Attempted to parse: {json_str[:500]}")
        raise ValueError(f"DeepSeek returned invalid JSON: {e}") from e

    # Handle both {"hacks": [...]} and [...] formats
    if isinstance(parsed, dict) and "hacks" in parsed:
        hacks = parsed["hacks"]
    elif isinstance(parsed, list):
        hacks = parsed
    else:
        # Try to find any list value in the dict
        for v in parsed.values():
            if isinstance(v, list):
                hacks = v
                break
        else:
            raise ValueError(f"Unexpected JSON structure: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")

    # Update de-duplication list
    new_titles = [h["title"] for h in hacks]
    used_hacks.extend(new_titles)
    _save_used_hacks(used_hacks)

    logger.info(f"Generated {len(hacks)} hacks: {new_titles}")
    return hacks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="kitchen and household")
    p.add_argument("--num-hacks", type=int, default=None)
    args = p.parse_args()
    result = generate_script(args.topic, args.num_hacks)
    print(json.dumps(result, indent=2, ensure_ascii=False))
