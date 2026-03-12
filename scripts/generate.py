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
stylized 3D animated everyday objects that come alive and explain life hacks.

RULES:
- Each life hack features a DIFFERENT object as the speaking character.
  Objects can be ANYTHING relevant to the hack: tools, gadgets, food items, cleaning supplies,
  office supplies, tech accessories, clothing items, bathroom items, outdoor gear, etc.
- The object IS the star. It has a face, arms, hands, and a big personality.
- Write in English only.
- Voice style: EXTREMELY expressive, manic, comedic, exaggerated, emotionally chaotic.
  Think: dramatic panic → smug realization → explosive excitement → mock seriousness → triumphant relief.
  Each sentence should have a DIFFERENT emotional energy. NEVER flat, NEVER monotone, NEVER calm.
  Use dramatic pauses, speed changes, whispers-to-shouts, and rhetorical questions.
- Each hack narration must be 80-120 words (about 30-45 seconds when spoken).
  This is CRITICAL — short narrations are boring. Pack in personality, reactions, and detail.
- Start each hack with an EXPLOSIVE emotional hook that grabs attention in the first 2 seconds.
  Examples: "WAIT WAIT WAIT—", "Oh. My. GOD.", "You've been doing WHAT?!", "Listen. LISTEN."
- Build tension in the middle with vivid descriptions and emotional reactions.
- End each hack with a satisfying punchline, callback, or mic-drop moment.
- Hacks MUST be genuinely surprising, counterintuitive, and make viewers say "NO WAY!"
  They should be the kind of facts/tricks people IMMEDIATELY want to share with friends.
- Cover ALL areas of life: kitchen, cleaning, tech, clothing, travel, health/wellness,
  productivity, money-saving, car care, gardening, social situations, DIY, beauty, etc.
- AVOID: dangerous, medical diagnoses, misleading, or vague/abstract-advice hacks.
- Do NOT include emotion tags like [PAUSE] or [EXCITED] in the narration text.
  Instead, convey emotion purely through word choice, punctuation, and sentence rhythm.
  Use ALL CAPS for emphasis and "—" for dramatic interruptions.
  Do NOT use "..." (ellipsis) — it breaks subtitle sync. Use "—" or short sentences instead.

OUTPUT FORMAT: Return a JSON object with a single key "hacks" containing the array. No markdown. Example: {"hacks": [...]}"""

HACK_TEMPLATE = """Generate {num_hacks} unique life-hack segments for a vertical short video.

Topic/category preference: {topic}
(But you are NOT limited to this — pick the most VIRAL, JAW-DROPPING hacks regardless of category.
Mix categories for variety: one might be kitchen, another tech, another clothing, etc.)

PREVIOUSLY USED HACKS (DO NOT REPEAT THESE):
{used_hacks}

QUALITY BAR: Each hack must pass ALL of these tests:
- Would someone say "NO WAY, I didn't know that!" when they hear it?
- Would someone screenshot this to send to a friend?
- Is this genuinely useful AND surprising at the same time?
- Does it include a SPECIFIC detail? (exact number, percentage, time, brand name, scientific name, etc.)
  BAD: "Toothpaste can clean stuff" → GOOD: "Toothpaste removes 90% of sneaker scuff marks in 30 seconds"
  BAD: "Rice helps with wet phones" → GOOD: "Silica gel packets absorb moisture 3x faster than rice for wet phones"
  BAD: "Freezing jeans cleans them" → GOOD: "Freezing jeans at -18°C for 24 hours kills 99% of odor bacteria WITHOUT washing"
If a hack fails ANY test, replace it with a better one.

BANNED HACK TYPES (too common, boring, everyone knows them):
- Putting phone in rice, banana peel shoe polish, frozen grapes as ice cubes,
  binder clip cable organizer, rubber band jar opener, lemon microwave cleaner

For each hack, output this JSON structure:
[
  {{
    "hack_number": 1,
    "title": "Short catchy title (3-5 words)",
    "object_character": "the main object that speaks — can be ANY everyday object relevant to the hack (e.g., shoe, phone charger, ice cube tray, zipper, wrench, sponge, lemon, rubber band, binder clip, dryer sheet, etc.)",
    "character_gender": "male or female — pick whichever fits the character's personality and voice better. Mix it up across hacks.",
    "narration": "The EXACT dialogue the object says. MUST be 80-120 words. Extremely expressive, comedic, emotional. Use ALL CAPS for emphasis, '—' for interruptions. Do NOT use [EMOTION] tags or '...' (they break subtitles). Every sentence must be a different emotional beat. MUST include at least ONE specific number, stat, or measurement. Include rhetorical questions, exclamations, dramatic reveals. End with a punchy one-liner.",
    "scene_description": "Detailed visual: what the character looks like, its expression, what it's holding or demonstrating, the specific background setting, warm cinematic lighting, shallow depth of field, camera angle (close-up or medium close-up). Be SPECIFIC about colors, materials, and textures.",
    "motion_prompt": "Extremely detailed movement: big arm gestures, exaggerated facial expressions (eyes widening, jaw dropping, eyebrows shooting up), body bouncing, leaning forward conspiratorially, pointing at camera, spinning in excitement, victory dance, face-palming. Describe SPECIFIC movements for SPECIFIC moments in the narration.",
    "sfx_cues": ["list", "of", "sound", "effects"],
    "emotional_arc": "e.g., outraged disbelief → conspiratorial whisper → explosive revelation → smug triumph"
  }}
]

Make each hack's character, topic category, tone, and emotional arc COMPLETELY DIFFERENT from the others.
The hacks should create variety and surprise when stitched together in sequence."""


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
    topic: str = "life hacks across all categories",
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
    p.add_argument("--topic", default="life hacks across all categories")
    p.add_argument("--num-hacks", type=int, default=None)
    args = p.parse_args()
    result = generate_script(args.topic, args.num_hacks)
    print(json.dumps(result, indent=2, ensure_ascii=False))
