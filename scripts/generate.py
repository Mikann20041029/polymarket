"""
Script generation using DeepSeek API (OpenAI-compatible).
Generates English life-hack scripts with anthropomorphic object characters.
Includes de-duplication to avoid repeating previously used hacks.
"""
import json
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

OUTPUT FORMAT: Return ONLY a valid JSON array. No markdown, no explanation."""

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
    use_together: bool = False,
) -> list[dict]:
    """
    Generate life-hack script segments using DeepSeek (or Together as backup).

    Returns list of hack dicts with narration, scene_description, motion_prompt, etc.
    """
    num_hacks = num_hacks or config.HACKS_PER_VIDEO
    used_hacks = _load_used_hacks()

    if use_together:
        client = OpenAI(
            api_key=config.TOGETHER_API_KEY,
            base_url=config.TOGETHER_BASE_URL,
        )
        model = config.TOGETHER_MODEL
        logger.info(f"Using Together API: {model}")
    else:
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
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw[:-3].rstrip()

    hacks = json.loads(raw)

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
    p.add_argument("--together", action="store_true", help="Use Together instead of DeepSeek")
    args = p.parse_args()
    result = generate_script(args.topic, args.num_hacks, args.together)
    print(json.dumps(result, indent=2, ensure_ascii=False))
