"""
Concept generation using DeepSeek API (OpenAI-compatible).
Generates "Impossible Satisfying" video concepts — surreal, physics-defying,
visually mesmerizing scenes with matching ASMR sound descriptions.
No language needed. Pure visual + audio satisfaction.
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
Every video gets millions of views because it follows a proven formula.

WHAT MAKES SATISFYING VIDEOS GO VIRAL (research-backed):
1. HOOK IN FIRST 0.3 SECONDS — the very first frame must show something unusual/impossible mid-action
2. CONTINUOUS ESCALATION — each moment should be more satisfying than the last, never plateau
3. ANTICIPATION BUILDUP — show the setup, let viewers predict what happens, then exceed expectations
4. IMPOSSIBLE PHYSICS — objects behave in ways that feel "wrong but perfect"
5. ASMR-TRIGGERING TEXTURES — glass cracking, liquid flowing, objects clicking into place
6. PERFECT LOOP — the ending should seamlessly restart for infinite rewatching (this 5x engagement)
7. HIGH CONTRAST VISUALS — dark backgrounds + vivid subjects = maximum visual pop on phone screens
8. SLOW MOTION — 0.5x to 0.25x perceived speed makes every detail satisfying

CHANNEL RULES:
- No words, no characters, no narration — pure visual + sound
- Every video must trigger the "I NEED to watch that again" response
- Dark/black backgrounds ONLY (maximizes visual contrast on phone screens)
- Cinematic lighting (rim light, volumetric, dramatic shadows)
- 9:16 vertical format, 5 seconds per clip
- Family-friendly, non-violent

WHAT FAILS (avoid these):
- Static scenes with no motion
- Gradual/slow reveals (viewers scroll away in 0.5s)
- Realistic physics (boring — must be IMPOSSIBLE)
- Busy/cluttered backgrounds
- Low contrast or flat lighting
- Sounds that don't match the visual (breaks immersion)

OUTPUT FORMAT: Return a JSON object with a single key "concepts" containing the array. No markdown."""

CONCEPT_TEMPLATE = """Generate {num_clips} "Impossible Satisfying" video concepts for theme: {theme}

PREVIOUSLY USED (DO NOT REPEAT ANYTHING SIMILAR):
{used_concepts}

Each concept MUST follow this exact JSON structure:
{{"concepts": [
  {{
    "clip_number": 1,
    "title": "2-4 word catchy title",
    "visual_prompt": "REQUIRED STRUCTURE for the AI video model prompt:
      SUBJECT: [specific object with material/texture detail]
      SETTING: [dark/black background, specific surface if any]
      LIGHTING: [cinematic lighting setup — rim light, volumetric, amber/cool tones]
      CAMERA: [specific movement — slow push-in, macro orbit, static close-up]
      ACTION: [the IMPOSSIBLE physics behavior, described frame-by-frame]
      PARTICLES: [secondary effects — sparkles, dust, droplets, light rays]
      FORMAT: 9:16 vertical, 5 seconds, cinematic shallow depth of field

      Write this as a SINGLE flowing paragraph, not as bullet points.
      Be HYPER-SPECIFIC about motion direction, speed, deformation, and timing.
      The first frame must already be mid-action (no setup time — instant hook).",
    "sound_prompt": "Detailed sound design prompt. Structure:
      PRIMARY: [main satisfying sound matching the visual action]
      TEXTURE: [subtle layered sounds adding depth — creaking, tinkling, whooshing]
      SPATIAL: [reverb/room size, stereo movement, distance]
      QUALITY: [ASMR quality, crisp highs, satisfying bass, no harshness]
      Write as a single flowing description.",
    "text_overlay": "null in most cases. Only use '???', 'wait for it', or 'HOW' when truly needed",
    "hook_type": "impossible_physics | satisfying_transformation | surreal_beauty | unexpected_reveal | perfect_loop",
    "color_palette": "2-3 colors (use high contrast: dark bg + vivid subject)",
    "loop_friendly": true (STRONGLY prefer true — loops get 3-5x more views)
  }}
]}}

CRITICAL QUALITY RULES:
- visual_prompt must be 80+ words with specific physics descriptions
- sound_prompt must be 40+ words describing layered audio
- First frame = already in action (NO setup, NO static start)
- Every concept must have a clear "impossible" element
- Vary materials: glass, metal, liquid, crystal, organic, magnetic, light
- Vary physics: reverse gravity, phase change, impossible geometry, scale shift, telekinesis"""


def _load_used_concepts() -> list[str]:
    """Load previously used concept titles to avoid repetition."""
    path = config.USED_CONCEPTS_FILE
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Keep only last 200 to prevent file bloat
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
    Validate a single concept has all required fields with sufficient detail.
    Raises ValueError if concept would waste API money on bad generation.
    """
    required = ["visual_prompt", "sound_prompt", "title", "hook_type"]
    for field in required:
        if field not in concept or not concept[field]:
            raise ValueError(f"Concept {index}: missing required field '{field}'")

    vp = concept["visual_prompt"]
    sp = concept["sound_prompt"]

    if len(vp.split()) < 30:
        raise ValueError(
            f"Concept {index} '{concept['title']}': visual_prompt too short "
            f"({len(vp.split())} words, need 30+). This will produce a bad video."
        )

    if len(sp.split()) < 15:
        raise ValueError(
            f"Concept {index} '{concept['title']}': sound_prompt too short "
            f"({len(sp.split())} words, need 15+). This will produce bad audio."
        )

    # Set defaults for optional fields
    concept.setdefault("clip_number", index + 1)
    concept.setdefault("text_overlay", None)
    concept.setdefault("color_palette", "white, black")
    concept.setdefault("loop_friendly", True)

    # Normalize null text_overlay
    if concept["text_overlay"] in ("null", "none", "", "None"):
        concept["text_overlay"] = None

    return concept


def generate_concepts(
    theme: str = "surreal physics",
    num_clips: int = None,
) -> list[dict]:
    """
    Generate Impossible Satisfying video concepts using DeepSeek.

    Returns list of validated concept dicts with visual_prompt, sound_prompt, etc.
    Raises ValueError if concepts don't meet quality bar (prevents wasting API money).
    """
    num_clips = num_clips or config.CLIPS_PER_VIDEO
    num_clips = max(1, min(num_clips, 10))  # Clamp to 1-10

    used_concepts = _load_used_concepts()

    # Sanitize theme input
    theme = re.sub(r'[^\w\s,.\-]', '', theme)[:200]

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    model = config.DEEPSEEK_MODEL
    logger.info(f"Using DeepSeek API: {model}")

    used_str = "\n".join(f"- {c}" for c in used_concepts[-50:]) if used_concepts else "(none yet)"
    user_msg = CONCEPT_TEMPLATE.format(
        num_clips=num_clips,
        theme=theme,
        used_concepts=used_str,
    )

    logger.info(f"Generating {num_clips} concepts, theme='{theme}'")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=config.SCRIPT_MAX_TOKENS,
        temperature=0.8,  # Balanced: creative but coherent
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    logger.debug(f"Raw LLM response (first 500 chars): {raw[:500]}")

    # Robust JSON extraction
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        if not match or not isinstance(parsed, dict):
            match = re.search(r'\[[\s\S]*\]', raw)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError(f"DeepSeek returned unparseable response: {raw[:300]}")

    # Handle {"concepts": [...]} and [...] formats
    if isinstance(parsed, dict) and "concepts" in parsed:
        concepts = parsed["concepts"]
    elif isinstance(parsed, list):
        concepts = parsed
    elif isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                concepts = v
                break
        else:
            raise ValueError(f"Unexpected JSON structure: {list(parsed.keys())}")
    else:
        raise ValueError(f"Unexpected response type: {type(parsed)}")

    if not concepts:
        raise ValueError("DeepSeek returned empty concepts list")

    # Validate every concept BEFORE saving or returning
    validated = []
    for i, concept in enumerate(concepts):
        validated.append(_validate_concept(concept, i))

    # Update de-duplication list only after validation passes
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
