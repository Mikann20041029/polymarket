"""
Concept generation using DeepSeek API (OpenAI-compatible).
Generates "Impossible Satisfying" video concepts — surreal, physics-defying,
visually mesmerizing scenes with matching ASMR sound descriptions.
No language needed. Pure visual + audio satisfaction.
"""
import json
import logging
import argparse
from pathlib import Path
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a creative director for "Impossible Satisfying" — a viral TikTok/Shorts channel
that creates surreal, physics-defying, oddly satisfying short videos using AI.

CHANNEL IDENTITY:
- Every video shows something IMPOSSIBLE yet deeply SATISFYING to watch
- Objects behave in ways that defy physics but feel visually perfect
- No words, no characters, no narration — pure visual + sound
- Think: "What if physics was beautiful and wrong?"

RULES:
- Each concept must be VISUALLY STUNNING and PHYSICALLY IMPOSSIBLE
- The visual prompt must be extremely detailed for AI video generation (Kling 3.0)
- The sound prompt must describe the exact ASMR/satisfying sounds to match
- Include specific details: materials, colors, lighting, camera movement, physics behavior
- Each concept should trigger the "wait, that's not possible..." reaction
- Keep it family-friendly and non-violent
- NEVER repeat the same concept type — each one must feel fresh

WINNING CONCEPTS (examples of what works):
- A watermelon sliced open reveals layers of different colored glass that shatter beautifully
- Mercury-like liquid metal flows upward in a spiral, then freezes into a chrome flower
- A cube of ice melts in reverse while cycling through rainbow colors
- Magnetic sand assembles itself into a perfect miniature city
- A soap bubble expands and inside it contains a tiny ocean with waves
- Honeycomb drips upward, each drop becoming a golden butterfly

OUTPUT FORMAT: Return ONLY a valid JSON array. No markdown, no explanation."""

CONCEPT_TEMPLATE = """Generate {num_clips} unique "Impossible Satisfying" video concepts.

Theme preference: {theme}

PREVIOUSLY USED CONCEPTS (DO NOT REPEAT):
{used_concepts}

For each concept, output this JSON structure:
[
  {{
    "clip_number": 1,
    "title": "Short catchy title (2-4 words, no language needed in video)",
    "visual_prompt": "Extremely detailed Kling 3.0 video generation prompt. Include: subject, materials, textures, colors, lighting (cinematic warm/cool), camera movement (slow zoom in, orbit, static close-up), the IMPOSSIBLE physics behavior, background (solid dark/gradient/minimal). 9:16 vertical format, 5 seconds. Be HYPER-SPECIFIC about motion: speed, direction, deformation, particle effects. Example quality: 'A perfect glass sphere sitting on a dark reflective surface, warm amber side lighting, slow camera push-in. The sphere suddenly fractures into hundreds of geometric shards that float upward in slow motion, each shard refracting rainbow light. Shards rotate independently and reassemble into a crystalline flower shape. Cinematic shallow depth of field, dark background, 9:16 vertical.'",
    "sound_prompt": "Detailed ElevenLabs Sound Effects prompt describing the exact ASMR/satisfying sounds. Include: primary sound, texture, intensity, spatial quality. Example quality: 'Deep resonant glass cracking followed by delicate tinkling of floating crystal shards, subtle reverb in a large space, each shard producing a soft chime as it rotates, building to a gentle harmonic chord as they reassemble. Crisp, detailed, ASMR quality.'",
    "text_overlay": "Optional 1-3 word text overlay for the video (or null if none needed). Use sparingly — only when it adds intrigue like '???' or 'wait for it'. Keep universal (no language-specific words).",
    "hook_type": "What makes viewers stop scrolling: 'impossible_physics' | 'satisfying_transformation' | 'surreal_beauty' | 'unexpected_reveal' | 'perfect_loop'",
    "color_palette": "2-3 dominant colors for visual consistency (e.g., 'amber, deep blue, gold')",
    "loop_friendly": true or false — whether the end can seamlessly connect to the beginning
  }}
]

Make each concept visually DISTINCT. Vary the materials (glass, metal, liquid, organic, crystal, fabric),
the impossible physics (reverse gravity, impossible geometry, material transformation, scale shift),
and the emotional tone (peaceful, dramatic, playful, mysterious)."""


def _load_used_concepts() -> list[str]:
    """Load previously used concept titles to avoid repetition."""
    path = config.USED_CONCEPTS_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_used_concepts(used: list[str]):
    """Save updated used concept titles."""
    with open(config.USED_CONCEPTS_FILE, "w") as f:
        json.dump(used, f, indent=2)


def generate_concepts(
    theme: str = "surreal physics",
    num_clips: int = None,
) -> list[dict]:
    """
    Generate Impossible Satisfying video concepts using DeepSeek.

    Returns list of concept dicts with visual_prompt, sound_prompt, etc.
    """
    num_clips = num_clips or config.CLIPS_PER_VIDEO
    used_concepts = _load_used_concepts()

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    model = config.DEEPSEEK_MODEL
    logger.info(f"Using DeepSeek API: {model}")

    used_str = "\n".join(f"- {c}" for c in used_concepts[-100:]) if used_concepts else "(none yet)"
    user_msg = CONCEPT_TEMPLATE.format(
        num_clips=num_clips,
        theme=theme,
        used_concepts=used_str,
    )

    logger.info(f"Generating {num_clips} concepts, theme='{theme}', {len(used_concepts)} previously used")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=config.SCRIPT_MAX_TOKENS,
        temperature=0.95,
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

    concepts = json.loads(raw)

    # Update de-duplication list
    new_titles = [c["title"] for c in concepts]
    used_concepts.extend(new_titles)
    _save_used_concepts(used_concepts)

    logger.info(f"Generated {len(concepts)} concepts: {new_titles}")
    return concepts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--theme", default="surreal physics")
    p.add_argument("--num-clips", type=int, default=None)
    args = p.parse_args()
    result = generate_concepts(args.theme, args.num_clips)
    print(json.dumps(result, indent=2, ensure_ascii=False))
