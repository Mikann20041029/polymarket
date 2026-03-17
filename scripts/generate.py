"""
Topic & scene generation using DeepSeek API.

Two content categories (randomly selected each run):
  1. Anime World — photorealistic "as if you entered the anime world"
  2. Historical Event — photorealistic "as if you witnessed it firsthand"

Each topic is broken into 8 scenes with:
  - Detailed FLUX image prompt (photorealistic, 9:16 vertical)
  - Camera movement type for Ken Burns effect
  - Ambient SFX description
  - Optional text overlay (year, location, etc.)
"""
import json
import re
import random
import logging
import argparse
from pathlib import Path
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a creative director for a viral YouTube Shorts channel.
You create 40-60 second photorealistic "world recreation" videos.

TWO CONTENT TYPES (you will be told which one):

TYPE A — ANIME WORLD RECREATION:
- Take a famous anime and recreate its world as photorealistic live-action
- The viewer should feel like they STEPPED INTO the anime world
- Locations, architecture, nature, atmosphere — all from the anime but looking 100% real
- Famous landmarks from the anime, iconic scenes turned photorealistic
- Examples: Studio Ghibli worlds, One Piece locations, Attack on Titan walls, Naruto's village

TYPE B — HISTORICAL EVENT WITNESS:
- Take a famous historical event and recreate it as if photographed on scene
- The viewer should feel like they TIME-TRAVELED to witness it
- Multiple angles/moments of the same event
- Examples: Construction of the pyramids, Pompeii eruption, D-Day landing, Moon landing, Fall of Berlin Wall

CRITICAL RULES FOR ALL SCENES:
1. SCENE 1 MUST BE THE HOOK — the single most visually stunning, jaw-dropping image that makes people STOP scrolling. This is life or death for the video.
2. Every image must look like a REAL PHOTOGRAPH — not CGI, not painting, not illustration
3. No text in the images themselves
4. Consistent visual style across all scenes (same lighting mood, color grading)
5. 9:16 vertical framing — compose for phone screens
6. Include people/figures where appropriate for scale and immersion (but no specific real people)
7. Rich environmental detail — weather, particles, atmospheric effects

OUTPUT: Return a JSON object. No markdown."""

TOPIC_TEMPLATE_ANIME = """Generate a video topic for ANIME WORLD RECREATION.

Pick a specific anime and recreate its world as photorealistic scenes.
Choose from well-known anime that have distinctive, visually rich worlds.

PREVIOUSLY USED TOPICS (DO NOT REPEAT):
{used_topics}

Return this JSON:
{{
  "topic_type": "anime",
  "title": "Short title for the video (Japanese + English, e.g. 'ハウルの動く城の世界をAIで実写化 / Howl's Moving Castle in Real Life')",
  "source_anime": "Name of the anime",
  "description": "1-2 sentence description of what viewers will see",
  "scenes": [
    {{
      "scene_number": 1,
      "image_prompt": "ULTRA-DETAILED prompt for FLUX image generation. Must produce photorealistic output. Include: exact subject, environment details, materials, lighting (golden hour/overcast/dramatic), weather, atmospheric effects (fog, dust, rays), camera angle (low angle/eye level/aerial), lens type (wide/telephoto/macro). 9:16 vertical composition. 80+ words. This prompt directly generates the image — be SPECIFIC. Start with 'Photorealistic photograph of...' or 'Ultra-realistic photo of...'",
      "camera_movement": "One of: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down",
      "sfx_prompt": "Ambient sound for this scene (wind, crowds, water, fire, etc.)",
      "text_overlay": "Optional short text (location name, year, etc.) or null"
    }}
  ]
}}

Generate exactly {num_scenes} scenes. Scene 1 = the HOOK (most visually striking)."""

TOPIC_TEMPLATE_HISTORY = """Generate a video topic for HISTORICAL EVENT WITNESS.

Pick a specific historical event and recreate key moments as if photographed on scene.
Choose events that are visually dramatic and widely known.

PREVIOUSLY USED TOPICS (DO NOT REPEAT):
{used_topics}

Return this JSON:
{{
  "topic_type": "historical",
  "title": "Short title (Japanese + English, e.g. 'ポンペイ最後の日をAIで再現 / AI Recreates the Last Day of Pompeii')",
  "event": "Name of the historical event",
  "era": "Time period (e.g. '79 AD', '1945', '1969')",
  "description": "1-2 sentence description",
  "scenes": [
    {{
      "scene_number": 1,
      "image_prompt": "ULTRA-DETAILED prompt for FLUX image generation. Photorealistic historical recreation. Include: exact subject, period-accurate details (clothing, architecture, technology), environment, lighting, weather, atmospheric effects, camera angle, lens type. 9:16 vertical. 80+ words. Start with 'Photorealistic photograph of...'",
      "camera_movement": "One of: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down",
      "sfx_prompt": "Period-appropriate ambient sound for this scene",
      "text_overlay": "Optional: year, location, or null"
    }}
  ]
}}

Generate exactly {num_scenes} scenes. Scene 1 = the HOOK (most visually dramatic moment)."""


def _load_used_topics() -> list[str]:
    """Load previously used topic titles."""
    path = config.USED_TOPICS_FILE
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if len(data) > 300:
            data = data[-300:]
            _save_used_topics(data)
        return data
    return []


def _save_used_topics(used: list[str]):
    config.USED_TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.USED_TOPICS_FILE, "w") as f:
        json.dump(used, f, indent=2, ensure_ascii=False)


def _parse_json_response(raw: str) -> dict | list:
    """Parse LLM JSON with multiple fallback strategies."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    if "```" in raw:
        for part in raw.split("```")[1::2]:
            clean = part.lstrip("json").strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON: {raw[:300]}")


def _validate_topic(topic: dict) -> dict:
    """Validate topic structure before spending money on images."""
    required = ["title", "scenes"]
    for field in required:
        if field not in topic or not topic[field]:
            raise ValueError(f"Topic missing '{field}'")

    if not topic.get("topic_type"):
        topic["topic_type"] = "unknown"

    scenes = topic["scenes"]
    if len(scenes) < 3:
        raise ValueError(f"Only {len(scenes)} scenes (need at least 3)")

    for i, scene in enumerate(scenes):
        if "image_prompt" not in scene or not scene["image_prompt"]:
            raise ValueError(f"Scene {i+1}: missing image_prompt")

        prompt_words = len(scene["image_prompt"].split())
        if prompt_words < 30:
            raise ValueError(
                f"Scene {i+1}: image_prompt only {prompt_words} words (need 30+). "
                f"This will produce a generic, low-quality image."
            )

        # Defaults
        scene.setdefault("scene_number", i + 1)
        scene.setdefault("camera_movement", "zoom_in")
        scene.setdefault("sfx_prompt", "ambient atmosphere")
        if scene.get("text_overlay") in ("null", "none", "", "None"):
            scene["text_overlay"] = None

        # Validate camera movement
        valid_movements = {"zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"}
        if scene["camera_movement"] not in valid_movements:
            scene["camera_movement"] = "zoom_in"

    return topic


def generate_topic(
    force_type: str = None,
    num_scenes: int = None,
) -> dict:
    """
    Generate a video topic with scene breakdown.

    Args:
        force_type: "anime" or "historical". None = random 50/50.
        num_scenes: Number of scenes. Default from config.

    Returns validated topic dict with scenes.
    """
    num_scenes = num_scenes or config.SCENES_PER_VIDEO
    num_scenes = max(4, min(num_scenes, 15))

    used_topics = _load_used_topics()

    # Random selection: anime or historical
    topic_type = force_type or random.choice(["anime", "historical"])

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    used_str = "\n".join(f"- {t}" for t in used_topics[-50:]) if used_topics else "(none)"

    if topic_type == "anime":
        user_msg = TOPIC_TEMPLATE_ANIME.format(
            num_scenes=num_scenes,
            used_topics=used_str,
        )
    else:
        user_msg = TOPIC_TEMPLATE_HISTORY.format(
            num_scenes=num_scenes,
            used_topics=used_str,
        )

    logger.info(f"Generating topic: type={topic_type}, scenes={num_scenes}")

    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=config.SCRIPT_MAX_TOKENS,
        temperature=0.85,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    logger.debug(f"Raw response: {raw[:500]}")

    topic = _parse_json_response(raw)
    if not isinstance(topic, dict):
        raise ValueError(f"Expected dict, got {type(topic)}")

    topic = _validate_topic(topic)

    # Save to dedup
    used_topics.append(topic["title"])
    _save_used_topics(used_topics)

    logger.info(f"Topic: {topic['title']} ({len(topic['scenes'])} scenes)")
    return topic


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["anime", "historical"], default=None)
    p.add_argument("--num-scenes", type=int, default=None)
    args = p.parse_args()
    result = generate_topic(args.type, args.num_scenes)
    print(json.dumps(result, indent=2, ensure_ascii=False))
