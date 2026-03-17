"""
Topic & scene generation using DeepSeek API.

Content selection (weighted random):
  - 70% Historical events (globally significant, visually dramatic)
  - 30% Anime worlds (top 20 internationally popular anime)

Titles: English only. For historical events, text overlays use the local
language of the country being recreated.
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

# Top 20 internationally popular anime with visually rich worlds
TOP_ANIME = [
    "Spirited Away",
    "My Neighbor Totoro",
    "Princess Mononoke",
    "Howl's Moving Castle",
    "Castle in the Sky (Laputa)",
    "Nausicaa of the Valley of the Wind",
    "Attack on Titan",
    "One Piece",
    "Naruto",
    "Demon Slayer",
    "Jujutsu Kaisen",
    "Dragon Ball Z",
    "Fullmetal Alchemist: Brotherhood",
    "Sword Art Online",
    "Weathering with You",
    "Your Name (Kimi no Na wa)",
    "Ghost in the Shell",
    "Akira",
    "Violet Evergarden",
    "Made in Abyss",
]

SYSTEM_PROMPT = """You are a creative director for a viral YouTube Shorts channel
that recreates worlds as photorealistic videos.

TARGET AUDIENCE: English-speaking global viewers.
ALL TITLES AND DESCRIPTIONS MUST BE IN ENGLISH.

TWO CONTENT TYPES:

TYPE A — ANIME WORLD RECREATION:
- Recreate a specific anime's world as photorealistic live-action
- The viewer feels like they STEPPED INTO the anime world
- Show iconic locations, architecture, nature from the anime — but looking 100% real
- You will be given a specific anime to recreate

TYPE B — HISTORICAL EVENT WITNESS:
- Recreate a famous historical event as if photographed on scene
- The viewer feels like they TIME-TRAVELED to witness it
- Show multiple moments/angles of the event
- Pick events that are VISUALLY DRAMATIC and will get millions of views
- For text overlays, use the LOCAL LANGUAGE of the country where the event happened
  (e.g. Latin for Ancient Rome, Japanese for events in Japan, German for WWII Germany)

CRITICAL RULES:
1. SCENE 1 = THE HOOK — the single most jaw-dropping image. Life or death for the video.
2. Every image MUST look like a REAL PHOTOGRAPH — not CGI, not a painting, not an illustration
3. No text burned into the images
4. Consistent visual style across all scenes
5. 9:16 vertical framing for phone screens
6. Include people/figures for scale (but no specific real people)
7. Rich detail: weather, particles, atmospheric effects, period-accurate details

OUTPUT: JSON object. No markdown."""

TOPIC_TEMPLATE_ANIME = """Generate a video recreating the world of: {anime_name}

Create photorealistic scenes showing iconic locations from this anime
as if they existed in real life and you walked through them.

PREVIOUSLY USED TOPICS (DO NOT REPEAT):
{used_topics}

Return this JSON:
{{
  "topic_type": "anime",
  "title": "English title (e.g. 'What if Spirited Away's World Was Real?' or 'AI Recreates the World of Attack on Titan')",
  "source_anime": "{anime_name}",
  "description": "1-2 sentence English description",
  "scenes": [
    {{
      "scene_number": 1,
      "image_prompt": "ULTRA-DETAILED FLUX prompt. Must produce PHOTOREALISTIC output that looks like a real photograph. Include: exact subject from the anime recreated realistically, real-world materials and textures, cinematic lighting (golden hour/overcast/dramatic), weather and atmospheric effects (fog, dust, light rays), camera angle (low/eye-level/aerial), lens (wide 24mm/telephoto 85mm). 9:16 vertical. 80+ words. Start with 'Photorealistic photograph of...'",
      "camera_movement": "zoom_in | zoom_out | pan_left | pan_right | pan_up | pan_down",
      "sfx_prompt": "Ambient environmental sound matching the scene",
      "text_overlay": null
    }}
  ]
}}

Generate exactly {num_scenes} scenes. Scene 1 = the HOOK (most visually stunning)."""

TOPIC_TEMPLATE_HISTORY = """Generate a video recreating a major historical event.

REQUIREMENTS:
- Pick an event that is VISUALLY DRAMATIC and globally known
- Events with destruction, massive scale, or human drama get the most views
- Think: events people have seen in paintings/movies but never as "real photos"
- The more visually spectacular, the better

HIGH-VIEW POTENTIAL CATEGORIES:
- Ancient wonders being built (Pyramids, Colosseum, Great Wall)
- Natural disasters (Pompeii, Krakatoa, 1906 San Francisco earthquake)
- Major battles and wars (D-Day, Pearl Harbor, Fall of Constantinople)
- Historic achievements (Moon landing, Wright Brothers first flight)
- Lost civilizations (Atlantis theories, Mayan cities at their peak, Ancient Egypt)
- Iconic moments (Fall of Berlin Wall, Titanic departure, first Olympic games)

PREVIOUSLY USED TOPICS (DO NOT REPEAT):
{used_topics}

Return this JSON:
{{
  "topic_type": "historical",
  "title": "English title (e.g. 'AI Recreates the Last Day of Pompeii' or 'What the Construction of the Pyramids Really Looked Like')",
  "event": "Name of the event",
  "era": "Time period",
  "description": "1-2 sentence English description",
  "scenes": [
    {{
      "scene_number": 1,
      "image_prompt": "ULTRA-DETAILED FLUX prompt. PHOTOREALISTIC historical recreation. Include: exact subject, period-accurate clothing/architecture/technology, real materials and textures, cinematic lighting, weather, atmospheric effects (smoke, dust, fire), camera angle, lens type. 9:16 vertical. 80+ words. Start with 'Photorealistic photograph of...'",
      "camera_movement": "zoom_in | zoom_out | pan_left | pan_right | pan_up | pan_down",
      "sfx_prompt": "Period-appropriate ambient sound",
      "text_overlay": "Text in the LOCAL LANGUAGE of the country (e.g. Latin for Rome, German for Berlin). Year + location. Or null."
    }}
  ]
}}

Generate exactly {num_scenes} scenes. Scene 1 = the HOOK (most dramatic moment)."""


def _load_used_topics() -> list[str]:
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
                f"Scene {i+1}: image_prompt only {prompt_words} words (need 30+)"
            )

        scene.setdefault("scene_number", i + 1)
        scene.setdefault("camera_movement", "zoom_in")
        scene.setdefault("sfx_prompt", "ambient atmosphere")
        if scene.get("text_overlay") in ("null", "none", "", "None"):
            scene["text_overlay"] = None

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
    70% historical, 30% anime (unless forced).
    """
    num_scenes = num_scenes or config.SCENES_PER_VIDEO
    num_scenes = max(4, min(num_scenes, 15))

    used_topics = _load_used_topics()

    # Weighted random: 70% historical, 30% anime
    if force_type:
        topic_type = force_type
    else:
        topic_type = random.choices(
            ["historical", "anime"],
            weights=[70, 30],
            k=1,
        )[0]

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    used_str = "\n".join(f"- {t}" for t in used_topics[-50:]) if used_topics else "(none)"

    if topic_type == "anime":
        # Pick from top 20 internationally popular anime
        anime_name = random.choice(TOP_ANIME)
        user_msg = TOPIC_TEMPLATE_ANIME.format(
            anime_name=anime_name,
            num_scenes=num_scenes,
            used_topics=used_str,
        )
        logger.info(f"Selected anime: {anime_name}")
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
