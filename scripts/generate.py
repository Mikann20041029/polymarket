"""
Topic & scene generation using DeepSeek API.

Content selection (weighted random):
  - 70% Historical events (globally significant, visually dramatic)
  - 30% Anime worlds (top 20 internationally popular anime)

Each scene generates a video_prompt for Wan 2.1 AI video generation.
All titles in English. Historical text overlays in local language.
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
that recreates worlds as photorealistic AI-generated VIDEO clips.

TARGET AUDIENCE: English-speaking global viewers.
ALL TITLES IN ENGLISH.

TWO CONTENT TYPES:

TYPE A — ANIME WORLD RECREATION:
- Recreate a specific anime's world as photorealistic live-action VIDEO
- The viewer feels like they are WALKING THROUGH the anime world with a camera
- Show iconic locations from the anime — looking indistinguishable from real footage
- Camera must MOVE through the scene (tracking shot, dolly, pan) — NOT a still image

TYPE B — HISTORICAL EVENT WITNESS:
- Recreate a famous historical event as if filmed with a camera on scene
- The viewer feels like they TIME-TRAVELED with a video camera
- Show key moments of the event as MOVING footage, not stills
- Pick events that are VISUALLY SPECTACULAR for maximum views
- Text overlays use the LOCAL LANGUAGE of the country

CRITICAL RULES FOR VIDEO PROMPTS:
1. SCENE 1 = THE HOOK — most visually stunning moment. People stop scrolling or leave.
2. Every clip must look like REAL VIDEO FOOTAGE — not CGI, not animation
3. Describe MOTION: camera movement, subject movement, environmental motion (wind, water, fire, crowds)
4. Include: lighting, weather, atmospheric effects (smoke, dust, fog, rain)
5. 9:16 vertical, 5 seconds per clip
6. People/figures for scale where appropriate
7. Consistent visual style across all scenes

OUTPUT: JSON object. No markdown."""

TOPIC_TEMPLATE_ANIME = """Generate a video recreating the world of: {anime_name}

Create 5-second photorealistic VIDEO CLIPS showing iconic locations from this anime
as if a camera crew walked through the real-life version.

PREVIOUSLY USED (DO NOT REPEAT):
{used_topics}

Return JSON:
{{
  "topic_type": "anime",
  "title": "English title (e.g. 'What if {anime_name} Was Real?')",
  "source_anime": "{anime_name}",
  "description": "1-2 sentence English description",
  "scenes": [
    {{
      "scene_number": 1,
      "video_prompt": "Prompt for AI video generation (Wan 2.1). Must describe MOVING footage, not a still image. Include: what the camera sees and how it moves (tracking forward, slow pan, dolly zoom), what is happening in the scene (people walking, wind blowing, water flowing), specific materials and textures looking photorealistic, cinematic lighting, atmospheric effects. 9:16 vertical, 5 seconds. 60+ words. Start with 'Photorealistic video footage of...'",
      "sfx_prompt": "Ambient sound matching the scene (wind, crowds, water, etc.)",
      "text_overlay": null
    }}
  ]
}}

Generate exactly {num_clips} scenes. Scene 1 = HOOK (most stunning)."""

TOPIC_TEMPLATE_HISTORY = """Generate a video recreating a major historical event.

PICK EVENTS THAT ARE:
- Visually SPECTACULAR (destruction, massive scale, dramatic moments)
- Globally known (millions will click)
- Never seen as "real footage" before

HIGH-VIEW CATEGORIES:
- Ancient wonders being built (Pyramids, Colosseum, Great Wall)
- Catastrophic events (Pompeii, Titanic sinking, Hindenburg)
- Epic battles (D-Day, Thermopylae, Waterloo)
- Historic firsts (Moon landing, Wright Brothers, first Olympics)
- Lost civilizations at peak (Ancient Rome, Maya, Ancient Egypt)

PREVIOUSLY USED (DO NOT REPEAT):
{used_topics}

Return JSON:
{{
  "topic_type": "historical",
  "title": "English title (e.g. 'AI Recreates the Last Day of Pompeii')",
  "event": "Name of event",
  "era": "Time period",
  "description": "1-2 sentence English description",
  "scenes": [
    {{
      "scene_number": 1,
      "video_prompt": "Prompt for AI video generation. Must describe MOVING footage. Include: camera movement (tracking, pan, dolly), action happening (explosions, crowds running, construction), period-accurate details (clothing, architecture, technology), lighting, weather, atmospheric effects (smoke, fire, dust). 9:16 vertical, 5 seconds. 60+ words. Start with 'Photorealistic video footage of...'",
      "sfx_prompt": "Period-appropriate ambient sound",
      "text_overlay": "In LOCAL LANGUAGE of the country (Latin for Rome, etc). Year + location. Or null."
    }}
  ]
}}

Generate exactly {num_clips} scenes. Scene 1 = HOOK (most dramatic)."""


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
        if "video_prompt" not in scene or not scene["video_prompt"]:
            raise ValueError(f"Scene {i+1}: missing video_prompt")

        prompt_words = len(scene["video_prompt"].split())
        if prompt_words < 25:
            raise ValueError(
                f"Scene {i+1}: video_prompt only {prompt_words} words (need 25+)"
            )

        scene.setdefault("scene_number", i + 1)
        scene.setdefault("sfx_prompt", "ambient atmosphere")
        if scene.get("text_overlay") in ("null", "none", "", "None"):
            scene["text_overlay"] = None

    return topic


def generate_topic(
    force_type: str = None,
    num_clips: int = None,
) -> dict:
    """
    Generate a video topic with scene breakdown.
    70% historical, 30% anime (unless forced).
    """
    num_clips = num_clips or config.CLIPS_PER_VIDEO
    num_clips = max(3, min(num_clips, 10))

    used_topics = _load_used_topics()

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
        anime_name = random.choice(TOP_ANIME)
        user_msg = TOPIC_TEMPLATE_ANIME.format(
            anime_name=anime_name,
            num_clips=num_clips,
            used_topics=used_str,
        )
        logger.info(f"Selected anime: {anime_name}")
    else:
        user_msg = TOPIC_TEMPLATE_HISTORY.format(
            num_clips=num_clips,
            used_topics=used_str,
        )

    logger.info(f"Generating topic: type={topic_type}, clips={num_clips}")

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
    p.add_argument("--num-clips", type=int, default=None)
    args = p.parse_args()
    result = generate_topic(args.type, args.num_clips)
    print(json.dumps(result, indent=2, ensure_ascii=False))
