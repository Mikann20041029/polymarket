"""
Sound effects generation using ElevenLabs Sound Effects V2 API.
Generates ASMR/satisfying sounds from text descriptions.
No voice, no narration — pure sound design for satisfying videos.
"""
import logging
import argparse
from pathlib import Path
import requests
import config

logger = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1/sound-generation"


def generate_sfx(
    prompt: str,
    output_path: Path,
    duration: float = None,
    loop: bool = False,
) -> dict:
    """
    Generate a satisfying sound effect from a text description.

    Args:
        prompt: Detailed description of the sound (e.g., "crisp glass shattering
                into tiny shards, each piece producing a delicate tinkling chime,
                subtle reverb, ASMR quality")
        output_path: Where to save the audio file
        duration: Duration in seconds (0.5-30). None = auto.
        loop: Whether to create a seamlessly looping sound.

    Returns dict with:
        - audio_path: path to generated audio
        - duration: requested duration
        - prompt: the prompt used
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = duration or config.SFX_DURATION_SECONDS

    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": prompt,
        "duration_seconds": duration,
        "prompt_influence": config.SFX_PROMPT_INFLUENCE,
        "loop": loop,
    }

    logger.info(f"Generating SFX ({duration}s): '{prompt[:80]}...'")

    resp = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        params={"output_format": "mp3_44100_128"},
        timeout=120,
    )
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(resp.content)

    logger.info(f"SFX saved: {output_path}")
    return {
        "audio_path": str(output_path),
        "duration": duration,
        "prompt": prompt,
    }


def generate_all_clip_sfx(concepts: list[dict], output_dir: Path) -> list[dict]:
    """Generate sound effects for all video concepts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, concept in enumerate(concepts):
        out = output_dir / f"clip_{i+1}_sfx.mp3"
        loop = concept.get("loop_friendly", False)
        result = generate_sfx(
            prompt=concept["sound_prompt"],
            output_path=out,
            loop=loop,
        )
        result["clip_number"] = concept.get("clip_number", i + 1)
        result["title"] = concept.get("title", "")
        results.append(result)
        logger.info(f"SFX {i+1}/{len(concepts)} done")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="Sound description")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--output", default="output/test_sfx.mp3")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()
    result = generate_sfx(args.prompt, Path(args.output), args.duration, args.loop)
    print(f"Saved: {result['audio_path']}")
