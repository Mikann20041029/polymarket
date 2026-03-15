"""
Sound effects generation using ElevenLabs Sound Effects V2 API.
Generates ASMR/satisfying sounds from text descriptions.
No voice, no narration — pure sound design for satisfying videos.
"""
import time
import logging
import argparse
from pathlib import Path
import requests
import config

logger = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1/sound-generation"
MAX_RETRIES = 3


def generate_sfx(
    prompt: str,
    output_path: Path,
    duration: float = None,
    loop: bool = False,
) -> dict:
    """
    Generate a satisfying sound effect from a text description.
    Retries up to 3 times on transient failures.
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

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                params={"output_format": "mp3_44100_128"},
                timeout=120,
            )

            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                raise RuntimeError("ElevenLabs auth failed — check ELEVENLABS_API_KEY")

            resp.raise_for_status()

            # Validate response is actual audio
            if len(resp.content) < 1000:
                raise RuntimeError(f"SFX response too small ({len(resp.content)} bytes)")

            with open(output_path, "wb") as f:
                f.write(resp.content)

            logger.info(f"SFX saved: {output_path}")
            return {
                "audio_path": str(output_path),
                "duration": duration,
                "prompt": prompt,
            }

        except requests.exceptions.ConnectionError as e:
            last_error = e
            wait = 2 ** (attempt + 1)
            logger.warning(f"Connection error (attempt {attempt+1}/{MAX_RETRIES}), retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"SFX generation failed after {MAX_RETRIES} attempts: {last_error}")


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
