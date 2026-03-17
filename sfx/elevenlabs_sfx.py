"""
Ambient sound effects generation using ElevenLabs Sound Effects V2 API.
Generates environmental/atmospheric sounds for scene immersion.
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
) -> dict:
    """Generate an ambient sound effect. Retries on transient failures."""
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
    }

    logger.info(f"Generating SFX ({duration}s): '{prompt[:60]}...'")

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
            logger.warning(f"Connection error (attempt {attempt+1}/{MAX_RETRIES}), retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"SFX generation failed after {MAX_RETRIES} attempts: {last_error}")


def generate_all_scene_sfx(scenes: list[dict], output_dir: Path) -> list[dict]:
    """Generate ambient SFX for all scenes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, scene in enumerate(scenes):
        out = output_dir / f"scene_{i+1:02d}_sfx.mp3"
        prompt = scene.get("sfx_prompt", "ambient atmosphere, gentle background")
        result = generate_sfx(prompt=prompt, output_path=out)
        result["scene_number"] = scene.get("scene_number", i + 1)
        results.append(result)
        logger.info(f"SFX {i+1}/{len(scenes)} done")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--output", default="output/test_sfx.mp3")
    args = p.parse_args()
    result = generate_sfx(args.prompt, Path(args.output), args.duration)
    print(f"Saved: {result['audio_path']}")
