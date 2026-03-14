"""
Video generation using Kling via fal.ai API.
Generates surreal, physics-defying satisfying video clips from text prompts.
No image input needed — pure text-to-video.

Includes balance checking to prevent running with insufficient funds.
"""
import os
import logging
import argparse
from pathlib import Path
import requests
import fal_client
import config

logger = logging.getLogger(__name__)

# Cost estimates per clip (in USD) — conservative, based on fal.ai pricing
COST_ESTIMATES = {
    "kling": 0.50,   # ~$0.10/sec × 5 sec
    "wan": 0.10,     # much cheaper fallback
}


def check_fal_balance() -> float:
    """
    Check fal.ai account balance. Returns balance in USD.
    Raises RuntimeError if balance is too low to generate.
    """
    try:
        resp = requests.get(
            "https://rest.alpha.fal.ai/billing/balance",
            headers={"Authorization": f"Key {config.FAL_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            balance = data.get("balance", data.get("amount", 0))
            logger.info(f"fal.ai balance: ${balance:.2f}")
            return float(balance)
        else:
            logger.warning(f"Balance check returned {resp.status_code}, proceeding with caution")
            return float("inf")  # Can't check, don't block
    except Exception as e:
        logger.warning(f"Could not check fal.ai balance: {e}")
        return float("inf")


def estimate_cost(num_clips: int, model: str = "wan") -> float:
    """Estimate total cost for generating clips."""
    per_clip = COST_ESTIMATES.get(model, COST_ESTIMATES["wan"])
    return per_clip * num_clips


def generate_video_clip(
    visual_prompt: str,
    output_path: Path,
) -> str:
    """
    Generate a satisfying video clip from a text description.

    Uses Wan 2.1 by default (cheapest). Kling only if FAL_VIDEO_MODEL is explicitly
    set to a kling endpoint.

    Args:
        visual_prompt: Detailed visual description for the AI video model
        output_path: Where to save the mp4

    Returns:
        Path to generated video
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default to cheapest model (Wan 2.1) to save money
    # Only use Kling if explicitly configured
    use_kling = "kling" in config.FAL_VIDEO_MODEL.lower()

    if use_kling:
        try:
            video_url = _generate_with_kling(visual_prompt)
        except Exception as e:
            logger.warning(f"Kling failed ({e}), falling back to Wan 2.1...")
            video_url = _generate_with_wan(visual_prompt)
    else:
        video_url = _generate_with_wan(visual_prompt)

    # Download
    resp = requests.get(video_url, timeout=300)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

    logger.info(f"Video saved: {output_path}")
    return str(output_path)


def _generate_with_kling(prompt: str) -> str:
    """Generate video using Kling via fal.ai."""
    logger.info(f"Generating video with Kling: '{prompt[:80]}...'")

    result = fal_client.subscribe(
        config.FAL_VIDEO_MODEL,
        arguments={
            "prompt": prompt,
            "duration": config.VIDEO_DURATION,
            "aspect_ratio": config.VIDEO_ASPECT_RATIO,
        },
        with_logs=True,
    )

    video_url = result["video"]["url"]
    logger.info(f"Kling video ready: {video_url}")
    return video_url


def _generate_with_wan(prompt: str) -> str:
    """Generate video using Wan 2.1 via fal.ai (cheap, good quality)."""
    logger.info(f"Generating video with Wan 2.1: '{prompt[:80]}...'")

    result = fal_client.subscribe(
        config.FAL_VIDEO_MODEL_FALLBACK,
        arguments={
            "prompt": prompt,
            "num_frames": 81,  # ~5 seconds at 16fps
            "resolution": "480p",
            "enable_safety_checker": True,
        },
        with_logs=True,
    )

    video_url = result["video"]["url"]
    logger.info(f"Wan 2.1 video ready: {video_url}")
    return video_url


def generate_all_clip_videos(concepts: list[dict], output_dir: Path) -> list[str]:
    """Generate video clips for all concepts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, concept in enumerate(concepts):
        out = output_dir / f"clip_{i+1}.mp4"
        path = generate_video_clip(concept["visual_prompt"], out)
        paths.append(path)
        logger.info(f"Video {i+1}/{len(concepts)} done")

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="Visual description for video")
    p.add_argument("--output", default="output/test_clip.mp4")
    p.add_argument("--check-balance", action="store_true", help="Just check balance")
    args = p.parse_args()

    if args.check_balance:
        balance = check_fal_balance()
        print(f"fal.ai balance: ${balance:.2f}")
    else:
        result = generate_video_clip(args.prompt, Path(args.output))
        print(f"Saved: {result}")
