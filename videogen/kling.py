"""
Video generation using Wan 2.1 / Kling via fal.ai API.
Generates surreal, physics-defying satisfying video clips from text prompts.

Includes:
- Balance checking to prevent running with insufficient funds
- Download validation to catch corrupt/empty videos
- Parallel clip generation for speed
"""
import os
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import fal_client
import config

logger = logging.getLogger(__name__)

# Cost estimates per clip (in USD) — conservative
COST_ESTIMATES = {
    "kling": 0.50,
    "wan": 0.10,
}

MIN_VIDEO_SIZE_BYTES = 50_000  # 50KB minimum — anything smaller is corrupt


def check_fal_balance() -> float:
    """
    Check fal.ai account balance. Returns balance in USD.
    Returns float("inf") only if balance endpoint is genuinely unavailable.
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
        elif resp.status_code == 401:
            logger.error("fal.ai auth failed — check FAL_KEY")
            return 0.0  # Auth failed = assume no balance
        else:
            logger.warning(f"Balance check returned {resp.status_code}")
            return float("inf")
    except requests.exceptions.ConnectionError:
        logger.warning("Could not reach fal.ai — network issue")
        return float("inf")
    except Exception as e:
        logger.warning(f"Balance check error: {e}")
        return float("inf")


def estimate_cost(num_clips: int, model: str = "wan") -> float:
    """Estimate total cost for generating clips."""
    per_clip = COST_ESTIMATES.get(model, COST_ESTIMATES["wan"])
    return per_clip * num_clips


def _validate_video(path: Path) -> bool:
    """Check that downloaded video is valid (not empty, not HTML error page)."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < MIN_VIDEO_SIZE_BYTES:
        logger.error(f"Video too small ({size} bytes): {path}")
        return False
    # Check for HTML error pages disguised as video
    with open(path, "rb") as f:
        header = f.read(16)
    if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
        logger.error(f"Video is HTML error page: {path}")
        return False
    return True


def generate_video_clip(
    visual_prompt: str,
    output_path: Path,
) -> str:
    """
    Generate a satisfying video clip from a text description.

    Uses Wan 2.1 by default (cheapest). Kling only if FAL_VIDEO_MODEL is
    explicitly set to a kling endpoint.

    Validates the downloaded video to prevent corrupt output.
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_kling = "kling" in config.FAL_VIDEO_MODEL.lower()

    if use_kling:
        try:
            video_url = _generate_with_kling(visual_prompt)
        except Exception as e:
            logger.warning(f"Kling failed ({e}), falling back to Wan 2.1...")
            video_url = _generate_with_wan(visual_prompt)
    else:
        video_url = _generate_with_wan(visual_prompt)

    # Download with retry
    for attempt in range(3):
        try:
            resp = requests.get(video_url, timeout=300)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)

            if _validate_video(output_path):
                logger.info(f"Video saved: {output_path} ({output_path.stat().st_size / 1024:.0f}KB)")
                return str(output_path)
            else:
                logger.warning(f"Video validation failed (attempt {attempt+1}/3)")
        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt+1}/3): {e}")

    raise RuntimeError(f"Failed to download valid video after 3 attempts: {video_url}")


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


def _generate_single_clip(args: tuple) -> tuple[int, str]:
    """Generate a single clip (for parallel execution). Returns (index, path)."""
    i, concept, output_dir = args
    out = output_dir / f"clip_{i+1}.mp4"
    path = generate_video_clip(concept["visual_prompt"], out)
    return i, path


def generate_all_clip_videos(concepts: list[dict], output_dir: Path) -> list[str]:
    """Generate video clips for all concepts in parallel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parallel generation — fal.ai handles queuing server-side
    tasks = [(i, concept, output_dir) for i, concept in enumerate(concepts)]
    paths = [None] * len(concepts)

    with ThreadPoolExecutor(max_workers=min(3, len(concepts))) as executor:
        futures = {executor.submit(_generate_single_clip, t): t[0] for t in tasks}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, path = future.result()
                paths[i] = path
                logger.info(f"Video {i+1}/{len(concepts)} done")
            except Exception as e:
                logger.error(f"Video {idx+1}/{len(concepts)} FAILED: {e}")
                raise

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
