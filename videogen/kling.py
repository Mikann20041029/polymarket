"""
Video generation using Wan 2.1 via fal.ai API.
Generates surreal, physics-defying satisfying video clips from text prompts.

Includes:
- Balance checking before spending money
- Download validation (reject corrupt/empty/HTML files)
- Parallel clip generation
- Retry on transient failures
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

# Cost per clip in USD (conservative estimates)
COST_PER_CLIP_WAN = 0.10
MIN_VIDEO_SIZE_BYTES = 50_000  # 50KB — anything smaller is corrupt


def check_fal_balance() -> float:
    """
    Check fal.ai account balance. Returns balance in USD.
    Returns 0.0 on auth failure, float("inf") if endpoint unreachable.
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
            return 0.0
        else:
            logger.warning(f"Balance check returned {resp.status_code}")
            return float("inf")
    except requests.exceptions.ConnectionError:
        logger.warning("Could not reach fal.ai — network issue")
        return float("inf")
    except Exception as e:
        logger.warning(f"Balance check error: {e}")
        return float("inf")


def estimate_cost(num_clips: int) -> float:
    """Estimate total fal.ai cost for generating clips."""
    return COST_PER_CLIP_WAN * num_clips


def _validate_video(path: Path) -> bool:
    """Check that downloaded video is valid."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < MIN_VIDEO_SIZE_BYTES:
        logger.error(f"Video too small ({size} bytes): {path}")
        return False
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
    Generate a video clip from a text prompt using Wan 2.1 via fal.ai.
    Validates the download. Retries up to 3 times on failure.
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def _generate_with_wan(prompt: str) -> str:
    """Generate video using Wan 2.1 via fal.ai."""
    logger.info(f"Generating video with Wan 2.1 ({config.WAN_RESOLUTION}): '{prompt[:80]}...'")

    result = fal_client.subscribe(
        config.FAL_VIDEO_MODEL,
        arguments={
            "prompt": prompt,
            "num_frames": config.WAN_NUM_FRAMES,
            "resolution": config.WAN_RESOLUTION,
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
    p.add_argument("--check-balance", action="store_true")
    args = p.parse_args()

    if args.check_balance:
        balance = check_fal_balance()
        print(f"fal.ai balance: ${balance:.2f}")
    else:
        result = generate_video_clip(args.prompt, Path(args.output))
        print(f"Saved: {result}")
