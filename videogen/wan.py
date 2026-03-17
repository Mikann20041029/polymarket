"""
AI video clip generation using Wan 2.1 via fal.ai.
Generates photorealistic 5-second video clips from text prompts.

Cost: ~$0.10 per clip (Wan 2.1)
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

COST_PER_CLIP = 0.10
MIN_VIDEO_SIZE_BYTES = 50_000


def check_fal_balance(num_clips: int) -> bool:
    """Check fal.ai balance BEFORE generating. Returns True if OK."""
    estimated = num_clips * COST_PER_CLIP
    try:
        resp = requests.get(
            "https://rest.alpha.fal.ai/billing/balance",
            headers={"Authorization": f"Key {config.FAL_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            balance = float(resp.json().get("balance", resp.json().get("amount", 0)))
            logger.info(f"fal.ai balance: ${balance:.2f} (need ~${estimated:.2f})")
            if balance < estimated:
                logger.error(
                    f"INSUFFICIENT BALANCE: ${balance:.2f} < ${estimated:.2f}. "
                    f"Top up at https://fal.ai/dashboard/billing"
                )
                return False
            return True
        elif resp.status_code == 401:
            logger.error("fal.ai auth failed — check FAL_KEY")
            return False
    except Exception as e:
        logger.warning(f"Balance check failed ({e}), proceeding")
    return True


def estimate_cost(num_clips: int) -> float:
    return num_clips * COST_PER_CLIP


def _validate_video(path: Path) -> bool:
    """Check downloaded video is valid."""
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


def generate_clip(prompt: str, output_path: Path) -> str:
    """
    Generate a single 5-second video clip using Wan 2.1.
    Validates download. Retries up to 3 times.
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating clip: '{prompt[:80]}...'")

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

    for attempt in range(3):
        try:
            resp = requests.get(video_url, timeout=300)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)

            if _validate_video(output_path):
                logger.info(f"Clip saved: {output_path} ({output_path.stat().st_size / 1024:.0f}KB)")
                return str(output_path)
            else:
                logger.warning(f"Validation failed (attempt {attempt+1}/3)")
        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt+1}/3): {e}")

    raise RuntimeError(f"Failed to download valid clip after 3 attempts")


def _generate_single(args: tuple) -> tuple[int, str]:
    i, scene, output_dir = args
    out = output_dir / f"clip_{i+1:02d}.mp4"
    path = generate_clip(scene["video_prompt"], out)
    return i, path


def generate_all_clips(scenes: list[dict], output_dir: Path) -> list[str]:
    """Generate video clips for all scenes in parallel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(i, scene, output_dir) for i, scene in enumerate(scenes)]
    paths = [None] * len(scenes)

    max_workers = min(3, len(scenes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_single, t): t[0] for t in tasks}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, path = future.result()
                paths[i] = path
                logger.info(f"Clip {i+1}/{len(scenes)} done")
            except Exception as e:
                logger.error(f"Clip {idx+1}/{len(scenes)} FAILED: {e}")
                raise

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="output/test_clip.mp4")
    p.add_argument("--check-balance", action="store_true")
    args = p.parse_args()

    if args.check_balance:
        check_fal_balance(1)
    else:
        generate_clip(args.prompt, Path(args.output))
