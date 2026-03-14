"""
Video generation using Kling via fal.ai API.
Generates surreal, physics-defying satisfying video clips from text prompts.
No image input needed — pure text-to-video.
"""
import os
import logging
import argparse
from pathlib import Path
import requests
import fal_client
import config

logger = logging.getLogger(__name__)


def generate_video_clip(
    visual_prompt: str,
    output_path: Path,
) -> str:
    """
    Generate a satisfying video clip from a text description.

    Primary: Kling via fal.ai (best physics simulation)
    Fallback: Wan 2.1 (open-source, decent quality)

    Args:
        visual_prompt: Detailed visual description for the AI video model
        output_path: Where to save the mp4

    Returns:
        Path to generated video
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        video_url = _generate_with_kling(visual_prompt)
    except Exception as e:
        logger.warning(f"Kling failed ({e}), falling back to Wan 2.1...")
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
    """Generate video using Wan 2.1 via fal.ai (fallback)."""
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
    args = p.parse_args()
    result = generate_video_clip(args.prompt, Path(args.output))
    print(f"Saved: {result}")
