"""
Video generation using FAL Hailuo 2.3 Fast API.
Converts static character images into animated video clips with
expressive body movement, facial animation, and hand gestures.
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
    image_path: str,
    motion_prompt: str,
    output_path: Path,
    duration: str = None,
) -> str:
    """
    Generate an animated video clip from a character image.

    Args:
        image_path: Path to character image
        motion_prompt: How the character should move/act
        output_path: Where to save the mp4
        duration: "6" or "10" seconds

    Returns:
        Path to generated video
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    duration = duration or config.HAILUO_DURATION
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Upload image
    image_url = fal_client.upload_file(image_path)
    logger.info(f"Uploaded: {image_url}")

    full_prompt = (
        f"3D Pixar-style animated character performing with exaggerated emotions. "
        f"{motion_prompt}. "
        f"The character talks animatedly with big mouth movements, "
        f"waves arms with expressive hand gestures, "
        f"bounces and leans with comedic energy. "
        f"Smooth fluid animation, cinematic warm lighting, shallow depth of field. "
        f"The character stays centered in the vertical frame."
    )

    logger.info(f"Generating video ({duration}s): '{motion_prompt[:60]}...'")

    result = fal_client.subscribe(
        config.FAL_VIDEO_MODEL,
        arguments={
            "image_url": image_url,
            "prompt": full_prompt,
            "duration": duration,
            "prompt_optimizer": config.HAILUO_PROMPT_OPTIMIZER,
        },
        with_logs=True,
    )

    video_url = result["video"]["url"]
    logger.info(f"Video ready: {video_url}")

    resp = requests.get(video_url, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

    logger.info(f"Video saved: {output_path}")
    return str(output_path)


def generate_all_hack_videos(
    hacks: list[dict],
    image_paths: list[str],
    output_dir: Path,
) -> list[str]:
    """Generate video clips for all hacks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, (hack, img) in enumerate(zip(hacks, image_paths)):
        out = output_dir / f"hack_{i+1}.mp4"
        motion = hack.get("motion_prompt", "character talking and gesturing energetically")
        path = generate_video_clip(img, motion, out)
        paths.append(path)
        logger.info(f"Video {i+1}/{len(hacks)} done")

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="output/test_video.mp4")
    p.add_argument("--duration", default=None)
    args = p.parse_args()
    result = generate_video_clip(args.image, args.prompt, Path(args.output), args.duration)
    print(f"Saved: {result}")
