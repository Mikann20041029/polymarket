"""
Lip-sync video generation using VEED Fabric 1.0 (primary) and SadTalker (fallback).

Takes a character image + TTS audio and generates a talking video where
the character's mouth moves in sync with the speech. Video duration
matches audio duration exactly — no more short clip loops.
"""
import os
import logging
import argparse
from pathlib import Path
import requests
import fal_client
import config

logger = logging.getLogger(__name__)


def generate_lipsync_video(
    image_path: str,
    audio_path: str,
    output_path: Path,
) -> str:
    """
    Generate a lip-synced talking video from character image + audio.

    Primary: VEED Fabric 1.0 (phoneme-driven mouth animation)
    Fallback: SadTalker (3D motion coefficient-based lip sync)

    Args:
        image_path: Path to character image (PNG)
        audio_path: Path to TTS audio (MP3)
        output_path: Where to save the mp4

    Returns:
        Path to generated video
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Upload image and audio to FAL
    image_url = fal_client.upload_file(image_path)
    audio_url = fal_client.upload_file(audio_path)
    logger.info(f"Uploaded image: {image_url}")
    logger.info(f"Uploaded audio: {audio_url}")

    # Try VEED Fabric 1.0 first (best quality for any image type)
    try:
        video_url = _generate_with_fabric(image_url, audio_url)
    except Exception as e:
        logger.warning(f"VEED Fabric failed ({e}), falling back to SadTalker...")
        video_url = _generate_with_sadtalker(image_url, audio_url)

    # Download the result
    resp = requests.get(video_url, timeout=300)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

    logger.info(f"Lip-sync video saved: {output_path}")
    return str(output_path)


def _generate_with_fabric(image_url: str, audio_url: str) -> str:
    """Generate lip-sync video using VEED Fabric 1.0."""
    logger.info("Generating lip-sync with VEED Fabric 1.0...")

    result = fal_client.subscribe(
        config.LIPSYNC_MODEL_PRIMARY,
        arguments={
            "image_url": image_url,
            "audio_url": audio_url,
            "resolution": config.LIPSYNC_RESOLUTION,
        },
        with_logs=True,
    )

    video_url = result["video"]["url"]
    logger.info(f"VEED Fabric video ready: {video_url}")
    return video_url


def _generate_with_sadtalker(image_url: str, audio_url: str) -> str:
    """Generate lip-sync video using SadTalker (fallback)."""
    logger.info("Generating lip-sync with SadTalker...")

    result = fal_client.subscribe(
        config.LIPSYNC_MODEL_FALLBACK,
        arguments={
            "source_image_url": image_url,
            "driven_audio_url": audio_url,
            "face_model_resolution": config.SADTALKER_FACE_RESOLUTION,
            "expression_scale": config.SADTALKER_EXPRESSION_SCALE,
            "preprocess": config.SADTALKER_PREPROCESS,
            "still_mode": False,
        },
        with_logs=True,
    )

    video_url = result["video"]["url"]
    logger.info(f"SadTalker video ready: {video_url}")
    return video_url


def generate_all_hack_videos(
    hacks: list[dict],
    image_paths: list[str],
    audio_paths: list[str],
    output_dir: Path,
) -> list[str]:
    """
    Generate lip-synced video clips for all hacks.

    Unlike the old Hailuo approach, this takes audio as input so
    the video duration matches the speech exactly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, (hack, img, aud) in enumerate(zip(hacks, image_paths, audio_paths)):
        out = output_dir / f"hack_{i+1}.mp4"
        path = generate_lipsync_video(img, aud, out)
        paths.append(path)
        logger.info(f"Lip-sync video {i+1}/{len(hacks)} done")

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to character image")
    p.add_argument("--audio", required=True, help="Path to TTS audio")
    p.add_argument("--output", default="output/test_lipsync.mp4")
    args = p.parse_args()
    result = generate_lipsync_video(args.image, args.audio, Path(args.output))
    print(f"Saved: {result}")
