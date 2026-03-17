"""
Photorealistic image generation using FLUX Dev via fal.ai.
Generates ultra-realistic images for anime world / historical event recreation.

Cost: ~$0.025 per image (FLUX Dev)
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

MIN_IMAGE_SIZE_BYTES = 10_000  # 10KB minimum


def _validate_image(path: Path) -> bool:
    """Check downloaded image is valid."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < MIN_IMAGE_SIZE_BYTES:
        logger.error(f"Image too small ({size} bytes): {path}")
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    # Check for PNG or JPEG magic bytes
    if not (header[:4] == b'\x89PNG' or header[:2] == b'\xff\xd8'):
        logger.error(f"Image has invalid header: {path}")
        return False
    return True


def generate_image(
    prompt: str,
    output_path: Path,
    width: int = None,
    height: int = None,
) -> str:
    """
    Generate a single photorealistic image using FLUX Dev.

    Args:
        prompt: Detailed image description (should start with "Photorealistic photograph of...")
        output_path: Where to save the image
        width: Image width (default from config)
        height: Image height (default from config)

    Returns path to saved image.
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = width or config.IMAGE_WIDTH
    height = height or config.IMAGE_HEIGHT

    logger.info(f"Generating image: '{prompt[:80]}...'")

    result = fal_client.subscribe(
        config.FAL_IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": {
                "width": width,
                "height": height,
            },
            "num_images": 1,
            "enable_safety_checker": True,
        },
        with_logs=True,
    )

    image_url = result["images"][0]["url"]

    # Download with retry
    for attempt in range(3):
        try:
            resp = requests.get(image_url, timeout=60)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)

            if _validate_image(output_path):
                logger.info(f"Image saved: {output_path} ({output_path.stat().st_size / 1024:.0f}KB)")
                return str(output_path)
            else:
                logger.warning(f"Image validation failed (attempt {attempt+1}/3)")
        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt+1}/3): {e}")

    raise RuntimeError(f"Failed to download valid image after 3 attempts")


def _generate_single(args: tuple) -> tuple[int, str]:
    """Generate one image (for parallel execution)."""
    i, scene, output_dir = args
    out = output_dir / f"scene_{i+1:02d}.png"
    path = generate_image(scene["image_prompt"], out)
    return i, path


def generate_all_scene_images(scenes: list[dict], output_dir: Path) -> list[str]:
    """
    Generate images for all scenes in parallel.
    Returns list of image paths in scene order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(i, scene, output_dir) for i, scene in enumerate(scenes)]
    paths = [None] * len(scenes)

    # Parallel: fal.ai handles queuing server-side
    max_workers = min(4, len(scenes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_single, t): t[0] for t in tasks}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, path = future.result()
                paths[i] = path
                logger.info(f"Image {i+1}/{len(scenes)} done")
            except Exception as e:
                logger.error(f"Image {idx+1}/{len(scenes)} FAILED: {e}")
                raise

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="output/test_image.png")
    args = p.parse_args()
    result = generate_image(args.prompt, Path(args.output))
    print(f"Saved: {result}")
