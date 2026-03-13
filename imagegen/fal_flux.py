"""
Image generation using FAL FLUX API.
Generates 3D Pixar-style anthropomorphic object character images.
"""
import os
import logging
import argparse
from pathlib import Path
import requests
import fal_client
import config

logger = logging.getLogger(__name__)

CHARACTER_STYLE = (
    "High-quality stylized 3D animated character render, "
    "Pixar/Disney movie quality, cute anthropomorphic object, "
    "the object has a LARGE face taking up most of the frame, "
    "big expressive eyes with thick eyebrows and a wide open mouth showing teeth, "
    "the face is clearly visible and front-facing for lip-sync animation, "
    "exactly two small stubby arms with hands attached to its body, "
    "warm indoor kitchen lighting, shallow depth of field, "
    "close-up portrait shot, clean solid-color background, "
    "vertical 9:16 portrait framing, face centered and large in frame, "
    "family-friendly, toy-like feel, professional 3D animation quality"
)


def generate_character_image(
    object_name: str,
    scene_description: str,
    output_path: Path,
) -> str:
    """
    Generate a 3D animated character image of an anthropomorphic object.

    Args:
        object_name: The object to anthropomorphize (e.g., "broccoli", "mug")
        scene_description: Full scene description from the script
        output_path: Where to save the PNG

    Returns:
        Path to saved image
    """
    os.environ["FAL_KEY"] = config.FAL_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Close-up portrait of a cute anthropomorphic {object_name} character, "
        f"the {object_name} has a large cartoon face filling most of the frame, "
        f"big expressive eyes and a wide open mouth, front-facing camera, "
        f"exactly two small stubby arms, "
        f"{scene_description}. "
        f"{CHARACTER_STYLE}"
    )

    negative_prompt = (
        "realistic photo, ugly, deformed, blurry, low quality, text, watermark, "
        "2D flat illustration, human, person, anime style, dark, scary, "
        "multiple characters, split screen, extra arms, extra limbs, "
        "multiple arms, many arms, extra hands, deformed hands"
    )

    logger.info(f"Generating image: {object_name} — '{scene_description[:50]}...'")

    result = fal_client.subscribe(
        config.FAL_IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": config.IMAGE_SIZE,
            "num_images": 1,
            "safety_tolerance": "5",
        },
        with_logs=True,
    )

    image_url = result["images"][0]["url"]
    logger.info(f"Image generated: {image_url}")

    # Download
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

    logger.info(f"Image saved: {output_path}")
    return str(output_path)


def generate_all_hack_images(hacks: list[dict], output_dir: Path) -> list[str]:
    """Generate character images for all hacks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, hack in enumerate(hacks):
        out = output_dir / f"hack_{i+1}.png"
        path = generate_character_image(
            object_name=hack["object_character"],
            scene_description=hack["scene_description"],
            output_path=out,
        )
        paths.append(path)
        logger.info(f"Image {i+1}/{len(hacks)} done")

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True, help="Object to anthropomorphize")
    p.add_argument("--scene", default="standing in a kitchen, looking excited")
    p.add_argument("--output", default="output/test_character.png")
    args = p.parse_args()
    result = generate_character_image(args.object, args.scene, Path(args.output))
    print(f"Saved: {result}")
