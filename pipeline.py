#!/usr/bin/env python3
"""
Life-Hack Short Video Generator — Main Pipeline

Generates a complete vertical short video (40-50 seconds) featuring
anthropomorphic 3D Pixar-style object characters presenting life hacks
with lip-synced speech animation.

Usage:
    python pipeline.py --topic "kitchen hacks"
    python pipeline.py --topic "cleaning hacks" --num-hacks 3

Pipeline:
    1. DeepSeek → script (hack ideas + narration + scene descriptions)
    2. ElevenLabs → expressive TTS audio with word-level timestamps
    3. FAL FLUX → 3D character images (anthropomorphic objects)
    4. VEED Fabric / SadTalker → lip-synced talking video (image+audio → video)
    5. FFmpeg → burn subtitles, add BGM, stitch into one short
"""
import json
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import config
from scripts.generate import generate_script
from tts.elevenlabs import generate_all_hack_audio
from imagegen.fal_flux import generate_all_hack_images
from videogen.lipsync import generate_all_hack_videos
from postprocess.effects import compose_final_video

logger = logging.getLogger(__name__)


def run_pipeline(
    topic: str = "kitchen and household",
    num_hacks: int = None,
    bgm_path: str = None,
    sfx_path: str = None,
    output_name: str = None,
) -> str:
    """
    Run the full video generation pipeline.

    Returns path to the final output video.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"lifehack_{timestamp}.mp4"

    final_output = config.OUTPUT_DIR / output_name

    logger.info(f"=== PIPELINE START: topic='{topic}' ===")
    logger.info(f"Run directory: {run_dir}")

    # ── Step 1: Generate script ──────────────────────────
    logger.info("── Step 1/5: Generating script...")
    hacks = generate_script(topic, num_hacks)

    script_file = run_dir / "script.json"
    with open(script_file, "w") as f:
        json.dump(hacks, f, indent=2, ensure_ascii=False)
    logger.info(f"Script: {len(hacks)} hacks generated")

    for h in hacks:
        logger.info(f"  #{h['hack_number']}: {h['title']} ({h['object_character']})")

    # ── Step 2: Generate TTS audio ───────────────────────
    logger.info("── Step 2/5: Generating TTS audio...")
    audio_dir = run_dir / "audio"
    tts_results = generate_all_hack_audio(hacks, audio_dir)

    tts_file = run_dir / "tts_results.json"
    with open(tts_file, "w") as f:
        json.dump(tts_results, f, indent=2, default=str)

    total_audio = sum(r["duration"] for r in tts_results)
    logger.info(f"TTS done: {total_audio:.1f}s total audio")

    # ── Step 3: Generate character images ────────────────
    logger.info("── Step 3/5: Generating character images...")
    images_dir = run_dir / "images"
    image_paths = generate_all_hack_images(hacks, images_dir)
    logger.info(f"Images done: {len(image_paths)} images")

    # ── Step 4: Generate lip-synced video clips ──────────
    # Unlike the old Hailuo approach, lip-sync takes BOTH image and audio
    # so the video duration matches the speech exactly.
    logger.info("── Step 4/5: Generating lip-synced video clips...")
    videos_dir = run_dir / "videos"
    audio_paths = [r["audio_path"] for r in tts_results]
    video_paths = generate_all_hack_videos(hacks, image_paths, audio_paths, videos_dir)
    logger.info(f"Videos done: {len(video_paths)} lip-synced clips")

    # ── Step 5: Post-process and compose ─────────────────
    # Lip-synced videos already have audio baked in from the model,
    # but we still overlay the original TTS for precise audio quality
    # and add subtitles + BGM.
    logger.info("── Step 5/5: Composing final video...")

    if not bgm_path:
        bgm_candidates = list(config.BGM_DIR.glob("*.*"))
        if bgm_candidates:
            bgm_path = str(bgm_candidates[0])

    if not sfx_path:
        sfx_candidates = list(config.SFX_DIR.glob("*.*"))
        if sfx_candidates:
            sfx_path = str(sfx_candidates[0])

    result = compose_final_video(
        video_paths=video_paths,
        tts_results=tts_results,
        audio_paths=audio_paths,
        output_path=final_output,
        bgm_path=bgm_path,
        sfx_transition_path=sfx_path,
    )

    logger.info(f"=== PIPELINE COMPLETE ===")
    logger.info(f"Final video: {result}")
    logger.info(f"Run data saved in: {run_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate life-hack short videos with lip-synced animated characters",
    )
    parser.add_argument(
        "--topic",
        default="kitchen and household",
        help="Topic for life hacks (default: kitchen and household)",
    )
    parser.add_argument(
        "--num-hacks",
        type=int,
        default=None,
        help=f"Number of hacks per video (default: {config.HACKS_PER_VIDEO})",
    )
    parser.add_argument(
        "--bgm",
        default=None,
        help="Path to background music file",
    )
    parser.add_argument(
        "--sfx",
        default=None,
        help="Path to transition SFX file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (saved in output/ directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_pipeline(
        topic=args.topic,
        num_hacks=args.num_hacks,
        bgm_path=args.bgm,
        sfx_path=args.sfx,
        output_name=args.output,
    )

    print(f"\nDone! Video saved to: {result}")


if __name__ == "__main__":
    main()
