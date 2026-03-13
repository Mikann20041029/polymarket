#!/usr/bin/env python3
"""
Life-Hack Short Video Generator — Main Pipeline

Generates a complete vertical short video (40-50 seconds) featuring
anthropomorphic 3D Pixar-style object characters presenting life hacks.

Optimized for speed:
- Image generation runs in parallel (all hacks at once)
- Video generation runs in parallel (all hacks at once)
- Total pipeline: ~10-15 minutes instead of 25-35 minutes

Usage:
    python pipeline.py --topic "kitchen hacks"
    python pipeline.py --topic "cleaning hacks" --num-hacks 3
"""
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import config
from scripts.generate import generate_script
from tts.elevenlabs import generate_all_hack_audio
from imagegen.fal_flux import generate_all_hack_images
from videogen.hailuo import generate_all_hack_videos
from postprocess.effects import compose_final_video

logger = logging.getLogger(__name__)


def _log_timing(step_name: str, start: float) -> float:
    """Log how long a step took and return current time."""
    elapsed = time.time() - start
    logger.info(f"  ⏱ {step_name}: {elapsed:.1f}s")
    return time.time()


def run_pipeline(
    topic: str = "life hacks across all categories",
    num_hacks: int = None,
    bgm_path: str = None,
    sfx_path: str = None,
    output_name: str = None,
) -> str:
    """
    Run the full video generation pipeline.

    Timeline (parallel execution):
      Step 1: Script gen (DeepSeek)      ~10-15s
      Step 2: TTS audio (Edge TTS)       ~15-30s
      Step 3: Images (FAL FLUX) x3       ~2-3min  (PARALLEL)
      Step 4: Videos (FAL Hailuo) x3     ~5-8min  (PARALLEL)
      Step 5: FFmpeg post-process         ~1-2min
      TOTAL:                              ~10-15min

    Returns path to the final output video.
    """
    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"lifehack_{timestamp}.mp4"

    final_output = config.OUTPUT_DIR / output_name

    logger.info(f"=== PIPELINE START: topic='{topic}' ===")
    logger.info(f"Run directory: {run_dir}")

    step_time = time.time()

    # ── Step 1: Generate script ──────────────────────────
    logger.info("── Step 1/5: Generating script...")
    hacks = generate_script(topic, num_hacks)

    script_file = run_dir / "script.json"
    with open(script_file, "w") as f:
        json.dump(hacks, f, indent=2, ensure_ascii=False)
    logger.info(f"Script: {len(hacks)} hacks generated")

    for h in hacks:
        logger.info(f"  #{h['hack_number']}: {h['title']} ({h['object_character']})")

    step_time = _log_timing("Script generation", step_time)

    # ── Step 2: Generate TTS audio ───────────────────────
    logger.info("── Step 2/5: Generating TTS audio...")
    audio_dir = run_dir / "audio"
    tts_results = generate_all_hack_audio(hacks, audio_dir)

    tts_file = run_dir / "tts_results.json"
    with open(tts_file, "w") as f:
        json.dump(tts_results, f, indent=2, default=str)

    total_audio = sum(r["duration"] for r in tts_results)
    logger.info(f"TTS done: {total_audio:.1f}s total audio")

    step_time = _log_timing("TTS generation", step_time)

    # ── Step 3: Generate character images (PARALLEL) ─────
    logger.info("── Step 3/5: Generating character images (parallel)...")
    images_dir = run_dir / "images"
    image_paths = generate_all_hack_images(hacks, images_dir)
    logger.info(f"Images done: {len(image_paths)} images")

    step_time = _log_timing("Image generation (parallel)", step_time)

    # ── Step 4: Generate animated video clips (PARALLEL) ─
    logger.info("── Step 4/5: Generating animated video clips (parallel)...")
    videos_dir = run_dir / "videos"
    video_paths = generate_all_hack_videos(hacks, image_paths, videos_dir)
    logger.info(f"Videos done: {len(video_paths)} clips")

    step_time = _log_timing("Video generation (parallel)", step_time)

    # ── Step 5: Post-process and compose ─────────────────
    logger.info("── Step 5/5: Composing final video...")

    # Find BGM and SFX files
    if not bgm_path:
        bgm_candidates = list(config.BGM_DIR.glob("*.*"))
        if bgm_candidates:
            bgm_path = str(bgm_candidates[0])

    if not sfx_path:
        sfx_candidates = list(config.SFX_DIR.glob("*.*"))
        if sfx_candidates:
            sfx_path = str(sfx_candidates[0])

    audio_paths = [r["audio_path"] for r in tts_results]

    result = compose_final_video(
        video_paths=video_paths,
        tts_results=tts_results,
        audio_paths=audio_paths,
        output_path=final_output,
        bgm_path=bgm_path,
        sfx_transition_path=sfx_path,
    )

    _log_timing("Post-processing", step_time)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"=== PIPELINE COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min) ===")
    logger.info(f"Final video: {result}")
    logger.info(f"Run data saved in: {run_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate life-hack short videos with animated object characters",
    )
    parser.add_argument(
        "--topic",
        default="life hacks across all categories",
        help="Topic for life hacks (default: life hacks across all categories)",
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
