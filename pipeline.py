#!/usr/bin/env python3
"""
Impossible Satisfying Video Generator — Main Pipeline

Generates surreal, physics-defying, oddly satisfying vertical short videos
(15-30 seconds) with matching ASMR sound effects. No language needed.

Usage:
    python pipeline.py --theme "glass and crystal"
    python pipeline.py --theme "liquid metal" --num-clips 5
    python pipeline.py --batch 3  # Generate 3 separate videos

Pipeline (3 steps, all via existing API keys):
    1. DeepSeek → concept (visual prompt + sound prompt)
    2. Kling 3.0 (fal.ai) → video clip  }  run in parallel
       ElevenLabs SFX V2 → ASMR audio   }  per concept
    3. FFmpeg → layer sound on video, optional text overlay, stitch + BGM
"""
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import config
from scripts.generate import generate_concepts
from sfx.elevenlabs_sfx import generate_all_clip_sfx
from videogen.kling import generate_all_clip_videos
from postprocess.effects import compose_final_video

logger = logging.getLogger(__name__)

# Theme ideas for variety across batches
THEMES = [
    "glass and crystal physics",
    "liquid metal and mercury",
    "impossible food transformations",
    "magnetic and gravitational anomalies",
    "ice and fire paradoxes",
    "organic growth and bloom",
    "geometric impossibilities",
    "miniature worlds inside objects",
    "color-shifting materials",
    "reverse entropy and time manipulation",
]


def _log_timing(step_name: str, start: float) -> float:
    """Log how long a step took and return current time."""
    elapsed = time.time() - start
    logger.info(f"  ⏱ {step_name}: {elapsed:.1f}s")
    return time.time()


def run_pipeline(
    theme: str = "surreal physics",
    num_clips: int = None,
    bgm_path: str = None,
    output_name: str = None,
) -> str:
    """
    Run the full video generation pipeline.

    Returns path to the final output video.
    """
    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"satisfying_{timestamp}.mp4"

    final_output = config.OUTPUT_DIR / output_name

    logger.info(f"=== PIPELINE START: theme='{theme}' ===")
    logger.info(f"Run directory: {run_dir}")

    step_time = time.time()

    # ── Step 1: Generate concepts ─────────────────────────
    logger.info("── Step 1/3: Generating concepts...")
    concepts = generate_concepts(theme, num_clips)

    concepts_file = run_dir / "concepts.json"
    with open(concepts_file, "w") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)
    logger.info(f"Concepts: {len(concepts)} generated")

    for c in concepts:
        logger.info(f"  #{c['clip_number']}: {c['title']} ({c['hook_type']})")

    step_time = _log_timing("Concept generation", step_time)

    # ── Step 2: Generate video + SFX in parallel ──────────
    logger.info("── Step 2/3: Generating videos + sound effects (parallel)...")
    videos_dir = run_dir / "videos"
    sfx_dir = run_dir / "sfx"

    with ThreadPoolExecutor(max_workers=2) as executor:
        video_future = executor.submit(generate_all_clip_videos, concepts, videos_dir)
        sfx_future = executor.submit(generate_all_clip_sfx, concepts, sfx_dir)

        video_paths = video_future.result()
        sfx_results = sfx_future.result()

    logger.info(f"Videos done: {len(video_paths)} clips")
    logger.info(f"SFX done: {len(sfx_results)} sounds")

    # Save SFX results
    sfx_file = run_dir / "sfx_results.json"
    with open(sfx_file, "w") as f:
        json.dump(sfx_results, f, indent=2, default=str)

    step_time = _log_timing("Video + SFX generation (parallel)", step_time)

    # ── Step 3: Post-process and compose ──────────────────
    logger.info("── Step 3/3: Composing final video...")

    if not bgm_path:
        bgm_candidates = list(config.BGM_DIR.glob("*.*"))
        if bgm_candidates:
            bgm_path = str(bgm_candidates[0])

    result = compose_final_video(
        video_paths=video_paths,
        sfx_results=sfx_results,
        concepts=concepts,
        output_path=final_output,
        bgm_path=bgm_path,
    )

    _log_timing("Post-processing", step_time)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"=== PIPELINE COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min) ===")
    logger.info(f"Final video: {result}")
    logger.info(f"Run data saved in: {run_dir}")

    return result


def run_batch(
    num_videos: int = 3,
    clips_per_video: int = None,
    bgm_path: str = None,
) -> list[str]:
    """Generate multiple videos with different themes for batch posting."""
    import random
    themes = random.sample(THEMES, min(num_videos, len(THEMES)))
    results = []

    for i, theme in enumerate(themes):
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {i+1}/{num_videos}: theme='{theme}'")
        logger.info(f"{'='*60}")

        result = run_pipeline(
            theme=theme,
            num_clips=clips_per_video,
            bgm_path=bgm_path,
        )
        results.append(result)

    logger.info(f"\nBatch complete! {len(results)} videos generated:")
    for r in results:
        logger.info(f"  → {r}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate 'Impossible Satisfying' short videos",
    )
    parser.add_argument(
        "--theme",
        default="surreal physics",
        help="Visual theme for concepts (default: surreal physics)",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=None,
        help=f"Number of clips per video (default: {config.CLIPS_PER_VIDEO})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Generate multiple videos with different themes",
    )
    parser.add_argument(
        "--bgm",
        default=None,
        help="Path to background music file",
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

    if args.batch:
        results = run_batch(
            num_videos=args.batch,
            clips_per_video=args.num_clips,
            bgm_path=args.bgm,
        )
        print(f"\nDone! {len(results)} videos generated.")
        for r in results:
            print(f"  → {r}")
    else:
        result = run_pipeline(
            theme=args.theme,
            num_clips=args.num_clips,
            bgm_path=args.bgm,
            output_name=args.output,
        )
        print(f"\nDone! Video saved to: {result}")


if __name__ == "__main__":
    main()
