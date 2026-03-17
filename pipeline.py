#!/usr/bin/env python3
"""
AI World Recreation Video Generator

Generates photorealistic "what if you were there" YouTube Shorts.
Two content types (randomly selected):
  - Anime worlds recreated as live-action
  - Historical events witnessed firsthand

Pipeline:
    1. DeepSeek → topic + scene breakdown              [~$0.001]
    2. FLUX Dev (fal.ai) → photorealistic images        [~$0.025/image]
       ElevenLabs SFX → ambient sounds                  [~$0.01/scene]
    3. FFmpeg → Ken Burns + crossfade + BGM              [free]

Cost per video (8 scenes): ~$0.22
Cost per day (3 videos): ~$0.66
Cost per month: ~$20 (≈¥3,000)

Usage:
    python pipeline.py                         # Random topic
    python pipeline.py --type anime            # Force anime topic
    python pipeline.py --type historical       # Force historical topic
    python pipeline.py --batch 3               # Generate 3 videos
    python pipeline.py --dry-run               # Topic only, no API spend
"""
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import config
from scripts.generate import generate_topic
from imagegen.flux import generate_all_scene_images
from sfx.elevenlabs_sfx import generate_all_scene_sfx
from postprocess.effects import compose_final_video

logger = logging.getLogger(__name__)


def _log_timing(step_name: str, start: float) -> float:
    elapsed = time.time() - start
    logger.info(f"  >> {step_name}: {elapsed:.1f}s")
    return time.time()


def run_pipeline(
    force_type: str = None,
    num_scenes: int = None,
    bgm_path: str = None,
    output_name: str = None,
    dry_run: bool = False,
) -> str:
    """
    Run the full video generation pipeline.
    dry_run=True: only generates topic (~$0.001), no images/SFX.
    """
    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"world_{timestamp}.mp4"
    final_output = config.OUTPUT_DIR / output_name

    logger.info(f"=== PIPELINE START {'[DRY RUN]' if dry_run else ''} ===")
    step_time = time.time()

    # ── Step 1: Generate topic + scenes (~$0.001) ─────────
    logger.info("── Step 1/3: Generating topic...")
    topic = generate_topic(force_type=force_type, num_scenes=num_scenes)

    topic_file = run_dir / "topic.json"
    with open(topic_file, "w") as f:
        json.dump(topic, f, indent=2, ensure_ascii=False)

    scenes = topic["scenes"]
    logger.info(f"  Topic: {topic['title']}")
    logger.info(f"  Type: {topic.get('topic_type', 'unknown')}")
    logger.info(f"  Scenes: {len(scenes)}")
    for s in scenes:
        logger.info(f"    #{s['scene_number']}: {s['image_prompt'][:80]}...")

    step_time = _log_timing("Topic generation", step_time)

    # ── Dry run: stop here ────────────────────────────────
    if dry_run:
        img_cost = len(scenes) * 0.025
        sfx_cost = len(scenes) * 0.01
        total = img_cost + sfx_cost
        logger.info(f"DRY RUN COMPLETE — no money spent")
        logger.info(f"Topic saved: {topic_file}")
        logger.info(f"Estimated cost: ${img_cost:.3f} (images) + ${sfx_cost:.3f} (SFX) = ${total:.3f}")
        return str(topic_file)

    # ── Step 2: Generate images + SFX in parallel ─────────
    logger.info("── Step 2/3: Generating images + SFX (parallel)...")
    images_dir = run_dir / "images"
    sfx_dir = run_dir / "sfx"

    with ThreadPoolExecutor(max_workers=2) as executor:
        img_future = executor.submit(generate_all_scene_images, scenes, images_dir)
        sfx_future = executor.submit(generate_all_scene_sfx, scenes, sfx_dir)

        image_paths = img_future.result()
        sfx_results = sfx_future.result()

    logger.info(f"Images: {len(image_paths)} | SFX: {len(sfx_results)}")

    sfx_file = run_dir / "sfx_results.json"
    with open(sfx_file, "w") as f:
        json.dump(sfx_results, f, indent=2, default=str)

    step_time = _log_timing("Image + SFX generation", step_time)

    # ── Step 3: Compose video ─────────────────────────────
    logger.info("── Step 3/3: Composing video...")

    if not bgm_path:
        bgm_candidates = list(config.BGM_DIR.glob("*.*"))
        if bgm_candidates:
            bgm_path = str(bgm_candidates[0])

    result = compose_final_video(
        image_paths=image_paths,
        sfx_results=sfx_results,
        scenes=scenes,
        output_path=final_output,
        bgm_path=bgm_path,
    )

    _log_timing("Post-processing", step_time)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"=== DONE in {total_elapsed:.1f}s === {result}")
    return result


def run_batch(
    num_videos: int = 3,
    num_scenes: int = None,
    bgm_path: str = None,
    dry_run: bool = False,
) -> list[str]:
    """Generate multiple videos (alternating anime/historical)."""
    results = []

    for i in range(num_videos):
        logger.info(f"\nBATCH {i+1}/{num_videos}")
        result = run_pipeline(
            force_type=None,  # Random each time
            num_scenes=num_scenes,
            bgm_path=bgm_path,
            dry_run=dry_run,
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate AI World Recreation shorts",
    )
    parser.add_argument("--type", choices=["anime", "historical"], default=None,
                        help="Force topic type (default: random)")
    parser.add_argument("--num-scenes", type=int, default=None,
                        help=f"Scenes per video (default: {config.SCENES_PER_VIDEO})")
    parser.add_argument("--batch", type=int, default=None,
                        help="Generate multiple videos")
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Topic only, no image/SFX generation")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.dry_run:
        config.validate_api_keys()
    elif not config.DEEPSEEK_API_KEY:
        print("FATAL: Missing DEEPSEEK_API_KEY (needed even for dry-run)")
        return

    if args.batch:
        results = run_batch(
            num_videos=args.batch,
            num_scenes=args.num_scenes,
            bgm_path=args.bgm,
            dry_run=args.dry_run,
        )
        print(f"\nDone! {len(results)} videos.")
        for r in results:
            print(f"  -> {r}")
    else:
        result = run_pipeline(
            force_type=args.type,
            num_scenes=args.num_scenes,
            bgm_path=args.bgm,
            output_name=args.output,
            dry_run=args.dry_run,
        )
        print(f"\nDone! {result}")


if __name__ == "__main__":
    main()
