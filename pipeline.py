#!/usr/bin/env python3
"""
AI World Recreation Video Generator

Photorealistic AI video shorts: anime worlds + historical events.

Pipeline:
    1. DeepSeek → topic + scene breakdown              [~$0.001]
    2. Wan 2.1 (fal.ai) → video clips (parallel)       [~$0.20/clip at 480p]
       ElevenLabs SFX → ambient sounds                  [~$0.01/clip]
    3. FFmpeg → scale + SFX + crossfade + BGM            [free]

Cost per video (5 clips): ~$1.05
Cost per day (1 video): ~$1.05
Cost per month (30 videos): ~$31.50

Usage:
    python pipeline.py                         # Random topic
    python pipeline.py --type anime            # Force anime
    python pipeline.py --type historical       # Force historical
    python pipeline.py --batch 2               # Generate 2 videos
    python pipeline.py --dry-run               # Topic only, no spend
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
from videogen.wan import generate_all_clips, check_fal_balance, estimate_cost
from sfx.elevenlabs_sfx import generate_all_scene_sfx
from postprocess.effects import compose_final_video

logger = logging.getLogger(__name__)


def _log_timing(step_name: str, start: float) -> float:
    elapsed = time.time() - start
    logger.info(f"  >> {step_name}: {elapsed:.1f}s")
    return time.time()


def run_pipeline(
    force_type: str = None,
    num_clips: int = None,
    bgm_path: str = None,
    output_name: str = None,
    dry_run: bool = False,
) -> str:
    """
    Run the full video generation pipeline.
    dry_run=True: only generates topic (~$0.001), no video/SFX.
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
    topic = generate_topic(force_type=force_type, num_clips=num_clips)

    topic_file = run_dir / "topic.json"
    with open(topic_file, "w") as f:
        json.dump(topic, f, indent=2, ensure_ascii=False)

    scenes = topic["scenes"]
    logger.info(f"  Topic: {topic['title']}")
    logger.info(f"  Type: {topic.get('topic_type', 'unknown')}")
    logger.info(f"  Clips: {len(scenes)}")
    for s in scenes:
        logger.info(f"    #{s['scene_number']}: {s['video_prompt'][:80]}...")

    step_time = _log_timing("Topic generation", step_time)

    # ── Dry run: stop here ────────────────────────────────
    if dry_run:
        cost = estimate_cost(len(scenes))
        sfx_cost = len(scenes) * 0.01
        logger.info(f"DRY RUN COMPLETE — no money spent")
        logger.info(f"Topic saved: {topic_file}")
        logger.info(f"Estimated cost: ${cost:.2f} (video) + ${sfx_cost:.2f} (SFX) = ${cost + sfx_cost:.2f}")
        return str(topic_file)

    # ── Preflight: check fal.ai balance ───────────────────
    if not check_fal_balance(len(scenes)):
        logger.error("Aborting — insufficient fal.ai balance")
        return str(topic_file)

    # ── Step 2: Generate video clips + SFX in parallel ────
    logger.info("── Step 2/3: Generating video clips + SFX (parallel)...")
    clips_dir = run_dir / "clips"
    sfx_dir = run_dir / "sfx"

    with ThreadPoolExecutor(max_workers=2) as executor:
        clip_future = executor.submit(generate_all_clips, scenes, clips_dir)
        sfx_future = executor.submit(generate_all_scene_sfx, scenes, sfx_dir)

        clip_paths = clip_future.result()
        sfx_results = sfx_future.result()

    logger.info(f"Clips: {len(clip_paths)} | SFX: {len(sfx_results)}")

    sfx_file = run_dir / "sfx_results.json"
    with open(sfx_file, "w") as f:
        json.dump(sfx_results, f, indent=2, default=str)

    step_time = _log_timing("Video + SFX generation", step_time)

    # ── Step 3: Compose final video ───────────────────────
    logger.info("── Step 3/3: Composing final video...")

    if not bgm_path:
        bgm_candidates = list(config.BGM_DIR.glob("*.*"))
        if bgm_candidates:
            bgm_path = str(bgm_candidates[0])

    result = compose_final_video(
        video_paths=clip_paths,
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
    num_videos: int = 2,
    num_clips: int = None,
    bgm_path: str = None,
    dry_run: bool = False,
) -> list[str]:
    """Generate multiple videos."""
    results = []
    for i in range(num_videos):
        logger.info(f"\nBATCH {i+1}/{num_videos}")
        result = run_pipeline(
            num_clips=num_clips,
            bgm_path=bgm_path,
            dry_run=dry_run,
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate AI World Recreation shorts",
    )
    parser.add_argument("--type", choices=["anime", "historical"], default=None)
    parser.add_argument("--num-clips", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Topic only, no video/SFX generation")
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
        print("FATAL: Missing DEEPSEEK_API_KEY")
        return

    if args.batch:
        results = run_batch(
            num_videos=args.batch,
            num_clips=args.num_clips,
            bgm_path=args.bgm,
            dry_run=args.dry_run,
        )
        print(f"\nDone! {len(results)} videos.")
        for r in results:
            print(f"  -> {r}")
    else:
        result = run_pipeline(
            force_type=args.type,
            num_clips=args.num_clips,
            bgm_path=args.bgm,
            output_name=args.output,
            dry_run=args.dry_run,
        )
        print(f"\nDone! {result}")


if __name__ == "__main__":
    main()
