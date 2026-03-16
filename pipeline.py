#!/usr/bin/env python3
"""
Impossible Satisfying Video Generator — Main Pipeline

Usage:
    python pipeline.py --theme "glass and crystal"
    python pipeline.py --batch 3
    python pipeline.py --dry-run  # Concepts only, no API spend

Pipeline:
    1. DeepSeek → concept (visual + sound prompt)    [~$0.001]
    2. Wan 2.1 (fal.ai) → video clip                 [~$0.10/clip]
       ElevenLabs SFX → ASMR audio                   [~$0.01/clip]
    3. FFmpeg → compose final video                   [free]

Estimated cost per video (3 clips): ~$0.33
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
from videogen.kling import generate_all_clip_videos, check_fal_balance, estimate_cost
from postprocess.effects import compose_final_video

logger = logging.getLogger(__name__)

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
    elapsed = time.time() - start
    logger.info(f"  >> {step_name}: {elapsed:.1f}s")
    return time.time()


def _preflight_check(num_clips: int) -> bool:
    """Check fal.ai balance before spending. Returns True if OK."""
    estimated = estimate_cost(num_clips)
    logger.info(f"Estimated fal.ai cost: ~${estimated:.2f} for {num_clips} clips")

    balance = check_fal_balance()
    if balance != float("inf") and balance < estimated:
        logger.error(
            f"INSUFFICIENT BALANCE: ${balance:.2f} available, "
            f"~${estimated:.2f} needed. Top up at https://fal.ai/dashboard/billing"
        )
        return False

    if balance != float("inf"):
        logger.info(f"Balance OK: ${balance:.2f} available")
    return True


def run_pipeline(
    theme: str = "surreal physics",
    num_clips: int = None,
    bgm_path: str = None,
    output_name: str = None,
    dry_run: bool = False,
) -> str:
    """
    Run the full video generation pipeline.
    dry_run=True: only concepts (~$0.001), no video/SFX APIs.
    """
    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    num_clips = num_clips or config.CLIPS_PER_VIDEO

    if not output_name:
        output_name = f"satisfying_{timestamp}.mp4"

    final_output = config.OUTPUT_DIR / output_name

    logger.info(f"=== PIPELINE START: theme='{theme}' {'[DRY RUN]' if dry_run else ''} ===")

    step_time = time.time()

    # ── Step 1: Generate concepts (~$0.001) ───────────────
    logger.info("── Step 1/3: Generating concepts...")
    concepts = generate_concepts(theme, num_clips)

    concepts_file = run_dir / "concepts.json"
    with open(concepts_file, "w") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)

    for c in concepts:
        logger.info(f"  #{c['clip_number']}: {c['title']} ({c['hook_type']})")
        logger.info(f"    Visual: {c['visual_prompt'][:100]}...")
        logger.info(f"    Sound: {c['sound_prompt'][:80]}...")

    step_time = _log_timing("Concept generation", step_time)

    # ── Dry run: stop here ────────────────────────────────
    if dry_run:
        estimated = estimate_cost(len(concepts))
        logger.info(f"DRY RUN COMPLETE — no money spent on video/SFX")
        logger.info(f"Concepts saved: {concepts_file}")
        logger.info(f"Estimated cost: ~${estimated:.2f} (video) + ~${len(concepts) * 0.01:.2f} (SFX) = ~${estimated + len(concepts) * 0.01:.2f}")
        return str(concepts_file)

    # ── Preflight: check balance ──────────────────────────
    if not _preflight_check(len(concepts)):
        return str(concepts_file)

    # ── Step 2: Generate video + SFX in parallel ──────────
    logger.info("── Step 2/3: Generating videos + sound effects (parallel)...")
    videos_dir = run_dir / "videos"
    sfx_dir = run_dir / "sfx"

    with ThreadPoolExecutor(max_workers=2) as executor:
        video_future = executor.submit(generate_all_clip_videos, concepts, videos_dir)
        sfx_future = executor.submit(generate_all_clip_sfx, concepts, sfx_dir)

        video_paths = video_future.result()
        sfx_results = sfx_future.result()

    logger.info(f"Videos: {len(video_paths)} | SFX: {len(sfx_results)}")

    sfx_file = run_dir / "sfx_results.json"
    with open(sfx_file, "w") as f:
        json.dump(sfx_results, f, indent=2, default=str)

    step_time = _log_timing("Video + SFX generation", step_time)

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
    logger.info(f"=== DONE in {total_elapsed:.1f}s === {result}")

    return result


def run_batch(
    num_videos: int = 3,
    clips_per_video: int = None,
    bgm_path: str = None,
    dry_run: bool = False,
) -> list[str]:
    """Generate multiple videos with different themes."""
    import random
    themes = random.sample(THEMES, min(num_videos, len(THEMES)))
    results = []

    for i, theme in enumerate(themes):
        logger.info(f"\nBATCH {i+1}/{num_videos}: theme='{theme}'")
        result = run_pipeline(
            theme=theme,
            num_clips=clips_per_video,
            bgm_path=bgm_path,
            dry_run=dry_run,
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate 'Impossible Satisfying' short videos",
    )
    parser.add_argument("--theme", default="surreal physics")
    parser.add_argument("--num-clips", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Concepts + cost estimate only, no API spend")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate API keys BEFORE doing anything
    if not args.dry_run:
        config.validate_api_keys()
    else:
        # Dry run only needs DeepSeek
        if not config.DEEPSEEK_API_KEY:
            print("FATAL: Missing DEEPSEEK_API_KEY (needed even for dry-run)")
            return

    if args.batch:
        results = run_batch(
            num_videos=args.batch,
            clips_per_video=args.num_clips,
            bgm_path=args.bgm,
            dry_run=args.dry_run,
        )
        print(f"\nDone! {len(results)} videos.")
        for r in results:
            print(f"  -> {r}")
    else:
        result = run_pipeline(
            theme=args.theme,
            num_clips=args.num_clips,
            bgm_path=args.bgm,
            output_name=args.output,
            dry_run=args.dry_run,
        )
        print(f"\nDone! {result}")


if __name__ == "__main__":
    main()
