#!/usr/bin/env python3
"""
Construction Timelapse Luxury Transformation Video Pipeline.

rebornspacestv-style: construction process + luxury reveal shorts.

Usage:
    # Full dry-run with hardcoded examples (no API needed)
    python build_pipeline.py --offline

    # Dry-run with LLM candidate generation (DeepSeek only, ~$0.003)
    python build_pipeline.py --dry-run

    # Generate video (requires approval + all API keys)
    python build_pipeline.py --generate
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Setup ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_PATH = BASE_DIR / "data" / "history.json"
CATEGORY_STATS_PATH = BASE_DIR / "data" / "category_stats.json"


def load_config() -> dict:
    """Load YAML configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_llm_client(config: dict):
    """Create OpenAI-compatible client for DeepSeek."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logging.error("DEEPSEEK_API_KEY not set. Use --offline for no-API mode.")
        sys.exit(1)

    llm_config = config.get("llm", {})
    return OpenAI(
        api_key=api_key,
        base_url=llm_config.get("base_url", "https://api.deepseek.com"),
    )


def run_offline(config: dict) -> None:
    """Fully offline dry-run. No API calls at all."""
    from scenario.generator import generate_candidates_dry
    from scenario.similarity import filter_candidates
    from scenario.scorer import score_candidates_heuristic, filter_by_score
    from scenario.balancer import (
        load_category_stats, adjust_scores, update_stats_after_selection, save_category_stats,
    )
    from scenario.selector import load_history, select_best, record_selection
    from prompts.video_prompt_builder import build_video_prompt_dry

    logger = logging.getLogger("build")

    history = load_history(HISTORY_PATH)
    stats = load_category_stats(CATEGORY_STATS_PATH)

    logger.info("=== OFFLINE DRY-RUN (no API calls) ===")
    logger.info("History: %d past builds", len(history))

    # Stage 1+2: Hardcoded example candidates
    candidates = generate_candidates_dry(config)
    logger.info("Candidates: %d (hardcoded examples)", len(candidates))

    # Similarity filter
    passed = filter_candidates(candidates, history, config)
    logger.info("After similarity filter: %d/%d", len(passed), len(candidates))

    # Heuristic scoring
    passed = score_candidates_heuristic(passed)
    passed = filter_by_score(passed, config)
    logger.info("After score filter: %d", len(passed))

    # Category balance
    categories_config = config.get("categories", {})
    passed = adjust_scores(passed, stats, categories_config)

    # Select best
    winner = select_best(passed)
    if not winner:
        logger.error("No candidates survived filtering. Adjust config thresholds.")
        sys.exit(1)

    # Record to history
    history = record_selection(winner, history, HISTORY_PATH)
    stats = update_stats_after_selection(winner, stats, categories_config)
    save_category_stats(stats, CATEGORY_STATS_PATH)

    # Generate video prompt (fallback, no LLM)
    prompts = build_video_prompt_dry(winner, config)

    # Output
    _print_results(winner, prompts, config)
    _save_run_output(winner, prompts)

    logger.info("=== OFFLINE DRY-RUN COMPLETE ===")
    logger.info("No money spent. No API calls made.")


def run_dry(config: dict) -> None:
    """Dry-run with LLM. DeepSeek only (~$0.003)."""
    from scenario.generator import generate_candidates
    from scenario.similarity import filter_candidates
    from scenario.scorer import score_candidates_llm, filter_by_score
    from scenario.balancer import (
        load_category_stats, adjust_scores, update_stats_after_selection, save_category_stats,
    )
    from scenario.selector import load_history, select_best, record_selection
    from prompts.video_prompt_builder import build_video_prompt

    logger = logging.getLogger("build")
    client = create_llm_client(config)

    history = load_history(HISTORY_PATH)
    stats = load_category_stats(CATEGORY_STATS_PATH)

    logger.info("=== DRY-RUN (DeepSeek LLM only, ~$0.003) ===")
    logger.info("History: %d past builds", len(history))

    # Generate candidates via LLM
    candidates = generate_candidates(client, config, history, stats)
    logger.info("Candidates generated: %d", len(candidates))

    if not candidates:
        logger.error("No candidates generated. Check API key and connectivity.")
        sys.exit(1)

    # Similarity filter
    passed = filter_candidates(candidates, history, config)
    logger.info("After similarity filter: %d/%d", len(passed), len(candidates))

    # LLM scoring
    passed = score_candidates_llm(passed, client, config)
    passed = filter_by_score(passed, config)
    logger.info("After score filter: %d", len(passed))

    # Category balance
    categories_config = config.get("categories", {})
    passed = adjust_scores(passed, stats, categories_config)

    # Select best
    winner = select_best(passed)
    if not winner:
        logger.error("No candidates survived filtering. Adjust config thresholds.")
        sys.exit(1)

    # Record
    history = record_selection(winner, history, HISTORY_PATH)
    stats = update_stats_after_selection(winner, stats, categories_config)
    save_category_stats(stats, CATEGORY_STATS_PATH)

    # Generate video prompt via LLM
    prompts = build_video_prompt(winner, client, config)

    # Output
    _print_results(winner, prompts, config)
    _save_run_output(winner, prompts)

    logger.info("=== DRY-RUN COMPLETE ===")
    logger.info("Estimated cost: ~$0.003 (DeepSeek LLM calls only)")


def run_generate(config: dict) -> None:
    """Full generation: scenario → video clip → SFX → compose final video."""
    from scenario.generator import generate_candidates
    from scenario.similarity import filter_candidates
    from scenario.scorer import score_candidates_llm, filter_by_score
    from scenario.balancer import (
        load_category_stats, adjust_scores, update_stats_after_selection, save_category_stats,
    )
    from scenario.selector import load_history, select_best, record_selection
    from prompts.video_prompt_builder import build_video_prompt
    from videogen.wan import generate_clip, check_fal_balance
    from sfx.elevenlabs_sfx import generate_sfx
    from postprocess.effects import compose_final_video

    logger = logging.getLogger("build")
    client = create_llm_client(config)

    history = load_history(HISTORY_PATH)
    stats = load_category_stats(CATEGORY_STATS_PATH)

    logger.info("=== FULL GENERATION MODE ===")
    logger.info("History: %d past builds", len(history))

    # ── Stage 1: Scenario selection (same as dry-run) ────
    candidates = generate_candidates(client, config, history, stats)
    logger.info("Candidates generated: %d", len(candidates))

    if not candidates:
        logger.error("No candidates generated. Check API key and connectivity.")
        sys.exit(1)

    passed = filter_candidates(candidates, history, config)
    logger.info("After similarity filter: %d/%d", len(passed), len(candidates))

    passed = score_candidates_llm(passed, client, config)
    passed = filter_by_score(passed, config)
    logger.info("After score filter: %d", len(passed))

    categories_config = config.get("categories", {})
    passed = adjust_scores(passed, stats, categories_config)

    winner = select_best(passed)
    if not winner:
        logger.error("No candidates survived filtering. Adjust config thresholds.")
        sys.exit(1)

    history = record_selection(winner, history, HISTORY_PATH)
    stats = update_stats_after_selection(winner, stats, categories_config)
    save_category_stats(stats, CATEGORY_STATS_PATH)

    # ── Stage 2: Generate video prompt via LLM ───────────
    prompts = build_video_prompt(winner, client, config)
    _print_results(winner, prompts, config)

    video_prompt = prompts.get("video_prompt", "")
    sfx_prompt = prompts.get("sfx_prompt", "")
    duration = prompts.get("duration_seconds", 15)

    if not video_prompt:
        logger.error("No video prompt generated.")
        sys.exit(1)

    # ── Stage 3: Check balance ───────────────────────────
    if not check_fal_balance(1):
        logger.error("Insufficient fal.ai balance. Top up and retry.")
        sys.exit(1)

    # ── Stage 4: Generate video clip via Wan 2.1 ─────────
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    clip_path = temp_dir / "clip_01.mp4"
    logger.info("Generating video clip via fal.ai Wan 2.1...")
    generate_clip(video_prompt, clip_path)
    logger.info("Video clip generated: %s", clip_path)

    # ── Stage 5: Generate SFX via ElevenLabs ─────────────
    sfx_path = temp_dir / "sfx_01.mp3"
    logger.info("Generating SFX via ElevenLabs...")
    sfx_result = generate_sfx(
        prompt=sfx_prompt,
        output_path=sfx_path,
        duration=duration,
    )
    logger.info("SFX generated: %s", sfx_result["audio_path"])

    # ── Stage 6: Compose final video ─────────────────────
    now = datetime.now(timezone.utc)
    final_filename = f"build_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
    final_path = output_dir / final_filename

    scene_data = [{
        "video_prompt": video_prompt,
        "sfx_prompt": sfx_prompt,
        "text_overlay": winner.get("one_line_concept", ""),
    }]

    logger.info("Composing final video...")
    compose_final_video(
        video_paths=[str(clip_path)],
        sfx_results=[sfx_result],
        scenes=scene_data,
        output_path=final_path,
    )
    logger.info("Final video saved: %s", final_path)

    # ── Stage 7: Upload to YouTube ───────────────────────
    import os
    if os.getenv("YOUTUBE_REFRESH_TOKEN"):
        from upload.youtube import upload_to_youtube
        logger.info("Uploading to YouTube Shorts...")
        try:
            yt_result = upload_to_youtube(
                video_path=str(final_path),
                scenario=winner,
                privacy="public",
            )
            logger.info("YouTube upload complete: %s", yt_result.get("url", ""))
        except Exception as e:
            logger.error("YouTube upload failed: %s", e)
            logger.info("Video saved locally — upload manually if needed.")
    else:
        logger.info("YOUTUBE_REFRESH_TOKEN not set — skipping YouTube upload.")
        logger.info("Run setup_youtube_auth.py to enable auto-upload.")

    # ── Save JSON output alongside ───────────────────────
    _save_run_output(winner, prompts)

    logger.info("=== GENERATION COMPLETE ===")
    logger.info("Output: %s", final_path)


def _print_results(scenario: dict, prompts: dict, config: dict) -> None:
    """Print formatted results to stdout."""
    gen_config = config.get("generation", {})
    duration = gen_config.get("duration_seconds", 15)

    print("\n" + "=" * 70)
    print("  SELECTED CONSTRUCTION TIMELAPSE SCENARIO")
    print("=" * 70)
    print(f"  Concept:     {scenario.get('one_line_concept', 'N/A')}")
    print(f"  Category:    {scenario.get('category', 'N/A')}")
    print(f"  Build type:  {scenario.get('construction_type', 'N/A')}")
    print(f"  Camera:      {scenario.get('camera_style', 'N/A')}")
    print(f"  Reveal:      {scenario.get('reveal_type', 'N/A')}")
    print(f"  Location:    {scenario.get('location_feel', 'N/A')}")
    print(f"  Duration:    {duration}s (single timelapse clip)")
    print()

    # Before space
    before = scenario.get("before_space", {})
    print(f"  BEFORE (0-1s):")
    print(f"    {before.get('description', 'N/A')}")
    print(f"    Visual: {before.get('visual', 'N/A')}")
    print()

    # Construction process
    proc = scenario.get("construction_process", {})
    print(f"  CONSTRUCTION PROCESS (1-10s):")
    for i, stage in enumerate(proc.get("stages", []), 1):
        print(f"    {i}. {stage}")
    print(f"    Machinery:  {', '.join(proc.get('heavy_machinery', [])) or 'none'}")
    print(f"    Workers:    {proc.get('worker_presence', 'N/A')}")
    print(f"    Materials:  {', '.join(proc.get('key_materials', []))}")
    print(f"    Excavation: {'yes' if proc.get('excavation_required') else 'no'}")
    print()

    # After space
    after = scenario.get("after_space", {})
    print(f"  REVEAL (10-15s):")
    print(f"    {after.get('description', 'N/A')}")
    print(f"    Luxury:     {after.get('luxury_level', 'N/A')}")
    print(f"    Water:      {'yes' if after.get('water_element') else 'no'}")
    print(f"    Hook:       {after.get('final_visual_hook', 'N/A')}")
    print()

    # Time structure
    ts = scenario.get("time_structure", {})
    if ts:
        print(f"  TIME STRUCTURE:")
        print(f"    0-1s:   {ts.get('0_1s', 'N/A')}")
        print(f"    1-4s:   {ts.get('1_4s', 'N/A')}")
        print(f"    4-10s:  {ts.get('4_10s', 'N/A')}")
        print(f"    10-15s: {ts.get('10_15s', 'N/A')}")
        print()

    # Scores
    buzz = scenario.get("buzz_score", {})
    if buzz:
        print("  SCORE:")
        for k, v in buzz.items():
            bar = "#" * v + "." * (10 - v)
            print(f"    {k:40s} [{bar}] {v}/10")
        print(f"    {'TOTAL':40s}       {scenario.get('buzz_total', 0)}/80")
        print(f"    {'Category bonus':40s}       {scenario.get('category_bonus', 0):+.1f}")
        print(f"    {'Adjusted score':40s}       {scenario.get('adjusted_score', 0):.1f}")
    print()

    # Tags
    print(f"  Tags: {', '.join(scenario.get('similarity_tags', []))}")
    print()

    # Video prompt
    video_prompt = prompts.get("video_prompt", "")
    sfx_prompt = prompts.get("sfx_prompt", "")
    if video_prompt:
        print("-" * 70)
        print("  VIDEO PROMPT (single 15s timelapse)")
        print("-" * 70)
        print(f"  {video_prompt}")
        print()
        print(f"  SFX: {sfx_prompt}")
        print()

    # Cost estimate
    video_config = config.get("video", {})
    max_cost = video_config.get("max_cost_per_video_usd", 0.35)
    print("-" * 70)
    print("  COST ESTIMATE (if generated)")
    print("-" * 70)
    print(f"  Video (1 clip, {duration}s):       model-dependent")
    print(f"  SFX   (1 clip):               ~$0.01")
    print(f"  LLM   (candidates + scoring): ~$0.003")
    print(f"  Hard ceiling:                 ${max_cost:.2f} (~50 JPY)")
    print("=" * 70 + "\n")


def _save_run_output(scenario: dict, prompts: dict) -> None:
    """Save run output to file."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = f"build_plan_{now.strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "created_at": now.isoformat(),
        "scenario": {
            k: v for k, v in scenario.items()
            if not k.startswith("_")
        },
        "prompts": prompts,
        "status": "planned",
    }

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.getLogger("build").info("Run output saved to: %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="Construction Timelapse Luxury Transformation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --offline    Fully offline. No API calls. Uses hardcoded examples.
  --dry-run    LLM-only mode. Generates real candidates via DeepSeek (~$0.003).
  --generate   Full video generation. LOCKED until config.yaml dry_run=false.
        """,
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Fully offline test (no API calls at all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="LLM candidate generation only, no video/audio (default)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Full video generation (requires approval)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--config", type=str, default=str(CONFIG_PATH),
        help="Path to config.yaml",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.generate:
        run_generate(config)
    elif args.offline:
        run_offline(config)
    else:
        run_dry(config)


if __name__ == "__main__":
    main()
