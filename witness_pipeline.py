#!/usr/bin/env python3
"""
Witness-type Shock Video Generation Pipeline.

CLI entry point for the scenario generation system.
Default mode is dry-run (no paid API calls).

Usage:
    # Full dry-run with hardcoded examples (no API needed)
    python witness_pipeline.py --offline

    # Dry-run with LLM candidate generation (DeepSeek only, ~$0.003)
    python witness_pipeline.py --dry-run

    # Generate video (requires approval + all API keys)
    python witness_pipeline.py --generate
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

    llm_config = config.get("llm", {})
    api_key = None

    # Try loading from env
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not api_key:
        logging.error("DEEPSEEK_API_KEY not set. Use --offline for no-API mode.")
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url=llm_config.get("base_url", "https://api.deepseek.com"),
    )


def run_offline(config: dict) -> None:
    """
    Fully offline dry-run. No API calls at all.
    Uses hardcoded example candidates to demonstrate the full pipeline.
    """
    from scenario.generator import generate_candidates_dry
    from scenario.similarity import filter_candidates
    from scenario.scorer import score_candidates_heuristic, filter_by_score
    from scenario.balancer import (
        load_category_stats, adjust_scores, update_stats_after_selection, save_category_stats,
    )
    from scenario.selector import load_history, select_best, record_selection
    from prompts.video_prompt_builder import build_video_prompt_dry

    logger = logging.getLogger("witness")

    # Load state
    history = load_history(HISTORY_PATH)
    stats = load_category_stats(CATEGORY_STATS_PATH)

    logger.info("=== OFFLINE DRY-RUN (no API calls) ===")
    logger.info("History: %d past videos", len(history))

    # Stage 1+2: Get example candidates
    candidates = generate_candidates_dry(config)
    logger.info("Candidates: %d (hardcoded examples)", len(candidates))

    # Similarity filter
    passed = filter_candidates(candidates, history, config)
    logger.info("After similarity filter: %d/%d", len(passed), len(candidates))

    # Heuristic scoring (no LLM)
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

    # Output results
    _print_results(winner, prompts, config)
    _save_run_output(winner, prompts)

    logger.info("=== OFFLINE DRY-RUN COMPLETE ===")
    logger.info("No money spent. No API calls made.")


def run_dry(config: dict) -> None:
    """
    Dry-run with LLM. Generates real candidates via DeepSeek (~$0.003 total).
    Does NOT generate any videos or audio (no fal.ai / ElevenLabs calls).
    """
    from scenario.generator import generate_candidates
    from scenario.similarity import filter_candidates
    from scenario.scorer import score_candidates_llm, filter_by_score
    from scenario.balancer import (
        load_category_stats, adjust_scores, update_stats_after_selection, save_category_stats,
    )
    from scenario.selector import load_history, select_best, record_selection
    from prompts.video_prompt_builder import build_video_prompt

    logger = logging.getLogger("witness")
    client = create_llm_client(config)

    # Load state
    history = load_history(HISTORY_PATH)
    stats = load_category_stats(CATEGORY_STATS_PATH)

    logger.info("=== DRY-RUN (DeepSeek LLM only, ~$0.003) ===")
    logger.info("History: %d past videos", len(history))

    # Stage 1+2: Generate candidates via LLM
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

    # Record to history
    history = record_selection(winner, history, HISTORY_PATH)
    stats = update_stats_after_selection(winner, stats, categories_config)
    save_category_stats(stats, CATEGORY_STATS_PATH)

    # Generate video prompt via LLM
    prompts = build_video_prompt(winner, client, config)

    # Output results
    _print_results(winner, prompts, config)
    _save_run_output(winner, prompts)

    logger.info("=== DRY-RUN COMPLETE ===")
    logger.info("Estimated cost: ~$0.003 (DeepSeek LLM calls only)")
    logger.info("No video/audio generation. No fal.ai or ElevenLabs charges.")


def run_generate(config: dict) -> None:
    """
    Full generation mode. LOCKED behind dry_run config flag.
    """
    logger = logging.getLogger("witness")

    if config.get("dry_run", True):
        logger.error(
            "BLOCKED: dry_run=true in config.yaml. "
            "Set dry_run: false ONLY after explicit owner approval."
        )
        sys.exit(1)

    logger.error(
        "Full video generation not yet implemented. "
        "This will be enabled after Phase 1 approval."
    )
    sys.exit(1)


def _print_results(scenario: dict, prompts: dict, config: dict) -> None:
    """Print formatted results to stdout."""
    gen_config = config.get("generation", {})
    duration = gen_config.get("duration_seconds", 14)

    print("\n" + "=" * 70)
    print("  SELECTED SCENARIO (single continuous shot)")
    print("=" * 70)
    print(f"  Category:    {scenario.get('category', 'N/A')}")
    print(f"  Event:       {scenario.get('event_type', 'N/A')}")
    print(f"  POV:         {scenario.get('camera_pov', 'N/A')}")
    print(f"  Camera:      {scenario.get('camera_movement', 'N/A')}")
    print(f"  Hook:        {scenario.get('opening_hook_type', 'N/A')}")
    print(f"  Location:    {scenario.get('location_style', 'N/A')}")
    print(f"  Time:        {scenario.get('time_of_day', 'N/A')}")
    print(f"  Weather:     {scenario.get('weather_atmosphere', 'N/A')}")
    print(f"  Duration:    {duration}s (single clip, no cuts)")
    print()
    print(f"  Summary:")
    print(f"    {scenario.get('scenario_summary', 'N/A')}")
    print()
    print(f"  0.0-0.5s  HOOK:")
    print(f"    {scenario.get('opening_hook_description', 'N/A')}")
    print(f"  5.0-10.0s PEAK:")
    print(f"    {scenario.get('peak_moment', 'N/A')}")
    print(f"  10.0-{duration}s AFTERMATH:")
    print(f"    {scenario.get('aftermath', 'N/A')}")
    print()

    # Scores
    buzz = scenario.get("buzz_score", {})
    if buzz:
        print("  Buzz Score:")
        for k, v in buzz.items():
            bar = "#" * v + "." * (10 - v)
            print(f"    {k:25s} [{bar}] {v}/10")
        print(f"    {'TOTAL':25s}       {scenario.get('buzz_total', 0)}/70")
        print(f"    {'Category bonus':25s}       {scenario.get('category_bonus', 0):+.1f}")
        print(f"    {'Adjusted score':25s}       {scenario.get('adjusted_score', 0):.1f}")
    print()

    # Visual/tone tags
    print(f"  Visual tags: {', '.join(scenario.get('visual_tags', []))}")
    print(f"  Tone tags:   {', '.join(scenario.get('tone_tags', []))}")
    print(f"  Colors:      {', '.join(scenario.get('dominant_colors', []))}")
    print(f"  Sound:       {scenario.get('sound_atmosphere', 'N/A')}")
    print()

    # Video prompt (single)
    video_prompt = prompts.get("video_prompt", "")
    sfx_prompt = prompts.get("sfx_prompt", "")
    if video_prompt:
        print("-" * 70)
        print("  VIDEO PROMPT (single continuous shot)")
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
    """Save run output to file for review."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = f"witness_plan_{now.strftime('%Y%m%d_%H%M%S')}.json"

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

    logging.getLogger("witness").info("Run output saved to: %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="Witness-type Shock Video Generation System",
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

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Route to appropriate mode
    if args.generate:
        run_generate(config)
    elif args.offline:
        run_offline(config)
    else:
        run_dry(config)


if __name__ == "__main__":
    main()
