#!/usr/bin/env python3
"""
Construction Timelapse Luxury Transformation — Kling 3.0 Prompt Generator.

rebornspacestv-style: construction process + luxury reveal shorts.

5-STAGE TEMPLATE ARCHITECTURE (Kling 3.0 manual workflow):
  5 clips × 5 seconds each = 25 seconds total
  Each clip = 1 copy-paste prompt for Kling 3.0

No API calls needed. Generates text prompts only.
You generate the videos manually in Kling 3.0, then stitch in CapCut/DaVinci.

Usage:
    # Random template lottery (default)
    python build_pipeline.py

    # Force a specific template
    python build_pipeline.py --force-template pool_vehicle

    # Dry-run with LLM candidate generation (DeepSeek only, ~$0.003)
    python build_pipeline.py --dry-run

    # Fully offline test with hardcoded examples
    python build_pipeline.py --offline
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
ANIMAL_HISTORY_PATH = BASE_DIR / "data" / "animal_history.json"


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


def run_template(config: dict, force_template: str | None = None) -> None:
    """Template-based prompt generation for Kling 3.0.

    NO API calls needed. Picks 1 of 5 templates, fills variables,
    outputs 5 copy-paste-ready prompts for manual Kling 3.0 generation.
    """
    from prompts.templates.lottery import TemplateLottery, print_lottery_info

    logger = logging.getLogger("build")

    logger.info("=== TEMPLATE LOTTERY → Kling 3.0 Prompts ===")
    print_lottery_info()

    lottery = TemplateLottery()
    result = lottery.draw(force_template=force_template)

    # Print header
    print("\n" + "=" * 70)
    print("  KLING 3.0 PROMPT SET")
    print("=" * 70)
    print(f"  Template:  {result['template_name']}")
    print(f"  Reference: {result['reference_video']}")
    print(f"  Structure: 5 clips × 5 seconds = 25 seconds total")
    print(f"  Cost:      50 credits (Standard) / 175 credits (Professional)")
    print(f"  Pro plan:  60 videos/month (Standard) / 17 videos/month (Professional)")
    print()

    print("  Variables:")
    for k, val in result["variables"].items():
        print(f"    {k}: {val}")
    print()

    # Print prompts in copy-paste format for Kling 3.0
    print("=" * 70)
    print("  COPY-PASTE PROMPTS FOR KLING 3.0")
    print("  Generate each as 5-second clip, then stitch in CapCut/DaVinci")
    print("=" * 70)

    neg_prompt = (
        "blurry, low quality, distorted faces, extra fingers, mutation, "
        "deformed, ugly, watermark, text overlay, logo, out of frame, "
        "bad anatomy, bad proportions, duplicate, glitch, noise"
    )

    for stage in result["stages"]:
        num = stage["stage"]
        name = stage["name"].upper()
        prompt = stage["video_prompt"]
        print(f"\n{'─' * 70}")
        print(f"  CLIP {num}/5: {name}")
        print(f"  Duration: 5 seconds | Aspect: 9:16 vertical | Mode: Standard")
        print(f"{'─' * 70}")
        print(f"\n[Prompt]\n{prompt}\n")
        print(f"[Negative Prompt]\n{neg_prompt}\n")
    print("=" * 70)

    # Credit summary
    print(f"\n  CREDIT SUMMARY:")
    print(f"  Standard mode:      5 clips × 10 credits = 50 credits")
    print(f"  Professional mode:  5 clips × 35 credits = 175 credits")
    print(f"  Tip: Generate in Standard first, re-generate best clips in Professional")
    print()

    # Save output
    _save_template_output(result)

    logger.info("=== DONE — Copy prompts above into Kling 3.0 ===")
    logger.info("Cost: 50 credits per video (Standard mode)")
    logger.info("Pro plan (3000 credits): up to 60 videos/month")


def _save_template_output(result: dict) -> None:
    """Save template lottery output."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = f"template_{result['template_id']}_{now.strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "created_at": now.isoformat(),
        "mode": "template_lottery",
        "template_id": result["template_id"],
        "template_name": result["template_name"],
        "reference_video": result["reference_video"],
        "total_duration_seconds": result["total_duration_seconds"],
        "variables": result["variables"],
        "stages": result["stages"],
        "status": "planned",
    }

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.getLogger("build").info("Template output saved to: %s", path)


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

    # Generate video prompts (3-stage, no LLM)
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

    # Generate 3-stage video prompts via LLM
    prompts = build_video_prompt(winner, client, config)

    # Output
    _print_results(winner, prompts, config)
    _save_run_output(winner, prompts)

    logger.info("=== DRY-RUN COMPLETE ===")
    logger.info("Estimated cost: ~$0.003 (DeepSeek LLM calls only)")



def _print_results(scenario: dict, prompts: dict, config: dict) -> None:
    """Print formatted results to stdout."""
    gen_config = config.get("generation", {})

    print("\n" + "=" * 70)
    print("  SELECTED CONSTRUCTION TRANSFORMATION SCENARIO")
    print("=" * 70)
    print(f"  Concept:     {scenario.get('one_line_concept', 'N/A')}")
    print(f"  Category:    {scenario.get('category', 'N/A')}")
    print(f"  Build type:  {scenario.get('construction_type', 'N/A')}")
    print(f"  Camera:      {scenario.get('camera_style', 'N/A')}")
    print(f"  Reveal:      {scenario.get('reveal_type', 'N/A')}")
    print(f"  Location:    {scenario.get('location_feel', 'N/A')}")
    print(f"  Architecture: 3-stage (5s × 3 = 15s total)")
    print()

    # Before space
    before = scenario.get("before_space", {})
    print(f"  BEFORE STATE:")
    print(f"    {before.get('description', 'N/A')}")
    print(f"    Visual: {before.get('visual', 'N/A')}")
    print()

    # Construction process
    proc = scenario.get("construction_process", {})
    print(f"  CONSTRUCTION PROCESS:")
    for i, stage in enumerate(proc.get("stages", []), 1):
        print(f"    {i}. {stage}")
    print(f"    Machinery:  {', '.join(proc.get('heavy_machinery', [])) or 'hand tools'}")
    print(f"    Workers:    {proc.get('worker_presence', 'N/A')}")
    materials = proc.get("key_materials", [])
    print(f"    Materials:  {', '.join(materials[:5])}")
    print()

    # After space
    after = scenario.get("after_space", {})
    print(f"  REVEAL:")
    print(f"    {after.get('description', 'N/A')}")
    print(f"    Luxury:     {after.get('luxury_level', 'N/A')}")
    print(f"    Water:      {'yes' if after.get('water_element') else 'no'}")
    print(f"    Hook:       {after.get('final_visual_hook', 'N/A')}")
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

    print(f"  Tags: {', '.join(scenario.get('similarity_tags', []))}")
    print()

    # 3-stage video prompts
    stages = prompts.get("stages", [])
    if stages:
        print("-" * 70)
        print("  3-STAGE VIDEO PROMPTS")
        print("-" * 70)
        print(f"  Camera: {prompts.get('camera_description', 'N/A')}")
        print()
        for stage in stages:
            name = stage.get("name", "unknown").upper()
            dur = stage.get("duration_seconds", 5)
            print(f"  [{name}] ({dur}s):")
            print(f"    Video: {stage.get('video_prompt', 'N/A')}")
            print(f"    SFX:   {stage.get('sfx_prompt', 'N/A')}")
            print()
    else:
        # Legacy single prompt
        video_prompt = prompts.get("video_prompt", "")
        sfx_prompt = prompts.get("sfx_prompt", "")
        if video_prompt:
            print("-" * 70)
            print("  VIDEO PROMPT")
            print("-" * 70)
            print(f"  {video_prompt}")
            print(f"  SFX: {sfx_prompt}")
            print()

    # Cost estimate
    video_config = config.get("video", {})
    max_cost = video_config.get("max_cost_per_video_usd", 0.75)
    num_clips = len(stages) if stages else 1
    print("-" * 70)
    print("  COST ESTIMATE (if generated)")
    print("-" * 70)
    print(f"  Video ({num_clips} clips × 5s):     ~${num_clips * 0.20:.2f}")
    print(f"  SFX   ({num_clips} clips):           ~${num_clips * 0.01:.2f}")
    print(f"  LLM   (candidates + scoring):  ~$0.003")
    print(f"  Hard ceiling:                  ${max_cost:.2f}")
    print("=" * 70 + "\n")


def _save_run_output(scenario: dict, prompts: dict) -> None:
    """Save run output to file."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = f"build_plan_{now.strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "created_at": now.isoformat(),
        "architecture": "3-stage (5s × 3)",
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


def _load_animal_history() -> list[dict]:
    """Load animal selection history."""
    if ANIMAL_HISTORY_PATH.exists():
        with open(ANIMAL_HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_animal_to_history(history: list[dict], animal_short: str, animal_full: str) -> None:
    """Save selected animal to history."""
    ANIMAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.append({
        "animal_short": animal_short,
        "animal_full": animal_full,
        "date": datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 200 entries
    history = history[-200:]
    with open(ANIMAL_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def run_animal_pov(config: dict) -> None:
    """Animal POV prompt generation for Kling 3.0.

    Generates 5 copy-paste-ready prompts showing the world from an animal's perspective.
    Uses weighted selection to reduce probability of recently used animals.
    """
    import random
    from prompts.templates.t6_animal_pov import AnimalPovTemplate

    logger = logging.getLogger("build")
    logger.info("=== ANIMAL POV → Kling 3.0 Prompts ===")

    template = AnimalPovTemplate()
    history = _load_animal_history()

    # Calculate animal weights based on history
    animals = template.get_variable_pools()["_animal_pair"]
    recent_shorts = [h["animal_short"] for h in history]
    last_5 = recent_shorts[-5:] if len(recent_shorts) >= 5 else recent_shorts
    last_10 = recent_shorts[-10:] if len(recent_shorts) >= 10 else recent_shorts
    last_20 = recent_shorts[-20:] if len(recent_shorts) >= 20 else recent_shorts

    weights = []
    for animal_tuple in animals:
        short = animal_tuple[1]
        w = 1.0
        if short in last_5:
            w = 0.3
        elif short in last_10:
            w = 0.6
        elif short in last_20:
            w = 0.8
        weights.append(w)

    # Generate with weighted animal selection
    result = template.generate_weighted(animal_weights=weights)

    # Save to history
    _save_animal_to_history(
        history,
        result["variables"]["animal_short"],
        result["variables"]["animal"],
    )

    # Print header
    print("\n" + "=" * 70)
    print("  KLING 3.0 PROMPT SET — ANIMAL POV")
    print("=" * 70)
    print(f"  Template:  {result['template_name']}")
    print(f"  Animal:    {result['variables']['animal']} ({result['variables']['biome']})")
    print(f"  Structure: 5 clips × 5 seconds = 25 seconds total")
    print(f"  Cost:      50 credits (Standard) / 175 credits (Professional)")
    print()

    print("  Variables:")
    for k, val in result["variables"].items():
        print(f"    {k}: {val}")
    print()

    # Print prompts
    print("=" * 70)
    print("  COPY-PASTE PROMPTS FOR KLING 3.0 — ANIMAL POV")
    print("  Generate each as 5-second clip, then stitch in CapCut/DaVinci")
    print("=" * 70)

    neg_prompt = (
        "blurry, low quality, distorted faces, extra fingers, mutation, "
        "deformed, ugly, watermark, text overlay, logo, out of frame, "
        "bad anatomy, bad proportions, duplicate, glitch, noise"
    )

    for stage in result["stages"]:
        num = stage["stage"]
        name = stage["name"].upper()
        prompt = stage["video_prompt"]
        print(f"\n{'─' * 70}")
        print(f"  CLIP {num}/5: {name}")
        print(f"  Duration: 5 seconds | Aspect: 9:16 vertical | Mode: Standard")
        print(f"{'─' * 70}")
        print(f"\n[Prompt]\n{prompt}\n")
        print(f"[Negative Prompt]\n{neg_prompt}\n")
    print("=" * 70)

    # Credit summary
    print(f"\n  CREDIT SUMMARY:")
    print(f"  Standard mode:      5 clips × 10 credits = 50 credits")
    print(f"  Professional mode:  5 clips × 35 credits = 175 credits")
    print()

    # Save output
    _save_animal_pov_output(result)

    logger.info("=== DONE — Animal POV prompts generated ===")


def _save_animal_pov_output(result: dict) -> None:
    """Save animal POV output."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    animal_short = result["variables"].get("animal_short", "unknown")
    filename = f"animal_pov_{animal_short}_{now.strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "created_at": now.isoformat(),
        "mode": "animal_pov",
        "template_id": result["template_id"],
        "template_name": result["template_name"],
        "animal": result["variables"].get("animal", ""),
        "animal_short": animal_short,
        "biome": result["variables"].get("biome", ""),
        "total_duration_seconds": result["total_duration_seconds"],
        "variables": result["variables"],
        "stages": result["stages"],
        "status": "planned",
    }

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.getLogger("build").info("Animal POV output saved to: %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="Kling 3.0 Prompt Generator — Construction Timelapse Luxury Transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (default: template lottery):
  (default)    Template lottery. Picks 1 of 5 templates, fills variables. No API. No cost.
  --dry-run    LLM-only mode. Generates real candidates via DeepSeek (~$0.003).
  --offline    Fully offline. No API calls. Uses hardcoded examples.

Kling 3.0 workflow:
  1. Run this script to generate 5 copy-paste prompts
  2. Paste each prompt into Kling 3.0 (5s, 9:16 vertical, Standard mode)
  3. Stitch 5 clips together in CapCut/DaVinci Resolve

Cost: 50 credits/video (Standard) or 175 credits/video (Professional)
Pro plan ($37/month, 3000 credits): ~60 videos/month (Standard)
        """,
    )
    parser.add_argument(
        "--force-template", type=str, default=None,
        choices=["pool_vehicle", "resin_table", "fiber_optic_floor", "garden_strip", "pool_megastructure"],
        help="Force a specific template instead of lottery",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Fully offline test with hardcoded examples (no API calls)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="LLM candidate generation via DeepSeek (~$0.003)",
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

    if args.dry_run:
        run_dry(config)
    elif args.offline:
        run_offline(config)
    else:
        run_template(config, force_template=args.force_template)
        run_animal_pov(config)


if __name__ == "__main__":
    main()
