"""
Template Lottery System.

1. Picks 1 of 5 templates at random (weighted or uniform)
2. Resolves all variables randomly from their pools
3. Returns a complete multi-stage prompt set ready for video generation

Usage:
    lottery = TemplateLottery()
    result = lottery.draw()
    # result["template_id"] = which template was chosen
    # result["variables"] = resolved variable values
    # result["stages"] = list of stage prompts
"""
import logging
import random

from prompts.templates.base import BaseTemplate
from prompts.templates.t1_pool_vehicle import PoolVehicleTemplate
from prompts.templates.t2_resin_table import ResinTableTemplate
from prompts.templates.t3_fiber_optic_floor import FiberOpticFloorTemplate
from prompts.templates.t4_garden_strip import GardenStripTemplate
from prompts.templates.t5_pool_megastructure import PoolMegastructureTemplate

logger = logging.getLogger(__name__)

# All available templates
ALL_TEMPLATES: list[BaseTemplate] = [
    PoolVehicleTemplate(),
    ResinTableTemplate(),
    FiberOpticFloorTemplate(),
    GardenStripTemplate(),
    PoolMegastructureTemplate(),
]

# Template weights (equal by default — adjust to favor certain types)
DEFAULT_WEIGHTS = {
    "pool_vehicle": 1.0,
    "resin_table": 1.0,
    "fiber_optic_floor": 1.0,
    "garden_strip": 1.0,
    "pool_megastructure": 1.0,
}


class TemplateLottery:
    """
    Lottery system that picks 1 of 5 templates, fills variables, and returns prompts.

    Features:
    - Weighted random selection (adjustable per template)
    - History-aware: avoids picking the same template consecutively
    - Variable overrides: force specific values if needed
    - Dry-run mode: returns result without API calls
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        history: list[str] | None = None,
    ):
        self.templates = {t.template_id: t for t in ALL_TEMPLATES}
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.history = list(history or [])

    def draw(
        self,
        exclude: list[str] | None = None,
        force_template: str | None = None,
        variable_overrides: dict | None = None,
    ) -> dict:
        """
        Draw one template from the lottery and generate prompts.

        Args:
            exclude: template IDs to exclude from this draw
            force_template: force a specific template (bypasses lottery)
            variable_overrides: dict of variable_name -> forced value

        Returns:
            Complete result dict with template info, variables, and stage prompts.
        """
        if force_template and force_template in self.templates:
            template = self.templates[force_template]
            logger.info("Forced template: %s", force_template)
        else:
            template = self._select(exclude)

        result = template.generate(overrides=variable_overrides)
        self.history.append(result["template_id"])

        logger.info(
            "Lottery result: template=%s, variables=%d, stages=%d",
            result["template_id"],
            len(result["variables"]),
            len(result["stages"]),
        )

        return result

    def draw_all_five(self, variable_overrides: dict | None = None) -> list[dict]:
        """Generate one result from each template (useful for preview/testing)."""
        results = []
        for template in ALL_TEMPLATES:
            result = template.generate(overrides=variable_overrides)
            results.append(result)
        return results

    def _select(self, exclude: list[str] | None = None) -> BaseTemplate:
        """Weighted random selection with history awareness."""
        exclude_set = set(exclude or [])

        # Reduce weight of recently used templates
        adjusted = {}
        for tid, weight in self.weights.items():
            if tid in exclude_set:
                adjusted[tid] = 0.0
                continue

            # Penalize if used in last 2 draws
            recent_penalty = 1.0
            if self.history and self.history[-1] == tid:
                recent_penalty = 0.1  # 90% reduction if just used
            elif len(self.history) >= 2 and self.history[-2] == tid:
                recent_penalty = 0.5  # 50% reduction if used 2 ago

            adjusted[tid] = weight * recent_penalty

        # Ensure at least some options are available
        available = {tid: w for tid, w in adjusted.items() if w > 0 and tid in self.templates}
        if not available:
            available = {tid: 1.0 for tid in self.templates}

        # Weighted random choice
        ids = list(available.keys())
        weights = [available[tid] for tid in ids]
        total = sum(weights)
        weights = [w / total for w in weights]

        chosen_id = random.choices(ids, weights=weights, k=1)[0]
        logger.info("Lottery selected: %s (from %d candidates)", chosen_id, len(ids))

        return self.templates[chosen_id]

    def get_template_info(self) -> list[dict]:
        """Return summary info for all templates."""
        info = []
        for t in ALL_TEMPLATES:
            pools = t.get_variable_pools()
            total_combos = 1
            for choices in pools.values():
                total_combos *= len(choices)

            info.append({
                "id": t.template_id,
                "name": t.template_name,
                "reference": t.reference_video,
                "duration": t.total_duration_seconds,
                "stages": t.num_stages,
                "variables": len(pools),
                "total_combinations": total_combos,
            })
        return info


def print_lottery_info():
    """Print template system summary."""
    lottery = TemplateLottery()
    info = lottery.get_template_info()

    print("\n" + "=" * 70)
    print("  TEMPLATE LOTTERY SYSTEM")
    print("=" * 70)
    total_vars = 0
    total_combos = 1

    for t in info:
        print(f"\n  [{t['id']}] {t['name']}")
        print(f"    Reference:    {t['reference']}")
        print(f"    Duration:     {t['duration']}s")
        print(f"    Stages:       {t['stages']}")
        print(f"    Variables:    {t['variables']}")
        print(f"    Combinations: {t['total_combinations']:,}")
        total_vars += t['variables']
        total_combos *= t['total_combinations']

    print(f"\n  TOTAL: {len(info)} templates, {total_vars} variables")
    print(f"  THEORETICAL COMBINATIONS: {total_combos:,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_lottery_info()

    print("\n--- Drawing from lottery ---\n")
    lottery = TemplateLottery()
    result = lottery.draw()

    print(f"Template: {result['template_name']}")
    print(f"Duration: {result['total_duration_seconds']}s")
    print(f"Stages:   {result['num_stages']}")
    print(f"\nVariables:")
    for k, val in result["variables"].items():
        print(f"  {k}: {val}")
    print(f"\nStage prompts:")
    for stage in result["stages"]:
        print(f"\n  [{stage['name']}] ({stage['duration_seconds']}s)")
        print(f"    Video: {stage['video_prompt'][:120]}...")
        print(f"    SFX:   {stage['sfx_prompt'][:80]}...")
