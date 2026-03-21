"""
Base class for all video prompt templates.

Each template encodes:
  - FIXED structure (camera angles, timing, phases) — never changes
  - VARIABLE slots (nouns, colors, materials) — randomized each run
"""
import random
from abc import ABC, abstractmethod


class BaseTemplate(ABC):
    """Base class for ultra-precise video prompt templates."""

    # Subclasses must define these
    template_id: str = ""
    template_name: str = ""
    reference_video: str = ""
    total_duration_seconds: float = 0
    num_stages: int = 0

    @abstractmethod
    def get_variable_pools(self) -> dict[str, list]:
        """Return dict of variable_name -> list of possible values.

        Each variable represents a swappable noun/color/material.
        The structure and camera work remain identical.
        """
        ...

    @abstractmethod
    def build_stages(self, variables: dict[str, str]) -> list[dict]:
        """Build the list of stage prompts using resolved variables.

        Each stage dict has:
          - stage: int (1-based)
          - name: str
          - duration_seconds: float
          - video_prompt: str (60-120 words, ultra-specific)
          - sfx_prompt: str (15-30 words)
          - camera: str (exact camera description)
        """
        ...

    def resolve_variables(self, overrides: dict | None = None) -> dict[str, str]:
        """Pick one random value per variable slot.

        Special handling for paired variables:
          Keys starting with '_' and ending with '_pair' contain tuples.
          E.g. "_vehicle_pair": [("luxury yacht", "yacht"), ...]
          → resolves to "vehicle": "luxury yacht" AND "vehicle_short": "yacht"
          The pair key prefix (after '_') and suffix (before '_pair') determines
          the output keys: "{name}" and "{name}_short".
        """
        pools = self.get_variable_pools()
        resolved = {}
        for key, choices in pools.items():
            if key.startswith("_") and key.endswith("_pair"):
                # Paired variable: pick one tuple, split into two keys
                pair = random.choice(choices)
                base_name = key[1:].replace("_pair", "")  # "_vehicle_pair" -> "vehicle"
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    resolved[base_name] = pair[0]
                    resolved[f"{base_name}_short"] = pair[1]
                else:
                    resolved[base_name] = str(pair)
            else:
                resolved[key] = random.choice(choices)
        if overrides:
            resolved.update(overrides)
        return resolved

    def generate(self, overrides: dict | None = None) -> dict:
        """Full generation: resolve variables → build stages → return result."""
        variables = self.resolve_variables(overrides)
        stages = self.build_stages(variables)
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "reference_video": self.reference_video,
            "total_duration_seconds": self.total_duration_seconds,
            "num_stages": self.num_stages,
            "variables": variables,
            "stages": stages,
            # Legacy compat
            "video_prompt": stages[0]["video_prompt"] if stages else "",
            "sfx_prompt": stages[0]["sfx_prompt"] if stages else "",
            "duration_seconds": self.total_duration_seconds,
        }
