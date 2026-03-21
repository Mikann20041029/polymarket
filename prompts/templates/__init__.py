"""
Ultra-precise video prompt templates derived from frame-by-frame analysis of
5 rebornspacestv reference videos (98 frames total).

Architecture:
  - 5 TEMPLATE classes, each faithfully reproducing one reference video
  - Each template has 20-40 VARIABLES that can be randomized
  - Variables ONLY change nouns/colors/materials — structure is LOCKED
  - A lottery system picks 1 of 5, fills variables, outputs stage prompts

Usage:
    from prompts.templates import TemplateLottery
    lottery = TemplateLottery()
    result = lottery.draw()  # picks 1 of 5, fills variables
    # result["stages"] = list of per-stage video prompts
"""

from prompts.templates.lottery import TemplateLottery
from prompts.templates.base import BaseTemplate

__all__ = ["TemplateLottery", "BaseTemplate"]
