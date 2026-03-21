"""
Template 4: Garden Strip Transformation (ref: Luxury Flooring / 04)

STRUCTURE (LOCKED):
  Stage 1: Worker digs up sod along house wall (low ground angle, forced perspective)
  Stage 2: Weed barrier + soil added (same low angle)
  Stage 3: Hard cut to BLACK (time skip representing weeks/months of growth)
  Stage 4: REVEAL — full bloom / completed garden in 3-frame crescendo, increasingly close

KEY INSIGHT: This template has the SIMPLEST structure but maximum emotional payoff.
  The same fixed camera angle throughout creates instant before/after comparison.
  The black frame is the structural hinge — it compresses time and creates anticipation.

VARIABLES:
  - What is GROWN (flowers, vegetables, herbs, succulents, vines)
  - The HOUSE EXTERIOR (siding type, color)
  - The soil/growing medium TYPE
  - The weed barrier METHOD
  - The color palette of the final garden
  - The worker appearance
"""
from prompts.templates.base import BaseTemplate


class GardenStripTemplate(BaseTemplate):
    template_id = "garden_strip"
    template_name = "Garden Strip Transformation"
    reference_video = "04_luxury_flooring.mp4"
    total_duration_seconds = 11.0
    num_stages = 4

    def get_variable_pools(self) -> dict[str, list]:
        return {
            # ── House exterior ──
            "house_siding": [
                "gray horizontal vinyl lap siding",
                "white painted clapboard siding",
                "warm red brick wall",
                "dark charcoal cedar shingle siding",
                "cream stucco wall",
                "natural stone wall with mortar joints",
                "weathered barn wood siding",
                "sage green horizontal board-and-batten",
            ],
            "foundation_type": [
                "concrete foundation strip", "exposed stone foundation",
                "painted gray block foundation", "brick foundation course",
            ],
            # ── Setting ──
            "lawn_type": [
                "green lawn", "patchy brown-green lawn",
                "clover-mixed lawn", "dry summer grass",
            ],
            "background": [
                "gray wooden fence, neighboring houses, and overcast sky",
                "white picket fence with rose bushes beyond",
                "chain-link fence with dense hedge behind",
                "stone wall with climbing ivy and blue sky",
                "cedar privacy fence with maple trees beyond",
                "wrought-iron fence with distant hills visible",
            ],
            "weather": [
                "overcast diffused daylight, neutral color temperature",
                "bright sunny day with sharp shadows",
                "soft golden afternoon light",
                "hazy warm summer light",
                "crisp autumn daylight with scattered clouds",
            ],
            # ── Worker ──
            "worker_top": [
                "bright red polo shirt", "navy blue work t-shirt",
                "olive green henley", "white linen button-up",
                "yellow safety vest over gray shirt", "coral tank top",
                "flannel plaid shirt", "black long-sleeve athletic shirt",
            ],
            "worker_pants": [
                "black work pants", "faded blue jeans",
                "khaki cargo pants", "dark green garden pants",
                "brown canvas work pants",
            ],
            "worker_gloves": [
                "black work gloves", "green garden gloves",
                "tan leather gloves", "orange rubber gloves",
            ],
            # ── Digging ──
            "dig_tool": [
                "flat-blade spade", "pointed garden shovel",
                "broad hoe", "mattock pick",
                "garden fork",
            ],
            "soil_type": [
                "dark rich black compost", "brown loamy garden soil",
                "red clay-rich soil", "sandy light-brown soil",
                "dark peat-based potting mix",
            ],
            "barrier_method": [
                "cardboard egg cartons as biodegradable weed barrier",
                "sheets of corrugated cardboard as weed suppression",
                "thick layer of newspaper as biodegradable mulch",
                "coconut coir fiber mats as planting medium",
                "burlap sacking as biodegradable ground cover",
            ],
            # ── What grows ──
            "plant_type": [
                "mixed cottage flowers",
                "vegetable garden",
                "herb garden",
                "succulent and cactus garden",
                "native wildflower meadow strip",
                "tropical foliage",
                "climbing vine wall",
                "ornamental grass garden",
            ],
            "bloom_palette": [
                "bright orange zinnias, white shasta daisies, pink cosmos, purple petunias, yellow marigolds, and magenta blooms — full rainbow spectrum",
                "deep red roses, pale pink peonies, white hydrangeas, and lavender spikes — romantic pastel palette",
                "bright yellow sunflowers, orange rudbeckia, red dahlias, and golden coreopsis — warm sunset tones",
                "blue delphiniums, purple salvia, white sweet alyssum, and pink foxglove — cool cottage tones",
                "lime green lettuces, purple cabbages, red tomatoes on stakes, yellow squash flowers, and emerald basil — edible rainbow",
                "rosemary bushes, purple lavender, mint patches, golden oregano, and sage gray-green — Mediterranean herbs",
                "jade green echeveria rosettes, purple aeonium, pink sedum, golden barrel cactus, and pale blue senecio — desert jewels",
                "tall pink muhly grass plumes, silver miscanthus, amber sedge, and green fountain grass — waving texture",
            ],
            "bloom_height": [
                "medium height, 6-12 inches, dense and full",
                "tall, 12-24 inches, reaching upward",
                "low and spreading, 3-6 inches, carpet-like",
                "mixed heights, layered from front to back",
            ],
            "reveal_lighting": [
                "warmer golden afternoon light than during construction",
                "soft morning light with dew visible on petals",
                "bright overhead midday sun with vivid saturated colors",
                "warm sunset sidelight creating long petal shadows",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        camera = (
            "Low angle, nearly ground level, 1-2 feet off the ground, 9:16 vertical portrait. "
            "Positioned at one end of the strip along the house wall, looking lengthwise down the strip. "
            "Strong one-point perspective — the horizontal siding lines converge to a vanishing point. "
            "Camera position is IDENTICAL across all stages for instant before/after comparison."
        )

        return [
            # ── STAGE 1: Dig + prep (0-4s) ──
            {
                "stage": 1,
                "name": "dig_and_prep",
                "duration_seconds": 4.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} A {v['house_siding']} wall dominates the right side with strong converging "
                    f"horizontal lines. {v['foundation_type']} visible at the base. {v['lawn_type']} extends "
                    f"to the left. {v['background']}. {v['weather']}. "
                    f"A worker in {v['worker_top']}, {v['worker_pants']}, and {v['worker_gloves']} is bent "
                    f"forward using a {v['dig_tool']} to dig up a strip of grass along the foundation. "
                    f"A large clump of grass and soil lifts toward the camera in dramatic forced perspective. "
                    f"A narrow trench 12-18 inches wide is exposed, showing bare brown soil. "
                    f"The worker crouches and lays {v['barrier_method']} into the trench, pressing down "
                    f"with both hands. Gray-beige barrier material visible in the trench."
                ),
                "sfx_prompt": (
                    "shovel slicing through sod with earthy thud, grass roots tearing, "
                    "soil falling, birds singing, gentle outdoor breeze, hands patting down material"
                ),
            },
            # ── STAGE 2: Soil addition (4-7s) ──
            {
                "stage": 2,
                "name": "soil_and_bed",
                "duration_seconds": 3.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} Same low angle along the house wall. The worker in {v['worker_top']} "
                    f"stands bent forward, pouring {v['soil_type']} from a large dark plastic bag "
                    f"onto the prepared trench. The dark soil creates a dramatic contrast band between "
                    f"the {v['lawn_type']} and the {v['house_siding']}. "
                    f"The worker walks away toward the vanishing point, spreading soil evenly. "
                    f"Then the worker exits frame. A static shot of the completed dark soil bed — "
                    f"a neat, defined strip of {v['soil_type']} running the full visible length of the "
                    f"house wall. The strip is empty, dark, and ready. "
                    f"The geometric composition is clean: green grass, dark soil strip, "
                    f"{v['house_siding']}, {v['background']}."
                ),
                "sfx_prompt": (
                    "soil pouring from bag with dry rustling cascade, footsteps on grass, "
                    "birds singing, wind in leaves, then quiet stillness as worker leaves"
                ),
            },
            # ── STAGE 3: BLACK SCREEN time skip (7-8s) ──
            {
                "stage": 3,
                "name": "time_skip_black",
                "duration_seconds": 1.0,
                "camera": "Complete black screen. No camera. This is a hard cut to black.",
                "video_prompt": (
                    "Complete black screen. Total darkness. No image content. "
                    "This is a hard cut representing weeks or months of growth passing. "
                    "The black frame separates the construction phase from the reveal phase. "
                    "It acts as a palate cleanser that makes the following color explosion more dramatic."
                ),
                "sfx_prompt": (
                    "all sound cuts to silence for half a second, then a subtle rising "
                    "magical shimmer tone builds anticipation for the reveal"
                ),
            },
            # ── STAGE 4: REVEAL — full bloom crescendo (8-11s) ──
            {
                "stage": 4,
                "name": "bloom_reveal",
                "duration_seconds": 3.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} Same low ground-level angle along the house wall — IDENTICAL position "
                    f"to all previous stages. {v['reveal_lighting']}. "
                    f"DRAMATIC TRANSFORMATION: the dark soil strip is now filled with a dense, "
                    f"spectacular garden of {v['bloom_palette']}. "
                    f"The plants are {v['bloom_height']}. The {v['house_siding']} acts as a neutral "
                    f"gallery wall making the colors pop intensely. "
                    f"The camera holds this angle, then pushes very slightly forward — the flowers "
                    f"fill more of the frame. The color density increases as the camera moves deeper "
                    f"into the garden corridor. Some petals are nearly at macro distance from the lens. "
                    f"No worker visible. The {v['foundation_type']} is now completely concealed by "
                    f"overflowing foliage. Individual petal textures catch warm backlighting. "
                    f"The transformation from dark soil to this explosion of color is the entire payoff."
                ),
                "sfx_prompt": (
                    "gentle orchestral swell or warm ambient pad, soft breeze rustling through "
                    "flower petals, honeybees buzzing faintly, birdsong, peaceful summer garden"
                ),
            },
        ]
