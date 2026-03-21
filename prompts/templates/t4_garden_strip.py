"""
Template 4: Garden Strip Transformation — 5×5s Kling 3.0 prompts

REFERENCE: Luxury Flooring (flower garden)
STRUCTURE: 5 clips × 5 seconds each = 25 seconds total

Clip 1 (0-5s):   Worker digs up sod along house wall (low forced perspective)
Clip 2 (5-10s):  Weed barrier + soil poured into trench
Clip 3 (10-15s): Time-lapse of seedlings sprouting and growing
Clip 4 (15-20s): Full bloom approaching — buds opening, color explosion building
Clip 5 (20-25s): Final spectacular reveal — full bloom garden in warm light

NOTE: Original was 11 seconds with a hard-cut black screen.
      For 25s, we expand the growth sequence into 3 clips (3-4-5)
      which creates a more satisfying progressive reveal.
"""
from prompts.templates.base import BaseTemplate


class GardenStripTemplate(BaseTemplate):
    template_id = "garden_strip"
    template_name = "Garden Strip Transformation"
    reference_video = "04_luxury_flooring.mp4"
    total_duration_seconds = 25.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
            "house_siding": [
                "gray horizontal vinyl lap siding",
                "white painted clapboard siding",
                "warm red brick wall",
                "dark charcoal cedar shingle siding",
                "cream stucco wall",
                "natural stone wall with mortar joints",
                "weathered barn wood siding",
                "sage green board-and-batten",
            ],
            "foundation_type": [
                "concrete foundation strip", "exposed stone foundation",
                "painted gray block foundation", "brick foundation course",
            ],
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
            ],
            "weather": [
                "overcast diffused daylight, neutral color temperature",
                "bright sunny day with sharp shadows",
                "soft golden afternoon light",
                "hazy warm summer light",
            ],
            "worker_top": [
                "bright red polo shirt", "navy blue work t-shirt",
                "olive green henley", "white linen button-up",
                "flannel plaid shirt", "black athletic shirt",
            ],
            "worker_pants": [
                "black work pants", "faded blue jeans",
                "khaki cargo pants", "dark green garden pants",
            ],
            "dig_tool": [
                "flat-blade spade", "pointed garden shovel",
                "broad hoe", "garden fork",
            ],
            "soil_type": [
                "dark rich black compost", "brown loamy garden soil",
                "red clay-rich soil", "dark peat-based potting mix",
            ],
            "barrier_method": [
                "cardboard egg cartons as biodegradable weed barrier",
                "sheets of corrugated cardboard as weed suppression",
                "thick layer of newspaper as biodegradable mulch",
                "coconut coir fiber mats as planting medium",
            ],
            "plant_type": [
                "mixed cottage flowers",
                "vegetable garden",
                "herb garden",
                "native wildflower meadow strip",
                "climbing vine wall",
            ],
            "bloom_palette": [
                "bright orange zinnias, white shasta daisies, pink cosmos, purple petunias, yellow marigolds, and magenta blooms — full rainbow spectrum",
                "deep red roses, pale pink peonies, white hydrangeas, and lavender spikes — romantic pastel palette",
                "bright yellow sunflowers, orange rudbeckia, red dahlias, and golden coreopsis — warm sunset tones",
                "blue delphiniums, purple salvia, white sweet alyssum, and pink foxglove — cool cottage tones",
                "lime green lettuces, purple cabbages, red tomatoes, yellow squash flowers, and emerald basil — edible rainbow",
                "rosemary bushes, purple lavender, mint patches, golden oregano, and sage — Mediterranean herbs",
            ],
            "bloom_height": [
                "medium height 6-12 inches, dense and full",
                "tall 12-24 inches, reaching upward",
                "low and spreading 3-6 inches, carpet-like",
                "mixed heights, layered from front to back",
            ],
            "reveal_lighting": [
                "warmer golden afternoon light",
                "soft morning light with dew visible on petals",
                "bright overhead midday sun with vivid saturated colors",
                "warm sunset sidelight creating long petal shadows",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        cam = (
            "Low angle nearly ground level, 1-2 feet off ground, 9:16 vertical portrait. "
            "Positioned at one end of the strip along the house wall, looking lengthwise. "
            "Strong one-point perspective — horizontal siding lines converge to vanishing point."
        )

        return [
            {
                "stage": 1,
                "name": "dig",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} A {v['house_siding']} wall dominates the right side with converging "
                    f"horizontal lines. {v['foundation_type']} at the base. {v['lawn_type']} on the left. "
                    f"{v['background']}. {v['weather']}. A worker in {v['worker_top']} and "
                    f"{v['worker_pants']} uses a {v['dig_tool']} to dig up a strip of grass along "
                    f"the foundation. A large clump of grass and soil lifts toward the camera in "
                    f"dramatic forced perspective. A narrow trench 12-18 inches wide exposed, "
                    f"bare brown soil visible. The worker crouches and lays {v['barrier_method']} "
                    f"into the trench, pressing down with both hands."
                ),
                "sfx_prompt": (
                    "shovel slicing through sod, grass roots tearing, soil falling, birds singing"
                ),
            },
            {
                "stage": 2,
                "name": "soil",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same low angle along the {v['house_siding']} wall. The worker in "
                    f"{v['worker_top']} pours {v['soil_type']} from a large dark plastic bag onto "
                    f"the prepared trench. Dark soil creates a dramatic contrast band between the "
                    f"{v['lawn_type']} and the {v['house_siding']}. The worker walks toward the "
                    f"vanishing point, spreading soil evenly. Then exits frame. "
                    f"Static shot of the completed dark soil bed — a neat defined strip of "
                    f"{v['soil_type']} running the full length of the wall. The strip is empty, dark, "
                    f"ready. Clean composition: green grass, dark soil strip, {v['house_siding']}."
                ),
                "sfx_prompt": (
                    "soil pouring from bag with rustling cascade, footsteps on grass, wind in leaves"
                ),
            },
            {
                "stage": 3,
                "name": "sprouting",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same low angle. TIME-LAPSE of weeks passing. The dark {v['soil_type']} "
                    f"strip begins to change. Tiny green seedling shoots push up through the soil — "
                    f"first a few scattered points of green, then more and more. Light shifts from "
                    f"overcast to sunny to overcast as days pass. The seedlings grow taller, developing "
                    f"first true leaves. By end of clip the strip has transformed from bare soil to "
                    f"a dense carpet of green foliage 3-4 inches tall. No flowers yet — just lush "
                    f"green growth. The contrast with the {v['house_siding']} is already striking."
                ),
                "sfx_prompt": (
                    "time-lapse ambience — shifting wind, day-night cycles, gentle nature sounds"
                ),
            },
            {
                "stage": 4,
                "name": "budding",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same low angle. TIME-LAPSE continues. The green foliage strip grows "
                    f"taller and denser — now {v['bloom_height']}. Flower buds form and begin to open. "
                    f"The first spots of color appear: hints of the {v['bloom_palette']}. "
                    f"More buds open rapidly in time-lapse — color spreads across the strip like "
                    f"paint being splashed. The transition from green to multicolored accelerates. "
                    f"By end of clip, 60-70 percent of buds have opened. The garden is becoming "
                    f"spectacular but hasn't reached peak bloom yet. Building anticipation."
                ),
                "sfx_prompt": (
                    "gentle orchestral swell building, soft breeze, bees buzzing faintly"
                ),
            },
            {
                "stage": 5,
                "name": "full_bloom_reveal",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same low angle — IDENTICAL position to all previous clips. "
                    f"{v['reveal_lighting']}. SPECTACULAR FULL BLOOM: the strip is packed dense with "
                    f"{v['bloom_palette']}. Every inch is overflowing with color and life. "
                    f"The {v['house_siding']} acts as a neutral gallery wall making colors pop intensely. "
                    f"Camera pushes very slightly forward — flowers fill more of the frame. "
                    f"Individual petal textures catch warm backlighting. Some petals nearly at macro "
                    f"distance from lens. The {v['foundation_type']} completely concealed by overflowing "
                    f"foliage. Butterflies and bees visible. The transformation from dark soil to "
                    f"this explosion of color is the entire emotional payoff. No worker visible."
                ),
                "sfx_prompt": (
                    "warm ambient pad, soft breeze through petals, honeybees buzzing, birdsong"
                ),
            },
        ]
