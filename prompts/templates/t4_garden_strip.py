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
                "pale yellow aluminum siding",
                "tan fiber-cement lap siding",
                "blue-gray Dutch colonial clapboard",
                "terracotta adobe render",
                "dark brown log cabin wall",
                "white-washed brick wall",
                "copper-patina standing-seam metal wall",
                "moss-stained concrete block wall",
                "peach stucco with decorative quoins",
            ],
            "foundation_type": [
                "concrete foundation strip", "exposed stone foundation",
                "painted gray block foundation", "brick foundation course",
                "parged and painted white foundation", "fieldstone rubble foundation",
                "stacked limestone block foundation", "smooth poured concrete footing",
                "river rock foundation veneer", "split-face concrete block foundation",
            ],
            "lawn_type": [
                "green lawn", "patchy brown-green lawn",
                "clover-mixed lawn", "dry summer grass",
                "thick zoysia turf", "freshly mowed bermuda grass",
                "dew-covered morning lawn", "fescue lawn with scattered dandelions",
                "overseeded ryegrass lawn", "crabgrass-speckled thin lawn",
            ],
            "background": [
                "gray wooden fence, neighboring houses, and overcast sky",
                "white picket fence with rose bushes beyond",
                "chain-link fence with dense hedge behind",
                "stone wall with climbing ivy and blue sky",
                "cedar privacy fence with maple trees beyond",
                "bamboo screen fence with jasmine climbing over",
                "corrugated metal fence with eucalyptus trees behind",
                "lattice trellis with wisteria and blue sky beyond",
                "split-rail fence with wildflower meadow behind",
                "wrought-iron fence with magnolia tree beyond",
                "brick garden wall with espaliered fruit tree",
                "stockade fence with tall ornamental grasses behind",
            ],
            "weather": [
                "overcast diffused daylight, neutral color temperature",
                "bright sunny day with sharp shadows",
                "soft golden afternoon light",
                "hazy warm summer light",
                "dappled light through tree canopy overhead",
                "cool blue morning light with long shadows",
                "flat midday light with thin high clouds",
                "warm sidelight after a rain shower with wet surfaces",
                "misty early spring morning with soft diffusion",
                "crisp autumn light with deep blue sky",
            ],
            "worker_top": [
                "bright red polo shirt", "navy blue work t-shirt",
                "olive green henley", "white linen button-up",
                "flannel plaid shirt", "black athletic shirt",
                "gray hoodie", "tan canvas work jacket",
                "maroon baseball henley", "sky blue chambray shirt",
                "mustard yellow rain jacket", "dark green garden smock",
                "striped Breton long-sleeve shirt", "heather gray thermal",
            ],
            "worker_pants": [
                "black work pants", "faded blue jeans",
                "khaki cargo pants", "dark green garden pants",
                "brown canvas overalls", "gray athletic joggers",
                "olive drab utility pants", "stone-washed carpenter jeans",
                "navy blue dickies", "tan linen drawstring pants",
            ],
            "dig_tool": [
                "flat-blade spade", "pointed garden shovel",
                "broad hoe", "garden fork",
                "half-moon edging tool", "mattock",
                "border spade", "trenching shovel",
                "draw hoe", "digging bar",
            ],
            "soil_type": [
                "dark rich black compost", "brown loamy garden soil",
                "red clay-rich soil", "dark peat-based potting mix",
                "sandy topsoil blend", "mushroom compost mix",
                "vermiculite-enriched seed starting mix", "aged horse manure compost",
                "worm-casting enriched soil", "forest humus and leaf mold",
            ],
            "barrier_method": [
                "cardboard egg cartons as biodegradable weed barrier",
                "sheets of corrugated cardboard as weed suppression",
                "thick layer of newspaper as biodegradable mulch",
                "coconut coir fiber mats as planting medium",
                "burlap sack strips as biodegradable liner",
                "wood chip mulch layer as weed suppression",
                "landscape fabric pinned with wire staples",
                "straw mat as moisture-retaining barrier",
                "shredded bark mulch as organic ground cover",
                "wool felt strips as weed-blocking layer",
            ],
            "plant_type": [
                "mixed cottage flowers",
                "vegetable garden",
                "herb garden",
                "native wildflower meadow strip",
                "climbing vine wall",
                "succulent rock garden",
                "ornamental grass border",
                "pollinator butterfly garden",
                "tropical foliage bed",
                "shade-loving fern garden",
                "cutting flower garden",
                "alpine rock garden",
            ],
            "bloom_palette": [
                "bright orange zinnias, white shasta daisies, pink cosmos, purple petunias, yellow marigolds, and magenta blooms — full rainbow spectrum",
                "deep red roses, pale pink peonies, white hydrangeas, and lavender spikes — romantic pastel palette",
                "bright yellow sunflowers, orange rudbeckia, red dahlias, and golden coreopsis — warm sunset tones",
                "blue delphiniums, purple salvia, white sweet alyssum, and pink foxglove — cool cottage tones",
                "lime green lettuces, purple cabbages, red tomatoes, yellow squash flowers, and emerald basil — edible rainbow",
                "rosemary bushes, purple lavender, mint patches, golden oregano, and sage — Mediterranean herbs",
                "white gardenias, pale pink camellias, cream jasmine, and soft green ferns — monochrome elegance",
                "electric purple alliums, magenta phlox, coral poppies, and yellow coneflowers — bold meadow tones",
                "silver lamb's ear, dusty blue catmint, white yarrow, and pale yellow sedum — muted prairie palette",
                "scarlet salvia, deep orange lantana, gold black-eyed Susans, and bronze helenium — fiery autumn hues",
                "pastel pink astilbe, white bleeding hearts, violet hostas, and chartreuse sweet potato vine — shade garden palette",
                "bright magenta bougainvillea, orange bird of paradise, red hibiscus, and yellow plumeria — tropical explosion",
            ],
            "bloom_height": [
                "medium height 6-12 inches, dense and full",
                "tall 12-24 inches, reaching upward",
                "low and spreading 3-6 inches, carpet-like",
                "mixed heights, layered from front to back",
                "very tall 24-36 inches, towering and dramatic",
                "compact 4-8 inches, tight mounding form",
                "cascading and trailing, spilling over the bed edge",
                "tiered staircase, ascending toward the wall",
                "undulating wave pattern, alternating heights",
                "uniform 10-14 inches, neat hedgerow-like form",
            ],
            "reveal_lighting": [
                "warmer golden afternoon light",
                "soft morning light with dew visible on petals",
                "bright overhead midday sun with vivid saturated colors",
                "warm sunset sidelight creating long petal shadows",
                "after-rain light with water droplets catching sun on petals",
                "diffused cloudy bright light with even color saturation",
                "early golden hour backlight making petals translucent",
                "blue-sky midmorning with crisp directional shadows",
                "magic hour with long warm horizontal rays",
                "high noon overhead sun with deep saturated colors",
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
