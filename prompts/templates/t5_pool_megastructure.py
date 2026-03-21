"""
Template 5: Pool Megastructure Build (ref: Submarine Base Pool / 05)

STRUCTURE (LOCKED):
  This is the PURE overhead drone construction video.
  Single locked-off drone angle throughout (except one welding close-up insert).
  Drain pool → stage materials → erect skeletal framework → skin hull → refill → twilight reveal.

  Differs from Template 1 (Pool Vehicle) because:
  - T1 has a furnished deck + night close-up finale
  - T5 is pure industrial construction + twilight beauty with NO furniture
  - T5 has a welding close-up insert shot
  - T5 focuses on the MASSIVE SCALE of the object

VARIABLES:
  - The MEGASTRUCTURE being built (submarine, rocket, whale sculpture, dinosaur, tank)
  - Material (steel, aluminum, carbon fiber)
  - Hull color
  - Setting (suburban, industrial, rural)
  - Reveal lighting
"""
from prompts.templates.base import BaseTemplate


class PoolMegastructureTemplate(BaseTemplate):
    template_id = "pool_megastructure"
    template_name = "Pool Megastructure Build"
    reference_video = "05_submarine_base_pool.mp4"
    total_duration_seconds = 25.0
    num_stages = 6

    def get_variable_pools(self) -> dict[str, list]:
        return {
            # ── The object (paired: full name, short name) ──
            "_megastructure_pair": [
                ("military submarine", "submarine"),
                ("space rocket", "rocket"),
                ("blue whale sculpture", "whale"),
                ("T-Rex dinosaur skeleton", "T-Rex"),
                ("medieval siege tower", "siege tower"),
                ("steampunk airship gondola", "airship"),
                ("giant chess piece set", "chess set"),
                ("full-scale fighter jet", "fighter jet"),
                ("Roman galley warship", "galley"),
                ("deep-sea diving bell", "diving bell"),
                ("armored tank", "tank"),
                ("giant electric guitar", "guitar"),
                ("lighthouse tower", "lighthouse"),
            ],
            # ── Construction material ──
            "frame_type": [
                "curved dark welded steel arches",
                "aluminum truss framework sections",
                "bent wood laminate ribs",
                "carbon fiber composite frames",
                "copper-riveted iron ribs",
                "galvanized steel tube framing",
            ],
            "skin_type": [
                "dark matte black steel plate panels",
                "brushed silver aluminum sheet panels",
                "riveted copper-patina panels",
                "white fiberglass composite panels",
                "camouflage-painted steel panels",
                "polished mirror-chrome panels",
            ],
            "skin_color": [
                "dark matte charcoal-black", "brushed silver",
                "verde-patina copper", "glossy white",
                "olive-drab military green", "mirror chrome",
            ],
            "distinctive_feature": [
                "conning tower with periscope mast",
                "nose cone with aerodynamic fairing",
                "tail flukes sculpted from sheet metal",
                "jaw with rows of steel teeth",
                "crenellated battlements at the top",
                "propeller nacelles and tail fins",
                "oversized crown piece",
                "canopy and vertical stabilizer",
                "ram bow and oar ports",
                "porthole windows and entry hatch",
                "turret with barrel",
                "tuning pegs and fretboard",
                "lamp housing and gallery railing",
            ],
            # ── Pool ──
            "pool_tile": [
                "blue square tile grid", "white mosaic tile",
                "dark gray slate tile", "turquoise pebble finish",
            ],
            "pool_water": [
                "turquoise-cyan", "deep teal", "crystal blue",
                "dark navy", "emerald green",
            ],
            "pool_surround": [
                "beige natural stone coping with dark mulch beds",
                "gray paver deck with boxwood hedging",
                "red brick coping with gravel border",
                "white concrete deck with potted palms",
                "dark slate coping with ornamental grasses",
            ],
            # ── Setting ──
            "background_house": [
                "beige two-story house with dark hip roof",
                "white modern cube house with flat roof",
                "red brick colonial with black shutters",
                "gray farmhouse with metal standing-seam roof",
                "stucco villa with terracotta roof tiles",
            ],
            "season_detail": [
                "golden-yellow autumn foliage on deciduous trees",
                "lush green summer canopy",
                "bare winter branches against gray sky",
                "pink cherry blossom branches",
            ],
            "fence_type": [
                "natural cedar horizontal slat fence",
                "white painted picket fence",
                "dark stained privacy fence",
                "stone wall with iron gate",
                "bamboo screen fence",
            ],
            # ── Workers ──
            "vest_color": [
                "hi-vis yellow-green", "bright orange",
                "reflective silver-stripe", "royal blue",
            ],
            "worker_count": ["3", "4", "5", "6"],
            # ── Heavy machinery ──
            "crane_type": [
                "mobile crane with gray boom and cable rigging",
                "compact telescopic handler with yellow arm",
                "overhead gantry crane on temporary rails",
                "small truck-mounted knuckle boom crane",
            ],
            "ground_machine": [
                "yellow skid-steer loader", "compact forklift",
                "mini excavator", "motorized wheelbarrow",
            ],
            # ── Reveal ──
            "reveal_sky": [
                "blue hour twilight with deep navy sky and warm amber house window lights",
                "dramatic sunset with orange and purple streaks",
                "clear starlit night with deep black sky",
                "moody overcast dusk with silver-gray clouds",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        drone = (
            "Elevated drone shot, approximately 30-35 feet high, 45 degrees from vertical, "
            "9:16 vertical portrait, hovering above the near end of the pool looking toward "
            f"a {v['background_house']} beyond a {v['fence_type']}. "
            "Camera position is FIXED and IDENTICAL across all stages."
        )

        return [
            # ── STAGE 1: Pool drain (0-4s) ──
            {
                "stage": 1,
                "name": "drain",
                "duration_seconds": 4.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} A rectangular in-ground pool filled with {v['pool_water']} water, "
                    f"{v['pool_tile']} visible on the floor. {v['pool_surround']}. "
                    f"{v['season_detail']} visible beyond the {v['fence_type']}. Overcast flat daylight. "
                    f"A yellow-orange gas pump sits at the near corner with black hoses in the water. "
                    f"Water turbulence and white foam where hoses operate. "
                    f"Water level drops steadily, exposing the pool walls. "
                    f"By end of clip: pool mostly empty, thin reflective puddle on {v['pool_tile']} floor, "
                    f"scattered debris visible."
                ),
                "sfx_prompt": (
                    "diesel pump engine rumbling, water gurgling through hoses, splashing, "
                    "water level dropping, outdoor ambience, distant traffic"
                ),
            },
            # ── STAGE 2: Materials + skeleton (4-10s) ──
            {
                "stage": 2,
                "name": "skeleton",
                "duration_seconds": 6.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} Pool is completely empty. {v['worker_count']} workers in {v['vest_color']} "
                    f"safety vests and hard hats carry lumber and metal structural beams into the pool. "
                    f"Materials staged in organized rows. Green laser alignment lines on the pool floor. "
                    f"A yellow DeWalt miter saw visible at the near end. Workers consult blueprints. "
                    f"Then: {v['frame_type']} begin rising from the pool floor. A central spine beam "
                    f"runs the full length of the pool. Transverse arches span the pool width, "
                    f"evenly spaced. 5-7 curved ribs create the unmistakable skeletal shape of a "
                    f"{v['megastructure']}. The arches near the ends have tighter curves simulating "
                    f"bow/stern tapering. Bright orange welding sparks visible at rib attachment points. "
                    f"The full skeletal framework is now clearly recognizable from above."
                ),
                "sfx_prompt": (
                    "metal clanging, welding crackle and hiss, power drill sequences, "
                    "workers shouting instructions, heavy beams being lifted and placed, "
                    "rhythmic construction sounds at timelapse speed"
                ),
            },
            # ── STAGE 3: Welding close-up INSERT (10-11s) ──
            {
                "stage": 3,
                "name": "welding_closeup",
                "duration_seconds": 1.5,
                "camera": (
                    "Ground-level extreme close-up, approximately 2-3 feet from welding action. "
                    "Shallow depth of field. 9:16 vertical. This breaks from the drone view."
                ),
                "video_prompt": (
                    f"Extreme close-up, 9:16 vertical, 2 feet from the action. A welder wearing "
                    f"a dark auto-darkening helmet with blue visor lens and yellow leather welding "
                    f"gloves arc-welds a {v['skin_color']} panel seam. Brilliant white-blue welding arc "
                    f"at center of frame creating intense light. Orange and white sparks spray outward "
                    f"in all directions from the weld point. Raw gray steel surface with visible weld "
                    f"bead lines. Worker wears {v['vest_color']} vest over dark clothing. "
                    f"Background is blurred — out-of-focus construction scene with bokeh highlights. "
                    f"This shot communicates craftsmanship and industrial power."
                ),
                "sfx_prompt": (
                    "intense welding arc crackling and buzzing, sparks sizzling on metal, "
                    "heavy breathing behind welding mask, muffled construction in background"
                ),
            },
            # ── STAGE 4: Skinning + completion (11-17s) ──
            {
                "stage": 4,
                "name": "skinning",
                "duration_seconds": 6.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} Back to overhead view. {v['skin_type']} are being attached to the "
                    f"rib framework. The far end shows significant skin coverage while the near end "
                    f"is still exposed skeleton. {v['crane_type']} extends over the pool. "
                    f"A {v['ground_machine']} sits on the concrete deck. Workers in {v['vest_color']} vests "
                    f"fasten panels — the skeleton transforms into a solid {v['skin_color']} hull. "
                    f"{v['distinctive_feature']} takes shape near the center. "
                    f"The {v['megastructure']} is now substantially complete — "
                    f"a {v['skin_color']} form fills most of the pool length. "
                    f"Workers and equipment on the {v['pool_surround']} deck around the perimeter."
                ),
                "sfx_prompt": (
                    "crane motor whirring, heavy panels clanking into place, "
                    "rivet guns firing, metallic hammering, power tools, wind over structure"
                ),
            },
            # ── STAGE 5: Water refill (17-21s) ──
            {
                "stage": 5,
                "name": "refill",
                "duration_seconds": 4.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} The {v['megastructure']} hull is fully complete — sleek {v['skin_color']} "
                    f"form with {v['distinctive_feature']} protruding prominently. "
                    f"{v['pool_water']} water is being pumped in — level rises around the hull. "
                    f"The water laps against the {v['skin_color']} surface. The waterline climbs. "
                    f"Workers clear tools and equipment from the deck in time-lapse motion blur. "
                    f"By end: pool is full, the {v['megastructure']} rises above the waterline by 2-3 feet. "
                    f"The {v['pool_water']} water contrasts beautifully with the {v['skin_color']} hull. "
                    f"All tools and equipment removed. Clean scene."
                ),
                "sfx_prompt": (
                    "water rushing from multiple hoses, pool filling and lapping against hull, "
                    "workers packing tools, footsteps on concrete, equipment being wheeled away"
                ),
            },
            # ── STAGE 6: Twilight reveal (21-25s) ──
            {
                "stage": 6,
                "name": "twilight_reveal",
                "duration_seconds": 4.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} Same exact angle. Time has jumped to twilight. "
                    f"{v['reveal_sky']}. The {v['megastructure']} sits in the {v['pool_water']} pool — "
                    f"its {v['skin_color']} hull catching subtle sky reflections along the curved top. "
                    f"{v['distinctive_feature']} silhouetted against the sky. "
                    f"The pool water has shifted to a deeper teal tone reflecting the twilight. "
                    f"House windows glow with warm amber interior light. {v['season_detail']} are darker "
                    f"silhouettes. The {v['fence_type']} barely visible. "
                    f"The scene deepens through blue hour — the sky darkens to deep navy, "
                    f"the hull becomes a dark silhouette with faint highlights along its spine, "
                    f"the water gains a subtle luminous teal quality. "
                    f"No workers, no tools. A {v['megastructure']} appears to surface in a suburban "
                    f"backyard pool at dusk — surreal, cinematic, magnificent. "
                    f"The final frame is the most atmospheric: near-darkness, warm window lights, "
                    f"dark hull silhouette, deep blue sky."
                ),
                "sfx_prompt": (
                    "serene evening ambience, gentle water lapping, distant crickets, "
                    "warm low-frequency atmospheric hum, occasional house sounds, deep peace"
                ),
            },
        ]
