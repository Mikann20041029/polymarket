"""
Template 2: Glowing Resin Craft Object (ref: Lava River Wood Table)

STRUCTURE (LOCKED):
  Stage 1: Outdoor torch-burning / surface treatment of raw material (high angle)
  Stage 2: Indoor workshop — LED strip installation into channels (low angle close-up)
  Stage 3: Colored epoxy resin pour over LEDs — the KEY satisfying moment (close-up)
  Stage 4: Torch de-gassing + orbital sander polishing (workshop shots)
  Stage 5: REVEAL — finished piece in luxury penthouse setting, night, three-light-source glamour

VARIABLES:
  - The BASE MATERIAL (charred wood, raw concrete, stone, metal)
  - The CHANNEL PATTERN (river, lightning bolt, fractal, circuit board)
  - The LED COLOR and EPOXY COLOR
  - The final OBJECT (coffee table, dining table, bar counter, desk)
  - The REVEAL SETTING (penthouse, gallery, rooftop lounge)
"""
from prompts.templates.base import BaseTemplate


class ResinTableTemplate(BaseTemplate):
    template_id = "resin_table"
    template_name = "Glowing Resin Craft Object"
    reference_video = "02_lava_river_wood_table.mp4"
    total_duration_seconds = 16.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
            # ── Base material ──
            "base_material": [
                "thick walnut wood slab", "raw concrete plank",
                "rough-hewn granite slab", "live-edge olive wood slab",
                "industrial steel plate", "reclaimed driftwood plank",
                "petrified wood slab", "black marble slab",
                "charred cypress wood slab", "rusted iron sheet",
            ],
            "base_color": [
                "medium-dark walnut brown", "cool industrial gray",
                "speckled granite gray", "warm olive-amber",
                "dark gunmetal", "sun-bleached silver-gray",
                "amber-brown with dark veins", "deep black with white veins",
                "jet black charred", "oxidized rust-orange",
            ],
            "surface_treatment": [
                "Shou Sugi Ban torch charring creating deep alligator-skin cracked texture",
                "acid-etching creating rough pitted lunar surface texture",
                "deep CNC-carved channels with clean geometric edges",
                "hand-chiseled channels with rough organic edges",
                "sandblasted channels revealing raw grain beneath",
                "laser-cut precision channels with smooth walls",
            ],
            # ── Channel pattern ──
            "channel_pattern": [
                "sinuous river-delta meandering pattern",
                "branching lightning-bolt fractal pattern",
                "geometric circuit-board trace pattern",
                "organic tree-root branching pattern",
                "cracked earth drought pattern",
                "topographic contour line pattern",
                "volcanic fissure jagged pattern",
                "coral reef branching pattern",
            ],
            # ── LED + Epoxy color ──
            "led_color": [
                "intense red-orange", "vivid cobalt blue",
                "bright emerald green", "deep violet-purple",
                "warm golden amber", "cool cyan-teal",
                "hot magenta-pink", "pure white",
            ],
            "epoxy_color": [
                "translucent blood-red crimson", "translucent deep cobalt blue",
                "translucent emerald green", "translucent violet-purple",
                "translucent honey-amber", "translucent teal-cyan",
                "translucent hot magenta", "translucent crystal clear",
            ],
            "glow_effect": [
                "molten lava river", "bioluminescent ocean trench",
                "toxic radioactive veins", "enchanted forest roots",
                "liquid gold flowing", "frozen northern lights",
                "neon cyberpunk circuits", "starlight crystal veins",
            ],
            # ── Final object ──
            "object_type": [
                "large rectangular coffee table", "long dining table",
                "bar counter top", "executive desk surface",
                "console table", "kitchen island countertop",
                "wall-mounted art panel", "outdoor fire pit surround",
            ],
            "table_legs": [
                "dark matte black metal hairpin legs",
                "brushed stainless steel X-frame base",
                "live-edge matching wood slab legs",
                "industrial cast-iron pedestal base",
                "transparent acrylic block supports",
                "raw welded steel A-frame legs",
            ],
            # ── Workshop details ──
            "workshop_lights": [
                "copper dome pendant lights", "exposed Edison bulb string",
                "industrial fluorescent tubes", "brass cage work lights",
                "chrome spot lights", "paper lantern pendants",
            ],
            "worker_apron": [
                "tan camel leather apron", "dark brown waxed canvas apron",
                "black rubber apron", "denim work apron",
                "olive canvas apron", "burgundy leather apron",
            ],
            # ── Reveal setting ──
            "reveal_room": [
                "luxury penthouse living room", "contemporary art gallery",
                "rooftop lounge bar", "private library with floor-to-ceiling bookshelves",
                "yacht interior salon", "mountain lodge great room",
                "underground speakeasy bar", "minimalist Japanese tea room",
            ],
            "reveal_chandelier": [
                "tiered crystal glass chandelier", "modern sputnik brass chandelier",
                "cascading blown-glass pendant cluster", "minimal track lighting",
                "paper globe lanterns", "antler chandelier",
                "geometric black wire chandelier", "floating candle display",
            ],
            "reveal_sofa": [
                "dark charcoal plush sectional sofa",
                "cream bouclé curved sofa",
                "cognac leather Chesterfield sofa",
                "navy velvet tufted sofa",
                "sage green linen sofa",
                "blush pink velvet sofa",
            ],
            "reveal_window_view": [
                "nighttime city skyline with scattered lights",
                "mountain range silhouette at dusk",
                "ocean horizon at blue hour",
                "dense forest canopy at twilight",
                "desert landscape under stars",
                "snowy rooftops under moonlight",
            ],
            # ── Outdoor setting for stage 1 ──
            "outdoor_setting": [
                "concrete patio with wooden privacy fence and green lawn",
                "gravel driveway beside a rustic barn",
                "stone terrace overlooking hills",
                "industrial loading dock with chain-link fence",
                "sandy workshop yard with corrugated metal wall",
                "flagstone courtyard with ivy-covered wall",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        return [
            # ── STAGE 1: Outdoor torch/treatment (0-3s) ──
            {
                "stage": 1,
                "name": "surface_treatment",
                "duration_seconds": 3.0,
                "camera": (
                    "High oblique angle, approximately 45 degrees from above, shooting down "
                    "at the tabletop surface. 9:16 vertical portrait. Then shifts to low angle "
                    "across the surface for the charred result shot."
                ),
                "video_prompt": (
                    f"High oblique angle, 9:16 vertical, looking down at a large {v['base_material']} "
                    f"arranged on a {v['outdoor_setting']}. Deep {v['channel_pattern']} channels carved "
                    f"into the surface. A male worker in black clothing, protective mask, ear protection, "
                    f"and black gloves leans over the piece, holding a propane torch pointed into the channels. "
                    f"Bright orange flame licks the {v['base_color']} surface. White-gray smoke rises from the "
                    f"point of contact. {v['surface_treatment']}. Overcast diffused daylight, cool neutral "
                    f"color temperature. The entire surface gradually darkens to deep charcoal black. "
                    f"Final low-angle shot across the fully treated surface showing the cracked volcanic texture."
                ),
                "sfx_prompt": (
                    "propane torch hissing and roaring, wood crackling and popping from heat, "
                    "smoke sizzling, outdoor ambient wind, distant birds"
                ),
            },
            # ── STAGE 2: Indoor LED installation (3-6s) ──
            {
                "stage": 2,
                "name": "led_install",
                "duration_seconds": 3.0,
                "camera": (
                    "Very low angle, nearly table-level, approximately 10-15 degrees above the "
                    "surface plane. Shallow depth of field. Indoor workshop. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Very low angle, nearly surface-level, 9:16 vertical, shallow depth of field. "
                    f"Indoor woodworking workshop with {v['workshop_lights']} overhead creating warm golden bokeh. "
                    f"The treated {v['base_material']} sits on a workbench — deeply cracked black charred texture "
                    f"like cooled volcanic rock. A female worker with {v['worker_apron']} and black nitrile gloves "
                    f"carefully presses {v['led_color']} LED strip lights into the {v['channel_pattern']} channels. "
                    f"The LED strips emit vivid {v['led_color']} pinpoint glow from individual diodes. "
                    f"The {v['led_color']} light reflects off the glossy black charred surface. "
                    f"The channels glow like {v['glow_effect']} against the dark volcanic texture. "
                    f"A brand neon sign glows in the soft background."
                ),
                "sfx_prompt": (
                    "quiet workshop ambience, soft electronic hum from LEDs activating, "
                    "gentle pressing sounds, worker breathing, faint background music"
                ),
            },
            # ── STAGE 3: Epoxy pour — the MONEY SHOT (6-10s) ──
            {
                "stage": 3,
                "name": "epoxy_pour",
                "duration_seconds": 4.0,
                "camera": (
                    "Close-up, approximately 30-40 degrees from above, focused on the pour point. "
                    "Shallow depth of field. Then pulls back to medium shot showing full table pour. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Close-up, 9:16 vertical, 30 degrees from above. Gloved hands tilt a white plastic "
                    f"mixing cup, pouring thick viscous {v['epoxy_color']} epoxy resin into the "
                    f"{v['channel_pattern']} channels. The resin streams downward in a thick glossy ribbon, "
                    f"pooling over the {v['led_color']} LED strips beneath. The LEDs backlight the translucent "
                    f"resin from below — the liquid glows intensely like {v['glow_effect']}. "
                    f"The charred black surface looks exactly like volcanic rock surrounding rivers of "
                    f"glowing {v['epoxy_color']} resin. Camera pulls back to reveal the full {v['object_type']} — "
                    f"a female worker pours from a larger pitcher. The {v['epoxy_color']} resin fills "
                    f"60-70 percent of all channels. The surface is now a dramatic glowing veined pattern. "
                    f"Mirror-like reflections on the wet glossy resin surface."
                ),
                "sfx_prompt": (
                    "thick viscous liquid pouring and pooling, satisfying gloopy ASMR sounds, "
                    "resin spreading with wet sheen sound, gentle workshop ambience"
                ),
            },
            # ── STAGE 4: Torch + polish (10-14s) ──
            {
                "stage": 4,
                "name": "degas_and_polish",
                "duration_seconds": 4.0,
                "camera": (
                    "Low-medium angle along the table length, shallow depth of field. "
                    "Then medium close-up from above for sanding. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Low-medium angle along the {v['object_type']} length, 9:16 vertical, shallow depth of field. "
                    f"A female worker holds a small silver handheld butane torch, passing flame over the "
                    f"{v['epoxy_color']} resin surface — popping micro-bubbles. The resin glows deep "
                    f"{v['led_color']} from the embedded LEDs. Warm {v['workshop_lights']} hang above. "
                    f"Then the epoxy has cured. The worker uses a random orbital sander with white pad "
                    f"on the surface, polishing to a mirror-smooth high-gloss finish. "
                    f"Small warm amber reflections dot the glossy surface. "
                    f"The {v['channel_pattern']} channels glow with deep internal luminescence through "
                    f"the polished {v['epoxy_color']} resin. The charred sections are matte black. "
                    f"The contrast between glossy glowing channels and matte dark surface is stunning."
                ),
                "sfx_prompt": (
                    "butane torch click and hiss, small bubbles popping, then orbital sander whirring, "
                    "smooth polishing rhythm, satisfying surface buffing sound"
                ),
            },
            # ── STAGE 5: Luxury reveal (14-16s) ──
            {
                "stage": 5,
                "name": "luxury_reveal",
                "duration_seconds": 2.0,
                "camera": (
                    "Medium wide shot, 25-30 degrees from above, looking down at the finished piece "
                    "in its room setting. 9:16 vertical. Piece centered in frame."
                ),
                "video_prompt": (
                    f"Medium wide shot, 9:16 vertical, 25 degrees from above. The finished "
                    f"{v['object_type']} with {v['table_legs']} sits in a dimly lit {v['reveal_room']}. "
                    f"The {v['channel_pattern']} channels glow vivid {v['led_color']} through the polished "
                    f"{v['epoxy_color']} resin — the piece radiates like {v['glow_effect']}. "
                    f"The {v['led_color']} light spills onto the surrounding floor. "
                    f"Above: {v['reveal_chandelier']} provides sparkling warm accent light. "
                    f"Flanking the table: {v['reveal_sofa']} on both sides. "
                    f"Floor-to-ceiling windows show {v['reveal_window_view']}. "
                    f"The room is deliberately dark so the glowing {v['object_type']} is the dominant "
                    f"visual element. The glossy resin surface creates mirror reflections of the chandelier. "
                    f"A hardcover book and decorative glass object sit on the surface. "
                    f"Three-point color temperature: warm chandelier, {v['led_color']} table glow, cool window light."
                ),
                "sfx_prompt": (
                    "deep atmospheric ambient hum, very faint city sounds through glass, "
                    "subtle electrical hum from the glowing table, serene luxurious silence"
                ),
            },
        ]
