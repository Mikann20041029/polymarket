"""
Template 2: Glowing Resin Craft Object — 5×5s Kling 3.0 prompts

REFERENCE: Lava River Wood Table
STRUCTURE: 5 clips × 5 seconds each = 25 seconds total

Clip 1 (0-5s):   Outdoor torch charring of wood surface
Clip 2 (5-10s):  Indoor LED strip installation into channels
Clip 3 (10-15s): Colored epoxy resin pour — ASMR money shot
Clip 4 (15-20s): Torch de-gassing + orbital sander polishing
Clip 5 (20-25s): Luxury room reveal with glowing table as hero light
"""
from prompts.templates.base import BaseTemplate


class ResinTableTemplate(BaseTemplate):
    template_id = "resin_table"
    template_name = "Glowing Resin Craft Object"
    reference_video = "02_lava_river_wood_table.mp4"
    total_duration_seconds = 25.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
            "base_material": [
                "thick walnut wood slab", "raw concrete plank",
                "rough-hewn granite slab", "live-edge olive wood slab",
                "industrial steel plate", "reclaimed driftwood plank",
                "charred cypress wood slab", "black marble slab",
                "spalted maple burl slab", "thick mesquite wood slab",
                "weathered teak beam", "white ash live-edge slab",
                "petrified wood plank", "sinker cypress slab",
                "African padauk wood slab", "quartzite stone plank",
                "reclaimed railroad tie", "burled redwood slab",
            ],
            "base_color": [
                "medium-dark walnut brown", "cool industrial gray",
                "speckled granite gray", "warm olive-amber",
                "dark gunmetal", "sun-bleached silver-gray",
                "jet black charred", "deep black with white veins",
                "rich reddish-brown mahogany", "pale blonde ash",
                "warm honey-gold pine", "streaked purple-brown padauk",
                "mottled cream-and-tan spalted", "dark chocolate espresso",
                "sandy driftwood beige", "rustic red-orange terracotta",
                "fossil gray with amber veins", "mossy green-gray patina",
            ],
            "surface_treatment": [
                "Shou Sugi Ban torch charring creating deep alligator-skin texture",
                "acid-etching creating rough pitted lunar surface texture",
                "deep CNC-carved channels with clean geometric edges",
                "hand-chiseled channels with rough organic edges",
                "sandblasted channels revealing raw grain beneath",
                "router-carved channels with smooth rounded edges",
                "plasma-cut steel grooves with heat-blued edges",
                "water-jet carved channels with precision-cut walls",
                "chainsaw-carved rough troughs with splintered texture",
                "laser-engraved channels with crisp hairline edges",
                "natural wood-split cracks along the grain",
                "freeze-thaw fractured channels with jagged crystalline edges",
            ],
            "channel_pattern": [
                "sinuous river-delta meandering pattern",
                "branching lightning-bolt fractal pattern",
                "geometric circuit-board trace pattern",
                "organic tree-root branching pattern",
                "cracked earth drought pattern",
                "volcanic fissure jagged pattern",
                "spider web radiating pattern",
                "topographic contour line pattern",
                "Celtic knotwork interlacing pattern",
                "river tributary branching pattern",
                "Lichtenberg fractal burn pattern",
                "hexagonal honeycomb pattern",
                "marble vein meandering pattern",
            ],
            "led_color": [
                "intense red-orange", "vivid cobalt blue",
                "bright emerald green", "deep violet-purple",
                "warm golden amber", "cool cyan-teal",
                "hot magenta-pink", "pure white",
                "sunset coral-orange", "lime neon green",
                "icy silver-blue", "cherry red",
                "soft rose-pink", "turquoise aqua",
                "burnt sienna amber", "ultraviolet blacklight",
            ],
            "epoxy_color": [
                "translucent blood-red crimson", "translucent deep cobalt blue",
                "translucent emerald green", "translucent violet-purple",
                "translucent honey-amber", "translucent teal-cyan",
                "translucent hot magenta", "translucent crystal clear",
                "translucent coral-peach", "translucent lime green",
                "translucent icy silver", "translucent deep cherry",
                "translucent soft rose", "translucent ocean turquoise",
                "translucent smoky topaz", "translucent midnight black",
            ],
            "glow_effect": [
                "molten lava river", "bioluminescent ocean trench",
                "toxic radioactive veins", "enchanted forest roots",
                "liquid gold flowing", "frozen northern lights",
                "neon cyberpunk circuits", "starlight crystal veins",
                "deep-sea anglerfish lure", "plasma lightning bolt",
                "aurora borealis ribbon", "volcanic magma fissure",
                "fairy dust trail", "nebula gas cloud",
                "phosphorescent tide pool", "dragon fire breath",
            ],
            "object_type": [
                "large rectangular coffee table", "long dining table",
                "bar counter top", "executive desk surface",
                "console table", "kitchen island countertop",
                "wall-mounted floating shelf", "bathroom vanity top",
                "round bistro table", "fireplace mantel piece",
                "headboard panel", "staircase tread set",
                "window sill plank", "outdoor bench seat",
            ],
            "table_legs": [
                "dark matte black metal hairpin legs",
                "brushed stainless steel X-frame base",
                "live-edge matching wood slab legs",
                "industrial cast-iron pedestal base",
                "transparent acrylic block supports",
                "raw welded steel slab legs",
                "tapered mid-century walnut legs",
                "concrete cylinder pedestal base",
                "blackened brass trapezoid legs",
                "reclaimed pipe fitting legs",
                "carved stone column supports",
                "chrome waterfall U-frame base",
            ],
            "workshop_lights": [
                "copper dome pendant lights", "exposed Edison bulb string",
                "industrial fluorescent tubes", "brass cage work lights",
                "black enamel barn pendants", "vintage porcelain socket lights",
                "LED panel overhead flood", "galvanized steel cone shades",
                "goose-neck wall-mount task lights", "frosted globe string lights",
            ],
            "worker_apron": [
                "tan camel leather apron", "dark brown waxed canvas apron",
                "black rubber apron", "denim work apron",
                "olive canvas shop apron", "charcoal suede leather apron",
                "heavy natural linen apron", "burgundy oilcloth apron",
                "raw selvedge denim apron", "gray herringbone tweed apron",
            ],
            "reveal_room": [
                "luxury penthouse living room", "contemporary art gallery",
                "rooftop lounge bar", "private library with bookshelves",
                "yacht interior salon", "mountain lodge great room",
                "mid-century modern den", "underground wine cellar lounge",
                "boutique hotel lobby", "converted church nave",
                "industrial loft recording studio", "Scandinavian minimalist studio",
                "cigar lounge with dark paneling", "Japanese zen tea room",
            ],
            "reveal_sofa": [
                "dark charcoal plush sectional sofa",
                "cream bouclé curved sofa",
                "cognac leather Chesterfield sofa",
                "navy velvet tufted sofa",
                "emerald green velvet settee",
                "caramel suede mid-century sofa",
                "white linen slipcovered sofa",
                "burgundy leather wingback settee",
                "slate gray wool modular sofa",
                "mustard yellow bouclé loveseat",
            ],
            "reveal_window_view": [
                "nighttime city skyline with scattered lights",
                "mountain range silhouette at dusk",
                "ocean horizon at blue hour",
                "dense forest canopy at twilight",
                "desert mesa silhouette at sunset",
                "snowy alpine peaks under moonlight",
                "river valley with twinkling village lights",
                "tropical harbor with anchored sailboats",
                "vineyard hillside at golden hour",
                "industrial skyline with bridge lights",
            ],
            "outdoor_setting": [
                "concrete patio with wooden privacy fence and green lawn",
                "gravel driveway beside a rustic barn",
                "stone terrace overlooking hills",
                "flagstone courtyard with ivy-covered wall",
                "asphalt driveway beside a brick garage",
                "weathered dock planks by a lake shore",
                "packed dirt yard with corrugated metal lean-to",
                "rooftop terrace with cityscape behind",
                "mossy brick patio with wrought-iron trellis",
                "sandy clearing beside a desert workshop shed",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        return [
            {
                "stage": 1,
                "name": "torch_charring",
                "duration_seconds": 5.0,
                "camera": (
                    "High oblique angle 45 degrees from above, then low across surface. "
                    "9:16 vertical portrait."
                ),
                "video_prompt": (
                    f"High oblique angle, 9:16 vertical, looking down at a large {v['base_material']} "
                    f"on a {v['outdoor_setting']}. Deep {v['channel_pattern']} channels carved into the "
                    f"{v['base_color']} surface. A male worker in black clothing, protective mask, and "
                    f"black gloves holds a propane torch pointed into the channels. Bright orange flame "
                    f"licks the surface. White-gray smoke rises from the contact point. "
                    f"{v['surface_treatment']}. The entire surface gradually darkens to deep charcoal black. "
                    f"Final shot: low angle across the fully treated surface showing volcanic cracked texture. "
                    f"Overcast diffused daylight, cool neutral color temperature."
                ),
                "sfx_prompt": (
                    "propane torch hissing and roaring, wood crackling from heat, smoke sizzling"
                ),
            },
            {
                "stage": 2,
                "name": "led_install",
                "duration_seconds": 5.0,
                "camera": (
                    "Very low angle nearly table-level, 10-15 degrees above surface. "
                    "Shallow depth of field. Indoor workshop. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Very low angle nearly surface-level, 9:16 vertical, shallow depth of field. "
                    f"Indoor woodworking workshop with {v['workshop_lights']} overhead creating warm "
                    f"golden bokeh. The treated {v['base_material']} on a workbench — deeply cracked black "
                    f"charred texture like cooled volcanic rock. A female worker in {v['worker_apron']} and "
                    f"black nitrile gloves carefully presses {v['led_color']} LED strip lights into the "
                    f"{v['channel_pattern']} channels. The LED strips emit vivid {v['led_color']} pinpoint "
                    f"glow from individual diodes. The {v['led_color']} light reflects off the glossy black "
                    f"charred surface. The channels glow like {v['glow_effect']} against volcanic texture."
                ),
                "sfx_prompt": (
                    "quiet workshop ambience, soft electronic hum from LEDs, gentle pressing sounds"
                ),
            },
            {
                "stage": 3,
                "name": "epoxy_pour",
                "duration_seconds": 5.0,
                "camera": (
                    "Close-up 30 degrees from above focused on pour point, then pulls back "
                    "to medium shot. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Close-up, 9:16 vertical, 30 degrees from above. Gloved hands tilt a white plastic "
                    f"mixing cup, pouring thick viscous {v['epoxy_color']} epoxy resin into the "
                    f"{v['channel_pattern']} channels. The resin streams in a thick glossy ribbon, pooling "
                    f"over the {v['led_color']} LED strips beneath. The LEDs backlight the translucent resin "
                    f"from below — the liquid glows intensely like {v['glow_effect']}. Camera pulls back "
                    f"to reveal the full {v['object_type']} — the {v['epoxy_color']} resin fills 70 percent "
                    f"of all channels. Mirror-like reflections on the wet glossy resin. The charred black "
                    f"surface looks like volcanic rock surrounding rivers of glowing resin."
                ),
                "sfx_prompt": (
                    "thick viscous liquid pouring and pooling, satisfying ASMR gloopy sounds, wet spreading"
                ),
            },
            {
                "stage": 4,
                "name": "polish",
                "duration_seconds": 5.0,
                "camera": (
                    "Low-medium angle along table length, then medium close-up from above. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Low-medium angle along the {v['object_type']} length, 9:16 vertical, shallow depth "
                    f"of field. A female worker holds a small silver butane torch, passing flame over the "
                    f"{v['epoxy_color']} resin surface — popping micro-bubbles. The resin glows deep "
                    f"{v['led_color']} from embedded LEDs. Then the epoxy has cured. The worker uses a "
                    f"random orbital sander on the surface, polishing to a mirror-smooth high-gloss finish. "
                    f"The {v['channel_pattern']} channels glow with deep internal luminescence through "
                    f"polished {v['epoxy_color']} resin. Charred sections are matte black. The contrast "
                    f"between glossy glowing channels and matte dark surface is stunning."
                ),
                "sfx_prompt": (
                    "butane torch click and hiss, bubbles popping, orbital sander whirring smoothly"
                ),
            },
            {
                "stage": 5,
                "name": "luxury_reveal",
                "duration_seconds": 5.0,
                "camera": (
                    "Medium wide shot 25 degrees from above, piece centered in room. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Medium wide shot, 9:16 vertical, 25 degrees from above. The finished "
                    f"{v['object_type']} with {v['table_legs']} in a dimly lit {v['reveal_room']}. "
                    f"The {v['channel_pattern']} channels glow vivid {v['led_color']} through polished "
                    f"{v['epoxy_color']} resin — radiating like {v['glow_effect']}. The {v['led_color']} "
                    f"light spills onto the surrounding floor. {v['reveal_sofa']} flanks the table. "
                    f"Floor-to-ceiling windows show {v['reveal_window_view']}. The room is deliberately "
                    f"dark so the glowing {v['object_type']} is the dominant light source. Glossy resin "
                    f"surface reflects everything. A hardcover book and glass object sit on the surface. "
                    f"Three-point color: warm overhead, {v['led_color']} glow, cool window light."
                ),
                "sfx_prompt": (
                    "deep atmospheric ambient hum, faint city sounds through glass, serene luxury"
                ),
            },
        ]
