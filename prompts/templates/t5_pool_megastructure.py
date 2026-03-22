"""
Template 5: Pool Megastructure Build — 5×5s Kling 3.0 prompts

REFERENCE: Submarine Base Pool
STRUCTURE: 5 clips × 5 seconds each = 25 seconds total

Clip 1 (0-5s):   Pool draining — water level drops
Clip 2 (5-10s):  Materials + skeletal framework erected in empty pool
Clip 3 (10-15s): Hull skinning + welding sparks close-up
Clip 4 (15-20s): Pool refilling around completed structure
Clip 5 (20-25s): Twilight/blue-hour beauty reveal
"""
from prompts.templates.base import BaseTemplate


class PoolMegastructureTemplate(BaseTemplate):
    template_id = "pool_megastructure"
    template_name = "Pool Megastructure Build"
    reference_video = "05_submarine_base_pool.mp4"
    total_duration_seconds = 25.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
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
                ("Viking longship", "longship"),
                ("giant octopus sculpture", "octopus"),
                ("Apollo space capsule", "capsule"),
                ("pirate ship", "pirate ship"),
                ("steam locomotive", "locomotive"),
                ("Ferris wheel section", "Ferris wheel"),
                ("hot air balloon basket", "balloon"),
                ("giant grand piano", "piano"),
                ("Egyptian obelisk", "obelisk"),
                ("windmill tower", "windmill"),
                ("giant human skull sculpture", "skull"),
                ("Trojan horse", "Trojan horse"),
                ("pagoda tower", "pagoda"),
                ("giant seashell sculpture", "seashell"),
                ("radio telescope dish", "telescope"),
                ("giant crown sculpture", "crown"),
                ("zeppelin gondola", "zeppelin"),
            ],
            "frame_type": [
                "curved dark welded steel arches",
                "aluminum truss framework sections",
                "bent wood laminate ribs",
                "carbon fiber composite frames",
                "copper-riveted iron ribs",
                "galvanized steel tube framing",
                "welded rebar cage framework",
                "laminated bamboo arch ribs",
                "stainless steel I-beam skeleton",
                "reclaimed railroad rail arches",
                "bronze-welded pipe framework",
                "titanium alloy truss structure",
                "cast-iron lattice framework",
            ],
            "skin_type": [
                "dark matte black steel plate panels",
                "brushed silver aluminum sheet panels",
                "riveted copper-patina panels",
                "white fiberglass composite panels",
                "camouflage-painted steel panels",
                "polished mirror-chrome panels",
                "weathered Corten steel panels",
                "hammered bronze sheet panels",
                "painted matte navy blue panels",
                "galvanized corrugated metal panels",
                "titanium-gray anodized panels",
                "powder-coated racing red panels",
            ],
            "skin_color": [
                "dark matte charcoal-black", "brushed silver",
                "verde-patina copper", "glossy white",
                "olive-drab military green", "mirror chrome",
                "rust-orange Corten", "hammered bronze-gold",
                "deep navy blue", "corrugated zinc gray",
                "titanium storm gray", "racing cherry red",
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
                "dragon head prow with carved teeth",
                "curling tentacle arms in welded steel",
                "docking hatch and heat shield tiles",
                "skull and crossbones flag on mast",
                "smokestack and cow-catcher front",
                "gondola cars and axle hub",
                "wicker basket and burner ring",
                "raised lid and exposed strings",
                "hieroglyphic-carved tapered shaft",
                "rotating blade assembly and nacelle",
                "hollow eye sockets with internal glow",
                "trap door belly and wooden legs",
                "tiered curved eave roofline",
                "spiraling nautilus chamber opening",
                "parabolic dish with receiver arm",
                "jewel-encrusted band with cross finial",
                "gondola cabin with mooring ropes",
            ],
            "pool_water": [
                "turquoise-cyan", "deep teal", "crystal blue",
                "dark navy", "emerald green",
                "sapphire blue", "milky jade",
                "warm aquamarine", "glacial ice blue",
                "pacific cerulean", "lagoon turquoise",
                "slate blue-gray",
            ],
            "pool_surround": [
                "beige natural stone coping with dark mulch beds",
                "gray paver deck with boxwood hedging",
                "red brick coping with gravel border",
                "white concrete deck with potted palms",
                "travertine tile deck with lavender planters",
                "bluestone flagstone with ornamental grasses",
                "dark slate coping with river rock border",
                "sandstone paver deck with terra cotta pots",
                "ipê hardwood decking with stainless steel rail",
                "limestone coping with boxwood topiaries",
            ],
            "background_house": [
                "beige two-story house with dark hip roof",
                "white modern cube house with flat roof",
                "red brick colonial with black shutters",
                "gray farmhouse with metal standing-seam roof",
                "tan stucco Mediterranean with clay tile roof",
                "dark wood A-frame cabin with steep gable",
                "cream Victorian with wraparound porch",
                "sage green craftsman with stone columns",
                "charcoal contemporary with floor-to-ceiling glass",
                "sandstone ranch with covered carport",
            ],
            "season_detail": [
                "golden-yellow autumn foliage on deciduous trees",
                "lush green summer canopy",
                "bare winter branches against gray sky",
                "pink cherry blossom branches",
                "bronze copper beech leaves",
                "deep green tropical fronds",
                "snow-laden evergreen boughs",
                "bright red maple foliage",
                "pale lavender wisteria draping",
                "russet oak leaves in late autumn",
            ],
            "fence_type": [
                "natural cedar horizontal slat fence",
                "white painted picket fence",
                "dark stained privacy fence",
                "stone wall with iron gate",
                "bamboo screen fence with twine lashing",
                "split-rail rustic fence",
                "black aluminum pool fence with glass panels",
                "rendered masonry wall with tile cap",
                "living hedge wall of privet",
                "gabion basket wall with river stones",
            ],
            "vest_color": [
                "hi-vis yellow-green", "bright orange",
                "reflective silver-stripe", "royal blue",
                "deep red", "white", "neon pink",
                "forest green", "charcoal gray", "tan khaki",
            ],
            "reveal_sky": [
                "blue hour twilight with deep navy sky and warm amber house window lights",
                "dramatic sunset with orange and purple streaks",
                "clear starlit night with deep black sky",
                "moody overcast dusk with silver-gray clouds",
                "fiery crimson sunset with silhouetted trees",
                "pastel pink and lavender twilight gradient",
                "tropical orange dusk with wispy cirrus clouds",
                "moonrise with indigo sky and silver highlights",
                "stormy purple sky with lightning in the distance",
                "soft peach dawn breaking behind rooftops",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        drone = (
            f"Elevated drone shot 30-35 feet high, 45 degrees from vertical, "
            f"9:16 vertical portrait, hovering above the near end of the pool "
            f"looking toward a {v['background_house']} beyond a {v['fence_type']}. FIXED."
        )

        return [
            {
                "stage": 1,
                "name": "drain",
                "duration_seconds": 5.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} A rectangular in-ground pool filled with {v['pool_water']} water. "
                    f"{v['pool_surround']}. {v['season_detail']} beyond the {v['fence_type']}. "
                    f"Overcast flat daylight. A yellow-orange gas pump at the near corner with black "
                    f"hoses in the water. Water turbulence and white foam where hoses operate. "
                    f"Water level drops steadily in time-lapse, exposing pool walls. "
                    f"By end of clip: pool completely empty — thin reflective puddle on the floor, "
                    f"bare concrete interior exposed, scattered debris visible."
                ),
                "sfx_prompt": (
                    "diesel pump rumbling, water gurgling through hoses, splashing, outdoor ambience"
                ),
            },
            {
                "stage": 2,
                "name": "skeleton",
                "duration_seconds": 5.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} Pool completely empty. 4 workers in {v['vest_color']} vests and hard hats "
                    f"carry structural beams into the pool. Materials staged in organized rows. "
                    f"Green laser alignment lines on pool floor. In rapid time-lapse: {v['frame_type']} "
                    f"rise from the pool floor. A central spine beam runs the full length. "
                    f"Transverse arches span the pool width, evenly spaced. 5-7 curved ribs create "
                    f"the unmistakable skeletal shape of a {v['megastructure']}. Bright orange welding "
                    f"sparks at rib attachment points. The full skeletal framework is clearly "
                    f"recognizable from above by end of clip."
                ),
                "sfx_prompt": (
                    "metal clanging, welding crackle, power drill sequences, workers shouting"
                ),
            },
            {
                "stage": 3,
                "name": "skinning",
                "duration_seconds": 5.0,
                "camera": (
                    "Starts as drone overhead, then CUT to ground-level extreme close-up "
                    "2-3 feet from welding action. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Drone overhead then cut to extreme close-up, 9:16 vertical. "
                    f"First: overhead view as {v['skin_type']} are attached to the rib framework. "
                    f"The skeleton transforms into a solid {v['skin_color']} hull. Workers fasten panels. "
                    f"{v['distinctive_feature']} takes shape. "
                    f"Then CUT TO: extreme close-up 2 feet away — a welder in dark auto-darkening "
                    f"helmet with blue visor arc-welds a {v['skin_color']} panel seam. Brilliant "
                    f"white-blue welding arc at center. Orange and white sparks spray outward. "
                    f"Raw steel surface with visible weld bead lines. Background is blurred bokeh. "
                    f"This shot communicates craftsmanship and industrial power."
                ),
                "sfx_prompt": (
                    "welding arc crackling, sparks sizzling on metal, heavy breathing behind mask"
                ),
            },
            {
                "stage": 4,
                "name": "refill",
                "duration_seconds": 5.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} The {v['megastructure']} hull is fully complete — sleek {v['skin_color']} "
                    f"form with {v['distinctive_feature']} protruding prominently. "
                    f"{v['pool_water']} water is being pumped in from tanker trucks on both sides. "
                    f"Water rises rapidly around the hull. The waterline climbs. "
                    f"Workers clear tools and equipment from the deck in time-lapse motion blur. "
                    f"By end: pool is full, the {v['megastructure']} rises 2-3 feet above waterline. "
                    f"The {v['pool_water']} water contrasts with the {v['skin_color']} hull. "
                    f"All tools removed. Clean scene. Late afternoon warm light."
                ),
                "sfx_prompt": (
                    "water rushing from hoses, pool filling, hull lapping sounds, tools being packed"
                ),
            },
            {
                "stage": 5,
                "name": "twilight_reveal",
                "duration_seconds": 5.0,
                "camera": drone,
                "video_prompt": (
                    f"{drone} Same exact angle. Time has jumped to twilight. "
                    f"{v['reveal_sky']}. The {v['megastructure']} sits in {v['pool_water']} water — "
                    f"its {v['skin_color']} hull catching subtle sky reflections along the curved top. "
                    f"{v['distinctive_feature']} silhouetted against the sky. Pool water shifted to "
                    f"deeper teal reflecting the twilight. House windows glow warm amber. "
                    f"{v['season_detail']} are darker silhouettes. The sky deepens through blue hour — "
                    f"hull becomes a dark silhouette with faint highlights along its spine, water gains "
                    f"a subtle luminous teal quality. No workers, no tools. A {v['megastructure']} "
                    f"surfaces in a suburban backyard pool at dusk — surreal, cinematic, magnificent."
                ),
                "sfx_prompt": (
                    "serene evening ambience, gentle water lapping, crickets, atmospheric hum"
                ),
            },
        ]
