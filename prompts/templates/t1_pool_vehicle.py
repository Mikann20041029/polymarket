"""
Template 1: Pool Vehicle Build (ref: Backyard Yacht Mansion + Submarine Base Pool)

STRUCTURE (LOCKED — never changes):
  Elevated drone shot, 9:16 vertical, fixed position throughout.
  Act 1: Existing pool shown, then drained empty
  Act 2: Workers arrive with materials, staging in empty pool
  Act 3: Skeletal framework erected (ribs/arches on keel/spine)
  Act 4: Exterior skin/planking applied, superstructure built
  Act 5: Details + furnishing, pool refilled with water
  Act 6: Twilight/night beauty shots with warm lighting

VARIABLES (what changes each run):
  - The VEHICLE being built (yacht, submarine, pirate ship, speedboat, etc.)
  - The MATERIAL (timber, steel, aluminum, fiberglass)
  - The pool SETTING (suburban backyard, desert compound, tropical villa, etc.)
  - The SEASON visible in background (autumn, summer, spring)
  - The FURNITURE on the finished vehicle
  - The ACCENT LIGHTING color
  - The reveal TIME OF DAY
"""
from prompts.templates.base import BaseTemplate


class PoolVehicleTemplate(BaseTemplate):
    template_id = "pool_vehicle"
    template_name = "Pool Vehicle Build"
    reference_video = "01_backyard_yacht_mansion.mp4 + 05_submarine_base_pool.mp4"
    total_duration_seconds = 25.0
    num_stages = 6

    def get_variable_pools(self) -> dict[str, list]:
        return {
            # ── What is being built (paired tuples: full name, short name) ──
            "_vehicle_pair": [
                ("luxury yacht", "yacht"),
                ("military submarine", "submarine"),
                ("pirate galleon", "galleon"),
                ("Viking longship", "longship"),
                ("speedboat", "speedboat"),
                ("houseboat", "houseboat"),
                ("gondola", "gondola"),
                ("fishing trawler", "trawler"),
                ("sailboat", "sailboat"),
                ("amphibious vehicle", "vehicle"),
                ("wooden ark", "ark"),
                ("catamaran", "catamaran"),
                ("steam paddle boat", "paddle boat"),
            ],
            # ── Construction material ──
            "frame_material": [
                "warm honey-toned timber", "dark welded steel", "brushed aluminum",
                "reclaimed barn wood", "white oak", "bamboo", "cedar planks",
                "mahogany", "teak", "copper-riveted steel",
            ],
            "frame_color": [
                "#D4B87A", "#3A3A3A", "#A0A0A0", "#8B6F4E", "#C8A85E",
                "#B8A060", "#A0522D", "#8B4513", "#D2B48C", "#B87333",
            ],
            "hull_exterior": [
                "crisp white", "matte black", "deep navy blue",
                "forest green", "burgundy red", "slate gray",
                "pearl white with gold pinstripe", "dark charcoal",
            ],
            # ── Setting / Location ──
            "setting": [
                "suburban backyard with cedar fence and autumn trees",
                "Mediterranean villa with terracotta walls and olive trees",
                "tropical compound with palm trees and coral stone wall",
                "desert oasis with adobe walls and Joshua trees",
                "English countryside with stone walls and ivy",
                "Japanese garden with bamboo fence and maple trees",
                "Scandinavian cabin with birch trees and grey sky",
                "coastal cliff property with ocean visible beyond fence",
            ],
            "season_trees": [
                "golden-orange autumn foliage", "lush green summer canopy",
                "pale pink cherry blossoms", "bare winter branches with frost",
                "deep green tropical fronds", "silver birch with yellow leaves",
            ],
            "neighbor_houses": [
                "beige two-story house with dark hip roof",
                "white stucco villa with terracotta roof",
                "modern glass-and-concrete cube house",
                "traditional stone cottage with slate roof",
                "wooden cabin with green metal roof",
                "pink Mediterranean house with blue shutters",
            ],
            # ── Pool details ──
            "pool_tile": [
                "blue square tile grid", "white mosaic tile",
                "dark slate gray tile", "turquoise pebble finish",
                "black granite tile", "cream travertine",
            ],
            "pool_water_color": [
                "turquoise-cyan", "deep teal", "crystal clear blue",
                "Caribbean aquamarine", "dark navy", "emerald green",
            ],
            "pool_surround": [
                "gray stone pavers", "cream travertine deck",
                "dark slate flagstone", "bleached teak decking",
                "red brick coping", "white concrete",
            ],
            # ── Superstructure details ──
            "cabin_feature": [
                "curved glass windows with dark tinting",
                "round porthole windows with brass frames",
                "open-air helm station with wooden wheel",
                "enclosed glass bridge with chrome fittings",
                "canvas awning over exposed wheelhouse",
                "sliding teak doors with frosted glass panels",
            ],
            "upper_deck": [
                "flybridge with parasol and string lights",
                "observation tower with telescope mount",
                "conning tower with periscope mast",
                "crow's nest with signal flags",
                "sun deck with retractable canopy",
                "radar dome and antenna array",
            ],
            # ── Furnishing ──
            "deck_sofa": [
                "cream L-shaped sectional with earth-tone pillows",
                "navy blue deep-seat sofa with white piping",
                "tan leather modular lounger with brass rivets",
                "white curved banquette with teal cushions",
                "charcoal linen daybed with mustard throws",
            ],
            "centerpiece": [
                "rectangular black fire pit with amber flames",
                "low teak coffee table with glass hurricane lanterns",
                "circular copper fire bowl with dancing flames",
                "marble-top serving island with brass fixtures",
                "cast-iron brazier with glowing embers",
            ],
            "accent_lighting": [
                "warm amber LED strips along deck edges",
                "cool white rope lights tracing the railings",
                "blue underwater LED strips along hull waterline",
                "copper lanterns with candle-like warm glow",
                "neon pink accent strips along cabin edges",
                "golden fairy lights draped over the rigging",
            ],
            "accent_color_hex": [
                "#F0C87A", "#E8E8F0", "#2040C0", "#B87333",
                "#FF69B4", "#F5D89A",
            ],
            # ── Worker details ──
            "worker_vest_color": [
                "hi-vis yellow-green", "bright orange", "reflective silver",
                "neon pink", "royal blue",
            ],
            "worker_count": ["3", "4", "5", "6"],
            # ── Reveal timing ──
            "reveal_sky": [
                "golden hour with warm pink-orange sunset strip",
                "blue hour twilight with deep navy sky",
                "purple dusk with magenta cloud streaks",
                "starlit clear night with deep black sky",
                "dramatic storm clearing with golden light breaking through",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        camera = (
            f"Elevated drone shot, 9:16 vertical portrait, approximately 35 degrees "
            f"from vertical, hovering above the near end of a rectangular in-ground pool, "
            f"looking down toward a {v['neighbor_houses']} beyond a {v['setting'].split(' with ')[0]} fence. "
            f"Camera position is FIXED and IDENTICAL across all stages."
        )

        return [
            # ── STAGE 1: Pool draining (0-4s) ──
            {
                "stage": 1,
                "name": "pool_drain",
                "duration_seconds": 4.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} A rectangular in-ground swimming pool filled with {v['pool_water_color']} water, "
                    f"{v['pool_tile']} visible on the floor. {v['pool_surround']} surrounds the edges. "
                    f"Landscaped borders with low hedging, {v['season_trees']} visible beyond the fence. "
                    f"A yellow-orange gas pump sits at the near corner with black hoses plunged into the water. "
                    f"{v['worker_count']} workers in {v['worker_vest_color']} vests manage hoses at the far end. "
                    f"Water turbulence and white foam where hoses operate. Water level drops rapidly, "
                    f"exposing {v['pool_tile']} walls. By end of clip the pool is nearly empty, "
                    f"shallow puddles on the floor, scattered debris and autumn leaves on gray concrete."
                ),
                "sfx_prompt": (
                    "loud diesel pump engine rumbling, water gurgling and draining through hoses, "
                    "splashing turbulence, distant birdsong, workers shouting faintly"
                ),
            },
            # ── STAGE 2: Material staging + framework begins (4-9s) ──
            {
                "stage": 2,
                "name": "framing",
                "duration_seconds": 5.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} The pool is completely empty, clean gray concrete interior exposed. "
                    f"{v['worker_count']} workers in {v['worker_vest_color']} vests and hard hats carry "
                    f"long {v['frame_material']} planks and metal structural beams into the pool. "
                    f"Materials are laid in parallel rows on the pool floor. A green laser alignment line "
                    f"is visible on the concrete. Workers consult blueprints spread on the floor. "
                    f"Then curved structural ribs begin rising from a central spine beam — "
                    f"15-20 curved {v['frame_material']} ribs erected symmetrically, creating the unmistakable "
                    f"skeleton of a {v['vehicle']}. Sawdust and wood shavings accumulate. "
                    f"The hull shape — pointed bow at the far end, wider stern at the near end — "
                    f"is now clearly recognizable from above."
                ),
                "sfx_prompt": (
                    "rhythmic hammering and drilling at timelapse speed, power saw buzzing, "
                    "wood creaking as ribs are raised, metallic clangs, workers calling out, sawdust settling"
                ),
            },
            # ── STAGE 3: Skin + superstructure (9-15s) ──
            {
                "stage": 3,
                "name": "skinning_and_structure",
                "duration_seconds": 6.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} Same angle. The {v['vehicle']} skeleton is now being covered. "
                    f"{v['hull_exterior']} exterior panels are attached to the rib framework, "
                    f"transforming the skeleton into a solid hull. Workers in {v['worker_vest_color']} vests "
                    f"weld and fasten panels — bright orange sparks cascade from weld points. "
                    f"A mobile crane boom extends over the pool, lifting heavy components. "
                    f"The deck is planked with {v['frame_material']}. A cabin superstructure rises — "
                    f"{v['cabin_feature']}. Above the cabin, {v['upper_deck']}. "
                    f"An external staircase with metallic handrails takes shape at the stern. "
                    f"The {v['vehicle']} now looks like a real vessel sitting in the pool."
                ),
                "sfx_prompt": (
                    "welding crackle and hiss, crane motor whirring, heavy panels clanking into place, "
                    "power drill sequences, metallic ringing, wind gusting over structure"
                ),
            },
            # ── STAGE 4: Furnishing + pool fill (15-20s) ──
            {
                "stage": 4,
                "name": "furnish_and_fill",
                "duration_seconds": 5.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} Same angle. The {v['vehicle']} exterior is complete — {v['hull_exterior']} hull, "
                    f"{v['cabin_feature']}, {v['upper_deck']}. Workers place luxury furniture on the stern deck: "
                    f"{v['deck_sofa']}, {v['centerpiece']}, decorative lanterns. "
                    f"{v['accent_lighting']} are installed along railings and deck edges. "
                    f"Then two large silver tanker trucks arrive on both sides of the pool. "
                    f"Black hoses pump {v['pool_water_color']} water into the pool. "
                    f"Water rises rapidly around the {v['hull_exterior']} hull. "
                    f"The waterline climbs until the {v['vehicle']} appears to float. "
                    f"Workers clear tools and debris from the {v['pool_surround']} deck. "
                    f"Trucks begin to depart."
                ),
                "sfx_prompt": (
                    "water rushing and filling from multiple hoses, truck engines idling, "
                    "pool water lapping against hull, furniture being placed with soft thuds, "
                    "workers finishing up, tools being packed"
                ),
            },
            # ── STAGE 5: Golden hour reveal (20-22s) ──
            {
                "stage": 5,
                "name": "reveal_golden",
                "duration_seconds": 3.0,
                "camera": camera,
                "video_prompt": (
                    f"{camera} Same angle. Pool is full of {v['pool_water_color']} water. "
                    f"The {v['vehicle']} sits majestically in the pool — {v['hull_exterior']} hull, "
                    f"warm {v['frame_material']} deck, {v['cabin_feature']}, {v['upper_deck']}. "
                    f"On the stern deck: {v['deck_sofa']}, {v['centerpiece']} with visible flames. "
                    f"{v['accent_lighting']} glow along every edge. "
                    f"The sky shifts to {v['reveal_sky']}. "
                    f"House windows in the background glow with warm amber interior light. "
                    f"{v['season_trees']} are silhouetted against the sky. "
                    f"The {v['pool_water_color']} water reflects the warm lighting, creating shimmering ripples. "
                    f"No workers visible. The scene is pristine, cinematic, aspirational."
                ),
                "sfx_prompt": (
                    "serene evening ambience, gentle water lapping against hull, "
                    "fire crackling softly from fire pit, distant crickets, warm atmospheric hum"
                ),
            },
            # ── STAGE 6: Night intimate close-up (22-25s) ──
            {
                "stage": 6,
                "name": "night_closeup",
                "duration_seconds": 3.0,
                "camera": (
                    "Camera descends from drone height to deck level, 9:16 vertical, "
                    "approximately 15-20 degrees from horizontal, positioned at stern railing "
                    "looking inward across the furnished deck toward the cabin."
                ),
                "video_prompt": (
                    f"Low intimate camera angle at deck level, 9:16 vertical. "
                    f"Looking across the {v['frame_material']} deck of the {v['vehicle']}. "
                    f"In the center: {v['centerpiece']} with dancing amber flames casting warm flickering light. "
                    f"Surrounding it: {v['deck_sofa']}. "
                    f"Decorative brass lanterns with candle-like glow sit on the deck. "
                    f"{v['accent_lighting']} trace every edge and railing. "
                    f"The {v['cabin_feature']} glow warmly from interior lights behind. "
                    f"Dark night sky with deep navy tones above. "
                    f"Pool water visible at frame edges, dark and reflective, catching warm light ripples. "
                    f"The scene is warm, intimate, luxurious."
                ),
                "sfx_prompt": (
                    "intimate fire crackling, very gentle water ripples, "
                    "soft warm ambient hum, distant night sounds, wind chime faintly"
                ),
            },
        ]
