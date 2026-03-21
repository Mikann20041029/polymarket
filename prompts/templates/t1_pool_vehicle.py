"""
Template 1: Pool Vehicle Build — 5×5s Kling 3.0 prompts

REFERENCE: Backyard Yacht Mansion + Submarine Base Pool
STRUCTURE: 5 clips × 5 seconds each = 25 seconds total

Clip 1 (0-5s):   Filled pool → draining (reverse-chronology hook)
Clip 2 (5-10s):  Empty pool → skeleton framework rising
Clip 3 (10-15s): Hull skinning + superstructure + welding sparks
Clip 4 (15-20s): Furnishing + pool refilling with water
Clip 5 (20-25s): Twilight/night glamour reveal with warm lighting
"""
from prompts.templates.base import BaseTemplate


class PoolVehicleTemplate(BaseTemplate):
    template_id = "pool_vehicle"
    template_name = "Pool Vehicle Build"
    reference_video = "01_backyard_yacht_mansion.mp4 + 05_submarine_base_pool.mp4"
    total_duration_seconds = 25.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
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
            "frame_material": [
                "warm honey-toned timber", "dark welded steel", "brushed aluminum",
                "reclaimed barn wood", "white oak", "bamboo", "cedar planks",
                "mahogany", "teak", "copper-riveted steel",
            ],
            "hull_exterior": [
                "crisp white", "matte black", "deep navy blue",
                "forest green", "burgundy red", "slate gray",
                "pearl white with gold pinstripe", "dark charcoal",
            ],
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
            ],
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
                "golden fairy lights draped over the rigging",
            ],
            "worker_vest_color": [
                "hi-vis yellow-green", "bright orange", "reflective silver",
            ],
            "reveal_sky": [
                "golden hour with warm pink-orange sunset strip",
                "blue hour twilight with deep navy sky",
                "purple dusk with magenta cloud streaks",
                "starlit clear night with deep black sky",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        cam = (
            f"Elevated drone shot, 9:16 vertical portrait, 35 degrees from vertical, "
            f"hovering above the near end of a rectangular in-ground pool, looking toward "
            f"a {v['neighbor_houses']} beyond a fence. FIXED position."
        )

        return [
            {
                "stage": 1,
                "name": "drain",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} A rectangular in-ground swimming pool filled with {v['pool_water_color']} "
                    f"water. {v['pool_surround']} edges. {v['setting']}. {v['season_trees']} beyond the fence. "
                    f"A yellow-orange gas pump at the near corner, black hoses plunged into the water. "
                    f"4 workers in {v['worker_vest_color']} vests manage hoses. Water turbulence and white "
                    f"foam where hoses operate. Water level drops rapidly in time-lapse, exposing the pool "
                    f"walls. By end of clip the pool is completely empty — bare gray concrete interior, "
                    f"shallow puddles, scattered debris on the floor. Overcast diffused daylight."
                ),
                "sfx_prompt": (
                    "diesel pump rumbling, water gurgling and draining, splashing, distant birdsong"
                ),
            },
            {
                "stage": 2,
                "name": "framing",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same angle. The pool is completely empty, clean gray concrete. "
                    f"Workers in {v['worker_vest_color']} vests carry long {v['frame_material']} beams "
                    f"into the pool, laying them in parallel rows. A green laser alignment line on the floor. "
                    f"In rapid time-lapse: curved structural ribs rise from a central spine beam — "
                    f"15-20 curved {v['frame_material']} ribs erected symmetrically, creating the unmistakable "
                    f"skeleton of a {v['vehicle']}. The hull shape becomes clear: pointed bow at far end, "
                    f"wider stern near camera. Sawdust accumulates on the pool floor. The iconic boat-skeleton "
                    f"silhouette is fully formed by end of clip. Overcast daylight."
                ),
                "sfx_prompt": (
                    "rapid hammering at timelapse speed, power saw buzzing, wood creaking, metal clangs"
                ),
            },
            {
                "stage": 3,
                "name": "skinning",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same angle. The {v['vehicle']} skeleton is being covered in time-lapse. "
                    f"{v['hull_exterior']} exterior panels attach to the rib framework. "
                    f"Bright orange welding sparks cascade from attachment points. "
                    f"The deck is planked with {v['frame_material']}. A cabin superstructure rises — "
                    f"{v['cabin_feature']}. Above the cabin, {v['upper_deck']}. "
                    f"An external staircase with metallic handrails at the stern. "
                    f"By end of clip the {v['vehicle']} is a complete vessel — {v['hull_exterior']} hull, "
                    f"warm deck, glass cabin, upper deck — sitting in the empty gray pool. "
                    f"Workers still visible around the structure. Overcast daylight."
                ),
                "sfx_prompt": (
                    "welding crackle and hiss, crane motor whirring, panels clanking, power drills"
                ),
            },
            {
                "stage": 4,
                "name": "furnish_and_fill",
                "duration_seconds": 5.0,
                "camera": cam,
                "video_prompt": (
                    f"{cam} Same angle. The {v['vehicle']} exterior is complete. Workers place luxury "
                    f"furniture on the stern deck: {v['deck_sofa']}, {v['centerpiece']}. "
                    f"{v['accent_lighting']} installed along railings and deck edges. "
                    f"Two large silver tanker trucks arrive on both sides of the pool. "
                    f"Black hoses pump {v['pool_water_color']} water into the pool. Water rises rapidly "
                    f"around the {v['hull_exterior']} hull. The waterline climbs until the {v['vehicle']} "
                    f"appears to float. Workers clear tools from the {v['pool_surround']} deck. "
                    f"By end of clip pool is full, trucks departing. Late afternoon warm light."
                ),
                "sfx_prompt": (
                    "water rushing from hoses, truck engines, pool water lapping against hull"
                ),
            },
            {
                "stage": 5,
                "name": "reveal",
                "duration_seconds": 5.0,
                "camera": (
                    "Camera descends from drone height to deck level, 9:16 vertical, "
                    "15-20 degrees from horizontal, at stern railing looking inward."
                ),
                "video_prompt": (
                    f"Low intimate camera at deck level, 9:16 vertical. The {v['vehicle']} "
                    f"in {v['pool_water_color']} pool at {v['reveal_sky']}. "
                    f"On the {v['frame_material']} deck: {v['centerpiece']} with dancing amber flames "
                    f"casting warm flickering light. {v['deck_sofa']} surrounds it. "
                    f"{v['accent_lighting']} trace every edge and railing. "
                    f"The {v['cabin_feature']} glow warmly from interior lights. "
                    f"Dark sky above, pool water dark and reflective, catching warm light ripples. "
                    f"House windows in background glow amber. No workers. "
                    f"The scene is warm, intimate, luxurious, cinematic."
                ),
                "sfx_prompt": (
                    "fire crackling softly, gentle water ripples, crickets, warm ambient hum"
                ),
            },
        ]
