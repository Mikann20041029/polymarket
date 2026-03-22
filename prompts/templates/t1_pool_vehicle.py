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
                ("Chinese junk", "junk"),
                ("tugboat", "tugboat"),
                ("pontoon party boat", "pontoon"),
                ("kayak", "kayak"),
                ("canoe", "canoe"),
                ("jet ski", "jet ski"),
                ("swan pedal boat", "pedal boat"),
                ("lobster boat", "lobster boat"),
                ("whaling ship", "whaling ship"),
                ("aircraft carrier", "carrier"),
                ("riverboat casino", "riverboat"),
                ("dragon boat", "dragon boat"),
                ("outrigger canoe", "outrigger"),
                ("clipper ship", "clipper"),
                ("narrow canal boat", "canal boat"),
                ("police patrol boat", "patrol boat"),
                ("fire rescue boat", "rescue boat"),
                ("glass-bottom tour boat", "tour boat"),
                ("cigarette racing boat", "cigarette boat"),
                ("dhow sailing vessel", "dhow"),
                ("war canoe", "war canoe"),
                ("Boston whaler", "whaler"),
                ("center console fishing boat", "center console"),
                ("trimaran", "trimaran"),
                ("Mississippi steamboat", "steamboat"),
                ("bass boat", "bass boat"),
                ("water taxi", "water taxi"),
                ("research vessel", "research vessel"),
                ("icebreaker ship", "icebreaker"),
                ("lifeboat", "lifeboat"),
                ("sampan", "sampan"),
                ("coracle", "coracle"),
                ("felucca", "felucca"),
                ("tall ship", "tall ship"),
                ("dinghy", "dinghy"),
                ("hydrofoil", "hydrofoil"),
                ("skiff", "skiff"),
                ("towboat", "towboat"),
            ],
            "frame_material": [
                "warm honey-toned timber", "dark welded steel", "brushed aluminum",
                "reclaimed barn wood", "white oak", "bamboo", "cedar planks",
                "mahogany", "teak", "copper-riveted steel",
                "ash wood", "galvanized iron", "stainless steel tube",
                "Douglas fir", "ipe hardwood", "walnut", "cherry wood",
                "powder-coated steel", "blackened wrought iron", "marine-grade plywood",
                "white oak plank", "spalted maple", "redwood timber",
                "titanium tube", "carbon fiber composite", "fiberglass panel",
                "cast bronze", "weathered cypress", "quarter-sawn sycamore",
                "birch plywood", "laminated pine beam", "ebony strip",
                "rosewood plank", "zinc-coated steel", "anodized aluminum tube",
                "larch timber", "hemlock beam",
            ],
            "hull_exterior": [
                "crisp white", "matte black", "deep navy blue",
                "forest green", "burgundy red", "slate gray",
                "pearl white with gold pinstripe", "dark charcoal",
                "bright cherry red", "sand beige", "copper metallic",
                "sky blue", "burnt sienna", "gunmetal gray",
                "midnight purple", "racing green with white stripe",
                "weathered teak brown", "arctic white with navy trim",
                "electric blue metallic", "olive drab military green",
                "canary yellow", "dusty rose pink", "raw steel unpainted",
                "mahogany varnish brown", "tangerine orange with white deck",
                "two-tone black over red", "champagne gold metallic",
                "deep plum purple", "seafoam green", "coral reef orange",
                "antique brass patina", "winter white with teal waterline",
                "graphite gray with red pinstripe", "pale robin egg blue",
                "candy apple red metallic", "faded turquoise vintage",
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
                "Tuscan farmhouse with cypress trees and terracotta pots",
                "Pacific Northwest lodge with Douglas fir and moss-covered rocks",
                "Balinese villa with thatched pavilion and frangipani trees",
                "Cape Cod cottage with dune grass and weathered shingle fence",
                "Texas Hill Country ranch with live oak and limestone wall",
                "Swiss chalet with pine forest and snow-dusted peaks behind",
                "Moroccan riad with zellige tile walls and date palms",
                "Australian homestead with eucalyptus trees and corrugated iron fence",
                "Greek island villa with whitewashed walls and bougainvillea",
                "Southern plantation house with magnolia trees and wrought-iron fence",
                "Hawaiian estate with plumeria trees and lava rock wall",
                "Colorado mountain lodge with aspen trees and log fence",
                "French Provençal farmhouse with lavender rows and stone wall",
                "New England saltbox with maple trees and split-rail fence",
                "Caribbean beach house with coconut palms and coral block wall",
                "Arizona ranch with saguaro cactus and adobe perimeter",
                "Georgian townhouse with box hedge and iron railings",
                "Vietnamese courtyard with banana trees and bamboo screen",
                "Argentine estancia with jacaranda trees and stucco wall",
                "Malibu cliffside home with agave plants and glass railing",
                "Portland craftsman with ferns and mossy cedar fence",
                "Dubai modern villa with date palms and white rendered wall",
                "Vermont farmstead with sugar maple trees and stone wall",
                "Kyoto machiya with pine tree and clay tile wall",
            ],
            "season_trees": [
                "golden-orange autumn foliage", "lush green summer canopy",
                "pale pink cherry blossoms", "bare winter branches with frost",
                "deep green tropical fronds", "silver birch with yellow leaves",
                "bright red maple leaves", "dusty olive Mediterranean canopy",
                "snow-laden evergreen boughs", "bronze copper beech leaves",
                "white magnolia blossoms", "vivid spring green new growth",
                "rust-red sweetgum leaves", "lavender jacaranda canopy",
                "dark emerald live oak foliage", "amber ginkgo fan leaves",
                "scarlet dogwood berries on bare limbs", "soft peach apricot blossoms",
                "ice-crusted bare black walnut branches", "lime-green weeping willow tendrils",
                "deep maroon Japanese plum leaves", "fiery orange sassafras foliage",
                "silver-green olive tree canopy", "purple wisteria cascades",
            ],
            "neighbor_houses": [
                "beige two-story house with dark hip roof",
                "white stucco villa with terracotta roof",
                "modern glass-and-concrete cube house",
                "traditional stone cottage with slate roof",
                "red brick colonial with black shutters",
                "gray farmhouse with metal standing-seam roof",
                "cedar-shingled Cape Cod with dormer windows",
                "pale yellow bungalow with wraparound porch",
                "dark brown Tudor with exposed timber framing",
                "tan split-level ranch with attached garage",
                "white Mediterranean villa with blue trim",
                "forest green craftsman with stone porch columns",
                "peach stucco ranch with clay tile roof",
                "slate blue Victorian with gingerbread trim",
                "charcoal modern farmhouse with board-and-batten",
                "sandstone townhouse with copper gutters",
                "brick Georgian with white columned portico",
                "sage green cottage with thatched roof",
                "cream prairie-style with wide overhanging eaves",
                "rust-red barn conversion with gambrel roof",
                "ivory Spanish revival with arched windows",
            ],
            "pool_water_color": [
                "turquoise-cyan", "deep teal", "crystal clear blue",
                "Caribbean aquamarine", "dark navy", "emerald green",
                "sapphire blue", "pale mint green", "lagoon turquoise",
                "steel blue-gray", "tropical cerulean", "deep indigo",
                "milky jade green", "warm aquamarine", "glacial ice blue",
                "cobalt blue", "seafoam green-blue", "bright chlorine blue",
                "pacific ocean blue", "translucent mint", "dusty periwinkle",
                "vivid azure", "dark forest pool green",
            ],
            "pool_surround": [
                "gray stone pavers", "cream travertine deck",
                "dark slate flagstone", "bleached teak decking",
                "red brick coping", "white concrete",
                "sandstone paver deck", "charcoal porcelain tile",
                "limestone coping", "weathered bluestone",
                "terracotta tile edge", "brushed concrete with pebble inlay",
                "quartzite flagstone coping", "coral stone deck",
                "black granite edge", "reclaimed brick paver",
                "honey onyx tile", "tumbled marble coping",
                "stained cedar decking", "buff sandstone paver",
                "salt-finished concrete", "basalt tile edge",
                "ipê hardwood decking", "cream limestone slab",
            ],
            "cabin_feature": [
                "curved glass windows with dark tinting",
                "round porthole windows with brass frames",
                "open-air helm station with wooden wheel",
                "enclosed glass bridge with chrome fittings",
                "canvas awning over exposed wheelhouse",
                "sliding teak doors with frosted glass panels",
                "wraparound windshield with stainless steel mullions",
                "louvered shutters with copper hinges",
                "retractable hardtop with smoked glass panels",
                "gothic arched windows with leaded glass",
                "panoramic bubble canopy with aluminum frame",
                "Dutch barn doors with iron strap hardware",
                "diamond-pane leaded windows with oak frame",
                "bi-fold glass doors with teak surround",
                "porthole clusters with polished nickel bezels",
                "stained glass transom with nautical motif",
                "clerestory windows with cedar trim",
                "roll-up canvas panels with brass grommets",
                "jalousie windows with anodized aluminum slats",
                "French doors with wrought-iron balconette",
                "smoked acrylic panels with rubber gasket seals",
                "sliding barn-style door with marine-grade track",
                "butterfly hatch with hydraulic struts",
                "octagonal window with mahogany mullions",
                "venetian blind shutters with bronze hinges",
            ],
            "upper_deck": [
                "flybridge with parasol and string lights",
                "observation tower with telescope mount",
                "conning tower with periscope mast",
                "crow's nest with signal flags",
                "sun deck with retractable canopy",
                "radar mast with spinning antenna dish",
                "rooftop hot tub with glass wind screen",
                "helicopter landing pad with painted markings",
                "captain's balcony with wrought-iron railing",
                "signal bridge with flag halyards and searchlight",
                "diving platform with retractable ladder",
                "weather station with anemometer and wind vane",
                "glass-floor observation deck with chrome railing",
                "sundial compass rose with brass fittings",
                "tiki bar with bamboo roof and string lights",
                "DJ booth with waterproof speakers and LED panels",
                "greenhouse dome with tropical plants inside",
                "miniature putting green with synthetic turf",
                "meditation platform with incense holder and gong",
                "astronomy deck with mounted binoculars",
                "rooftop cinema screen with projector mount",
            ],
            "deck_sofa": [
                "cream L-shaped sectional with earth-tone pillows",
                "navy blue deep-seat sofa with white piping",
                "tan leather modular lounger with brass rivets",
                "white curved banquette with teal cushions",
                "charcoal linen daybed with mustard throws",
                "olive green canvas bench with rope-trim pillows",
                "wicker loveseat with coral cushions",
                "teak built-in bench with navy stripe upholstery",
                "burgundy velvet chaise longue with gold fringe",
                "gray rattan modular set with sage green cushions",
                "black leather director's chairs with chrome frames",
                "white canvas hammock bench with rope suspension",
                "driftwood frame loveseat with ocean-blue cushions",
                "mahogany steamer chairs with striped canvas seats",
                "cast aluminum settee with slate gray Sunbrella fabric",
                "cedar Adirondack chairs with cream sheepskin throws",
                "teak porch swing with indigo ikat cushions",
                "rope-weave lounge chair with sand-colored pad",
                "bamboo daybed with linen canopy and cream bolsters",
                "powder-coated steel bench with burnt orange cushion",
                "reclaimed pallet sofa with denim-blue upholstery",
            ],
            "centerpiece": [
                "rectangular black fire pit with amber flames",
                "low teak coffee table with glass hurricane lanterns",
                "circular copper fire bowl with dancing flames",
                "marble-top serving island with brass fixtures",
                "cast-iron brazier with glowing embers",
                "driftwood sculpture with embedded LED candles",
                "stone slab table with succulent planter center",
                "brass telescope on a mahogany tripod stand",
                "ship wheel coffee table with glass top",
                "slate water feature with cascading stream",
                "antique diving helmet repurposed as lamp",
                "hand-blown glass orb terrarium with succulents",
                "nautical chart under glass on teak frame table",
                "bronze anchor sculpture on marble pedestal",
                "ceramic sake set on lacquered tray",
                "vintage compass rose inlaid into wood tabletop",
                "wrought-iron candelabra with pillar candles",
                "hammered copper bowl with floating gardenias",
                "petrified wood slab table with resin fill",
                "ship model in glass display case on low stand",
                "mosaic tile table with Mediterranean pattern",
            ],
            "accent_lighting": [
                "warm amber LED strips along deck edges",
                "cool white rope lights tracing the railings",
                "blue underwater LED strips along hull waterline",
                "copper lanterns with candle-like warm glow",
                "golden fairy lights draped over the rigging",
                "recessed deck floor lights with warm white glow",
                "paper lanterns in pastel colors hung from mast",
                "fiber optic star-field embedded in canopy ceiling",
                "tiki torch flames along the stern railing",
                "neon tube accent in soft pink along the gunwale",
                "vintage brass oil lanterns on wall brackets",
                "solar-powered mason jar lights on rope netting",
                "color-changing RGB LED strips under bench seats",
                "hurricane candle lamps on each stair tread",
            ],
            "worker_vest_color": [
                "hi-vis yellow-green", "bright orange", "reflective silver",
                "royal blue", "deep red", "white", "neon pink",
                "forest green", "charcoal gray", "tan khaki",
            ],
            "reveal_sky": [
                "golden hour with warm pink-orange sunset strip",
                "blue hour twilight with deep navy sky",
                "purple dusk with magenta cloud streaks",
                "starlit clear night with deep black sky",
                "fiery sunset with crimson and gold bands",
                "overcast dusk with moody steel-gray clouds",
                "tropical twilight with peach and lavender gradient",
                "moonlit night with silver cloud edges",
                "aurora-streaked polar twilight with green curtains",
                "hazy warm evening with amber horizon glow",
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
