"""
Template 3: Fiber Optic Epoxy Floor — 5×5s Kling 3.0 prompts

REFERENCE: Flower Garden (fiber optic floor)
STRUCTURE: 5 clips × 5 seconds each = 25 seconds total

Clip 1 (0-5s):   Workers unpack, install fiber optic strands into floor
Clip 2 (5-10s):  Colored epoxy pour over fiber field — ASMR close-up
Clip 3 (10-15s): Torch de-gassing + floor buffer polishing
Clip 4 (15-20s): Furniture staging in time-lapse
Clip 5 (20-25s): Dusk/night reveal with fiber floor cycling brightness
"""
from prompts.templates.base import BaseTemplate


class FiberOpticFloorTemplate(BaseTemplate):
    template_id = "fiber_optic_floor"
    template_name = "Fiber Optic Epoxy Floor"
    reference_video = "03_flower_garden.mp4"
    total_duration_seconds = 25.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
            "room_type": [
                "luxury high-rise penthouse living room",
                "converted warehouse loft",
                "contemporary art gallery main hall",
                "upscale restaurant dining room",
                "hotel presidential suite",
                "rooftop glass-enclosed lounge",
                "high-end spa meditation room",
                "private nightclub VIP section",
                "modern museum exhibition hall",
                "luxury yacht owner's stateroom",
                "boutique cinema screening room",
                "underground cocktail bar",
                "corporate penthouse boardroom",
                "bridal suite dressing room",
                "recording studio control room",
            ],
            "window_type": [
                "massive floor-to-ceiling corner windows",
                "panoramic curved glass wall",
                "industrial steel-frame windows",
                "arched floor-to-ceiling windows",
                "narrow vertical slit windows with deep reveals",
                "hexagonal geometric glass panels",
                "skylights with motorized shades",
                "porthole-style circular windows",
                "bi-fold glass doors opening to terrace",
                "clerestory ribbon windows near ceiling",
            ],
            "window_view": [
                "city skyline at high elevation",
                "ocean horizon and coastal cliffs",
                "mountain range with snow caps",
                "dense urban night lights",
                "tropical beach with palm silhouettes",
                "rolling vineyard hills at golden hour",
                "desert canyon with red rock formations",
                "lakeside forest with morning mist",
                "harbor with illuminated bridge at night",
                "snowy rooftop landscape with chimney smoke",
            ],
            "wall_feature": [
                "built-in linear gas fireplace with cream surround",
                "floor-to-ceiling bookshelf wall",
                "raw exposed brick wall with steel beams",
                "white marble feature wall with indirect lighting",
                "green living plant wall",
                "black slate accent wall with gold inlay",
                "reclaimed wood plank feature wall",
                "concrete wall with geometric relief pattern",
                "mirrored wall with antiqued bronze tint",
                "cork tile wall with warm amber tone",
                "terrazzo accent wall with brass trim",
                "backlit onyx stone panel wall",
            ],
            "base_floor": [
                "bare gray concrete", "raw polished concrete",
                "dark slate gray concrete", "light cream concrete",
                "stained charcoal concrete", "warm terracotta-tinted concrete",
                "sandy buff concrete", "blue-gray troweled concrete",
                "white self-leveling compound", "acid-stained mottled concrete",
            ],
            "epoxy_color": [
                "deep teal-blue", "rich emerald green",
                "midnight purple", "warm honey-amber",
                "ocean navy blue", "dark smoky black",
                "deep crimson red", "charcoal graphite",
                "translucent ice white", "copper-flecked bronze",
                "dark plum purple", "forest moss green",
                "slate blue-gray", "warm cinnamon brown",
            ],
            "fiber_glow_color": [
                "cyan-teal", "emerald green", "violet-purple",
                "warm amber-gold", "ice blue-white", "magenta-pink",
                "coral-orange", "lime neon green", "soft lavender",
                "ruby red", "electric indigo", "champagne gold",
                "aquamarine", "tangerine orange",
            ],
            "fiber_effect": [
                "bioluminescent deep-ocean garden", "captured starfield constellation",
                "northern aurora trapped in glass", "firefly field at midnight",
                "circuit board of light", "scattered diamond dust",
                "glowing coral reef at night", "enchanted forest floor",
                "meteor shower frozen in time", "glowing jellyfish swarm",
                "galaxy arm spiral", "pixelated digital rain",
                "frozen champagne bubbles", "luminous pollen field",
            ],
            "worker_outfit": [
                "all-black work uniforms with white gloves",
                "gray coveralls with black nitrile gloves",
                "white cleanroom suits with blue gloves",
                "navy blue jumpsuits with latex gloves",
                "black polo shirts with clear safety glasses",
                "tan cargo pants and hi-vis yellow vests",
                "charcoal work shirts with rubber knee pads",
                "forest green scrubs with purple nitrile gloves",
                "denim overalls with leather tool belt",
                "khaki work pants and white hard hats",
            ],
            "sofa_style": [
                "light gray L-shaped sectional sofa",
                "cream bouclé modular sofa",
                "dark charcoal velvet curved sofa",
                "cognac leather sectional",
                "emerald velvet Chesterfield sofa",
                "black leather low-profile platform sofa",
                "blush pink linen curved sofa",
                "camel suede three-seater with brass legs",
                "ivory sherpa cloud sofa",
                "navy corduroy modular pit sofa",
            ],
            "accent_furniture": [
                "dark walnut coffee table with glass top",
                "black marble plinth side table",
                "brass and glass round coffee table",
                "live-edge wood slab table",
                "chrome and smoked glass console table",
                "terrazzo pedestal side table",
                "onyx-top brass tripod table",
                "acrylic waterfall coffee table",
                "hammered copper drum table",
                "travertine block end table",
            ],
            "fire_element": [
                "linear gas fireplace blazing with warm amber flames",
                "suspended bioethanol fireplace with dancing flames",
                "cast-iron wood-burning stove with glowing embers",
                "copper fire bowl on the floor with low flames",
                "wall-mounted ribbon fireplace with blue flame",
                "freestanding glass cylinder ethanol fire column",
                "recessed floor fire pit with lava rocks",
                "double-sided see-through gas fireplace",
                "tabletop concrete fire bowl with white stones",
                "hanging orb fireplace suspended from ceiling",
            ],
            "reveal_time": [
                "dusk with blue-hour sky transitioning to deep blue",
                "late evening with dark sky and city lights",
                "overcast twilight with moody gray-purple sky",
                "clear midnight with scattered stars",
                "golden sunset with warm amber flooding through glass",
                "rainy evening with water droplets on windows",
                "foggy night with diffused city glow",
                "pink-orange dawn breaking on the horizon",
                "stormy dusk with dramatic cloud formations",
                "full moon night with silver light on surfaces",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        low_cam = (
            "Very low angle 6-12 inches off the floor, looking across the floor "
            "toward the far wall. 9:16 vertical portrait."
        )

        return [
            {
                "stage": 1,
                "name": "fiber_install",
                "duration_seconds": 5.0,
                "camera": low_cam,
                "video_prompt": (
                    f"Low angle 6 inches off the floor, 9:16 vertical, looking across a "
                    f"{v['base_floor']} floor in a {v['room_type']}. {v['window_type']} on the left "
                    f"flood the room with overcast daylight. {v['wall_feature']} on the right wall. "
                    f"2 workers in {v['worker_outfit']} kneel on the floor, unpacking tools from "
                    f"a black tool bag. They unfurl a white translucent template. "
                    f"In time-lapse: hundreds of tiny luminous {v['fiber_glow_color']} points appear "
                    f"emerging vertically from the floor — fiber optic strands standing upright. "
                    f"Workers move rapidly, installing more strands. The field of {v['fiber_glow_color']} "
                    f"points grows denser, creating a {v['fiber_effect']} effect across the dark floor."
                ),
                "sfx_prompt": (
                    "quiet room ambience, electronic hum building gradually, tools tapping on concrete"
                ),
            },
            {
                "stage": 2,
                "name": "epoxy_pour",
                "duration_seconds": 5.0,
                "camera": (
                    "Close-up at slight low angle for the pour, then wider low floor angle. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Close-up, 9:16 vertical, slightly low angle. A worker pours thick viscous "
                    f"{v['epoxy_color']} epoxy resin from a clear plastic bucket. The liquid streams "
                    f"downward in a thick glossy ribbon. Fiber optic strands protrude upward through "
                    f"the substrate, and a pool of {v['epoxy_color']} epoxy forms around their bases. "
                    f"The {v['fiber_glow_color']} fiber tips glow through the translucent resin creating "
                    f"a {v['fiber_effect']} effect. Camera pulls back — the resin spreads and self-levels "
                    f"across the floor. Rippled fluid dynamics on the wet surface. Reflections of "
                    f"the {v['window_type']} streak across the glossy wet surface."
                ),
                "sfx_prompt": (
                    "thick viscous liquid pouring slowly, satisfying wet ASMR spreading sound"
                ),
            },
            {
                "stage": 3,
                "name": "torch_and_polish",
                "duration_seconds": 5.0,
                "camera": (
                    "Low-medium angle from the foreground corner, then wider room shot. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Low-medium angle, 9:16 vertical. A worker in {v['worker_outfit']} holds a "
                    f"long-handled propane torch. Bright orange-yellow flame sweeps across the "
                    f"{v['epoxy_color']} floor surface — popping micro-bubbles. The {v['fiber_glow_color']} "
                    f"fiber glow creates a three-way color interaction: warm flame, {v['epoxy_color']} resin, "
                    f"{v['fiber_glow_color']} fiber points. Then the worker uses a floor buffer-polisher "
                    f"machine, pushing it across the surface. The polished section becomes hyper-reflective, "
                    f"mirror-like — the {v['window_type']} reflections appear perfectly on the glossy surface."
                ),
                "sfx_prompt": (
                    "propane torch hissing, bubbles popping, then floor buffer humming and whirring"
                ),
            },
            {
                "stage": 4,
                "name": "furnishing",
                "duration_seconds": 5.0,
                "camera": (
                    "Slightly higher angle 3-4 feet off ground, diagonal across room. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Medium height, 9:16 vertical, diagonal across the {v['room_type']}. The floor "
                    f"is fully finished — a {v['epoxy_color']} mirror surface with {v['fiber_glow_color']} "
                    f"fiber optic points glowing throughout like {v['fiber_effect']}. In rapid time-lapse: "
                    f"a {v['sofa_style']} appears in the back corner against the {v['window_type']}. "
                    f"Workers motion-blurred, arranging furniture. A {v['accent_furniture']} is placed. "
                    f"The {v['wall_feature']} is visible. Daylight floods through windows, competing with "
                    f"the {v['fiber_glow_color']} floor glow that casts colored light upward onto the "
                    f"furniture undersides. By end of clip the room is fully furnished."
                ),
                "sfx_prompt": (
                    "furniture sliding on smooth floor, fabric rustling, footsteps echoing"
                ),
            },
            {
                "stage": 5,
                "name": "reveal",
                "duration_seconds": 5.0,
                "camera": (
                    "Medium height, diagonal across room, showing lighting modes. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Medium height, 9:16 vertical, diagonal view of the furnished {v['room_type']}. "
                    f"{v['reveal_time']} visible through {v['window_type']}. "
                    f"The {v['fire_element']}. {v['sofa_style']} flanks the room. "
                    f"The {v['epoxy_color']} fiber optic floor cycles through brightness: first DIM — "
                    f"deep dark surface with subtle {v['fiber_glow_color']} points, moody and intimate. "
                    f"Then FULL BRIGHTNESS — the entire floor ablaze with dense {v['fiber_glow_color']} "
                    f"luminous points like {v['fiber_effect']}. The floor becomes the dominant light source, "
                    f"casting {v['fiber_glow_color']} uplight onto walls and ceiling. Mirror surface "
                    f"reflects furniture legs, fire glow, window frames. No workers. Futuristic luxury."
                ),
                "sfx_prompt": (
                    "fireplace crackling, deep atmospheric hum, subtle electrical shimmer, evening ambience"
                ),
            },
        ]
