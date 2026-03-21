"""
Template 3: Fiber Optic Epoxy Floor Installation (ref: Flower Garden / 03)

STRUCTURE (LOCKED):
  Stage 1: Workers unpack tools on bare concrete floor in high-rise penthouse
  Stage 2: Fiber optic strands installed vertically into floor (time-lapse, low floor angle)
  Stage 3: Colored epoxy poured over fiber field — viscous close-up ASMR shots
  Stage 4: Torch de-gassing + polishing + furniture staging
  Stage 5: Reveal — evening/dusk, furnished room, fiber floor in multiple brightness modes

VARIABLES:
  - Epoxy COLOR (blue, green, purple, amber)
  - Fiber optic GLOW COLOR
  - Room TYPE (penthouse, loft, gallery, restaurant)
  - Furniture STYLE
  - Fireplace / accent feature
  - Window VIEW
"""
from prompts.templates.base import BaseTemplate


class FiberOpticFloorTemplate(BaseTemplate):
    template_id = "fiber_optic_floor"
    template_name = "Fiber Optic Epoxy Floor"
    reference_video = "03_flower_garden.mp4"
    total_duration_seconds = 16.0
    num_stages = 5

    def get_variable_pools(self) -> dict[str, list]:
        return {
            # ── Room setting ──
            "room_type": [
                "luxury high-rise penthouse living room",
                "converted warehouse loft",
                "contemporary art gallery main hall",
                "upscale restaurant dining room",
                "private cinema lounge",
                "hotel presidential suite",
                "rooftop glass-enclosed lounge",
                "underground speakeasy bar",
            ],
            "window_type": [
                "massive floor-to-ceiling corner windows",
                "panoramic curved glass wall",
                "industrial steel-frame windows",
                "arched floor-to-ceiling windows",
                "narrow vertical slot windows",
                "glass dome skylight overhead",
            ],
            "window_view": [
                "city skyline at high elevation",
                "ocean horizon and coastal cliffs",
                "mountain range with snow caps",
                "dense urban night lights",
                "forest canopy at treetop level",
                "river bend with bridge lights",
            ],
            "wall_feature": [
                "built-in linear gas fireplace with cream surround",
                "floor-to-ceiling bookshelf wall",
                "raw exposed brick wall with steel beams",
                "white marble feature wall with indirect lighting",
                "green living plant wall",
                "dark timber slat acoustic wall",
            ],
            # ── Floor material ──
            "base_floor": [
                "bare gray concrete", "raw polished concrete",
                "dark slate gray concrete", "light cream concrete",
                "black epoxy-coated concrete", "terrazzo-flecked gray concrete",
            ],
            "epoxy_color": [
                "deep teal-blue", "rich emerald green",
                "midnight purple", "warm honey-amber",
                "ocean navy blue", "dark smoky black",
                "ruby crimson", "frosted pearl white",
            ],
            "fiber_glow_color": [
                "cyan-teal", "emerald green", "violet-purple",
                "warm amber-gold", "ice blue-white", "magenta-pink",
                "pure white", "shifting rainbow spectrum",
            ],
            "fiber_effect": [
                "bioluminescent deep-ocean garden", "captured starfield constellation map",
                "northern aurora trapped in glass", "firefly field at midnight",
                "circuit board of light", "scattered diamond dust",
                "underwater coral spawning", "galaxy nebula swirl",
            ],
            # ── Workers ──
            "worker_outfit": [
                "all-black work uniforms with white gloves",
                "gray coveralls with black nitrile gloves",
                "white cleanroom suits with blue gloves",
                "navy work shirts with leather tool belts",
            ],
            "worker_count": ["2", "3"],
            # ── Furniture ──
            "sofa_style": [
                "light gray L-shaped sectional sofa",
                "cream bouclé modular sofa",
                "dark charcoal velvet curved sofa",
                "cognac leather sectional",
                "sage green linen corner sofa",
                "ivory white low-profile sofa",
            ],
            "accent_furniture": [
                "dark walnut coffee table with glass top",
                "black marble plinth side table",
                "brass and glass round coffee table",
                "live-edge wood slab table",
                "white lacquer minimalist console",
            ],
            # ── Reveal lighting modes ──
            "fire_element": [
                "linear gas fireplace blazing with warm amber flames",
                "suspended bioethanol fireplace with dancing flames",
                "cast-iron wood-burning stove with glowing embers",
                "candle arrangement on mantlepiece with warm glow",
                "copper fire bowl on the floor with low flames",
            ],
            "reveal_time": [
                "dusk with blue-hour sky transitioning to deep blue",
                "late evening with dark sky and city lights",
                "pre-dawn with pale indigo eastern sky",
                "overcast twilight with moody gray-purple sky",
            ],
        }

    def build_stages(self, v: dict[str, str]) -> list[dict]:
        low_camera = (
            "Very low angle, approximately 6-12 inches off the floor surface, "
            "looking across the floor toward the far wall. 9:16 vertical portrait. "
            "This angle makes the fiber optic strands look like a luminous field."
        )

        return [
            # ── STAGE 1: Setup + fiber install (0-3s) ──
            {
                "stage": 1,
                "name": "fiber_install",
                "duration_seconds": 3.0,
                "camera": low_camera,
                "video_prompt": (
                    f"Low angle 6 inches off the floor, 9:16 vertical, looking across a {v['base_floor']} "
                    f"floor in a {v['room_type']}. {v['window_type']} on the left flood the room with "
                    f"overcast natural daylight. {v['wall_feature']} visible on the right wall. "
                    f"{v['worker_count']} workers in {v['worker_outfit']} kneel on the floor, unpacking "
                    f"tools from a black tool bag with colorful handles. They unfurl a white translucent "
                    f"template material. Then hundreds of tiny luminous {v['fiber_glow_color']} points "
                    f"appear emerging vertically from the floor substrate — fiber optic strands standing "
                    f"upright. Workers move rapidly in time-lapse, installing more strands. "
                    f"The field of {v['fiber_glow_color']} points grows denser each moment, creating a "
                    f"{v['fiber_effect']} effect across the dark floor. Cool cyan-blue glow fills the room."
                ),
                "sfx_prompt": (
                    "quiet room ambience, gentle electronic hum building gradually, "
                    "soft tapping of tools on concrete, workers shuffling, fabric rustling"
                ),
            },
            # ── STAGE 2: Epoxy pour — close-up ASMR (3-7s) ──
            {
                "stage": 2,
                "name": "epoxy_pour",
                "duration_seconds": 4.0,
                "camera": (
                    "Close-up at slight low angle for the pour, showing viscous liquid "
                    "streaming from bucket. Then wider low floor angle showing spread. "
                    "9:16 vertical."
                ),
                "video_prompt": (
                    f"Close-up, 9:16 vertical, slightly low angle. A worker pours thick viscous "
                    f"{v['epoxy_color']} epoxy resin from a clear plastic bucket. The liquid streams "
                    f"downward in a thick glossy ribbon. On the floor, fiber optic strands protrude "
                    f"upward through the gray substrate, and a pool of {v['epoxy_color']} epoxy forms "
                    f"around their bases. The {v['fiber_glow_color']} fiber tips glow through the "
                    f"translucent resin creating a stunning {v['fiber_effect']} effect. "
                    f"Camera pulls back to low floor angle — the {v['epoxy_color']} epoxy spreads and "
                    f"self-levels across 60-70 percent of the floor. Rippled fluid dynamics visible on "
                    f"the wet surface. The fiber strands at the perimeter create a border of "
                    f"{v['fiber_glow_color']} points around the spreading pool. Reflections of the "
                    f"{v['window_type']} streak across the wet surface."
                ),
                "sfx_prompt": (
                    "thick viscous liquid pouring slowly, satisfying wet spreading sound, "
                    "ASMR gloopy resin pooling, gentle squelching as epoxy self-levels"
                ),
            },
            # ── STAGE 3: Torch + polish (7-10s) ──
            {
                "stage": 3,
                "name": "torch_and_polish",
                "duration_seconds": 3.0,
                "camera": (
                    "Low-medium angle from the foreground corner. Then transitions to "
                    "wider room shot. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Low-medium angle, 9:16 vertical. A worker in {v['worker_outfit']} stands holding "
                    f"a long-handled propane torch. A bright orange-yellow flame sweeps across the "
                    f"{v['epoxy_color']} floor surface — popping micro-bubbles. The {v['fiber_glow_color']} "
                    f"fiber glow from below creates a three-way color interaction: warm orange flame, "
                    f"{v['epoxy_color']} resin, {v['fiber_glow_color']} fiber points. "
                    f"Then the worker uses a floor buffer-polisher machine, pushing it across the left "
                    f"portion. The polished section becomes hyper-reflective, mirror-like — "
                    f"the {v['window_type']} reflections appear perfectly on the glossy surface. "
                    f"Daylight through windows provides ambient fill while the {v['fiber_glow_color']} "
                    f"floor glow is the primary illumination."
                ),
                "sfx_prompt": (
                    "propane torch hissing with flame roar, bubbles popping, "
                    "then floor buffer machine humming and whirring, smooth polishing rhythm"
                ),
            },
            # ── STAGE 4: Furniture staging (10-12s) ──
            {
                "stage": 4,
                "name": "furnishing",
                "duration_seconds": 2.0,
                "camera": (
                    "Slightly higher angle, approximately 3-4 feet off the ground, "
                    "diagonal across the room. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Medium height, 9:16 vertical, diagonal across the {v['room_type']}. "
                    f"The floor is fully finished — a {v['epoxy_color']} mirror surface with "
                    f"{v['fiber_glow_color']} fiber optic points glowing throughout like {v['fiber_effect']}. "
                    f"A {v['sofa_style']} appears in the back corner, positioned against the {v['window_type']}. "
                    f"A worker is motion-blurred, arranging it in time-lapse. "
                    f"A {v['accent_furniture']} is placed. The {v['wall_feature']} is visible. "
                    f"Daylight floods through windows, competing with the {v['fiber_glow_color']} "
                    f"floor glow that casts colored light upward onto the furniture undersides."
                ),
                "sfx_prompt": (
                    "furniture sliding on smooth floor, fabric rustling, "
                    "workers footsteps echoing in large room, quiet efficiency"
                ),
            },
            # ── STAGE 5: Evening reveal with lighting modes (12-16s) ──
            {
                "stage": 5,
                "name": "reveal_modes",
                "duration_seconds": 4.0,
                "camera": (
                    "Medium height, slightly lower, diagonal across the room. "
                    "Shows multiple lighting modes. 9:16 vertical."
                ),
                "video_prompt": (
                    f"Medium height, 9:16 vertical, diagonal view of the fully furnished {v['room_type']}. "
                    f"{v['reveal_time']} visible through {v['window_type']}. "
                    f"The {v['wall_feature']} — the {v['fire_element']}. "
                    f"{v['sofa_style']} flanks the room. The {v['epoxy_color']} fiber optic floor cycles "
                    f"through brightness modes: first DIM — deep dark surface with subtle scattered "
                    f"{v['fiber_glow_color']} points, moody and intimate. Then FULL BRIGHTNESS — "
                    f"the entire floor ablaze with dense bright {v['fiber_glow_color']} luminous points, "
                    f"like {v['fiber_effect']}. The floor becomes the dominant light source, "
                    f"casting {v['fiber_glow_color']} uplight onto furniture and walls. "
                    f"Three distinct color temperatures: warm fire, {v['fiber_glow_color']} floor glow, "
                    f"cool window light. The mirror-smooth surface reflects everything — furniture legs, "
                    f"fire glow, window frames. Then dims again to moody — the living-with-it moment. "
                    f"No workers. Pristine, futuristic luxury."
                ),
                "sfx_prompt": (
                    "fireplace crackling warmly, deep atmospheric ambient hum, "
                    "very subtle electrical shimmer from floor, evening city sounds through glass, serene"
                ),
            },
        ]
