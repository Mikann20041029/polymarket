"""
rebornspacestv Style Guide — Deep Visual DNA.

This module encodes the exact visual language, pacing, and cinematic rules
that define the rebornspacestv construction-to-luxury transformation genre.

Every prompt and scenario MUST pass through these rules.
"""

# ── CORE IDENTITY ────────────────────────────────────────────────
# rebornspacestv videos are NOT random AI slop.
# They are HYPER-SPECIFIC construction timelapses with ONE rule:
#   "Boring/ugly/impossible space → visible construction → jaw-dropping luxury reveal"
#
# The viewer experience:
#   Second 0: "What is this ugly/boring place?"
#   Second 1-10: "Holy shit they're actually building that"
#   Second 10-15: "I NEED that in my life"

CHANNEL_DNA = {
    "core_formula": "mundane_space + impossible_construction + luxury_reveal = viral",
    "emotional_arc": ["confusion/curiosity", "satisfaction/awe", "desire/disbelief"],
    "key_differentiator": "the PROCESS is the content, not just before/after",
    "realism_level": "aspirational_but_believable",  # NOT fantasy, NOT boring reality
}

# ── VISUAL RULES (NON-NEGOTIABLE) ────────────────────────────────

CAMERA_RULES = {
    "primary": "FIXED camera position throughout entire sequence",
    "angle": "slightly elevated 3/4 view OR clean overhead drone",
    "movement": "ZERO camera movement during construction phases",
    "reveal_exception": "subtle slow push-in or descent ONLY in final 3 seconds",
    "aspect_ratio": "9:16 vertical (portrait mode for Shorts/Reels)",
    "framing": "entire workspace visible, subject centered, 15% breathing room on edges",
    "consistency": "identical focal length, height, and tilt across ALL stages",
}

LIGHTING_RULES = {
    "construction_phases": "natural daylight, consistent sun angle, visible shadow movement for time passage",
    "reveal_phase": "dramatic shift — warm golden hour OR strategic artificial lighting turns on",
    "shadow_continuity": "shadows must rotate naturally to suggest hours passing",
    "night_reveal": "if night reveal, warm interior glow contrasting with blue-hour exterior",
    "forbidden": "NO random lighting changes mid-construction, NO neon unless part of the reveal",
}

TIMELAPSE_PHYSICS = {
    "speed": "8-16x real-time feel — workers move fast but recognizably human",
    "motion_blur": "subtle streak on fast-moving tools and materials",
    "shadow_sweep": "sun shadows visibly rotate across the ground",
    "dust_particles": "visible in sunlight during excavation and demolition",
    "material_accumulation": "gradual — not instant. Piles grow, holes deepen progressively",
    "weather_consistency": "same weather throughout (clear day preferred)",
}

# ── 3-STAGE VIDEO STRUCTURE ──────────────────────────────────────
# Each video = 3 clips of 5 seconds, stitched together.
# This is the KEY to coherence. One 15s clip = incoherent mess.

STAGE_STRUCTURE = {
    "stage_1_before": {
        "duration_seconds": 5,
        "name": "THE UGLY TRUTH",
        "purpose": "show the boring/ugly/empty space that hooks curiosity",
        "timing": "0-5 seconds",
        "visual_rules": [
            "static shot of the ENTIRE space from the chosen camera angle",
            "nothing happening for first 0.5s — let viewer absorb the before state",
            "then: first signs of work — marking lines, first shovel, truck arrives",
            "by end of clip: demolition/excavation visibly underway",
            "dust, debris, first material removal",
        ],
        "prompt_keywords": [
            "static camera", "establishing shot", "empty", "abandoned",
            "mundane", "then construction begins", "workers arrive",
            "first excavation", "marking ground", "demolition starts",
        ],
    },
    "stage_2_construction": {
        "duration_seconds": 5,
        "name": "THE BUILD",
        "purpose": "the satisfying construction process — this IS the content",
        "timing": "5-10 seconds",
        "visual_rules": [
            "SAME camera angle as stage 1 — zero change in position",
            "rapid construction: framing, pouring, welding, tiling, installing",
            "workers clearly visible — 2-4 people moving at timelapse speed",
            "heavy machinery if applicable (excavator, crane, concrete truck)",
            "materials visibly transforming: raw → structured → finished",
            "shadows rotating to show time passage",
        ],
        "prompt_keywords": [
            "same camera angle", "timelapse construction", "workers moving fast",
            "rapid building", "framing", "pouring concrete", "welding sparks",
            "materials arriving", "structure rising", "shadows rotating",
        ],
    },
    "stage_3_reveal": {
        "duration_seconds": 5,
        "name": "THE REVEAL",
        "purpose": "the jaw-dropping completed space — make them want it",
        "timing": "10-15 seconds",
        "visual_rules": [
            "SAME camera angle for first 2 seconds showing near-completion",
            "then: dramatic lighting shift (golden hour / lights turn on)",
            "optional: subtle camera push-in for final 2 seconds",
            "water fills in, lights illuminate, door opens, etc.",
            "final frame must be BEAUTIFUL — this is the thumbnail moment",
            "luxury materials clearly visible: marble, wood, glass, water, greenery",
        ],
        "prompt_keywords": [
            "same camera angle", "finishing touches", "lights turning on",
            "golden hour lighting", "luxury reveal", "water filling",
            "dramatic transformation", "premium materials", "stunning completion",
        ],
    },
}

# ── WHAT MAKES REBORNSPACESTV UNIQUE ─────────────────────────────

CONTENT_PILLARS = [
    {
        "id": "impossible_underground",
        "name": "Impossible Underground Spaces",
        "description": "Digging into the earth to create hidden luxury rooms, pools, cinemas",
        "visual_hook": "overhead drone shot of ground being excavated, descending into finished room",
        "examples": [
            "Underground cinema beneath the garden — excavator digs, concrete walls, LED ceiling becomes starry sky",
            "Buried shipping container becomes luxury sauna with glass roof showing roots above",
            "Basement parking spot becomes hidden Japanese onsen with natural stone and steam",
            "Backyard excavation reveals multi-room underground lounge with fireplace and skylight",
            "Garden shed floor opens to reveal underground cocktail bar with neon accents",
        ],
    },
    {
        "id": "water_transformation",
        "name": "Water Where There Was None",
        "description": "Dry land transforms into stunning water features through visible construction",
        "visual_hook": "the money shot is always the moment water fills the completed structure",
        "examples": [
            "Empty concrete backyard becomes infinity pool overlooking city lights",
            "Flat grass becomes multi-level koi pond with glass viewing panels",
            "Abandoned tennis court becomes natural swimming lagoon with waterfall",
            "Rooftop gravel becomes glass-bottom pool with living room visible below",
            "Hillside carved into cascading hot spring pools with stone edges",
        ],
    },
    {
        "id": "hidden_reveal",
        "name": "Secret Rooms Behind Ordinary Surfaces",
        "description": "Normal wall/floor/shelf conceals a luxury space — the reveal is mechanical",
        "visual_hook": "the moment the hidden door opens and warm light spills out",
        "examples": [
            "Wine rack slides to reveal underground tasting room with barrel-vault ceiling",
            "Bathroom mirror is a door to a hidden spa with rain shower and sauna",
            "Garage floor hydraulically lifts to reveal underground car gallery with LED strips",
            "Kitchen island counter slides to reveal hidden staircase to rooftop bar",
            "Bedroom closet wall rotates to reveal home theater with stadium seating",
        ],
    },
    {
        "id": "container_transformation",
        "name": "Industrial Object Becomes Luxury",
        "description": "Shipping container, bus, silo, water tank → luxury living space",
        "visual_hook": "the external ugliness contrasting with the interior luxury",
        "examples": [
            "Rusted shipping container → minimalist Japanese tea room with garden window",
            "Old school bus → mobile luxury cabin with wood-clad interior and fireplace",
            "Abandoned grain silo → vertical loft with spiral staircase and panoramic windows",
            "Decommissioned water tank → artist studio with domed skylight",
            "Old train car → boutique hotel suite with velvet and brass accents",
        ],
    },
    {
        "id": "extreme_narrow",
        "name": "Impossibly Tight Space Becomes Grand",
        "description": "Tiny/narrow/awkward space transformed into something surprisingly luxurious",
        "visual_hook": "disbelief at how much fits in so little space",
        "examples": [
            "2-meter alley between buildings becomes vertical Japanese garden with tea room",
            "Under-staircase closet becomes fully functional cocktail bar with copper sink",
            "Rooftop utility shed becomes glass-walled meditation room with zen garden",
            "Narrow garage side-space becomes outdoor kitchen with pizza oven and bar seating",
            "Tiny balcony becomes multi-level indoor garden with seating nook",
        ],
    },
    {
        "id": "landscape_sculpting",
        "name": "Reshaping the Earth Itself",
        "description": "Heavy earthmoving creates dramatic landscape features",
        "visual_hook": "overhead drone watching earth move and reshape in timelapse",
        "examples": [
            "Flat field sculpted into sunken fire pit amphitheater with stone seating",
            "Hillside carved into terraced infinity pools connected by waterfalls",
            "Backyard leveled and reshaped into Japanese dry garden with raked gravel",
            "Sloped yard becomes multi-level deck system with hot tub at lowest point",
            "Empty lot transformed into sunken courtyard with mature trees and lighting",
        ],
    },
    {
        "id": "rooftop_paradise",
        "name": "Boring Rooftop Becomes Paradise",
        "description": "Empty/ugly flat roof transformed into luxury outdoor living",
        "visual_hook": "city skyline backdrop contrasting with the intimate luxury space",
        "examples": [
            "Gravel rooftop becomes rooftop pool with transparent glass walls showing city below",
            "Empty flat roof becomes Mediterranean garden with pergola and outdoor kitchen",
            "Industrial rooftop becomes Japanese rooftop onsen with wooden deck and bamboo screens",
            "Bare concrete roof becomes cinema under the stars with modular seating and screen",
            "Building top becomes greenhouse restaurant with hanging plants and fairy lights",
        ],
    },
    {
        "id": "abandoned_luxury",
        "name": "Abandoned Ruin to Ultra-Luxury",
        "description": "Completely trashed/abandoned space restored to stunning luxury",
        "visual_hook": "the extreme contrast between the ruin and the finished space",
        "examples": [
            "Abandoned church becomes loft apartment with original stained glass preserved",
            "Derelict factory floor becomes open-plan penthouse with exposed brick and steel",
            "Burned-out barn becomes glass-and-timber artist retreat with wraparound deck",
            "Collapsed greenhouse becomes indoor tropical garden with koi pond and hammock",
            "Flooded basement becomes underwater-themed lounge with porthole windows and blue LED",
        ],
    },
]

# ── PROMPT ENGINEERING RULES ─────────────────────────────────────
# These rules produce COHERENT video from AI models.

PROMPT_RULES = {
    "structure": [
        "Start with camera angle and format: 'Fixed overhead drone shot, 9:16 vertical'",
        "Describe the ENVIRONMENT first: location, weather, time of day",
        "Then describe WHAT IS VISIBLE: the space, its current state",
        "Then describe THE ACTION: what changes happen, in what order",
        "End with ATMOSPHERE: lighting, particles, mood",
    ],
    "forbidden_words": [
        "timelapse",  # AI models don't understand this — describe the SPEED instead
        "fast-forward",  # same problem
        "scene change",  # causes cuts
        "cut to",  # causes cuts
        "next scene",  # causes cuts
        "transition",  # causes unwanted effects
        "suddenly",  # causes jarring jumps
        "magically",  # breaks realism
        "instantly",  # breaks realism
    ],
    "required_elements": [
        "specific camera angle description",
        "consistent lighting description",
        "visible human workers (not ghosts)",
        "specific material names (not generic 'materials')",
        "specific color palette",
        "atmosphere (dust, light rays, shadows)",
    ],
    "prompt_length": {
        "min_words": 60,
        "max_words": 120,
        "sweet_spot": "80-100 words per stage prompt",
    },
}

# ── SFX RULES ────────────────────────────────────────────────────

SFX_STYLE = {
    "stage_1": "ambient outdoor atmosphere, gentle wind, distant birds, then first construction impact sounds — shovel hitting earth, truck reversing beep",
    "stage_2": "layered construction sounds at high speed — rhythmic hammering, power drill whine, concrete pouring, metal clanging, worker shouts muffled",
    "stage_3": "construction fading out, replaced by atmospheric reveal — water flowing, warm ambient hum, soft music undertone, final satisfying click or splash",
    "overall": "continuous audio flow, no silence gaps, construction sounds should feel rhythmic and satisfying like ASMR",
}

# ── ANTI-PATTERNS (what makes videos SHIT) ───────────────────────

ANTI_PATTERNS = [
    "Generic prompts like 'construction timelapse video' — model doesn't know what to do",
    "Listing materials without visual context — 'wood, steel, glass' means nothing",
    "Changing camera angle between stages — destroys continuity",
    "No workers visible — space transforms 'magically' which looks fake",
    "Too many things happening — focus on ONE clear transformation",
    "Fantasy/sci-fi elements — must feel BUILDABLE even if ambitious",
    "Same lighting in before and after — the reveal needs a lighting SHIFT",
    "No atmosphere — no dust, no shadows, no particles = flat lifeless video",
    "Prompt says 'beautiful' or 'stunning' — these words mean nothing to AI models",
    "One 15-second clip trying to show everything — always incoherent",
]


def get_stage_prompt_template(stage: int) -> dict:
    """Get the prompt template for a specific stage (1, 2, or 3)."""
    key = {1: "stage_1_before", 2: "stage_2_construction", 3: "stage_3_reveal"}[stage]
    return STAGE_STRUCTURE[key]


def get_random_content_pillar(exclude_ids: list[str] | None = None) -> dict:
    """Get a random content pillar, optionally excluding recent ones."""
    import random
    exclude = set(exclude_ids or [])
    available = [p for p in CONTENT_PILLARS if p["id"] not in exclude]
    if not available:
        available = CONTENT_PILLARS
    return random.choice(available)


def get_sfx_prompt(stage: int) -> str:
    """Get SFX prompt for a stage."""
    return SFX_STYLE.get(f"stage_{stage}", SFX_STYLE["overall"])


def validate_prompt(prompt: str) -> list[str]:
    """Check a prompt against anti-pattern rules. Returns list of violations."""
    violations = []
    lower = prompt.lower()

    for word in PROMPT_RULES["forbidden_words"]:
        if word.lower() in lower:
            violations.append(f"Forbidden word: '{word}'")

    if len(prompt.split()) < PROMPT_RULES["prompt_length"]["min_words"]:
        violations.append(f"Too short: {len(prompt.split())} words (min {PROMPT_RULES['prompt_length']['min_words']})")

    if len(prompt.split()) > PROMPT_RULES["prompt_length"]["max_words"]:
        violations.append(f"Too long: {len(prompt.split())} words (max {PROMPT_RULES['prompt_length']['max_words']})")

    return violations
