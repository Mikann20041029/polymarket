"""
Configuration for Impossible Satisfying Video Generator.
APIs: DeepSeek (concept) / ElevenLabs (SFX) / FAL + Wan 2.1 (video)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
FAL_KEY = os.getenv("FAL_KEY", "")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
BGM_DIR = ASSETS_DIR / "bgm"
SFX_DIR = ASSETS_DIR / "sfx"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
DATA_DIR = BASE_DIR / "data"
USED_CONCEPTS_FILE = DATA_DIR / "used_concepts.json"

for d in [OUTPUT_DIR, TEMP_DIR, DATA_DIR, BGM_DIR, SFX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── DeepSeek LLM (concept generation, ~$0.001/call) ──────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SCRIPT_MAX_TOKENS = int(os.getenv("SCRIPT_MAX_TOKENS", "4096"))
CLIPS_PER_VIDEO = int(os.getenv("CLIPS_PER_VIDEO", "3"))

# ── ElevenLabs Sound Effects (~$0.01/clip) ────────────────
# Duration MUST match VIDEO_DURATION — paying for longer audio is wasting money
SFX_DURATION_SECONDS = float(os.getenv("SFX_DURATION_SECONDS", "5.0"))
SFX_PROMPT_INFLUENCE = float(os.getenv("SFX_PROMPT_INFLUENCE", "0.7"))

# ── FAL Video Generation ─────────────────────────────────
# Wan 2.1 via fal.ai. Use the larger model for better quality.
FAL_VIDEO_MODEL = os.getenv("FAL_VIDEO_MODEL", "fal-ai/wan/v2.1/1.3b/text-to-video")
VIDEO_DURATION = int(os.getenv("VIDEO_DURATION", "5"))  # seconds, must be int
VIDEO_ASPECT_RATIO = os.getenv("VIDEO_ASPECT_RATIO", "9:16")
# Wan resolution: "480p" or "720p". 480p is garbage when upscaled to 1080x1920.
WAN_RESOLUTION = os.getenv("WAN_RESOLUTION", "720p")
# Derive frame count from duration: 16fps * seconds + 1
WAN_NUM_FRAMES = int(os.getenv("WAN_NUM_FRAMES", str(VIDEO_DURATION * 16 + 1)))

# ── Post-processing (FFmpeg, free) ────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
OVERLAY_FONT_SIZE = int(os.getenv("OVERLAY_FONT_SIZE", "72"))
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.15"))
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.85"))


def validate_api_keys():
    """
    Validate all required API keys are set BEFORE any work starts.
    Call this at pipeline startup to fail fast, not after spending money.
    """
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not FAL_KEY:
        missing.append("FAL_KEY")

    if missing:
        print(f"FATAL: Missing API keys: {', '.join(missing)}", file=sys.stderr)
        print("Set them in .env or as environment variables.", file=sys.stderr)
        sys.exit(1)
