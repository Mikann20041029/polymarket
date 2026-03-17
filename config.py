"""
Configuration for AI World Recreation Video Generator.
Generates photorealistic video shorts from anime worlds and historical events.

APIs: DeepSeek (topic) / Wan 2.1 via fal.ai (video) / ElevenLabs (SFX)
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
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
DATA_DIR = BASE_DIR / "data"
USED_TOPICS_FILE = DATA_DIR / "used_topics.json"

for d in [OUTPUT_DIR, TEMP_DIR, DATA_DIR, BGM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── DeepSeek LLM (~$0.001/call) ──────────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SCRIPT_MAX_TOKENS = int(os.getenv("SCRIPT_MAX_TOKENS", "4096"))

# ── Wan 2.1 Video Generation via fal.ai (~$0.10/clip) ────
FAL_VIDEO_MODEL = os.getenv("FAL_VIDEO_MODEL", "fal-ai/wan/v2.1/1.3b/text-to-video")
WAN_RESOLUTION = os.getenv("WAN_RESOLUTION", "720p")
VIDEO_DURATION = int(os.getenv("VIDEO_DURATION", "5"))
WAN_NUM_FRAMES = int(os.getenv("WAN_NUM_FRAMES", str(VIDEO_DURATION * 16 + 1)))
CLIPS_PER_VIDEO = int(os.getenv("CLIPS_PER_VIDEO", "5"))

# ── ElevenLabs Sound Effects (~$0.01/clip) ────────────────
SFX_DURATION_SECONDS = float(os.getenv("SFX_DURATION_SECONDS", "5.0"))
SFX_PROMPT_INFLUENCE = float(os.getenv("SFX_PROMPT_INFLUENCE", "0.7"))

# ── Post-processing (FFmpeg, free) ────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
CROSSFADE_DURATION = float(os.getenv("CROSSFADE_DURATION", "0.5"))
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.12"))
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.70"))
OVERLAY_FONT_SIZE = int(os.getenv("OVERLAY_FONT_SIZE", "48"))

# ── Budget ────────────────────────────────────────────────
VIDEOS_PER_DAY = int(os.getenv("VIDEOS_PER_DAY", "2"))


def validate_api_keys():
    """Validate all required API keys before any work starts."""
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not FAL_KEY:
        missing.append("FAL_KEY")
    if missing:
        print(f"FATAL: Missing API keys: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
