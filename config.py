"""
Configuration for AI World Recreation Video Generator.
Generates photorealistic "what if you were there" shorts from
anime worlds and historical events.

APIs: DeepSeek (topic/scenes) / FLUX Dev (images) / ElevenLabs (SFX) / FFmpeg (compose)
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

# ── DeepSeek LLM (topic + scene generation, ~$0.001/call) ─
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SCRIPT_MAX_TOKENS = int(os.getenv("SCRIPT_MAX_TOKENS", "4096"))

# ── FLUX Image Generation (fal.ai) ───────────────────────
# FLUX Dev: ~$0.025/image, photorealistic quality
# FLUX Pro 1.1: ~$0.05/image, best quality (use if budget allows)
FAL_IMAGE_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/dev")
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1344   # ~9:16 ratio, good for FLUX
SCENES_PER_VIDEO = int(os.getenv("SCENES_PER_VIDEO", "8"))

# ── ElevenLabs Sound Effects (~$0.01/clip) ────────────────
SFX_DURATION_SECONDS = float(os.getenv("SFX_DURATION_SECONDS", "5.0"))
SFX_PROMPT_INFLUENCE = float(os.getenv("SFX_PROMPT_INFLUENCE", "0.7"))

# ── Post-processing ──────────────────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
SECONDS_PER_SCENE = float(os.getenv("SECONDS_PER_SCENE", "6.0"))
CROSSFADE_DURATION = float(os.getenv("CROSSFADE_DURATION", "0.8"))
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.12"))
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.70"))
OVERLAY_FONT_SIZE = int(os.getenv("OVERLAY_FONT_SIZE", "48"))

# ── Batch settings ────────────────────────────────────────
VIDEOS_PER_DAY = int(os.getenv("VIDEOS_PER_DAY", "3"))


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
