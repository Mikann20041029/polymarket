"""
Configuration for Impossible Satisfying Video Generator.
APIs: DeepSeek (concept) / ElevenLabs (SFX) / FAL + Wan 2.1 (video)
"""
import os
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
ELEVENLABS_SFX_MODEL = os.getenv("ELEVENLABS_SFX_MODEL", "eleven_sfx_v2")
SFX_DURATION_SECONDS = float(os.getenv("SFX_DURATION_SECONDS", "10.0"))
SFX_PROMPT_INFLUENCE = float(os.getenv("SFX_PROMPT_INFLUENCE", "0.7"))

# ── FAL Video Generation (~$0.10/clip Wan, ~$0.50/clip Kling) ──
# Default: Wan 2.1 (cheapest). Override with kling endpoint when validated.
FAL_VIDEO_MODEL = os.getenv("FAL_VIDEO_MODEL", "fal-ai/wan/v2.1/1.3b/text-to-video")
FAL_VIDEO_MODEL_FALLBACK = os.getenv("FAL_VIDEO_MODEL_FALLBACK", "fal-ai/wan/v2.1/1.3b/text-to-video")
VIDEO_DURATION = os.getenv("VIDEO_DURATION", "5")
VIDEO_ASPECT_RATIO = os.getenv("VIDEO_ASPECT_RATIO", "9:16")

# ── Post-processing (FFmpeg, free) ────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
OVERLAY_FONT_SIZE = int(os.getenv("OVERLAY_FONT_SIZE", "72"))
OVERLAY_FONT = os.getenv("OVERLAY_FONT", "Arial")
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.15"))  # Subtle but audible
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.85"))
