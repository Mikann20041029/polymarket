"""
Configuration for Life-Hack Short Video Generator.
APIs: DeepSeek (LLM) / ElevenLabs (TTS) / FAL (image+video) / Together (backup LLM)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
FAL_KEY = os.getenv("FAL_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
BGM_DIR = ASSETS_DIR / "bgm"
SFX_DIR = ASSETS_DIR / "sfx"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
DATA_DIR = BASE_DIR / "data"
USED_HACKS_FILE = DATA_DIR / "used_hacks.json"

for d in [OUTPUT_DIR, TEMP_DIR, DATA_DIR, BGM_DIR, SFX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── DeepSeek LLM (script generation) ─────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SCRIPT_MAX_TOKENS = int(os.getenv("SCRIPT_MAX_TOKENS", "4096"))
HACKS_PER_VIDEO = int(os.getenv("HACKS_PER_VIDEO", "3"))

# ── Together (backup LLM) ────────────────────────────────
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

# ── ElevenLabs TTS ───────────────────────────────────────
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.3"))
TTS_SIMILARITY_BOOST = float(os.getenv("TTS_SIMILARITY_BOOST", "0.85"))
TTS_STYLE = float(os.getenv("TTS_STYLE", "0.8"))
TTS_USE_SPEAKER_BOOST = True

# ── FAL (image generation - FLUX) ────────────────────────
FAL_IMAGE_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux-pro/v1.1")
IMAGE_SIZE = "portrait_16_9"  # 9:16 vertical

# ── FAL (video generation - Hailuo 2.3 Fast) ─────────────
FAL_VIDEO_MODEL = os.getenv(
    "FAL_VIDEO_MODEL",
    "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video",
)
HAILUO_DURATION = os.getenv("HAILUO_DURATION", "6")  # "6" or "10"
HAILUO_PROMPT_OPTIMIZER = True

# ── Post-processing (FFmpeg) ─────────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "58"))
SUBTITLE_FONT = os.getenv("SUBTITLE_FONT", "Arial")
SUBTITLE_MARGIN_V = int(os.getenv("SUBTITLE_MARGIN_V", "160"))
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.12"))
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.6"))
