"""
Configuration for Life-Hack Short Video Generator.
APIs: DeepSeek (LLM) / ElevenLabs (TTS) / FAL (image+video)
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
USED_HACKS_FILE = DATA_DIR / "used_hacks.json"

for d in [OUTPUT_DIR, TEMP_DIR, DATA_DIR, BGM_DIR, SFX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── DeepSeek LLM (script generation) ─────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SCRIPT_MAX_TOKENS = int(os.getenv("SCRIPT_MAX_TOKENS", "4096"))
HACKS_PER_VIDEO = int(os.getenv("HACKS_PER_VIDEO", "3"))

# ── ElevenLabs TTS ───────────────────────────────────────
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_VOICE_ID_FEMALE = os.getenv("ELEVENLABS_VOICE_ID_FEMALE", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
# Lower stability = more expressive/varied. 0.15 gives punchy, dynamic delivery.
TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.15"))
TTS_SIMILARITY_BOOST = float(os.getenv("TTS_SIMILARITY_BOOST", "0.90"))
# Max style exaggeration for high-energy, comedic delivery
TTS_STYLE = float(os.getenv("TTS_STYLE", "1.0"))
TTS_USE_SPEAKER_BOOST = True

# ── FAL (image generation - FLUX) ────────────────────────
FAL_IMAGE_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux-pro/v1.1")
IMAGE_SIZE = "portrait_16_9"  # 9:16 vertical

# ── FAL (lip-sync video generation) ───────────────────────
# Primary: VEED Fabric 1.0 — phoneme-driven mouth animation, works with any image
LIPSYNC_MODEL_PRIMARY = os.getenv("LIPSYNC_MODEL_PRIMARY", "veed/fabric-1.0")
LIPSYNC_RESOLUTION = os.getenv("LIPSYNC_RESOLUTION", "720p")
# Fallback: SadTalker — 3D motion coefficient lip sync
LIPSYNC_MODEL_FALLBACK = os.getenv("LIPSYNC_MODEL_FALLBACK", "fal-ai/sadtalker")
SADTALKER_FACE_RESOLUTION = os.getenv("SADTALKER_FACE_RESOLUTION", "512")
SADTALKER_EXPRESSION_SCALE = float(os.getenv("SADTALKER_EXPRESSION_SCALE", "1.5"))
SADTALKER_PREPROCESS = os.getenv("SADTALKER_PREPROCESS", "full")

# ── Post-processing (FFmpeg) ─────────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "58"))
SUBTITLE_FONT = os.getenv("SUBTITLE_FONT", "Arial")
SUBTITLE_MARGIN_V = int(os.getenv("SUBTITLE_MARGIN_V", "160"))
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.12"))
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.6"))
