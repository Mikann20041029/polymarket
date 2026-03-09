"""
Configuration for TikTok/Shorts Lip-Sync Video Generator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DID_API_KEY = os.getenv("DID_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
CHARACTERS_DIR = ASSETS_DIR / "characters"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Script Generation ────────────────────────────────────
SCRIPT_MODEL = os.getenv("SCRIPT_MODEL", "claude-haiku-4-5-20251001")
SCRIPT_MAX_TOKENS = int(os.getenv("SCRIPT_MAX_TOKENS", "1024"))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ja")

# ── ElevenLabs TTS ───────────────────────────────────────
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.5"))
TTS_SIMILARITY_BOOST = float(os.getenv("TTS_SIMILARITY_BOOST", "0.75"))

# ── D-ID Lip Sync ───────────────────────────────────────
DID_API_URL = "https://api.d-id.com"
DID_POLL_INTERVAL = int(os.getenv("DID_POLL_INTERVAL", "5"))
DID_POLL_TIMEOUT = int(os.getenv("DID_POLL_TIMEOUT", "300"))

# ── Post-processing ─────────────────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920  # 9:16 aspect ratio
SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "48"))
SUBTITLE_FONT_COLOR = os.getenv("SUBTITLE_FONT_COLOR", "white")
SUBTITLE_BG_COLOR = os.getenv("SUBTITLE_BG_COLOR", "black@0.6")
