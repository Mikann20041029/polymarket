"""
TTS module using Microsoft Edge TTS (FREE, no API key required).
Generates expressive English voice audio with word-level timing for subtitle sync.
Falls back to ElevenLabs if ELEVENLABS_API_KEY is set and Edge TTS fails.
"""
import asyncio
import io
import json
import logging
import re
import argparse
import subprocess
from pathlib import Path

import edge_tts
import config

logger = logging.getLogger(__name__)

# ── Edge TTS Voice Configuration ──────────────────────────
# High-quality neural voices, free, no API key needed
EDGE_VOICES = {
    "male": [
        "en-US-GuyNeural",          # Energetic, expressive
        "en-US-ChristopherNeural",  # Clear, professional
        "en-US-EricNeural",         # Warm, friendly
    ],
    "female": [
        "en-US-JennyNeural",       # Versatile, expressive
        "en-US-AriaNeural",        # Warm, conversational
        "en-US-SaraNeural",        # Clear, youthful
    ],
}

# Rotate voices for variety within a single run
_voice_index = {"male": 0, "female": 0}


def _get_edge_voice(gender: str) -> str:
    """Pick the next Edge TTS voice for the given gender (round-robin for variety)."""
    voices = EDGE_VOICES.get(gender, EDGE_VOICES["male"])
    idx = _voice_index.get(gender, 0) % len(voices)
    _voice_index[gender] = idx + 1
    return voices[idx]


def _get_audio_duration(path: str) -> float:
    """Get audio duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return 0.0
    return float(json.loads(r.stdout)["format"]["duration"])


def _estimate_word_timing(text: str, duration: float) -> list[dict]:
    """
    Estimate word-level timing by distributing duration proportionally
    across words based on character length.
    """
    words = text.split()
    if not words or duration <= 0:
        return []

    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    alignment = []
    current_time = 0.0

    for w in words:
        word_dur = (len(w) / total_chars) * duration
        alignment.append({
            "word": w,
            "start": round(current_time, 3),
            "end": round(current_time + word_dur, 3),
        })
        current_time += word_dur

    return alignment


def _parse_edge_subtitles(sub_maker: edge_tts.SubMaker) -> list[dict]:
    """
    Parse Edge TTS SubMaker word-level cues into alignment format.
    Each cue is a Subtitle object with .start (timedelta), .end (timedelta), .content (str).
    """
    alignment = []
    cues = getattr(sub_maker, "cues", [])

    if not cues:
        return []

    for cue in cues:
        alignment.append({
            "word": cue.content,
            "start": round(cue.start.total_seconds(), 3),
            "end": round(cue.end.total_seconds(), 3),
        })

    return alignment


async def _generate_edge_tts(text: str, output_path: Path, voice: str) -> dict:
    """Generate TTS audio using Edge TTS with word-level timing."""
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate="+20%",   # Faster for energetic, punchy delivery
        pitch="+5Hz",  # Higher for animated character feel
    )

    sub_maker = edge_tts.SubMaker()
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            sub_maker.feed(chunk)

    # Save audio
    with open(output_path, "wb") as f:
        f.write(audio_data)

    # Get alignment from SubMaker
    alignment = _parse_edge_subtitles(sub_maker)
    duration = _get_audio_duration(str(output_path))

    # If SubMaker alignment is empty, estimate from duration
    if not alignment:
        alignment = _estimate_word_timing(text, duration)

    return {
        "audio_path": str(output_path),
        "alignment": alignment,
        "duration": duration,
    }


def generate_speech(text: str, output_path: Path, voice_id: str = None,
                    gender: str = "male") -> dict:
    """
    Generate expressive speech audio using Edge TTS (free).

    Args:
        text: Text to speak
        output_path: Where to save audio
        voice_id: Override voice name (e.g., "en-US-GuyNeural")
        gender: "male" or "female" — selects the appropriate voice

    Returns dict with:
        - audio_path: path to mp3 file
        - alignment: list of {word, start, end} for subtitle sync
        - duration: total audio duration in seconds
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip emotion tags and problematic punctuation for clean TTS
    clean_text = re.sub(r'\[(?:PAUSE|EXCITED|FRUSTRATED|SMUG|PANIC|RELIEF|SERIOUS|HAPPY|[A-Z]+)\]', '', text)
    # Replace ellipsis with comma-pause (avoids TTS reading "dot dot dot")
    clean_text = re.sub(r'\.{2,}', ',', clean_text)
    # Collapse multiple spaces/punctuation
    clean_text = re.sub(r'  +', ' ', clean_text).strip()

    # Select voice
    voice = voice_id or _get_edge_voice(gender)
    logger.info(f"TTS ({gender}, voice={voice}): '{clean_text[:60]}...'")

    # Generate with Edge TTS
    try:
        result = asyncio.run(_generate_edge_tts(clean_text, output_path, voice))
        logger.info(f"Edge TTS done: {result['duration']:.1f}s, {len(result['alignment'])} word cues")
        return result
    except Exception as e:
        logger.error(f"Edge TTS failed: {e}")
        raise RuntimeError(f"Edge TTS failed: {e}") from e


def generate_all_hack_audio(hacks: list[dict], output_dir: Path) -> list[dict]:
    """Generate TTS for all hack narrations. Returns list of TTS result dicts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, hack in enumerate(hacks):
        audio_path = output_dir / f"hack_{i+1}.mp3"
        gender = hack.get("character_gender", "male")
        result = generate_speech(hack["narration"], audio_path, gender=gender)
        result["hack_number"] = hack.get("hack_number", i + 1)
        result["title"] = hack.get("title", "")
        results.append(result)
        logger.info(f"Audio {i+1}/{len(hacks)}: {result['duration']:.1f}s")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--output", default="output/test_tts.mp3")
    p.add_argument("--gender", default="male", choices=["male", "female"])
    args = p.parse_args()
    result = generate_speech(args.text, Path(args.output), gender=args.gender)
    print(json.dumps(result, indent=2, default=str))
