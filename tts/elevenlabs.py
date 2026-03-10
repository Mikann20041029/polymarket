"""
TTS module using ElevenLabs API.
Generates expressive, high-energy English voice audio.
Estimates word-level timing from audio duration for subtitle sync.
"""
import io
import json
import logging
import argparse
import subprocess
from pathlib import Path
import requests
import config

logger = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1"


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


def generate_speech(text: str, output_path: Path, voice_id: str = None,
                    gender: str = "male") -> dict:
    """
    Generate expressive speech audio.

    Tries /with-timestamps first; falls back to standard TTS + estimated timing.

    Args:
        text: Text to speak
        output_path: Where to save audio
        voice_id: Override voice ID (if None, picks based on gender)
        gender: "male" or "female" — selects the appropriate voice

    Returns dict with:
        - audio_path: path to mp3 file
        - alignment: list of {word, start, end} for subtitle sync
        - duration: total audio duration in seconds
    """
    if not voice_id:
        if gender == "female" and config.ELEVENLABS_VOICE_ID_FEMALE:
            voice_id = config.ELEVENLABS_VOICE_ID_FEMALE
        else:
            voice_id = config.ELEVENLABS_VOICE_ID
    if not voice_id:
        raise ValueError("ELEVENLABS_VOICE_ID not set in .env")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip emotion tags for TTS
    clean_text = text
    for tag in ["[PAUSE]", "[EXCITED]", "[FRUSTRATED]", "[SMUG]",
                "[PANIC]", "[RELIEF]", "[SERIOUS]", "[HAPPY]"]:
        clean_text = clean_text.replace(tag, "...")
    # Clean up multiple dots
    while "......" in clean_text:
        clean_text = clean_text.replace("......", "...")
    clean_text = clean_text.strip()

    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": clean_text,
        "model_id": config.ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": config.TTS_STABILITY,
            "similarity_boost": config.TTS_SIMILARITY_BOOST,
            "style": config.TTS_STYLE,
            "use_speaker_boost": config.TTS_USE_SPEAKER_BOOST,
        },
    }

    logger.info(f"TTS ({gender}): '{clean_text[:60]}...'")

    # Try with-timestamps first
    alignment = []
    try:
        url = f"{API_URL}/text-to-speech/{voice_id}/with-timestamps"
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        import base64
        audio_bytes = base64.b64decode(data["audio_base64"])
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        alignment = _parse_alignment(data.get("alignment", {}))
        duration = alignment[-1]["end"] if alignment else _get_audio_duration(str(output_path))
        logger.info(f"TTS done (with-timestamps): {duration:.1f}s")

    except requests.exceptions.HTTPError as e:
        logger.warning(f"with-timestamps failed ({e}), using standard TTS...")

        # Fallback: standard TTS endpoint (returns raw audio bytes)
        url = f"{API_URL}/text-to-speech/{voice_id}"
        headers_stream = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        resp = requests.post(url, headers=headers_stream, json=payload, timeout=60)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(resp.content)

        duration = _get_audio_duration(str(output_path))
        alignment = _estimate_word_timing(clean_text, duration)
        logger.info(f"TTS done (standard + estimated timing): {duration:.1f}s")

    return {
        "audio_path": str(output_path),
        "alignment": alignment,
        "duration": duration,
    }


def _parse_alignment(raw_alignment: dict) -> list[dict]:
    """Parse ElevenLabs character-level alignment into word-level alignment."""
    chars = raw_alignment.get("characters", [])
    starts = raw_alignment.get("character_start_times_seconds", [])
    ends = raw_alignment.get("character_end_times_seconds", [])

    if not chars:
        return []

    words = []
    current_word = ""
    word_start = None
    word_end = None

    for i, ch in enumerate(chars):
        if ch == " ":
            if current_word:
                words.append({"word": current_word, "start": word_start, "end": word_end})
                current_word = ""
                word_start = None
        else:
            if word_start is None:
                word_start = starts[i] if i < len(starts) else 0
            word_end = ends[i] if i < len(ends) else word_start
            current_word += ch

    if current_word:
        words.append({"word": current_word, "start": word_start, "end": word_end})

    return words


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
    args = p.parse_args()
    result = generate_speech(args.text, Path(args.output))
    print(json.dumps(result, indent=2, default=str))
