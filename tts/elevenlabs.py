"""
TTS module using ElevenLabs API.
Generates expressive, high-energy English voice audio.
Estimates word-level timing from audio duration for subtitle sync.
"""
import io
import json
import logging
import re
import argparse
import subprocess
from pathlib import Path
import requests
import config

logger = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1"

# Cache for available voices (fetched once per run)
_available_voices_cache = None


def _get_available_voices(api_key: str) -> list[dict]:
    """Fetch all voices available to this ElevenLabs account."""
    global _available_voices_cache
    if _available_voices_cache is not None:
        return _available_voices_cache

    try:
        resp = requests.get(
            f"{API_URL}/voices",
            headers={"xi-api-key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        voices = resp.json().get("voices", [])
        _available_voices_cache = voices
        logger.info(f"Found {len(voices)} available voices on account")
        for v in voices[:5]:
            labels = v.get("labels", {})
            logger.info(f"  Voice: '{v['name']}' ({v['voice_id']}) gender={labels.get('gender', '?')}")
        return voices
    except Exception as e:
        logger.error(f"Failed to fetch available voices: {e}")
        _available_voices_cache = []
        return []


def _pick_voice(api_key: str, gender: str = "male") -> str | None:
    """Pick a suitable voice from the account's available voices."""
    voices = _get_available_voices(api_key)
    if not voices:
        return None

    # Prefer voices matching the requested gender
    for v in voices:
        labels = v.get("labels", {})
        v_gender = labels.get("gender", "").lower()
        if v_gender == gender:
            logger.info(f"Selected voice: '{v['name']}' ({v['voice_id']}) for gender={gender}")
            return v["voice_id"]

    # No gender match — just use the first available voice
    v = voices[0]
    logger.info(f"Selected voice (no gender match): '{v['name']}' ({v['voice_id']})")
    return v["voice_id"]


def _resolve_voice_id(voice_id: str | None, api_key: str, gender: str) -> str:
    """
    Resolve a working voice ID. Tries in order:
    1. The provided voice_id (if valid)
    2. A voice from the account matching the gender
    3. Any available voice from the account
    Raises ValueError if no voice can be found at all.
    """
    # Try the provided voice ID first
    if voice_id:
        try:
            resp = requests.get(
                f"{API_URL}/voices/{voice_id}",
                headers={"xi-api-key": api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                return voice_id
            logger.warning(f"Voice ID '{voice_id}' returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Voice ID validation failed: {e}")

    # Fall back to account voices
    fallback = _pick_voice(api_key, gender)
    if fallback:
        return fallback

    raise ValueError(
        "No usable voice found. Check ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in secrets."
    )


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
    # Resolve voice ID: try configured → account voices → error
    if not voice_id:
        if gender == "female" and config.ELEVENLABS_VOICE_ID_FEMALE:
            voice_id = config.ELEVENLABS_VOICE_ID_FEMALE
        else:
            voice_id = config.ELEVENLABS_VOICE_ID

    voice_id = _resolve_voice_id(voice_id, config.ELEVENLABS_API_KEY, gender)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip any leftover emotion tags completely (not replacing with dots)
    clean_text = re.sub(r'\[(?:PAUSE|EXCITED|FRUSTRATED|SMUG|PANIC|RELIEF|SERIOUS|HAPPY|[A-Z]+)\]', '', text)
    # Collapse multiple spaces and clean up
    clean_text = re.sub(r'  +', ' ', clean_text).strip()

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

    # Models to try in order (multilingual first, then monolingual as fallback)
    models_to_try = [config.ELEVENLABS_MODEL]
    if "multilingual" in config.ELEVENLABS_MODEL:
        models_to_try.append("eleven_monolingual_v1")
    elif "monolingual" in config.ELEVENLABS_MODEL:
        models_to_try.append("eleven_multilingual_v2")

    alignment = []
    last_error = None

    for model_id in models_to_try:
        payload["model_id"] = model_id
        logger.info(f"Trying TTS model: {model_id}")

        # Try with-timestamps first
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
            logger.info(f"TTS done (with-timestamps, model={model_id}): {duration:.1f}s")

            return {
                "audio_path": str(output_path),
                "alignment": alignment,
                "duration": duration,
            }

        except requests.exceptions.HTTPError as e:
            logger.warning(f"with-timestamps failed ({e}), trying standard TTS...")

        # Fallback: standard TTS endpoint
        try:
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
            logger.info(f"TTS done (standard, model={model_id}): {duration:.1f}s")

            return {
                "audio_path": str(output_path),
                "alignment": alignment,
                "duration": duration,
            }

        except requests.exceptions.HTTPError as e:
            last_error = e
            logger.warning(f"Standard TTS also failed with model={model_id}: {e}")
            continue

    # All attempts failed
    raise RuntimeError(
        f"ElevenLabs TTS failed with all models {models_to_try} and voice {voice_id}. "
        f"Last error: {last_error}. "
        f"Check: 1) ELEVENLABS_API_KEY is valid 2) Account has TTS credits "
        f"3) Subscription plan supports TTS"
    )


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
