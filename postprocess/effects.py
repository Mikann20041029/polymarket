"""
Post-processing with FFmpeg.
- Overlays TTS audio onto video clips
- Burns in word-synced subtitles (center-bottom, large, readable on mobile)
- Adds SFX at transitions
- Adds looping BGM at low volume
- Stitches all mini-hack clips into one final vertical short
"""
import json
import logging
import re
import subprocess
from pathlib import Path
import config

logger = logging.getLogger(__name__)


def _ffprobe_duration(path: str) -> float:
    """Get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def _seconds_to_ass(s: float) -> str:
    """Convert seconds to ASS subtitle timestamp H:MM:SS.cc"""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def _build_ass_file(
    tts_results: list[dict],
    clip_offsets: list[float],
    output_path: Path,
) -> str:
    """
    Build ASS subtitle file with word-group timing synced to speech.
    Subtitles appear center-bottom, large font, with a semi-transparent background box.
    """
    font_size = config.SUBTITLE_FONT_SIZE
    margin_v = config.SUBTITLE_MARGIN_V
    font = config.SUBTITLE_FONT

    header = f"""[Script Info]
Title: Life Hacks
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {config.VIDEO_WIDTH}
PlayResY: {config.VIDEO_HEIGHT}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,3,4,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    words_per_group = 4

    for hack_idx, (tts, offset) in enumerate(zip(tts_results, clip_offsets)):
        alignment = tts.get("alignment", [])
        if not alignment:
            continue

        # Filter: keep only entries that contain at least one letter or digit
        clean_alignment = []
        for w in alignment:
            raw_word = w.get("word", "").strip()
            # Must contain at least one alphanumeric character
            if re.search(r'[a-zA-Z0-9]', raw_word):
                clean_alignment.append(w)

        for i in range(0, len(clean_alignment), words_per_group):
            group = clean_alignment[i : i + words_per_group]
            # Strip leading/trailing punctuation for display
            words = []
            for w in group:
                # Remove surrounding quotes, dots, dashes but keep apostrophes/hyphens inside words
                display = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', w["word"])
                if display:
                    words.append(display)
            if not words:
                continue
            text = " ".join(words)
            t_start = offset + group[0]["start"]
            t_end = offset + group[-1]["end"] + 0.05
            lines.append(
                f"Dialogue: 0,{_seconds_to_ass(t_start)},{_seconds_to_ass(t_end)},"
                f"Default,,0,0,0,,{text}"
            )

    content = header + "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Subtitles written: {output_path} ({len(lines)} groups)")
    return str(output_path)


def _overlay_audio_on_clip(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """
    Loop video clip to match TTS audio duration, then overlay audio.
    Video clips are typically 6-10s but narration can be 20-40s,
    so the video is looped seamlessly to fill the audio length.
    """
    # Get audio duration to know how long video should be
    audio_dur = _ffprobe_duration(audio_path)
    video_dur = _ffprobe_duration(video_path)

    if audio_dur <= video_dur:
        # Audio fits within video — just overlay and trim to audio length
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(audio_dur),
            output_path,
        ]
    else:
        # Audio is longer — loop video to match audio duration
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",  # Loop video infinitely
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(audio_dur),  # Cut to exact audio length
            output_path,
        ]

    logger.info(f"Overlay: video={video_dur:.1f}s, audio={audio_dur:.1f}s → output={audio_dur:.1f}s")
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def compose_final_video(
    video_paths: list[str],
    tts_results: list[dict],
    audio_paths: list[str],
    output_path: Path,
    bgm_path: str = None,
    sfx_transition_path: str = None,
) -> str:
    """
    Compose the final short video:
    1. Overlay TTS audio on each video clip
    2. Concatenate clips
    3. Burn in word-synced subtitles
    4. Mix in BGM + optional transition SFX
    5. Output final 9:16 mp4
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = config.TEMP_DIR

    # 1. Overlay audio
    processed = []
    for i, (vp, ap) in enumerate(zip(video_paths, audio_paths)):
        out = str(temp / f"av_{i}.mp4")
        _overlay_audio_on_clip(vp, ap, out)
        processed.append(out)
        logger.info(f"Audio overlaid on clip {i+1}")

    # 2. Compute subtitle offsets
    durations = [_ffprobe_duration(p) for p in processed]
    offsets = []
    running = 0.0
    for d in durations:
        offsets.append(running)
        running += d
    total_dur = running
    logger.info(f"Total: {total_dur:.1f}s across {len(processed)} clips")

    # 3. Build ASS subtitles
    ass_path = temp / "subs.ass"
    _build_ass_file(tts_results, offsets, ass_path)

    # 4. Concatenate clips
    concat_list = temp / "concat.txt"
    with open(concat_list, "w") as f:
        for p in processed:
            f.write(f"file '{p}'\n")

    concat_out = str(temp / "concat.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", concat_out],
        capture_output=True, check=True,
    )

    # 5. Burn subs + BGM
    inputs = ["-i", concat_out]
    has_bgm = bgm_path and Path(bgm_path).exists()
    if has_bgm:
        inputs.extend(["-stream_loop", "-1", "-i", bgm_path])

    vf = f"[0:v]ass='{ass_path}'[vout]"
    if has_bgm:
        af = (
            f"[1:a]atrim=0:{total_dur},asetpts=PTS-STARTPTS,"
            f"volume={config.BGM_VOLUME}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
    else:
        af = "[0:a]acopy[aout]"

    full_filter = f"{vf};{af}"

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", full_filter,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-r", "25",
        str(output_path),
    ]

    logger.info("Composing final video with subs + audio...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"FFmpeg error:\n{r.stderr[-800:]}")
        raise RuntimeError(f"FFmpeg failed:\n{r.stderr[-500:]}")

    logger.info(f"Final video: {output_path} ({total_dur:.1f}s)")
    return str(output_path)
