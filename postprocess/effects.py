"""
Post-processing for AI World Recreation videos.

Takes AI-generated video clips and composes them into a final short:
1. Scale all clips to consistent 1080x1920 (9:16)
2. Layer ambient SFX on each clip
3. Add optional text overlays
4. Concatenate clips with crossfade transitions
5. Mix in BGM
6. Clean up temp files
"""
import json
import logging
import subprocess
from pathlib import Path
import config

logger = logging.getLogger(__name__)


def _ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def _escape_ffmpeg_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("\n", " ")
    text = text.replace("%", "%%")
    return text


def _scale_to_vertical(video_path: str, output_path: str) -> str:
    """Scale and pad video to exact 1080x1920 with black bars if needed."""
    w, h = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"Scale failed: {r.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg scale failed for {video_path}")
    return output_path


def _layer_sfx_on_clip(video_path: str, sfx_path: str, output_path: str) -> str:
    """Layer ambient SFX onto video clip with fade in/out."""
    duration = _ffprobe_duration(video_path)
    fade_start = max(0, duration - 0.5)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", sfx_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-af", (
            f"volume={config.SFX_VOLUME},"
            f"afade=t=in:st=0:d=0.3,"
            f"afade=t=out:st={fade_start}:d=0.5"
        ),
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def _add_text_overlay(video_path: str, text: str, output_path: str) -> str:
    """Add text overlay (location, year in local language)."""
    safe_text = _escape_ffmpeg_text(text)
    font_size = config.OVERLAY_FONT_SIZE
    duration = _ffprobe_duration(video_path)

    drawtext = (
        f"drawtext=text='{safe_text}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize={font_size}:fontcolor=white@0.85:"
        f"shadowcolor=black@0.6:shadowx=2:shadowy=2:"
        f"x=(w-text_w)/2:y=h*0.88:"
        f"enable='between(t,0.5,{duration - 0.5})'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", drawtext,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def compose_final_video(
    video_paths: list[str],
    sfx_results: list[dict],
    scenes: list[dict],
    output_path: Path,
    bgm_path: str = None,
) -> str:
    """
    Compose final short from AI-generated video clips + SFX + BGM.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = config.TEMP_DIR

    sfx_paths = [r["audio_path"] for r in sfx_results]

    try:
        # 1. Scale + SFX + text overlay per clip
        processed = []
        for i, (vp, sp) in enumerate(zip(video_paths, sfx_paths)):
            scene = scenes[i] if i < len(scenes) else {}

            # Scale to 1080x1920
            scaled = str(temp / f"scaled_{i:02d}.mp4")
            _scale_to_vertical(vp, scaled)

            # Layer SFX
            av = str(temp / f"av_{i:02d}.mp4")
            _layer_sfx_on_clip(scaled, sp, av)

            # Text overlay
            text = scene.get("text_overlay")
            if text and text not in ("null", "none", "None", ""):
                overlay = str(temp / f"overlay_{i:02d}.mp4")
                _add_text_overlay(av, text, overlay)
                processed.append(overlay)
            else:
                processed.append(av)

            logger.info(f"Processed clip {i+1}/{len(video_paths)}")

        # 2. Concatenate with crossfade
        if len(processed) == 1:
            concat_out = processed[0]
        else:
            concat_out = str(temp / "concat.mp4")
            _concatenate_with_crossfade(processed, concat_out)

        total_dur = _ffprobe_duration(concat_out)
        logger.info(f"Total duration: {total_dur:.1f}s")

        # 3. Add BGM
        has_bgm = bgm_path and Path(bgm_path).exists()
        if has_bgm:
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-stream_loop", "-1", "-i", bgm_path,
                "-filter_complex",
                f"[1:a]atrim=0:{total_dur},asetpts=PTS-STARTPTS,"
                f"volume={config.BGM_VOLUME},"
                f"afade=t=in:st=0:d=1.5,"
                f"afade=t=out:st={max(0, total_dur-2)}:d=2[bgm];"
                f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error(f"FFmpeg error:\n{r.stderr[-800:]}")
            raise RuntimeError("FFmpeg final compose failed")

        logger.info(f"Final video: {output_path} ({total_dur:.1f}s)")
        return str(output_path)

    finally:
        try:
            for f in temp.glob("*.mp4"):
                f.unlink()
            for f in temp.glob("*.txt"):
                f.unlink()
        except Exception as e:
            logger.warning(f"Temp cleanup: {e}")


def _concatenate_with_crossfade(clips: list[str], output_path: str) -> str:
    """Concatenate clips with crossfade transitions."""
    fade_dur = config.CROSSFADE_DURATION

    if len(clips) == 2:
        dur0 = _ffprobe_duration(clips[0])
        offset = dur0 - fade_dur
        cmd = [
            "ffmpeg", "-y",
            "-i", clips[0], "-i", clips[1],
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={fade_dur}:offset={offset}[v];"
            f"[0:a][1:a]acrossfade=d={fade_dur}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    # Multiple clips: chain xfade
    inputs = []
    for clip in clips:
        inputs.extend(["-i", clip])

    durations = [_ffprobe_duration(c) for c in clips]

    v_filters = []
    a_filters = []
    cumulative_offset = 0

    for i in range(len(clips) - 1):
        cumulative_offset += durations[i] - fade_dur

        v_in = f"[0:v][1:v]" if i == 0 else f"[vfade{i}][{i+1}:v]"
        a_in = f"[0:a][1:a]" if i == 0 else f"[afade{i}][{i+1}:a]"
        v_out = "[v]" if i == len(clips) - 2 else f"[vfade{i+1}]"
        a_out = "[a]" if i == len(clips) - 2 else f"[afade{i+1}]"

        v_filters.append(
            f"{v_in}xfade=transition=fade:duration={fade_dur}:offset={cumulative_offset}{v_out}"
        )
        a_filters.append(f"{a_in}acrossfade=d={fade_dur}{a_out}")

    filter_complex = ";".join(v_filters + a_filters)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
    )

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.warning("Crossfade failed, falling back to simple concat")
        return _simple_concat(clips, output_path)

    return output_path


def _simple_concat(clips: list[str], output_path: str) -> str:
    """Fallback: simple concatenation without crossfade."""
    temp = config.TEMP_DIR
    concat_list = temp / "concat.txt"
    with open(concat_list, "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_list),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart",
         output_path],
        capture_output=True, check=True,
    )
    return output_path
