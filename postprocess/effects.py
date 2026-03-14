"""
Post-processing with FFmpeg for Impossible Satisfying videos.
- Layers ASMR sound effects onto silent video clips
- Adds optional minimal text overlay (no subtitles needed)
- Adds looping BGM at low volume underneath SFX
- Stitches clips into one final 15-30s vertical short
- Ensures consistent 9:16 format and audio levels
"""
import json
import logging
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


def _layer_sfx_on_clip(
    video_path: str,
    sfx_path: str,
    output_path: str,
) -> str:
    """Layer sound effect onto video clip, trimmed to video duration."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", sfx_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-af", f"volume={config.SFX_VOLUME}",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def _add_text_overlay(
    video_path: str,
    text: str,
    output_path: str,
) -> str:
    """Add minimal text overlay (e.g., '???' or 'wait for it') to video."""
    font_size = config.OVERLAY_FONT_SIZE
    font = config.OVERLAY_FONT

    # Centered, with a subtle shadow for readability
    drawtext = (
        f"drawtext=text='{text}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize={font_size}:fontcolor=white:"
        f"shadowcolor=black@0.6:shadowx=3:shadowy=3:"
        f"x=(w-text_w)/2:y=h*0.85"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", drawtext,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def compose_final_video(
    video_paths: list[str],
    sfx_results: list[dict],
    concepts: list[dict],
    output_path: Path,
    bgm_path: str = None,
) -> str:
    """
    Compose the final satisfying short video:
    1. Layer SFX audio on each video clip
    2. Add optional text overlays
    3. Concatenate clips with brief crossfade
    4. Mix in BGM at low volume
    5. Output final 9:16 mp4
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = config.TEMP_DIR

    # 1. Layer SFX + text overlay on each clip
    processed = []
    sfx_paths = [r["audio_path"] for r in sfx_results]

    for i, (vp, sp) in enumerate(zip(video_paths, sfx_paths)):
        # Layer SFX
        av_out = str(temp / f"av_{i}.mp4")
        _layer_sfx_on_clip(vp, sp, av_out)

        # Add text overlay if specified
        concept = concepts[i] if i < len(concepts) else {}
        text_overlay = concept.get("text_overlay")
        if text_overlay:
            overlay_out = str(temp / f"overlay_{i}.mp4")
            _add_text_overlay(av_out, text_overlay, overlay_out)
            processed.append(overlay_out)
        else:
            processed.append(av_out)

        logger.info(f"Processed clip {i+1}/{len(video_paths)}")

    # 2. Compute total duration
    durations = [_ffprobe_duration(p) for p in processed]
    total_dur = sum(durations)
    logger.info(f"Total: {total_dur:.1f}s across {len(processed)} clips")

    # 3. Concatenate clips
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

    # 4. Add BGM if available
    has_bgm = bgm_path and Path(bgm_path).exists()
    if has_bgm:
        cmd = [
            "ffmpeg", "-y",
            "-i", concat_out,
            "-stream_loop", "-1", "-i", bgm_path,
            "-filter_complex",
            f"[1:a]atrim=0:{total_dur},asetpts=PTS-STARTPTS,"
            f"volume={config.BGM_VOLUME}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", concat_out,
            "-c", "copy",
            str(output_path),
        ]

    logger.info("Composing final video...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"FFmpeg error:\n{r.stderr[-800:]}")
        raise RuntimeError(f"FFmpeg failed:\n{r.stderr[-500:]}")

    logger.info(f"Final video: {output_path} ({total_dur:.1f}s)")
    return str(output_path)
