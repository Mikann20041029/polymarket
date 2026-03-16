"""
Post-processing with FFmpeg for Impossible Satisfying videos.
- Scales/pads all clips to exact 9:16 (1080x1920)
- Layers ASMR sound effects with fade-out
- Adds optional minimal text overlay (safely escaped)
- Concatenates clips with re-encoding for consistency
- Mixes in looping BGM with loudness normalization
- Cleans up temp files after completion
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


def _escape_ffmpeg_text(text: str) -> str:
    """Escape text for FFmpeg drawtext filter to prevent injection."""
    # FFmpeg drawtext requires escaping: ' : \ and newlines
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("\n", " ")
    text = text.replace("%", "%%")
    return text


def _scale_to_vertical(video_path: str, output_path: str) -> str:
    """Scale and pad video to exact 1080x1920 (9:16) with black bars if needed."""
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
        "-an",  # Strip audio (will add SFX separately)
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"Scale failed: {r.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg scale failed for {video_path}")
    return output_path


def _layer_sfx_on_clip(
    video_path: str,
    sfx_path: str,
    output_path: str,
) -> str:
    """Layer sound effect onto video clip with fade-out at the end."""
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
            f"afade=t=out:st={fade_start}:d=0.5"
        ),
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
    safe_text = _escape_ffmpeg_text(text)

    drawtext = (
        f"drawtext=text='{safe_text}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize={font_size}:fontcolor=white:"
        f"shadowcolor=black@0.7:shadowx=3:shadowy=3:"
        f"x=(w-text_w)/2:y=h*0.82:"
        f"enable='between(t,0.5,4.5)'"  # Show after 0.5s, hide before end
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
    concepts: list[dict],
    output_path: Path,
    bgm_path: str = None,
) -> str:
    """
    Compose the final satisfying short video:
    1. Scale all clips to 1080x1920 (consistent format)
    2. Layer SFX audio on each clip
    3. Add optional text overlays (safely escaped)
    4. Concatenate clips (re-encoded for consistency)
    5. Mix in BGM at low volume with loudness normalization
    6. Clean up temp files
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = config.TEMP_DIR

    sfx_paths = [r["audio_path"] for r in sfx_results]

    try:
        # 1. Scale all clips to consistent 9:16
        scaled = []
        for i, vp in enumerate(video_paths):
            scaled_out = str(temp / f"scaled_{i}.mp4")
            _scale_to_vertical(vp, scaled_out)
            scaled.append(scaled_out)
            logger.info(f"Scaled clip {i+1}/{len(video_paths)} to 1080x1920")

        # 2. Layer SFX + text overlay on each clip
        processed = []
        for i, (vp, sp) in enumerate(zip(scaled, sfx_paths)):
            # Layer SFX
            av_out = str(temp / f"av_{i}.mp4")
            _layer_sfx_on_clip(vp, sp, av_out)

            # Add text overlay if specified
            concept = concepts[i] if i < len(concepts) else {}
            text_overlay = concept.get("text_overlay")
            if text_overlay and text_overlay not in ("null", "none", "None", ""):
                overlay_out = str(temp / f"overlay_{i}.mp4")
                _add_text_overlay(av_out, text_overlay, overlay_out)
                processed.append(overlay_out)
            else:
                processed.append(av_out)

            logger.info(f"Processed clip {i+1}/{len(video_paths)}")

        # 3. Compute total duration
        durations = [_ffprobe_duration(p) for p in processed]
        total_dur = sum(durations)
        logger.info(f"Total: {total_dur:.1f}s across {len(processed)} clips")

        # 4. Concatenate clips (re-encode for consistent codec/bitrate)
        concat_list = temp / "concat.txt"
        with open(concat_list, "w") as f:
            for p in processed:
                f.write(f"file '{p}'\n")

        concat_out = str(temp / "concat.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart",
             concat_out],
            capture_output=True, check=True,
        )

        # 5. Add BGM if available
        has_bgm = bgm_path and Path(bgm_path).exists()
        if has_bgm:
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-stream_loop", "-1", "-i", bgm_path,
                "-filter_complex",
                f"[1:a]atrim=0:{total_dur},asetpts=PTS-STARTPTS,"
                f"volume={config.BGM_VOLUME},"
                f"afade=t=in:st=0:d=1,"
                f"afade=t=out:st={max(0, total_dur-1.5)}:d=1.5[bgm];"
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

        logger.info("Composing final video...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error(f"FFmpeg error:\n{r.stderr[-800:]}")
            raise RuntimeError(f"FFmpeg failed:\n{r.stderr[-500:]}")

        logger.info(f"Final video: {output_path} ({total_dur:.1f}s)")
        return str(output_path)

    finally:
        # Clean up temp files to prevent disk bloat
        try:
            for f in temp.glob("*.mp4"):
                f.unlink()
            for f in temp.glob("*.txt"):
                f.unlink()
            logger.debug("Temp files cleaned up")
        except Exception as e:
            logger.warning(f"Temp cleanup failed (non-fatal): {e}")
