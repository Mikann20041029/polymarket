"""
Post-processing for AI World Recreation videos.

Takes photorealistic images and composes them into a cinematic short:
1. Ken Burns effect on each image (zoom/pan creating motion from stills)
2. Crossfade transitions between scenes
3. Layer ambient SFX per scene
4. Mix in low-volume BGM
5. Optional text overlays (year, location)
6. Output 9:16 vertical, 40-60 seconds
"""
import json
import logging
import subprocess
from pathlib import Path
import config

logger = logging.getLogger(__name__)

# Ken Burns presets: FFmpeg zoompan filter parameters
# Each produces different camera motion from a still image
CAMERA_MOVEMENTS = {
    "zoom_in": {
        # Slow zoom into center
        "z": "min(zoom+0.001,1.15)",
        "x": "iw/2-(iw/zoom/2)",
        "y": "ih/2-(ih/zoom/2)",
    },
    "zoom_out": {
        # Start zoomed in, slowly zoom out
        "z": "if(eq(on,1),1.15,max(zoom-0.001,1.0))",
        "x": "iw/2-(iw/zoom/2)",
        "y": "ih/2-(ih/zoom/2)",
    },
    "pan_left": {
        # Pan from right to left
        "z": "1.10",
        "x": "iw*0.10*(1-on/(duration*fps))",
        "y": "ih/2-(ih/zoom/2)",
    },
    "pan_right": {
        # Pan from left to right
        "z": "1.10",
        "x": "iw*0.10*(on/(duration*fps))",
        "y": "ih/2-(ih/zoom/2)",
    },
    "pan_up": {
        # Pan from bottom to top
        "z": "1.10",
        "x": "iw/2-(iw/zoom/2)",
        "y": "ih*0.10*(1-on/(duration*fps))",
    },
    "pan_down": {
        # Pan from top to bottom
        "z": "1.10",
        "x": "iw/2-(iw/zoom/2)",
        "y": "ih*0.10*(on/(duration*fps))",
    },
}


def _ffprobe_duration(path: str) -> float:
    """Get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def _escape_ffmpeg_text(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("\n", " ")
    text = text.replace("%", "%%")
    return text


def _image_to_clip(
    image_path: str,
    output_path: str,
    camera_movement: str = "zoom_in",
    duration: float = None,
) -> str:
    """
    Convert a still image into a video clip with Ken Burns camera movement.

    Args:
        image_path: Path to input image
        output_path: Where to save the clip
        camera_movement: Type of camera motion
        duration: Clip duration in seconds
    """
    duration = duration or config.SECONDS_PER_SCENE
    fps = 30
    total_frames = int(duration * fps)
    w, h = config.VIDEO_WIDTH, config.VIDEO_HEIGHT

    movement = CAMERA_MOVEMENTS.get(camera_movement, CAMERA_MOVEMENTS["zoom_in"])

    # Replace 'duration' placeholder in pan formulas
    z = movement["z"]
    x = movement["x"].replace("duration", str(duration))
    y = movement["y"].replace("duration", str(duration))

    zoompan = (
        f"zoompan=z='{z}':x='{x}':y='{y}'"
        f":d={total_frames}:s={w}x{h}:fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", zoompan,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"Ken Burns failed: {r.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg Ken Burns failed for {image_path}")

    return output_path


def _add_text_overlay(
    video_path: str,
    text: str,
    output_path: str,
) -> str:
    """Add subtle text overlay (location, year) to clip."""
    safe_text = _escape_ffmpeg_text(text)
    font_size = config.OVERLAY_FONT_SIZE

    drawtext = (
        f"drawtext=text='{safe_text}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize={font_size}:fontcolor=white@0.85:"
        f"shadowcolor=black@0.6:shadowx=2:shadowy=2:"
        f"x=(w-text_w)/2:y=h*0.88:"
        f"enable='between(t,0.5,{config.SECONDS_PER_SCENE - 0.5})'"
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


def _layer_sfx_on_clip(
    video_path: str,
    sfx_path: str,
    output_path: str,
) -> str:
    """Layer ambient SFX onto a video clip."""
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


def compose_final_video(
    image_paths: list[str],
    sfx_results: list[dict],
    scenes: list[dict],
    output_path: Path,
    bgm_path: str = None,
) -> str:
    """
    Compose the final short video from images + SFX + BGM.

    1. Each image → Ken Burns clip (6 seconds)
    2. Layer SFX on each clip
    3. Add text overlays where specified
    4. Concatenate with crossfade transitions
    5. Mix in BGM
    6. Output final 9:16 video
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = config.TEMP_DIR

    sfx_paths = [r["audio_path"] for r in sfx_results]

    try:
        # 1. Convert images to Ken Burns clips + layer SFX + text overlay
        processed = []
        for i, (img_path, sfx_path) in enumerate(zip(image_paths, sfx_paths)):
            scene = scenes[i] if i < len(scenes) else {}
            movement = scene.get("camera_movement", "zoom_in")

            # Image → Ken Burns clip
            clip_path = str(temp / f"clip_{i:02d}.mp4")
            _image_to_clip(img_path, clip_path, camera_movement=movement)

            # Layer SFX
            av_path = str(temp / f"av_{i:02d}.mp4")
            _layer_sfx_on_clip(clip_path, sfx_path, av_path)

            # Text overlay
            text = scene.get("text_overlay")
            if text and text not in ("null", "none", "None", ""):
                overlay_path = str(temp / f"overlay_{i:02d}.mp4")
                _add_text_overlay(av_path, text, overlay_path)
                processed.append(overlay_path)
            else:
                processed.append(av_path)

            logger.info(f"Processed scene {i+1}/{len(image_paths)}: {movement}")

        # 2. Concatenate with crossfade transitions
        if len(processed) == 1:
            concat_out = processed[0]
        else:
            concat_out = str(temp / "final_concat.mp4")
            _concatenate_with_crossfade(processed, concat_out)

        # 3. Compute total duration
        total_dur = _ffprobe_duration(concat_out)
        logger.info(f"Total duration: {total_dur:.1f}s")

        # 4. Add BGM
        has_bgm = bgm_path and Path(bgm_path).exists()
        if has_bgm:
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-stream_loop", "-1", "-i", bgm_path,
                "-filter_complex",
                f"[1:a]atrim=0:{total_dur},asetpts=PTS-STARTPTS,"
                f"volume={config.BGM_VOLUME},"
                f"afade=t=in:st=0:d=2,"
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
            raise RuntimeError(f"FFmpeg final compose failed")

        logger.info(f"Final video: {output_path} ({total_dur:.1f}s)")
        return str(output_path)

    finally:
        # Cleanup temp files
        try:
            for f in temp.glob("*.mp4"):
                f.unlink()
            for f in temp.glob("*.txt"):
                f.unlink()
        except Exception as e:
            logger.warning(f"Temp cleanup: {e}")


def _concatenate_with_crossfade(clips: list[str], output_path: str) -> str:
    """
    Concatenate clips with crossfade transitions using xfade filter.
    Each transition is CROSSFADE_DURATION seconds.
    """
    fade_dur = config.CROSSFADE_DURATION

    if len(clips) == 2:
        # Simple case: two clips
        dur0 = _ffprobe_duration(clips[0])
        offset = dur0 - fade_dur
        cmd = [
            "ffmpeg", "-y",
            "-i", clips[0],
            "-i", clips[1],
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

    # Multiple clips: chain xfade filters
    # Build complex filter graph
    inputs = []
    for clip in clips:
        inputs.extend(["-i", clip])

    durations = [_ffprobe_duration(c) for c in clips]

    # Video xfade chain
    v_filters = []
    a_filters = []
    cumulative_offset = 0

    for i in range(len(clips) - 1):
        cumulative_offset += durations[i] - fade_dur

        if i == 0:
            v_in = f"[0:v][1:v]"
            a_in = f"[0:a][1:a]"
        else:
            v_in = f"[vfade{i}][{i+1}:v]"
            a_in = f"[afade{i}][{i+1}:a]"

        if i == len(clips) - 2:
            v_out = "[v]"
            a_out = "[a]"
        else:
            v_out = f"[vfade{i+1}]"
            a_out = f"[afade{i+1}]"

        v_filters.append(
            f"{v_in}xfade=transition=fade:duration={fade_dur}:offset={cumulative_offset}{v_out}"
        )
        a_filters.append(
            f"{a_in}acrossfade=d={fade_dur}{a_out}"
        )

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
        logger.error(f"Crossfade concat failed: {r.stderr[-800:]}")
        # Fallback: simple concat without crossfade
        logger.warning("Falling back to simple concatenation")
        return _simple_concat(clips, output_path)

    return output_path


def _simple_concat(clips: list[str], output_path: str) -> str:
    """Simple concatenation without crossfade (fallback)."""
    temp = config.TEMP_DIR
    concat_list = temp / "concat_fallback.txt"
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
