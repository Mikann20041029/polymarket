"""
YouTube Shorts auto-upload via YouTube Data API v3.

Requires OAuth2 credentials (one-time setup via setup_youtube_auth.py).
Secrets needed: YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN
"""
import os
import logging
import json
import random
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Hashtags that boost Shorts discovery
HASHTAGS = [
    "#construction", "#timelapse", "#renovation", "#luxuryhome",
    "#beforeandafter", "#diy", "#homedesign", "#architecture",
    "#satisfying", "#transformation", "#shorts",
]


def _get_access_token() -> str:
    """Exchange refresh token for a fresh access token."""
    client_id = os.getenv("YOUTUBE_CLIENT_ID", "")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN", "")

    if not all([client_id, client_secret, refresh_token]):
        missing = []
        if not client_id:
            missing.append("YOUTUBE_CLIENT_ID")
        if not client_secret:
            missing.append("YOUTUBE_CLIENT_SECRET")
        if not refresh_token:
            missing.append("YOUTUBE_REFRESH_TOKEN")
        raise RuntimeError(f"Missing YouTube secrets: {', '.join(missing)}")

    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(f"YouTube token refresh failed: {resp.status_code} {resp.text}")

    return resp.json()["access_token"]


def _build_title(scenario: dict) -> str:
    """Build a short, catchy title for the YouTube Short."""
    concept = scenario.get("one_line_concept", "")
    if concept:
        # Trim to 90 chars max (YouTube limit is 100, leave room for emoji)
        title = concept[:90]
        return title
    return "Construction Timelapse Transformation"


def _build_description(scenario: dict) -> str:
    """Build description with hashtags for SEO."""
    concept = scenario.get("one_line_concept", "")
    category = scenario.get("category", "construction")
    before = scenario.get("before_space", {}).get("description", "")
    after = scenario.get("after_space", {}).get("description", "")

    lines = []
    if concept:
        lines.append(concept)
    if before and after:
        lines.append(f"Before: {before}")
        lines.append(f"After: {after}")
    lines.append("")

    # Add 5-8 random hashtags
    tags = random.sample(HASHTAGS, min(8, len(HASHTAGS)))
    if f"#{category.replace('_', '')}" not in tags:
        tags.append(f"#{category.replace('_', '')}")
    lines.append(" ".join(tags))

    return "\n".join(lines)


def _build_tags(scenario: dict) -> list[str]:
    """Build keyword tags for the video."""
    base_tags = [
        "construction timelapse", "renovation", "home transformation",
        "luxury home", "before and after", "satisfying",
        "construction", "timelapse", "shorts",
    ]
    category = scenario.get("category", "")
    if category:
        base_tags.append(category.replace("_", " "))

    sim_tags = scenario.get("similarity_tags", [])
    base_tags.extend(sim_tags[:5])

    # YouTube allows max 500 chars total for tags
    return base_tags[:15]


def upload_to_youtube(
    video_path: str,
    scenario: dict,
    privacy: str = "public",
) -> dict:
    """
    Upload a video to YouTube as a Short.

    Args:
        video_path: Path to the .mp4 file
        scenario: The scenario dict with metadata
        privacy: "public", "unlisted", or "private"

    Returns:
        dict with video_id, url, title
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    file_size = video_path.stat().st_size
    if file_size < 10_000:
        raise RuntimeError(f"Video too small ({file_size} bytes), likely corrupted")

    title = _build_title(scenario)
    description = _build_description(scenario)
    tags = _build_tags(scenario)

    logger.info(f"Uploading to YouTube: '{title}' ({file_size / 1024 / 1024:.1f} MB)")

    access_token = _get_access_token()

    # Metadata for the upload
    metadata = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
            "shorts": {"shortsEligibility": "eligible"},
        },
    }

    # Resumable upload: init
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size),
    }

    init_resp = requests.post(
        f"{UPLOAD_URL}?uploadType=resumable&part=snippet,status",
        headers=headers,
        json=metadata,
        timeout=30,
    )

    if init_resp.status_code not in (200, 308):
        raise RuntimeError(
            f"YouTube upload init failed: {init_resp.status_code} {init_resp.text}"
        )

    upload_url = init_resp.headers.get("Location")
    if not upload_url:
        raise RuntimeError("No upload URL returned from YouTube")

    # Upload the video file
    with open(video_path, "rb") as f:
        upload_resp = requests.put(
            upload_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
            },
            data=f,
            timeout=600,
        )

    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"YouTube upload failed: {upload_resp.status_code} {upload_resp.text}"
        )

    result = upload_resp.json()
    video_id = result.get("id", "")
    url = f"https://youtube.com/shorts/{video_id}" if video_id else ""

    logger.info(f"YouTube upload complete: {url}")

    return {
        "video_id": video_id,
        "url": url,
        "title": title,
        "privacy": privacy,
    }
