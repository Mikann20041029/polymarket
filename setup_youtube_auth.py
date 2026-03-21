#!/usr/bin/env python3
"""
One-time setup: Get YouTube OAuth2 refresh token.

Run this ONCE on your local machine:
    python setup_youtube_auth.py

Prerequisites:
    1. Go to https://console.cloud.google.com/
    2. Create a project (or use existing)
    3. Enable "YouTube Data API v3"
    4. Go to Credentials → Create OAuth 2.0 Client ID (Desktop app)
    5. Download the JSON → save as client_secret.json in this directory

After running this script, it will print the REFRESH_TOKEN.
Add these to GitHub Secrets:
    - YOUTUBE_CLIENT_ID
    - YOUTUBE_CLIENT_SECRET
    - YOUTUBE_REFRESH_TOKEN
"""
import json
import sys
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("Install: pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET_FILE = Path(__file__).parent / "client_secret.json"


def main():
    if not CLIENT_SECRET_FILE.exists():
        print(f"ERROR: {CLIENT_SECRET_FILE} not found.")
        print()
        print("Steps:")
        print("  1. Go to https://console.cloud.google.com/apis/credentials")
        print("  2. Create OAuth 2.0 Client ID (type: Desktop)")
        print("  3. Download JSON → save as client_secret.json here")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE), SCOPES
    )
    credentials = flow.run_local_server(port=8080)

    # Read client_id and client_secret from the file
    with open(CLIENT_SECRET_FILE) as f:
        client_data = json.load(f)

    # Handle both "installed" and "web" credential types
    key = "installed" if "installed" in client_data else "web"
    client_id = client_data[key]["client_id"]
    client_secret = client_data[key]["client_secret"]

    print()
    print("=" * 60)
    print("  YouTube OAuth Setup Complete!")
    print("=" * 60)
    print()
    print("Add these to GitHub Secrets:")
    print(f"  YOUTUBE_CLIENT_ID     = {client_id}")
    print(f"  YOUTUBE_CLIENT_SECRET = {client_secret}")
    print(f"  YOUTUBE_REFRESH_TOKEN = {credentials.refresh_token}")
    print()
    print("GitHub Secrets URL:")
    print("  https://github.com/Mikann20041029/polymarket/settings/secrets/actions")
    print("=" * 60)


if __name__ == "__main__":
    main()
