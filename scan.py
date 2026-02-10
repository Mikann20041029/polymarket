import os
import datetime
import requests

GAMMA = "https://gamma-api.polymarket.com"

def create_issue(title: str, body: str):
    repo = os.environ["REPO"]
    token = os.environ["GITHUB_TOKEN"]
    url = f"https://api.github.com/repos/{repo}/issues"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body},
        timeout=30,
    )
    r.raise_for_status()

def fetch_events(limit: int = 200):
    # 重要：active=true & closed=false（＝いま取引できるイベントに寄せる）
    url = f"{GAMMA}/events"
    params = {
        "active": "true",
        "closed": "false",
        "archived": "false",
        "limit": str(limit),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    events = fetch_events(200)

    # events -> markets を集める（最大500件）
    markets = []
    for e in events:
        for m in e.get("markets", []) or []:
            markets.append(m)
            if len(markets) >= 500:
                break
        if len(markets) >= 500:
            break

    lines = []
    for m in markets[:5]:
        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        end_time = m.get("endDate") or m.get("closeTime") or m.get("resolutionTime") or ""
        lines.append(f"- {q}\n  slug: {slug}\n  time: {end_time}")

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = f"Scanned {len(markets)} markets from active & not-closed events.\n\n" + "\n".join(lines)
    create_issue(f"[{now}] scan 500 (live)", body)

if __name__ == "__main__":
    main()
