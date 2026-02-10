import os
import datetime
import requests

def create_issue(title: str, body: str):
    repo = os.environ["REPO"]
    token = os.environ["GITHUB_TOKEN"]
    url = f"https://api.github.com/repos/{repo}/issues"

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"title": title, "body": body},
        timeout=30,
    )
    r.raise_for_status()

def main():
    # 500件取得（active=true）
    url = "https://gamma-api.polymarket.com/markets"
    params = {"limit": "500", "active": "true"}

    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    markets = res.json()

    # 先頭5件をIssueに出す
    lines = []
    for m in markets[:5]:
        title_txt = m.get("question") or m.get("title") or "Polymarket Market"
        slug = m.get("slug", "")
        lines.append(f"- {title_txt}\n  slug: {slug}")

    now = datetime.datetime.utcnow().isoformat() + "Z"
    issue_title = f"[{now}] scan 500"
    body = f"Scanned {len(markets)} active markets.\n\n" + "\n".join(lines)

    create_issue(issue_title, body)

if __name__ == "__main__":
    main()
