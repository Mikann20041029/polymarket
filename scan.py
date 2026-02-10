import os, datetime, requests

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

def main():
    # Public endpoint例：Gamma API(市場一覧)を叩く。取引・キー不要。
    # Polymarketは市場データ取得用に公開API(Gamma)を提供しています。
    # ※もしこのURLが通らなければ、次のメッセージであなたの画面/エラーに合わせて差し替えます。
    url = "https://gamma-api.polymarket.com/markets?limit=20&active=true"
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    markets = res.json()

    # 1つだけ拾ってログ
    m = markets[0]
    title_txt = m.get("question") or m.get("title") or "Polymarket Scan"
    slug = m.get("slug", "")
    now = datetime.datetime.utcnow().isoformat() + "Z"

    title = f"[{now}] scan"
    body = f"Scanned 20 active markets. Example: {title_txt}\n\nslug: {slug}"

    create_issue(title, body)

if __name__ == "__main__":
    main()
