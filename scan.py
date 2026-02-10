import os
import datetime
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"  # 価格はここから取る

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
    url = f"{GAMMA}/events"
    params = {"active": "true", "closed": "false", "archived": "false", "limit": str(limit)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def clob_price(token_id: str) -> str:
    # Public: /price?token_id=...
    r = requests.get(f"{CLOB}/price", params={"token_id": token_id}, timeout=30)
    r.raise_for_status()
    j = r.json()
    # {"price":"0.42"} みたいな形式想定
    return str(j.get("price", ""))

def get_yes_no_token_ids(m: dict):
    # Gammaのmarketに clobTokenIds が入ってることがある
    # 例: {"Yes": "...token...", "No": "...token..."} or {"YES":..., "NO":...}
    ct = m.get("clobTokenIds")
    if not ct:
        return None, None

    if isinstance(ct, dict):
        # キー揺れ吸収
        keys = {str(k).upper(): v for k, v in ct.items()}
        yes = keys.get("YES")
        no = keys.get("NO")
        if yes and no:
            return str(yes), str(no)

    # たまに配列の可能性もあるので最低限
    return None, None

def main():
    events = fetch_events(200)

    markets = []
    for e in events:
        for m in (e.get("markets") or []):
            markets.append(m)
            if len(markets) >= 500:
                break
        if len(markets) >= 500:
            break

    # 先頭10件だけ、YES/NO価格をCLOBから引いて表示
    lines = []
    shown = 0
    tried = 0

    for m in markets:
        if shown >= 10:
            break
        tried += 1

        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        end_time = m.get("endDate") or m.get("closeTime") or m.get("resolutionTime") or ""

        yes_id, no_id = get_yes_no_token_ids(m)
        if not yes_id or not no_id:
            # token idが無い市場は飛ばす
            continue

        try:
            yes_px = clob_price(yes_id)
            no_px = clob_price(no_id)
            prices = f"YES: {yes_px} | NO: {no_px}"
        except Exception as ex:
            prices = f"(price fetch failed)"

        lines.append(
            f"- {q}\n"
            f"  {prices}\n"
            f"  slug: {slug}\n"
            f"  time: {end_time}"
        )
        shown += 1

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = (
        f"Scanned {len(markets)} markets from active & not-closed events.\n"
        f"Displayed {shown} markets with CLOB YES/NO prices (tried {tried}).\n\n"
        + "\n\n".join(lines)
    )
    create_issue(f"[{now}] scan 500 (live + CLOB prices)", body)

if __name__ == "__main__":
    main()
