import os
import datetime
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

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

def fetch_markets(limit: int = 500):
    # enableOrderBook=true の市場だけ拾う（CLOBで取引できる条件）:contentReference[oaicite:3]{index=3}
    r = requests.get(
        f"{GAMMA}/markets",
        params={"active": "true", "closed": "false", "archived": "false", "limit": str(limit)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def parse_token_ids(m: dict):
    # Gammaの clobTokenIds は文字列JSONのことがあるので両対応
    v = m.get("clobTokenIds")
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            import json
            v = json.loads(v)
        except Exception:
            return None

    # dict: {"YES":"...","NO":"..."} 想定
    if isinstance(v, dict):
        yes = v.get("YES") or v.get("Yes") or v.get("yes")
        no  = v.get("NO")  or v.get("No")  or v.get("no")
        if yes and no:
            return str(yes), str(no)

    # list: outcomes と突き合わせ（無理なら先頭2つ）
    if isinstance(v, list) and len(v) >= 2:
        outs = m.get("outcomes")
        if isinstance(outs, str):
            try:
                import json
                outs = json.loads(outs)
            except Exception:
                outs = None
        if isinstance(outs, list) and len(outs) >= 2:
            up = [str(x).upper() for x in outs]
            if "YES" in up and "NO" in up:
                yi = up.index("YES")
                ni = up.index("NO")
                return str(v[yi]), str(v[ni])
        return str(v[0]), str(v[1])

    return None

def get_price(token_id: str, side: str) -> str:
    # 公開API: GET /price?token_id=...&side=BUY|SELL :contentReference[oaicite:4]{index=4}
    r = requests.get(f"{CLOB}/price", params={"token_id": token_id, "side": side}, timeout=30)
    if r.status_code != 200:
        return f"ERR({r.status_code})"
    j = r.json()
    return str(j.get("price", ""))

def main():
    markets = fetch_markets(500)

    picked = []
    for m in markets:
        if not m.get("enableOrderBook", False):
            continue
        ids = parse_token_ids(m)
        if not ids:
            continue
        picked.append((m, ids[0], ids[1]))
        if len(picked) >= 10:
            break

    lines = []
    for m, yes_id, no_id in picked:
        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        yes_px = get_price(yes_id, "BUY")
        no_px  = get_price(no_id, "BUY")
        lines.append(
            f"- {q}\n"
            f"  YES(BUY): {yes_px} | NO(BUY): {no_px}\n"
            f"  slug: {slug}\n"
            f"  yes_token_id: {yes_id}\n"
            f"  no_token_id: {no_id}"
        )

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = (
        f"Scanned {len(markets)} markets.\n"
        f"Displayed {len(picked)} with enableOrderBook + YES/NO token ids.\n\n"
        + ("\n\n".join(lines) if lines else "(no rows)")
    )
    create_issue(f"[{now}] scan 500 (gamma + public CLOB price)", body)

if __name__ == "__main__":
    main()
