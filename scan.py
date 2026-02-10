import os
import json
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
    # Gamma: list markets
    url = f"{GAMMA}/markets"
    params = {"active": "true", "closed": "false", "archived": "false", "limit": str(limit)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_jsonish(value):
    """GammaはJSONを文字列で返すことがあるので吸収"""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def extract_yes_no_token_ids(m: dict):
    raw = m.get("clobTokenIds")
    parsed = parse_jsonish(raw)

    # 1) {"YES":"...","NO":"..."} みたいな形
    if isinstance(parsed, dict):
        keys = {str(k).upper(): v for k, v in parsed.items()}
        yes = keys.get("YES")
        no = keys.get("NO")
        if yes and no:
            return str(yes), str(no)

        # {"Yes": "...", "No": "..."} 等も吸収
        yes = parsed.get("Yes") or parsed.get("yes")
        no = parsed.get("No") or parsed.get("no")
        if yes and no:
            return str(yes), str(no)

    # 2) ["tokenA","tokenB"] みたいな形の可能性 → outcomesで対応付け
    if isinstance(parsed, list) and len(parsed) >= 2:
        outcomes = parse_jsonish(m.get("outcomes"))
        if isinstance(outcomes, list) and len(outcomes) >= 2:
            up = [str(x).upper() for x in outcomes]
            try:
                yi = up.index("YES")
                ni = up.index("NO")
                return str(parsed[yi]), str(parsed[ni])
            except Exception:
                pass
        # outcomesで判別できないなら先頭2つを返す（表示用）
        return str(parsed[0]), str(parsed[1])

    return None, None

def fetch_prices_bulk(token_ids):
    # CLOB: POST /prices (BUY側の価格を取得)
    url = f"{CLOB}/prices"
    payload = [{"token_id": tid, "side": "BUY"} for tid in token_ids]
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()  # {token_id: {"BUY": "...", "SELL":"..."}}

def main():
    markets = fetch_markets(500)

    rows = []
    token_ids = []
    meta = []

    found_with_ids = 0
    for m in markets:
        yes_id, no_id = extract_yes_no_token_ids(m)
        if not yes_id or not no_id:
            continue
        found_with_ids += 1
        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        end_time = m.get("endDate") or ""
        meta.append((q, slug, end_time, yes_id, no_id))
        token_ids.extend([yes_id, no_id])

    prices_map = {}
    displayed = 0

    if token_ids:
        # 重複除去してまとめて価格取得
        uniq = list(dict.fromkeys(token_ids))
        prices_map = fetch_prices_bulk(uniq)

        for (q, slug, end_time, yes_id, no_id) in meta[:10]:
            yes_buy = prices_map.get(yes_id, {}).get("BUY")
            no_buy = prices_map.get(no_id, {}).get("BUY")
            prices = f"YES(BUY): {yes_buy} | NO(BUY): {no_buy}"
            rows.append(
                f"- {q}\n"
                f"  {prices}\n"
                f"  slug: {slug}\n"
                f"  time: {end_time}\n"
                f"  yes_token_id: {yes_id}\n"
                f"  no_token_id: {no_id}"
            )
            displayed += 1

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = (
        f"Fetched {len(markets)} markets (active & not-closed).\n"
        f"Found {found_with_ids} markets with clobTokenIds, displayed {displayed} with CLOB prices.\n\n"
        + ("\n\n".join(rows) if rows else "(no rows)")
    )
    create_issue(f"[{now}] scan 500 (markets + CLOB prices)", body)

if __name__ == "__main__":
    main()
