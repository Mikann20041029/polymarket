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
    url = f"{GAMMA}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "archived": "false",
        "limit": str(limit),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_jsonish(value):
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
    """
    Gammaの clobTokenIds が dict の時もあれば JSON文字列の時もあるので両対応。
    期待：{"YES":"...","NO":"..."} もしくは ["yesToken","noToken"] + outcomes
    """
    parsed = parse_jsonish(m.get("clobTokenIds"))

    if isinstance(parsed, dict):
        keys = {str(k).upper(): v for k, v in parsed.items()}
        yes = keys.get("YES") or parsed.get("Yes") or parsed.get("yes")
        no = keys.get("NO") or parsed.get("No") or parsed.get("no")
        if yes and no:
            return str(yes), str(no)
        return None, None

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
        # outcomesで判別できない場合は先頭2つを返す（表示用）
        return str(parsed[0]), str(parsed[1])

    return None, None

def clob_price(token_id: str) -> str:
    # 一番確実：GET /price?token_id=...
    r = requests.get(f"{CLOB}/price", params={"token_id": token_id}, timeout=30)
    r.raise_for_status()
    j = r.json()
    # {"price":"0.42"} の想定
    p = j.get("price")
    return "" if p is None else str(p)

def main():
    markets = fetch_markets(500)

    picked = []
    found_with_ids = 0

    for m in markets:
        yes_id, no_id = extract_yes_no_token_ids(m)
        if not yes_id or not no_id:
            continue
        found_with_ids += 1
        picked.append((m, yes_id, no_id))
        if len(picked) >= 10:
            break

    rows = []
    displayed = 0

    for (m, yes_id, no_id) in picked:
        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        end_time = m.get("endDate") or m.get("closeTime") or m.get("resolutionTime") or ""

        try:
            yes_px = clob_price(yes_id)
        except Exception:
            yes_px = "ERR"
        try:
            no_px = clob_price(no_id)
        except Exception:
            no_px = "ERR"

        rows.append(
            f"- {q}\n"
            f"  YES: {yes_px} | NO: {no_px}\n"
            f"  slug: {slug}\n"
            f"  time: {end_time}\n"
            f"  yes_token_id: {yes_id}\n"
            f"  no_token_id: {no_id}"
        )
        displayed += 1

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = (
        f"Fetched {len(markets)} markets (active & not-closed).\n"
        f"Found {found_with_ids} markets with token ids, displayed {displayed} with CLOB prices.\n\n"
        + ("\n\n".join(rows) if rows else "(no rows)")
    )

    # ここで例外を投げない。必ずIssueを書く（落ちるのが一番ダメ）
    create_issue(f"[{now}] scan 500 (markets + CLOB prices)", body)

if __name__ == "__main__":
    main()
