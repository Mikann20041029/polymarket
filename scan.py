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
    # live寄せ：active=true & closed=false
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

def fmt_prices(m: dict) -> str:
    # Gammaのmarketsには outcomePrices が入ってることが多い（["0.42","0.58"]みたいな文字列配列）
    op = m.get("outcomePrices")
    outcomes = m.get("outcomes")

    if not op or not isinstance(op, list) or len(op) < 2:
        return "(no prices)"

    # outcomesがあればYES/NOを確実に対応させる
    if isinstance(outcomes, list) and len(outcomes) == len(op):
        pairs = list(zip(outcomes, op))
        # YES/NOがあるならそれを優先表示
        d = {str(k).upper(): v for k, v in pairs}
        if "YES" in d and "NO" in d:
            return f"YES: {d['YES']} | NO: {d['NO']}"
        # それ以外は全部並べる
        return " | ".join([f"{k}: {v}" for k, v in pairs])

    # outcomesが無い/ズレてる場合は先頭2つをYES/NOっぽく扱う
    return f"YES?: {op[0]} | NO?: {op[1]}"

def main():
    events = fetch_events(200)

    markets = []
    for e in events:
        for m in e.get("markets", []) or []:
            markets.append(m)
            if len(markets) >= 500:
                break
        if len(markets) >= 500:
            break

    lines = []
    for m in markets[:10]:  # 10件表示（多すぎるとIssueが読みにくい）
        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        end_time = m.get("endDate") or m.get("closeTime") or m.get("resolutionTime") or ""
        prices = fmt_prices(m)
        lines.append(
            f"- {q}\n"
            f"  {prices}\n"
            f"  slug: {slug}\n"
            f"  time: {end_time}"
        )

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = (
        f"Scanned {len(markets)} markets from active & not-closed events.\n"
        f"Showing first {min(10, len(markets))} with YES/NO prices when available.\n\n"
        + "\n\n".join(lines)
    )
    create_issue(f"[{now}] scan 500 (live + prices)", body)

if __name__ == "__main__":
    main()
