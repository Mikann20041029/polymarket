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
    # markets を直接。ここに clobTokenIds が入りやすい
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

def get_yes_no_token_ids(m: dict):
    ct = m.get("clobTokenIds")
    if not isinstance(ct, dict):
        return None, None
    keys = {str(k).upper(): v for k, v in ct.items()}
    yes = keys.get("YES") or keys.get("Yes")
    no = keys.get("NO") or keys.get("No")
    if yes and no:
        return str(yes), str(no)
    return None, None

def clob_price(token_id: str) -> str:
    r = requests.get(f"{CLOB}/price", params={"token_id": token_id}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return str(j.get("price", ""))

def main():
    markets = fetch_markets(500)

    lines = []
    shown = 0
    with_token = 0

    for m in markets:
        if shown >= 10:
            break

        yes_id, no_id = get_yes_no_token_ids(m)
        if not yes_id or not no_id:
            continue

        with_token += 1
        q = m.get("question") or m.get("title") or "(no title)"
        slug = m.get("slug", "")
        end_time = m.get("endDate") or m.get("closeTime") or m.get("resolutionTime") or ""

        try:
            yes_px = clob_price(yes_id)
            no_px = clob_price(no_id)
            prices = f"YES: {yes_px} | NO: {no_px}"
        except Exception:
            prices = "(price fetch failed)"

        lines.append(
            f"- {q}\n"
            f"  {prices}\n"
            f"  slug: {slug}\n"
            f"  time: {end_time}"
        )
        shown += 1

    now = datetime.datetime.utcnow().isoformat() + "Z"
    body = (
        f"Fetched {len(markets)} markets (active & not-closed).\n"
        f"Found {with_token} markets with clobTokenIds, displayed {shown} with CLOB prices.\n\n"
        + "\n\n".join(lines)
    )
    create_issue(f"[{now}] scan 500 (markets + CLOB prices)", body)

if __name__ == "__main__":
    main()
