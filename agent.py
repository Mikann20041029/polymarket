import os
import json
import time
import math
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA = "https://gamma-api.polymarket.com"

EDGE_MIN = 0.08
KELLY_MAX = 0.06

def env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"missing env: {k}")
    return v

def is_hex_bytes(s: str, n_bytes: int) -> bool:
    s = s.strip()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) != n_bytes * 2:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False

def gh_issue(title: str, body: str):
    token = env("GITHUB_TOKEN")
    repo = env("GITHUB_REPOSITORY")
    r = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body},
        timeout=30,
    )
    r.raise_for_status()

def gamma_markets(limit: int):
    # Gamma markets API（activeのみ）
    r = requests.get(
        f"{GAMMA}/markets",
        params={"active": "true", "closed": "false", "archived": "false", "limit": str(limit)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def extract_yes_token_ids(markets, max_tokens: int):
    token_ids = []
    picked = []  # (token_id, title)
    for m in markets:
        if len(token_ids) >= max_tokens:
            break
        if not m.get("enableOrderBook", False):
            continue

        v = m.get("clobTokenIds")
        if v is None:
            continue

        # clobTokenIdsが文字列JSONのケースを吸収
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                continue

        yes = None
        if isinstance(v, dict):
            yes = v.get("YES") or v.get("Yes") or v.get("yes")
        elif isinstance(v, list) and len(v) > 0:
            yes = v[0]

        if yes:
            tid = str(yes)
            token_ids.append(tid)
            picked.append((tid, str(m.get("question") or m.get("title") or "unknown")))
    return token_ids, picked

def clob_prices(token_ids):
    # まとめて価格取得（BUY/SELL）: /prices :contentReference[oaicite:4]{index=4}
    # 返り値: { token_id: { "BUY": "...", "SELL": "..." }, ... }
    r = requests.post(
        f"{HOST}/prices",
        json={"token_ids": token_ids, "sides": ["BUY", "SELL"]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def claude_fair_prob(title: str, yes_buy: float, yes_sell: float) -> float:
    # Anthropic Messages API（最小）
    # 必要env: CLAUDE_API_KEY, CLAUDE_MODEL
    api_key = env("CLAUDE_API_KEY")
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")

    prompt = (
        "You estimate fair probability for a Polymarket YES/NO event.\n"
        "Return ONLY a number between 0 and 1 as fair YES probability.\n"
        f"Title: {title}\n"
        f"Current YES BUY price: {yes_buy}\n"
        f"Current YES SELL price: {yes_sell}\n"
    )

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    text = data["content"][0]["text"].strip()
    p = float(text)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"Claude returned out-of-range prob: {p}")
    return p

def kelly_fraction(p: float, price: float) -> float:
    # binary: 1口=1$, price=$p_mkt, payout b = (1/price - 1)
    if price <= 0.0 or price >= 1.0:
        return 0.0
    b = (1.0 / price) - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    if f < 0:
        f = 0.0
    if f > KELLY_MAX:
        f = KELLY_MAX
    return f

def main():
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    scan_markets = int(os.getenv("SCAN_MARKETS", "1000"))     # 500-1000
    max_tokens = int(os.getenv("MAX_TOKENS", "500"))          # 500-1000
    min_order_usd = float(os.getenv("MIN_ORDER_USD", "1.0"))  # 小さく
    max_orders = int(os.getenv("MAX_ORDERS_PER_RUN", "1"))    # まず1

    # Secrets/keys
    private_key = env("PM_PRIVATE_KEY").strip()
    funder = env("PM_FUNDER").strip()
    signature_type = int(env("PM_SIGNATURE_TYPE"))

    # 入力チェック（取り違え即死防止）
    if not is_hex_bytes(private_key, 32):
        raise RuntimeError("PM_PRIVATE_KEY must be 32 bytes hex (0x + 64 hex chars)")
    if not is_hex_bytes(funder, 20):
        raise RuntimeError("PM_FUNDER must be 20 bytes address hex (0x + 40 hex chars)")

    client = ClobClient(
        HOST,
        key=private_key,
        chain_id=CHAIN_ID,
        signature_type=signature_type,
        funder=funder,
    )
    # L2 credsを導出（既存があればそれ、なければ作成） :contentReference[oaicite:5]{index=5}
    client.set_api_creds(client.create_or_derive_api_creds())

    markets = gamma_markets(scan_markets)
    token_ids, picked = extract_yes_token_ids(markets, max_tokens)
    if not token_ids:
        gh_issue("run: no tradable markets", "enableOrderBook=true の市場が見つかりませんでした。")
        return

    prices = clob_prices(token_ids)  # {tid: {"BUY": "...", "SELL": "..."}, ...}

    decisions = []
    for tid, title in picked:
        p = prices.get(tid)
        if not p:
            continue
        yes_buy = float(p.get("BUY", "nan"))
        yes_sell = float(p.get("SELL", "nan"))
        if not (0.0 < yes_buy < 1.0 and 0.0 < yes_sell < 1.0):
            continue

        fair = claude_fair_prob(title, yes_buy, yes_sell)

        # mispricing: fair - buy_price
        edge = (fair - yes_buy) / yes_buy
        if edge < EDGE_MIN:
            continue

        decisions.append((edge, tid, title, fair, yes_buy, yes_sell))

    decisions.sort(reverse=True, key=lambda x: x[0])
    if not decisions:
        gh_issue("run: no edge >= 8%", "乖離8%超が見つかりませんでした（Claude推定ベース）。")
        return

    # 残高（bankroll）: まずは「balance/allowance」をSDK側で参照するのが筋
    # ただし返却形式は環境で揺れるため、ここは安全策として env を優先
    bankroll = float(os.getenv("BANKROLL_USD", "50.0"))

    executed = 0
    logs = []
    for edge, tid, title, fair, yes_buy, yes_sell in decisions[:max_orders]:
        f = kelly_fraction(fair, yes_buy)
        usd = bankroll * f

        if usd < min_order_usd:
            logs.append(f"skip (too small): edge={edge:.3f} kelly={f:.4f} usd={usd:.2f} title={title}")
            continue

        # shares = usd / price
        size = usd / yes_buy

        order = OrderArgs(
            price=yes_buy,
            size=size,
            side=BUY,
            token_id=tid,
        )

        if dry_run:
            signed = client.create_order(order)
            logs.append(
                f"DRY_RUN build: edge={edge:.3f} fair={fair:.3f} buy={yes_buy:.3f} kelly={f:.4f} usd={usd:.2f} size={size:.4f} token={tid} title={title}"
            )
        else:
            signed = client.create_order(order)
            resp = client.post_order(signed)
            logs.append(
                f"LIVE post: edge={edge:.3f} fair={fair:.3f} buy={yes_buy:.3f} kelly={f:.4f} usd={usd:.2f} size={size:.4f} token={tid} title={title}\nresp={resp}"
            )
            executed += 1

    title = f"run: {'DRY_RUN' if dry_run else 'LIVE'} decisions={len(decisions)} executed={executed}"
    body = "\n".join(logs[:50])  # 長すぎ防止
    gh_issue(title, body)

if __name__ == "__main__":
    main()
