# agent.py
import os
import json
import time
import requests

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import OrderArgs

HOST = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"missing env: {name}")
    return v


def validate_hex(name: str, v: str, n_bytes: int):
    if not v.startswith("0x"):
        raise RuntimeError(f"{name} must start with 0x")
    hexpart = v[2:]
    if len(hexpart) != n_bytes * 2:
        raise RuntimeError(f"{name} length wrong: expected {n_bytes} bytes (0x + {n_bytes*2} hex), got {len(hexpart)//2} bytes")
    int(hexpart, 16)  # validate hex


def gh_issue(title: str, body: str):
    # これが一番安全：Actions の GITHUB_TOKEN を使う
    token = env("GITHUB_TOKEN")
    repo = env("REPO")

    r = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"title": title, "body": body},
        timeout=30,
    )
    r.raise_for_status()


def gamma_list_markets(limit: int = 500):
    r = requests.get(
        f"{GAMMA}/markets",
        params={
            "active": "true",
            "closed": "false",
            "archived": "false",
            "limit": str(limit),
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def pick_one_tradable_yes_token_id(markets):
    """
    enableOrderBook=true かつ clobTokenIds から YES の token_id を拾う。
    """
    for m in markets:
        if not m.get("enableOrderBook", False):
            continue

        v = m.get("clobTokenIds")
        if v is None:
            continue

        # v が str のことも list/dict のこともあるので吸収
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            try:
                v = json.loads(s)
            except Exception:
                continue

        if isinstance(v, dict):
            yes = v.get("YES") or v.get("Yes") or v.get("yes")
            if yes:
                return str(yes), m

        if isinstance(v, list) and len(v) >= 1:
            return str(v[0]), m

    raise RuntimeError("no tradable token_id found (enableOrderBook market)")


def main():
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    max_markets = int(os.getenv("MAX_MARKETS", "500"))

    # --- Secrets (GitHub Actions env) ---
    private_key = env("PM_PRIVATE_KEY")
    funder = env("PM_FUNDER")

    # 厳密チェック：private key は 32 bytes、funder(address) は 20 bytes
    validate_hex("PM_PRIVATE_KEY", private_key, 32)
    validate_hex("PM_FUNDER", funder, 20)

    # --- Scan markets ---
    markets = gamma_list_markets(limit=max_markets)
    token_id, picked_market = pick_one_tradable_yes_token_id(markets)

    # --- Build client ---
    client = ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        private_key=private_key,
        funder=funder,
        signature_type=POLYGON,  # これで通る構成が多い。環境によっては不要/別指定が要る。
    )

    # --- Fetch orderbook ---
    ob = client.get_order_book(token_id)
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        raise RuntimeError("orderbook empty (no bids/asks)")

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])

    # ここは「外人仕様」の前段：まだ LLM の fair value 推定は入れてない（まず取引が通る土台）
    # ひとまず最小サイズで DRY_RUN を確認
    side = "BUY"
    price = best_ask if side == "BUY" else best_bid
    size = float(os.getenv("TEST_SIZE", "0.1"))

    # --- Build order ---
    order_args = OrderArgs(
        price=price,
        size=size,
        side=side,
        token_id=token_id,
    )

    if dry_run:
        signed = client.create_order(order_args)
        title = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] DRY_RUN order built (not posted)"
        body = "\n".join([
            f"token_id: {token_id}",
            f"price: {price}",
            f"size: {size}",
            f"side: {side}",
            "",
            "Result: built+signed OK (not posted).",
        ])
        gh_issue(title, body)
        return

    # 本番：発注
    signed = client.create_order(order_args)
    resp = client.post_order(signed)

    title = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] LIVE order posted"
    body = "\n".join([
        f"token_id: {token_id}",
        f"price: {price}",
        f"size: {size}",
        f"side: {side}",
        "",
        f"response: {resp}",
    ])
    gh_issue(title, body)


if __name__ == "__main__":
    main()
