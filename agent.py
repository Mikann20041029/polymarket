import os
import json
import datetime
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
EDGE_MIN = 0.08   # 8%
KELLY_MAX = 0.06  # bankrollの6%上限

def should_trade(fair_yes_prob: float, yes_price: float):
    # yes_price が 0 の場合は除外
    if yes_price <= 0:
        return False, 0.0
    edge = (fair_yes_prob - yes_price) / yes_price
    return edge >= EDGE_MIN, edge

def kelly_fraction(edge: float, odds: float):
    # 簡易・安全側。oddsは仮で1.0固定でも可
    if odds <= 0:
        return 0.0
    f = edge / odds
    if f < 0:
        f = 0.0
    if f > KELLY_MAX:
        f = KELLY_MAX
    return f

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA = "https://gamma-api.polymarket.com"

def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"missing env: {name}")
    return v

def gh_issue(title: str, body: str):
    repo = env("REPO")
    token = env("GITHUB_TOKEN")
    r = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body},
        timeout=30,
    )
    r.raise_for_status()

def gamma_pick_one_token_id() -> str:
    # enableOrderBook=true & active の市場からYES token_idを1つ拾う
    r = requests.get(
        f"{GAMMA}/markets",
        params={"active": "true", "closed": "false", "archived": "false", "limit": "200"},
        timeout=30,
    )
    r.raise_for_status()
    markets = r.json()

    for m in markets:
        if not m.get("enableOrderBook", False):
            continue

        v = m.get("clobTokenIds")
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
            try:
                v = json.loads(v)
            except Exception:
                continue

        # dict {"YES": "...", "NO": "..."} を優先
        if isinstance(v, dict):
            yes = v.get("YES") or v.get("Yes") or v.get("yes")
            if yes:
                return str(yes)

        # list の場合は先頭をYES扱い（outcomesが取れるなら厳密化可能）
        if isinstance(v, list) and len(v) >= 1:
            return str(v[0])

    raise RuntimeError("no tradable token_id found (enableOrderBook market)")

def main():
    # ---- knobs ----
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    # まずは「絶対に刺さりにくい」超保守：0.01で0.1株（実弾でも損失最小）
    price = float(os.getenv("TEST_PRICE", "0.01"))
    size  = float(os.getenv("TEST_SIZE", "0.1"))

    # ---- secrets ----
    l1_key = env("PM_PRIVATE_KEY")
    funder = env("PM_FUNDER")
    sig_t  = int(env("PM_SIGNATURE_TYPE"))

    # ---- client init (trading) ----
    client = ClobClient(
        HOST,
        key=l1_key,
        chain_id=CHAIN_ID,
        signature_type=sig_t,
        funder=funder,
    )
    # L2 creds は「既存があれば取得、なければ作成」なので env にAPIキーを持たなくても進められます
    client.set_api_creds(client.create_or_derive_api_creds())

    token_id = gamma_pick_one_token_id()

    # ---- create order ----
    order = OrderArgs(token_id=token_id, price=price, size=size, side=BUY)
    signed = client.create_order(order)

    now = datetime.datetime.utcnow().isoformat() + "Z"

    if dry_run:
        gh_issue(
            f"[{now}] DRY_RUN order built (not posted)",
            f"token_id: {token_id}\nprice: {price}\nsize: {size}\nside: BUY\n\nResult: built+signed OK (not posted).",
        )
        return

    # ---- post order ----
    resp = client.post_order(signed, OrderType.GTC)
    gh_issue(
        f"[{now}] LIVE order posted",
        f"token_id: {token_id}\nprice: {price}\nsize: {size}\nside: BUY\norder_type: GTC\n\nresp:\n{resp}",
    )

if __name__ == "__main__":
    main()
