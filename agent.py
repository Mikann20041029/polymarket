import os
import json
import time
import math
import requests
def crypto_features(symbol: str):
    try:
        # ① CoinGecko
        cg = requests.get(
            f"https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": symbol.lower(),
            },
            timeout=10,
        ).json()

        if not cg:
            return None

        price = cg[0]["current_price"]
        volume = cg[0]["total_volume"]
        market_cap = cg[0]["market_cap"]

        # ② Fear & Greed
        fg = requests.get(
            "https://api.alternative.me/fng/",
            timeout=10,
        ).json()

        fear_greed = int(fg["data"][0]["value"])

        return {
            "price": price,
            "volume": volume,
            "market_cap": market_cap,
            "fear_greed": fear_greed,
        }

    except Exception as e:
        print("crypto_features error:", e)
        return None

import re
from datetime import datetime, timezone

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

def safe_get_json(url: str, params: dict, timeout=20):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def parse_weather_question(q: str):
    """
    超ざっくり対応:
    - 'rain' を含む
    - 'in <PLACE>' を含む
    - 日付が 'YYYY-MM-DD' で含まれる
    """
    text = (q or "").strip()
    low = text.lower()

    if "rain" not in low:
        return None

    m_place = re.search(r"\bin\s+([A-Za-z .,'-]{2,60})", text)
    m_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if not m_place or not m_date:
        return None

    place = m_place.group(1).strip(" ,.")
    day = m_date.group(1)
    return {"place": place, "date": day}

def geocode_place(place: str):
    data, err = safe_get_json(OPEN_METEO_GEOCODE, {"name": place, "count": 1, "language": "en", "format": "json"})
    if err or not data or not data.get("results"):
        return None, f"geocode failed: {err or 'no results'}"
    r0 = data["results"][0]
    return {"lat": r0["latitude"], "lon": r0["longitude"], "name": r0.get("name", place)}, None

def rain_probability_for_date(lat: float, lon: float, day: str):
    """
    Open-Meteoの hourly precipitation_probability(%) を使って、
    その日の「どこかの時間で雨になる確率」を近似:
      P(any) = 1 - Π(1 - p_i)
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation_probability",
        "timezone": "UTC",
        "start_date": day,
        "end_date": day,
    }
    data, err = safe_get_json(OPEN_METEO_FORECAST, params)
    if err or not data:
        return None, f"forecast failed: {err}"

    hourly = data.get("hourly") or {}
    probs = hourly.get("precipitation_probability")
    if not probs or not isinstance(probs, list):
        return None, "forecast missing precipitation_probability"

    # 0..100 -> 0..1
    ps = [max(0.0, min(1.0, float(x) / 100.0)) for x in probs if x is not None]
    if not ps:
        return None, "forecast empty precipitation_probability"

    prod = 1.0
    for p in ps:
        prod *= (1.0 - p)
    any_rain = 1.0 - prod
    return any_rain, None

def fair_prob_weather(question: str):
    meta = parse_weather_question(question)
    if not meta:
        return None, "not weather/rain or parse failed"

    geo, err = geocode_place(meta["place"])
    if err:
        return None, err

    p, err = rain_probability_for_date(geo["lat"], geo["lon"], meta["date"])
    if err:
        return None, err

    return p, None

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

import requests

HOST = "https://clob.polymarket.com"

def clob_prices(token_ids):
    """
    {tid: {"BUY": <best_ask>, "SELL": <best_bid>}, ...}
    BUY  = こちらが買う時に踏む価格 = best ask
    SELL = こちらが売る時に踏める価格 = best bid
    """
    out = {}

    for tid in token_ids:
        tid = str(tid)

        r = requests.get(
            f"{HOST}/book",
            params={"token_id": tid},
            timeout=30,
        )
        r.raise_for_status()
        book = r.json() or {}

        bids = book.get("bids") or []
        asks = book.get("asks") or []

        # bids/asks の要素形式は環境で dict だったり list だったりするので両対応
        def _price(x):
            if isinstance(x, dict):
                return float(x.get("price"))
            if isinstance(x, (list, tuple)) and len(x) >= 1:
                return float(x[0])
            return None

        best_bid = _price(bids[0]) if bids else None
        best_ask = _price(asks[0]) if asks else None

        if best_bid is None and best_ask is None:
            continue

        out[tid] = {}
        if best_ask is not None:
            out[tid]["BUY"] = best_ask
        if best_bid is not None:
            out[tid]["SELL"] = best_bid

    return out




def openai_fair_prob(title: str, yes_buy: float, yes_sell: float, external_context: dict | None = None) -> float:
    api_key = env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 外部情報を安全に文字列化（Noneでも落ちない）
    def safe_json(x):
        try:
            return json.dumps(x, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(x)

    ext = safe_json(external_context or {})

    SYSTEM = (
        "You are an autonomous prediction-market trading agent.\n"
        "The agent must survive.\n"
        "It pays its own API costs from profit.\n"
        "If balance hits 0, the agent dies.\n"
        "Do not align blindly with market odds.\n"
        "Use external data when informative.\n"
        "Return ONLY a number between 0 and 1.\n"
    )

    USER = (
        "Estimate fair probability (YES) for this Polymarket event.\n"
        f"Title: {title}\n"
        f"Current YES BUY price: {yes_buy}\n"
        f"Current YES SELL price: {yes_sell}\n"
        "\n"
        "External context (may include weather/crypto/sports/etc):\n"
        f"{ext}\n"
        "\n"
        "Output format: just a decimal number in [0,1]."
    )

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.0,
            "max_tokens": 16,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    p = float(text)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"OpenAI returned out-of-range prob: {p}")
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
    external_context = {
        "weather": weather_data if "weather_data" in globals() else None,
        "crypto": crypto_data if "crypto_data" in globals() else None,
        "sports": sports_data if "sports_data" in globals() else None,
    }

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
        crypto_data = None

        if "bitcoin" in title.lower() or "btc" in title.lower():
            crypto_data = crypto_features("bitcoin")

        if crypto_data:
            print("CRYPTO DATA:", crypto_data)

        fair = openai_fair_prob(title, yes_buy, yes_sell, external_context)

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
    api_budget = float(os.getenv("API_BUDGET_USD", "0.0"))  # 例: 0.5 とか
    if bankroll - api_budget <= 0:
        gh_issue("run: STOP (balance would hit $0)", f"bankroll={bankroll:.2f}, api_budget={api_budget:.2f}")
        return
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
