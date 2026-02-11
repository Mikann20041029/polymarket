import os
import json
import time
import math
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY

# =========================
# Config (env)
# =========================
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA = "https://gamma-api.polymarket.com"

def getenv_float(k: str, default: float) -> float:
    v = os.getenv(k)
    if v is None or v == "":
        return default
    return float(v)

def getenv_int(k: str, default: int) -> int:
    v = os.getenv(k)
    if v is None or v == "":
        return default
    return int(v)

def getenv_str(k: str, default: str) -> str:
    v = os.getenv(k)
    if v is None or v == "":
        return default
    return str(v)

EDGE_MIN = getenv_float("EDGE_MIN", 0.08)          # base required edge
KELLY_MAX = getenv_float("KELLY_MAX", 0.06)        # cap fraction
FEE_RATE = getenv_float("FEE_RATE", 0.00)          # per side fee rate (unknown -> start 0)
SLIPPAGE_MAX = getenv_float("SLIPPAGE_MAX", 0.01)  # max acceptable slippage for planned size
SPREAD_MAX = getenv_float("SPREAD_MAX", 0.05)      # drop if spread too wide
STATE_DIR = getenv_str("STATE_DIR", "state")

# =========================
# Small utils
# =========================
def ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)

def state_path(name: str) -> str:
    ensure_state_dir()
    return os.path.join(STATE_DIR, name)

def load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any):
    ensure_state_dir()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)

def env_required(k: str) -> str:
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

# =========================
# GitHub Issue
# =========================
def gh_issue(title: str, body: str):
    token = env_required("GITHUB_TOKEN")
    repo = env_required("GITHUB_REPOSITORY")
    r = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body},
        timeout=30,
    )
    r.raise_for_status()

# =========================
# Gamma markets
# =========================
def gamma_markets(limit: int):
    r = requests.get(
        f"{GAMMA}/markets",
        params={"active": "true", "closed": "false", "archived": "false", "limit": str(limit)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def extract_yes_token_ids(markets, max_tokens: int):
    token_ids = []
    picked = []  # (token_id, title, market_id)
    for m in markets:
        if len(token_ids) >= max_tokens:
            break
        if not m.get("enableOrderBook", False):
            continue

        clob_ids = m.get("clobTokenIds")
        if clob_ids is None:
            continue
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                continue

        yes = None
        if isinstance(clob_ids, dict):
            yes = clob_ids.get("YES") or clob_ids.get("Yes") or clob_ids.get("yes")
        elif isinstance(clob_ids, list) and len(clob_ids) > 0:
            yes = clob_ids[0]

        if not yes:
            continue

        tid = str(yes)
        title = str(m.get("question") or m.get("title") or "unknown")
        mid = str(m.get("id") or m.get("marketId") or "")
        token_ids.append(tid)
        picked.append((tid, title, mid))
    return token_ids, picked

# =========================
# External context (crypto/sports/weather)
# =========================
def fetch_crypto_context() -> Optional[dict]:
    out = {"ts_utc": datetime.now(timezone.utc).isoformat()}
    ok = False

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin,ethereum,solana", "vs_currencies": "usd", "include_24hr_change": "true"},
            timeout=20,
        )
        r.raise_for_status()
        out["coingecko"] = r.json()
        ok = True
    except Exception as e:
        out["coingecko_error"] = f"{type(e).__name__}: {e}"

    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": "1"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        out["fear_greed"] = data.get("data", [None])[0]
        ok = True
    except Exception as e:
        out["fear_greed_error"] = f"{type(e).__name__}: {e}"

    return out if ok else None

def fetch_sports_context() -> Optional[dict]:
    out = {"ts_utc": datetime.now(timezone.utc).isoformat()}
    feeds = [
        ("nfl", "https://www.espn.com/espn/rss/nfl/news"),
        ("nba", "https://www.espn.com/espn/rss/nba/news"),
        ("mlb", "https://www.espn.com/espn/rss/mlb/news"),
        ("nhl", "https://www.espn.com/espn/rss/nhl/news"),
    ]
    keywords = ["injury", "injured", "out", "questionable", "doubtful", "IL", "concussion", "hamstring", "ankle", "knee"]

    hits, errors = [], []
    for league, url in feeds:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            xml = r.text
            titles = []
            start = 0
            while True:
                a = xml.find("<title>", start)
                if a == -1:
                    break
                b = xml.find("</title>", a)
                if b == -1:
                    break
                t = xml[a + 7 : b].strip()
                titles.append(t)
                start = b + 8
            titles = titles[1:50]
            for t in titles:
                low = t.lower()
                if any(k in low for k in keywords):
                    hits.append({"league": league, "title": t})
        except Exception as e:
            errors.append({"league": league, "error": f"{type(e).__name__}: {e}"})

    if hits:
        out["injury_news_titles"] = hits[:40]
    if errors:
        out["errors"] = errors
    return out if "injury_news_titles" in out else None

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
    ps = [max(0.0, min(1.0, float(x) / 100.0)) for x in probs if x is not None]
    if not ps:
        return None, "forecast empty precipitation_probability"
    prod = 1.0
    for p in ps:
        prod *= (1.0 - p)
    return 1.0 - prod, None

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

def classify_market_type(title: str) -> Optional[str]:
    t = (title or "").lower()
    if "rain" in t and re.search(r"\b\d{4}-\d{2}-\d{2}\b", t) and " in " in t:
        return "weather"
    if (" vs " in t) or (" v " in t) or (" at " in t):
        return "sports"
    crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "sol", "solana", "etf", "sec", "cpi", "fed", "rate cut", "rate hike"]
    if any(k in t for k in crypto_kw):
        return "crypto"
    return None

# =========================
# Orderbook depth & slippage
# =========================
def fetch_book(token_id: str) -> dict:
    r = requests.get(f"{HOST}/book", params={"token_id": str(token_id)}, timeout=30)
    r.raise_for_status()
    return r.json() or {}

def parse_levels(levels) -> List[Tuple[float, float]]:
    """
    Returns list of (price, size) sorted as given by API.
    bids: high -> low, asks: low -> high (usually)
    """
    out = []
    if not levels:
        return out
    for x in levels:
        if isinstance(x, dict):
            p = x.get("price")
            s = x.get("size") or x.get("amount") or x.get("quantity")
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            p, s = x[0], x[1]
        else:
            continue
        try:
            fp = float(p)
            fs = float(s)
            if fp > 0 and fs > 0:
                out.append((fp, fs))
        except Exception:
            continue
    return out

def best_bid_ask(book: dict) -> Tuple[Optional[float], Optional[float]]:
    bids = parse_levels(book.get("bids") or [])
    asks = parse_levels(book.get("asks") or [])
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    return best_bid, best_ask

def vwap_for_size(levels: List[Tuple[float, float]], size: float) -> Tuple[Optional[float], float]:
    """
    Compute VWAP if we take 'size' shares from given side levels.
    Returns (vwap_price, filled_size)
    """
    if size <= 0:
        return None, 0.0
    remain = size
    notional = 0.0
    filled = 0.0
    for price, avail in levels:
        take = min(remain, avail)
        notional += take * price
        filled += take
        remain -= take
        if remain <= 1e-12:
            break
    if filled <= 0:
        return None, 0.0
    return notional / filled, filled

def estimate_fill_prob(best_px: float, spread: float, size: float, top_depth: float) -> float:
    """
    Approx fill probability proxy:
    - more top_depth relative to size => higher
    - wider spread => lower
    """
    if best_px is None:
        return 0.0
    if size <= 0:
        return 0.0
    # depth ratio
    dr = max(0.0, min(5.0, top_depth / max(1e-9, size)))
    # map dr to 0..1
    p_depth = 1.0 - math.exp(-dr)  # fast saturating
    # spread penalty (0..1)
    sp = max(0.0, min(1.0, spread / 0.05))  # 5% spread => heavy
    p = p_depth * (1.0 - 0.7 * sp)
    return max(0.0, min(1.0, p))

# =========================
# Fair probability via LLM (with modes)
# =========================
def openai_fair_prob(title: str, yes_buy: float, yes_sell: float, external_context: Optional[dict]) -> float:
    api_key = env_required("OPENAI_API_KEY")
    model = getenv_str("OPENAI_MODEL", "gpt-4o-mini")

    FAIR_MODE = getenv_str("FAIR_MODE", "FULL").strip().upper()
    ctx = external_context or {}

    if FAIR_MODE == "TITLE_ONLY":
        title_for_llm = title
        ctx_for_llm = {}
        mode_hint = ""
    elif FAIR_MODE == "EXTERNAL_ONLY":
        title_for_llm = title
        ctx_for_llm = ctx
        mode_hint = "IMPORTANT: In EXTERNAL_ONLY mode, IGNORE the Title and rely ONLY on External context + prices.\n"
    else:
        title_for_llm = title
        ctx_for_llm = ctx
        mode_hint = ""

    try:
        ext = json.dumps(ctx_for_llm, ensure_ascii=False, sort_keys=True)
    except Exception:
        ext = str(ctx_for_llm)

    try:
        ext_flags = (
            f"ext(weather={'Y' if ctx_for_llm.get('weather') else 'N'},"
            f" sports={'Y' if ctx_for_llm.get('sports') else 'N'},"
            f" crypto={'Y' if ctx_for_llm.get('crypto') else 'N'})"
        )
    except Exception:
        ext_flags = "ext(weather=N, sports=N, crypto=N)"

    SYSTEM = (
        "You are an autonomous prediction-market trading agent.\n"
        "The agent must survive.\n"
        "It pays its own API costs from profit.\n"
        "If balance hits 0, the agent dies.\n"
        "Do not align blindly with market odds.\n"
        "Use external data when informative.\n"
        f"{ext_flags}\n"
        "Return ONLY a number between 0 and 1.\n"
    )

    user_lines = [
        "Estimate fair probability (YES) for this Polymarket event.",
        f"Title: {title_for_llm}",
        f"FAIR_MODE: {FAIR_MODE}",
    ]
    if mode_hint:
        user_lines.append(mode_hint.strip())

    user_lines += [
        f"Current YES BUY price: {yes_buy}",
        f"Current YES SELL price: {yes_sell}",
        "",
        "External context (may include weather/crypto/sports/etc):",
        ext,
        "",
        "Output format: just a decimal number in [0,1].",
    ]
    USER = "\n".join(user_lines)

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

# =========================
# Risk model: uncertainty/volatility, shrinkage, costs
# =========================
def market_mid(best_bid: float, best_ask: float) -> float:
    return (best_bid + best_ask) / 2.0

def proxy_volatility(mtype: Optional[str], crypto_ctx: Optional[dict]) -> float:
    """
    Rough vol proxy:
    - crypto: avg abs 24h change across BTC/ETH/SOL (scaled)
    - else: small baseline
    """
    base = 0.03  # default uncertainty
    if mtype != "crypto" or not crypto_ctx:
        return base

    cg = (crypto_ctx.get("coingecko") or {})
    changes = []
    for k in ["bitcoin", "ethereum", "solana"]:
        v = cg.get(k) or {}
        ch = v.get("usd_24h_change")
        if ch is None:
            continue
        try:
            changes.append(abs(float(ch)) / 100.0)
        except Exception:
            continue
    if not changes:
        return base
    # clamp
    return max(base, min(0.20, sum(changes) / len(changes)))

def shrink_prob_to_market(p: float, mid: float, uncertainty: float, liquidity_score: float) -> float:
    """
    Uncertainty + low liquidity => stronger shrinkage toward market mid.
    """
    # uncertainty 0.03..0.20 => alpha 0.1..0.6
    u = max(0.01, min(0.25, uncertainty))
    alpha_u = (u - 0.01) / (0.25 - 0.01)  # 0..1
    alpha_u = 0.10 + 0.50 * alpha_u       # 0.10..0.60

    # liquidity_score 0..1 (higher = more liquid) => less shrink
    liq = max(0.0, min(1.0, liquidity_score))
    alpha = alpha_u * (1.0 - 0.7 * liq)   # liquid => smaller alpha

    p2 = (1.0 - alpha) * p + alpha * mid
    return max(0.0, min(1.0, p2))

def effective_costs(spread: float, slippage: float, fee_rate: float) -> float:
    """
    Combine penalties into "edge threshold add-on" scale.
    """
    # treat spread/slippage/fee roughly additive in required edge space
    return max(0.0, spread) + max(0.0, slippage) + max(0.0, fee_rate)

def liquidity_score_from_spread(spread: float) -> float:
    # small spread => high liquidity
    # spread 0..0.05 => score ~1..0
    s = max(0.0, min(0.05, spread))
    return max(0.0, min(1.0, 1.0 - s / 0.05))

# =========================
# Kelly with liquidity constraint
# =========================
def kelly_fraction_binary(p: float, price: float) -> float:
    if price <= 0.0 or price >= 1.0:
        return 0.0
    b = (1.0 / price) - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    if f < 0.0:
        f = 0.0
    return min(f, KELLY_MAX)

def apply_liquidity_constraints(
    f: float,
    bankroll: float,
    price: float,
    asks: List[Tuple[float, float]],
    spread: float,
    fee_rate: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Cap position by:
    - slippage max
    - spread max (handled earlier, but keep safe)
    - available depth
    """
    info = {}
    if bankroll <= 0 or f <= 0:
        return 0.0, {"cap_reason": 1.0}

    usd_target = bankroll * f
    size_target = usd_target / price

    # compute vwap at target
    vwap, filled = vwap_for_size(asks, size_target)
    if vwap is None or filled <= 0:
        return 0.0, {"cap_reason": 2.0}

    slippage = max(0.0, (vwap - price) / price)

    info["usd_target"] = usd_target
    info["size_target"] = size_target
    info["vwap"] = vwap
    info["filled"] = filled
    info["slippage"] = slippage

    # cap by slippage max: find max size such that slippage <= SLIPPAGE_MAX
    if slippage > SLIPPAGE_MAX:
        # binary search size
        lo, hi = 0.0, size_target
        for _ in range(30):
            mid = (lo + hi) / 2.0
            v, fd = vwap_for_size(asks, mid)
            if v is None or fd <= 0:
                hi = mid
                continue
            sl = max(0.0, (v - price) / price)
            if sl <= SLIPPAGE_MAX:
                lo = mid
            else:
                hi = mid
        size_cap = lo
    else:
        size_cap = size_target

    # also cap by available depth we can fill at all
    # (use what vwap_for_size can fill)
    v2, f2 = vwap_for_size(asks, size_cap)
    if v2 is None or f2 <= 0:
        return 0.0, {"cap_reason": 3.0}

    usd_cap = f2 * price
    f_cap = usd_cap / bankroll

    # fee reduces effective budget a bit
    # (simple: pay fee on notional)
    f_cap = max(0.0, f_cap * (1.0 - fee_rate))

    info["size_cap"] = size_cap
    info["usd_cap"] = usd_cap
    info["f_cap"] = f_cap

    return max(0.0, min(f, f_cap)), info

# =========================
# Feedback calibration
# =========================
def record_prediction(market_id: str, token_id: str, p: float, mid: float, ts: str):
    db = load_json(state_path("preds.json"), {})
    key = market_id or token_id
    db[key] = {"p": p, "mid": mid, "ts": ts, "market_id": market_id, "token_id": token_id}
    save_json(state_path("preds.json"), db)

def load_calibration() -> dict:
    return load_json(state_path("calibration.json"), {"bias": 0.0, "n": 0})

def save_calibration(cal: dict):
    save_json(state_path("calibration.json"), cal)

def apply_calibration(p: float, cal: dict) -> float:
    """
    Simple logit-bias calibration: logit(p') = logit(p) + bias
    """
    bias = float(cal.get("bias", 0.0))
    eps = 1e-6
    p0 = max(eps, min(1.0 - eps, p))
    logit = math.log(p0 / (1.0 - p0))
    logit2 = logit + bias
    p2 = 1.0 / (1.0 + math.exp(-logit2))
    return max(0.0, min(1.0, p2))

def update_calibration_from_closed_markets():
    """
    If Gamma provides recent closed outcomes, use stored preds to update bias.
    This is "best-effort"; if API doesn't provide outcome, it will do nothing.
    """
    preds = load_json(state_path("preds.json"), {})
    if not preds:
        return

    cal = load_calibration()
    bias = float(cal.get("bias", 0.0))
    n = int(cal.get("n", 0))

    # Try fetch some closed markets (best-effort)
    try:
        r = requests.get(
            f"{GAMMA}/markets",
            params={"active": "false", "closed": "true", "archived": "false", "limit": "200"},
            timeout=30,
        )
        r.raise_for_status()
        closed = r.json() or []
    except Exception:
        return

    # Update with outcomes if available
    # (Different deployments expose fields differently; check common patterns)
    updated = 0
    for m in closed:
        mid = str(m.get("id") or m.get("marketId") or "")
        if not mid:
            continue
        rec = preds.get(mid)
        if not rec:
            continue

        outcome = m.get("outcome") or m.get("resolvedOutcome") or m.get("result")
        if outcome is None:
            continue

        # normalize outcome to 0/1
        y = None
        if isinstance(outcome, (int, float)):
            y = 1.0 if float(outcome) >= 0.5 else 0.0
        elif isinstance(outcome, str):
            low = outcome.lower().strip()
            if low in ["yes", "true", "1", "y"]:
                y = 1.0
            elif low in ["no", "false", "0", "n"]:
                y = 0.0
        if y is None:
            continue

        p = float(rec.get("p", 0.5))
        eps = 1e-6
        p = max(eps, min(1.0 - eps, p))

        # One-step bias gradient on log-loss: d/d(bias) = (y - p)
        # Use small learning rate
        lr = 0.02
        bias += lr * (y - p)
        n += 1
        updated += 1

        # remove once consumed
        preds.pop(mid, None)

    if updated > 0:
        cal["bias"] = bias
        cal["n"] = n
        save_calibration(cal)
        save_json(state_path("preds.json"), preds)

# =========================
# Main
# =========================
def main():
    dry_run = getenv_str("DRY_RUN", "1") == "1"
    scan_markets = getenv_int("SCAN_MARKETS", 1000)
    max_tokens = getenv_int("MAX_TOKENS", 500)
    min_order_usd = getenv_float("MIN_USD_ORDER", 1.0)
    max_orders = getenv_int("MAX_ORDERS_PER_RUN", 1)
    bankroll = getenv_float("BANKROLL_USD", 50.0)
    api_budget = getenv_float("API_BUDGET_USD", 0.0)

    # update calibration from past resolved outcomes (best-effort)
    update_calibration_from_closed_markets()
    cal = load_calibration()

    if bankroll - api_budget <= 0:
        gh_issue("run: STOP (balance would hit $0)", f"bankroll={bankroll:.2f}, api_budget={api_budget:.2f}")
        return

    # Keys
    private_key = env_required("PM_PRIVATE_KEY").strip()
    funder = env_required("PM_FUNDER").strip()
    signature_type = int(env_required("PM_SIGNATURE_TYPE"))

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
    client.set_api_creds(client.create_or_derive_api_creds())

    markets = gamma_markets(scan_markets)
    token_ids, picked = extract_yes_token_ids(markets, max_tokens)

    # external context (best-effort)
    try:
        sports_data = fetch_sports_context()
    except Exception as e:
        sports_data = {"error": f"{type(e).__name__}: {e}"}

    try:
        crypto_data = fetch_crypto_context()
    except Exception as e:
        crypto_data = {"error": f"{type(e).__name__}: {e}"}

    evaluated = 0
    decisions = []
    best = None  # dict
 stats = {
    "picked": len(picked),
    "book_fail": 0,
    "no_bids_or_asks": 0,
    "bad_prices": 0,
    "spread_too_wide": 0,
}


    for tid, title, market_id in picked:
     evaluated += 1

        # Fetch full book (depth)
        try:
            book = fetch_book(tid)
        except Exception:
           stats["book_fail"] += 1
           continue


        bids = parse_levels(book.get("bids") or [])
        asks = parse_levels(book.get("asks") or [])
        if not bids or not asks:
            stats["no_bids_or_asks"] += 1
            continue


        best_bid = bids[0][0]
        best_ask = asks[0][0]
        if not (0.0 < best_bid < 1.0 and 0.0 < best_ask < 1.0):
            stats["bad_prices"] += 1
            continue


        spread = max(0.0, best_ask - best_bid)
        if spread > SPREAD_MAX:
            stats["spread_too_wide"] += 1
            continue


        mid = market_mid(best_bid, best_ask)
        mtype = classify_market_type(title)

        # weather: computed per market
        weather_data = None
        if mtype == "weather":
            wp, werr = fair_prob_weather(title)
            if wp is not None:
                weather_data = {"p_any_rain": wp, "source": "open-meteo precipitation_probability -> any-rain"}
            else:
                weather_data = {"error": werr or "weather prob unavailable"}

        ctx = {
            "weather": weather_data,
            "sports": sports_data,
            "crypto": crypto_data,
            "prices": {"best_bid": best_bid, "best_ask": best_ask, "mid": mid, "spread": spread},
        }

        # LLM fair prob
        try:
            p_raw = openai_fair_prob(title, best_ask, best_bid, external_context=ctx)
        except Exception:
            evaluated += 1
            continue

        # Apply feedback calibration
        p_cal = apply_calibration(p_raw, cal)

        # Uncertainty/volatility proxy
        vol = proxy_volatility(mtype, crypto_data if isinstance(crypto_data, dict) else None)

        # Liquidity score from spread
        liq_score = liquidity_score_from_spread(spread)

        # Shrink toward market mid based on uncertainty + liquidity
        p_shrunk = shrink_prob_to_market(p_cal, mid, vol, liq_score)

        # Immediate adverse move risk penalty:
        # reduce p when spread/vol high (proxy for adverse selection)
        adverse = min(0.08, 0.5 * spread + 0.5 * vol)
        p_eff = max(0.0, min(1.0, p_shrunk - adverse))

        # Decide BUY YES at best_ask
        buy_price = best_ask

        # compute kelly (pre-liquidity cap)
        f0 = kelly_fraction_binary(p_eff, buy_price)

        # liquidity constraints (slippage cap using full asks)
        f1, liq_info = apply_liquidity_constraints(f0, bankroll, buy_price, asks, spread, FEE_RATE)

        usd = bankroll * f1
        size = usd / buy_price if buy_price > 0 else 0.0

        # compute expected vwap for that size (slippage)
        vwap, filled = vwap_for_size(asks, size)
        if vwap is None or filled <= 0:
            evaluated += 1
            continue
        slippage = max(0.0, (vwap - buy_price) / buy_price)

        # approximate fill probability using top depth at best ask
        top_depth = asks[0][1]
        fill_p = estimate_fill_prob(buy_price, spread, size, top_depth)

        # Effective threshold includes costs and fill risk
        costs = effective_costs(spread, slippage, FEE_RATE)
        fill_penalty = (1.0 - fill_p) * 0.05  # up to +5% required edge if fill risk high
        th = EDGE_MIN + costs + fill_penalty

        # edge in same scale as your prior: (p - price)/price
        edge = (p_eff - buy_price) / buy_price if buy_price > 0 else -1.0

        # record for later calibration (best-effort)
        record_prediction(market_id, tid, p_eff, mid, datetime.now(timezone.utc).isoformat())

        if best is None or edge > best["edge"]:
            best = {
                "edge": edge,
                "th": th,
                "title": title,
                "tid": tid,
                "mid": mid,
                "bid": best_bid,
                "ask": best_ask,
                "p_raw": p_raw,
                "p_eff": p_eff,
                "p_cal": p_cal,
                "p_shrunk": p_shrunk,
                "vol": vol,
                "spread": spread,
                "slippage": slippage,
                "fill_p": fill_p,
                "f0": f0,
                "f1": f1,
                "usd": usd,
                "size": size,
            }

        if edge >= th and usd >= min_order_usd and size > 0:
            decisions.append((edge, th, tid, title, p_eff, buy_price, best_bid, usd, size, slippage, spread, fill_p, f0, f1))

        evaluated += 1

    # If no decisions
    if not decisions:
        if best:
            body = (
                "乖離が見つかりませんでした（推定ベース）。\n"
                f"Evaluated: {evaluated} markets\n"
                f"Best edge: {best['edge']:.4f} (needed {best['th']:.4f})\n\n"
                f"Best market:\n"
                f"- Title: {best['title']}\n"
                f"- Token: {best['tid']}\n"
                f"- Bid/Ask/Mid: {best['bid']:.4f}/{best['ask']:.4f}/{best['mid']:.4f}\n"
                f"- p_raw/p_cal/p_shrunk/p_eff: {best['p_raw']:.4f}/{best['p_cal']:.4f}/{best['p_shrunk']:.4f}/{best['p_eff']:.4f}\n"
                f"- spread/slippage/fill_p/vol: {best['spread']:.4f}/{best['slippage']:.4f}/{best['fill_p']:.2f}/{best['vol']:.3f}\n"
                f"- kelly f0/f1 usd size: {best['f0']:.4f}/{best['f1']:.4f} {best['usd']:.2f} {best['size']:.4f}\n"
                f"- threshold components: EDGE_MIN={EDGE_MIN:.4f} + costs(spread+slip+fee) + fill_penalty\n"
                f"  -> th={best['th']:.4f}\n"
            )
        else:
            body = (
                "No valid markets (missing books/bid-ask).\n\n"
                f"stats={json.dumps(stats, ensure_ascii=False, sort_keys=True)}\n"
                f"evaluated={evaluated}\n"
            )


        gh_issue("run: no edge (full depth model)", body)
        return

    # Execute top decisions (sorted by edge)
    decisions.sort(key=lambda x: x[0], reverse=True)
    executed = 0
    logs = []

    for (edge, th, tid, title, p_eff, buy_price, best_bid, usd, size, slippage, spread, fill_p, f0, f1) in decisions[:max_orders]:
        order = OrderArgs(
            price=buy_price,
            size=size,
            side=BUY,
            token_id=tid,
        )
        if dry_run:
            signed = client.create_order(order)
            logs.append(
                f"DRY_RUN: edge={edge:.4f} th={th:.4f} p_eff={p_eff:.4f} ask={buy_price:.4f} "
                f"usd={usd:.2f} size={size:.4f} spread={spread:.4f} slip={slippage:.4f} fill_p={fill_p:.2f} "
                f"kelly f0={f0:.4f} f1={f1:.4f} token={tid} title={title}"
            )
        else:
            signed = client.create_order(order)
            resp = client.post_order(signed)
            executed += 1
            logs.append(
                f"LIVE: edge={edge:.4f} th={th:.4f} p_eff={p_eff:.4f} ask={buy_price:.4f} "
                f"usd={usd:.2f} size={size:.4f} spread={spread:.4f} slip={slippage:.4f} fill_p={fill_p:.2f} "
                f"kelly f0={f0:.4f} f1={f1:.4f} token={tid} title={title}\nresp={resp}"
            )

    issue_title = f"run: {'DRY_RUN' if dry_run else 'LIVE'} decisions={len(decisions)} executed={executed}"
    gh_issue(issue_title, "\n".join(logs[:50]))

if __name__ == "__main__":
    main()
