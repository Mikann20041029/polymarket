import os
import json
import time
import math
import re
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# ── Constants ──────────────────────────────────────────────
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA = "https://gamma-api.polymarket.com"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

EDGE_MIN = float(os.getenv("EDGE_MIN", "0.05"))
KELLY_MAX = float(os.getenv("KELLY_MAX", "0.06"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.00"))
SLIPPAGE_MAX = float(os.getenv("SLIPPAGE_MAX", "0.01"))
SPREAD_MAX = float(os.getenv("SPREAD_MAX", "0.15"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "1000"))
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "500"))
CLOB_WORKERS = int(os.getenv("CLOB_WORKERS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
CANDIDATE_MIN_DIFF = float(os.getenv("CANDIDATE_MIN_DIFF", "0.06"))

# LLM Advantage: Political analysis, policy prediction, long-term events
PREFER_TICKER_PATTERNS = [
    "presidential-",         # 2028 presidential election
    "democratic-",           # Democratic nomination
    "republican-",           # Republican nomination
    "prime-minister",        # International politics
    "how-much-",             # Policy impact (DOGE cuts, tariffs)
    "how-many-",             # Deportation estimates
    "senate-",               # Senate races
    "house-",                # House races
    "-ceasefire",            # International relations
    "-out-before",           # Leader removal (Putin, Xi)
]

# LLM Disadvantage: Sports, real-time events, unpredictable individual cases
AVOID_TICKER_PATTERNS = [
    "2026-nhl-",             # NHL championships
    "2026-nba-",             # NBA championships
    "2026-fifa-world-cup-winner",  # World Cup winner
    "nba-rookie-",           # Rookie of the year
    "-convicted",            # Criminal convictions
    "-guilty",               # Criminal verdicts
    "megaeth-",              # Crypto market cap
    "-champion",             # Sports championships (generic)
]


# ── Utility ────────────────────────────────────────────────
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


def safe_json(x) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def safe_get_json(url: str, params: dict, timeout=20):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── GitHub Issue Reporter ──────────────────────────────────
def gh_issue(title: str, body: str):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        print(f"[GH Issue] (skipped, no token/repo) {title}")
        return
    try:
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
    except Exception as e:
        print(f"[GH Issue] Failed to post: {e}")


# ── External Data: Crypto ─────────────────────────────────
def crypto_features(symbol: str):
    try:
        cg = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "ids": symbol.lower()},
            timeout=10,
        ).json()

        if not cg:
            return None

        price = cg[0]["current_price"]
        volume = cg[0]["total_volume"]
        market_cap = cg[0]["market_cap"]

        fg = requests.get("https://api.alternative.me/fng/", timeout=10).json()
        fear_greed = int(fg["data"][0]["value"])

        return {
            "price": price,
            "volume": volume,
            "market_cap": market_cap,
            "fear_greed": fear_greed,
        }
    except Exception as e:
        print("  crypto_features error:", e)
        return None


def fetch_crypto_context() -> dict | None:
    out = {"ts_utc": datetime.now(timezone.utc).isoformat()}

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "bitcoin,ethereum,solana",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
            timeout=20,
        )
        r.raise_for_status()
        out["coingecko"] = r.json()
    except Exception as e:
        out["coingecko_error"] = f"{type(e).__name__}: {e}"

    try:
        r = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": "1"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        out["fear_greed"] = data.get("data", [None])[0]
    except Exception as e:
        out["fear_greed_error"] = f"{type(e).__name__}: {e}"

    if "coingecko" not in out and "fear_greed" not in out:
        return None
    return out


# ── External Data: Sports ─────────────────────────────────
def fetch_sports_context() -> dict | None:
    out = {"ts_utc": datetime.now(timezone.utc).isoformat()}

    feeds = [
        ("nfl", "https://www.espn.com/espn/rss/nfl/news"),
        ("nba", "https://www.espn.com/espn/rss/nba/news"),
        ("mlb", "https://www.espn.com/espn/rss/mlb/news"),
        ("nhl", "https://www.espn.com/espn/rss/nhl/news"),
    ]

    keywords = [
        "injury", "injured", "out", "questionable", "doubtful",
        "IL", "concussion", "hamstring", "ankle", "knee",
    ]

    hits = []
    errors = []

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

    if "injury_news_titles" not in out:
        return None
    return out


# ── External Data: Weather ────────────────────────────────
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
    data, err = safe_get_json(
        OPEN_METEO_GEOCODE,
        {"name": place, "count": 1, "language": "en", "format": "json"},
    )
    if err or not data or not data.get("results"):
        return None, f"geocode failed: {err or 'no results'}"
    r0 = data["results"][0]
    return {
        "lat": r0["latitude"],
        "lon": r0["longitude"],
        "name": r0.get("name", place),
    }, None


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
        prod *= 1.0 - p
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


# ── Market Classification ─────────────────────────────────
def classify_market_type(title: str) -> str | None:
    t = (title or "").lower()

    if "rain" in t and re.search(r"\b\d{4}-\d{2}-\d{2}\b", t) and " in " in t:
        return "weather"

    if (" vs " in t) or (" v " in t) or (" at " in t):
        return "sports"

    crypto_kw = [
        "bitcoin", "btc", "ethereum", "eth", "sol", "solana",
        "etf", "sec", "cpi", "fed", "rate cut", "rate hike",
    ]
    if any(k in t for k in crypto_kw):
        return "crypto"

    return None


# ── Polymarket API ─────────────────────────────────────────
def gamma_markets(limit: int):
    all_markets = []
    offset = 0
    page_size = min(limit, 100)

    while len(all_markets) < limit:
        try:
            r = requests.get(
                f"{GAMMA}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "enableOrderBook": "true",
                    "limit": str(page_size),
                    "offset": str(offset),
                },
                timeout=30,
            )
            r.raise_for_status()
            page = r.json()
        except Exception as e:
            print(f"  gamma_markets page error (offset={offset}): {e}")
            break

        if not page or not isinstance(page, list):
            break

        all_markets.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)

    print(f"[Gamma] Fetched {len(all_markets)} markets (requested up to {limit})")

    if all_markets:
        sample = all_markets[0]
        keys = list(sample.keys())
        print(f"[Gamma] Sample keys: {keys[:20]}")
        print(f"[Gamma] enableOrderBook={sample.get('enableOrderBook')!r}")
        print(f"[Gamma] clobTokenIds={str(sample.get('clobTokenIds', 'MISSING'))[:80]}")
        print(f"[Gamma] bestBid={sample.get('bestBid')!r}, bestAsk={sample.get('bestAsk')!r}")
        print(f"[Gamma] outcomePrices={str(sample.get('outcomePrices', 'MISSING'))[:80]}")
        print(f"[Gamma] volume={sample.get('volume')!r}, liquidity={sample.get('liquidity')!r}")

    return all_markets


def extract_tradable_markets(markets, max_tokens: int):
    results = []
    skipped_eob = 0
    skipped_no_clob = 0
    skipped_parse = 0
    skipped_no_price = 0
    skipped_spread = 0
    skipped_low_quality = 0
    skipped_category = 0

    for m in markets:
        if len(results) >= max_tokens:
            break

        eob = m.get("enableOrderBook")
        if eob is None:
            eob = m.get("enable_order_book")
        if not (eob is True or str(eob).lower() == "true"):
            skipped_eob += 1
            continue

        v = m.get("clobTokenIds")
        if v is None:
            v = m.get("clob_token_ids")
        if v is None:
            skipped_no_clob += 1
            continue

        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                skipped_parse += 1
                continue

        yes_tid = None
        if isinstance(v, dict):
            yes_tid = v.get("YES") or v.get("Yes") or v.get("yes")
        elif isinstance(v, list) and len(v) > 0:
            yes_tid = v[0]

        if not yes_tid:
            skipped_parse += 1
            continue

        tid = str(yes_tid)

        best_bid = safe_float(m.get("bestBid"))
        best_ask = safe_float(m.get("bestAsk"))

        if best_bid <= 0 or best_ask <= 0:
            op = m.get("outcomePrices")
            if op:
                try:
                    if isinstance(op, str):
                        op = json.loads(op)
                    if isinstance(op, list) and len(op) >= 1:
                        yes_price = safe_float(op[0])
                        if yes_price > 0:
                            best_ask = yes_price + 0.005
                            best_bid = yes_price - 0.005
                except Exception:
                    pass

        if best_bid <= 0 or best_ask <= 0:
            skipped_no_price += 1
            continue

        if not (0.0 < best_ask < 1.0 and 0.0 < best_bid < 1.0):
            skipped_no_price += 1
            continue

        spread = best_ask - best_bid
        if spread > SPREAD_MAX:
            skipped_spread += 1
            continue

        volume = safe_float(m.get("volume"))
        liquidity = safe_float(m.get("liquidity"))
        if volume < MIN_VOLUME or liquidity < MIN_LIQUIDITY:
            skipped_low_quality += 1
            continue

        # Category filter: prefer LLM-advantage markets
        should_trade, reason = should_trade_market(m)
        if not should_trade:
            skipped_category += 1
            continue

        title = str(m.get("question") or m.get("title") or "unknown")

        results.append({
            "tid": tid,
            "title": title,
            "yes_buy": best_ask,
            "yes_sell": best_bid,
            "volume": volume,
            "liquidity": liquidity,
            "spread": spread,
        })

    print(f"[Extract] Passed: {len(results)}")
    print(f"[Extract] Skipped: eob={skipped_eob}, no_clob={skipped_no_clob}, "
          f"parse={skipped_parse}, no_price={skipped_no_price}, "
          f"spread={skipped_spread}, low_quality={skipped_low_quality}, "
          f"category={skipped_category}")
    return results


def _fetch_one_book(tid: str):
    tid = str(tid)
    try:
        r = requests.get(
            f"{HOST}/book",
            params={"token_id": tid},
            timeout=30,
        )
        r.raise_for_status()
        book = r.json() or {}
    except Exception:
        return tid, None

    bids = book.get("bids") or []
    asks = book.get("asks") or []

    def _price(x):
        if isinstance(x, dict):
            raw = x.get("price")
            if raw is not None:
                return float(raw)
            return None
        if isinstance(x, (list, tuple)) and len(x) >= 1:
            return float(x[0])
        return None

    best_bid = _price(bids[0]) if bids else None
    best_ask = _price(asks[0]) if asks else None

    if best_bid is None and best_ask is None:
        return tid, None

    result = {}
    if best_ask is not None:
        result["BUY"] = best_ask
    if best_bid is not None:
        result["SELL"] = best_bid
    return tid, result


def clob_prices(token_ids):
    out = {}
    if not token_ids:
        return out

    with ThreadPoolExecutor(max_workers=CLOB_WORKERS) as executor:
        futures = {executor.submit(_fetch_one_book, tid): tid for tid in token_ids}
        for future in as_completed(futures):
            try:
                tid, result = future.result()
                if result:
                    out[tid] = result
            except Exception:
                pass

    return out


# ── Dynamic Edge Threshold ─────────────────────────────────
def dynamic_edge_threshold(yes_buy: float, yes_sell: float, has_data: bool = False) -> float:
    """
    Adapts minimum edge threshold based on market conditions.
    has_data=True means we have objective external data (weather, crypto prices)
    which gives us more confidence in our estimate.
    """
    if yes_buy <= 0.03 or yes_buy >= 0.97:
        return 0.12
    if yes_sell <= 0 or yes_sell >= 1:
        return 0.12

    spread = max(0.0, yes_buy - yes_sell)

    # Base thresholds
    if spread <= 0.01:
        base = 0.04
    elif spread <= 0.03:
        base = 0.06
    elif spread <= 0.06:
        base = 0.08
    else:
        base = 0.10

    # Data-backed markets get lower thresholds (higher confidence)
    if has_data:
        base = max(0.03, base - 0.02)

    return base


# ── Individual Blind Evaluation (NO market prices shown) ───
def individual_blind_eval(
    title: str,
    external_context: dict | None,
    api_key: str,
    model: str,
) -> float | None:
    """
    High-quality individual probability estimation.
    CRITICAL: Does NOT show market prices. The LLM forms a completely
    independent view based on the question and external data only.
    The resulting estimate is compared to market price AFTER the LLM returns.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ctx = safe_json(external_context or {})

    system = (
        "You are an expert probability estimator for prediction markets.\n"
        f"Today's date: {today}\n"
        "\n"
        "Your task: Estimate the probability that the given event resolves YES.\n"
        "\n"
        "Think carefully about:\n"
        "1. BASE RATES: What is the historical frequency of similar events?\n"
        "2. CURRENT CONTEXT: What do you know about the current situation?\n"
        "3. EXTERNAL DATA: If provided, use real data (prices, weather, news) heavily.\n"
        "4. TIME: How much time remains? More time = more uncertainty toward 50%.\n"
        "5. SPECIFICITY: Narrow outcomes (exact ranges) are less likely than broad ones.\n"
        "\n"
        "Be calibrated and precise. Use the full range [0.05, 0.95].\n"
        "Think step-by-step in 2-3 sentences, then give your probability.\n"
        "Final line must be: PROBABILITY: X.XX\n"
    )

    user = (
        f"Event: {title}\n"
        f"\nExternal data (if available):\n{ctx}\n"
        f"\nAnalyze and estimate the YES probability:"
    )

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.3,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()

        # Try to find "PROBABILITY: X.XX" pattern first
        match = re.search(r"PROBABILITY:\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, re.IGNORECASE)
        if not match:
            # Fallback: find any decimal number in the last line
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", lines[-1])
        if not match:
            # Final fallback: find any decimal in the whole text
            match = re.search(r"(0\.\d+)", text)

        if not match:
            print(f"    individual_blind_eval: no probability found in: {text[:100]!r}")
            return None

        p = float(match.group(1))
        if 0.0 <= p <= 1.0:
            return p
        return None

    except Exception as e:
        print(f"    individual_blind_eval error: {e}")
        return None


# ── Batch Blind Screening (Phase 1 - quick filter) ────────
def batch_blind_screen(markets_batch, api_key, model):
    """
    Quick batch screening WITHOUT market prices.
    Uses smaller batches (5) with today's date for better quality.
    Returns: dict of {batch_index: probability}
    """
    if not markets_batch:
        return {}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = []
    for i, m in enumerate(markets_batch):
        lines.append(f"{i+1}. {m['title']}")

    system = (
        "You are a calibrated probability estimator for prediction markets.\n"
        f"Today's date: {today}\n"
        "\n"
        "For each question, estimate the probability that YES is correct.\n"
        "\n"
        "Guidelines:\n"
        "- Use your knowledge of current events, base rates, and domain expertise.\n"
        "- Be INDEPENDENT. Form your own view.\n"
        "- Think about what is actually likely given everything you know.\n"
        "- Consider: Is this event common or rare? What are the base rates?\n"
        "- For narrow ranges (e.g., 'between X and Y'), probabilities are usually 0.05-0.30.\n"
        "- For broad events that seem likely, use 0.60-0.90.\n"
        "- For things that probably won't happen, use 0.05-0.25.\n"
        "- Be specific and use the FULL range. 0.73 is better than 0.70.\n"
        "\n"
        "Output: one probability per line.\n"
        "Format: NUMBER: PROBABILITY\n"
        "Example:\n1: 0.35\n2: 0.72\n3: 0.08\n"
    )

    user = (
        "Estimate the YES probability for each:\n\n"
        + "\n".join(lines)
        + "\n\nProbabilities:"
    )

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.3,
                "max_tokens": len(markets_batch) * 15 + 50,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=90,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()

        results = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\d+)\s*[:\.]\s*(0(?:\.\d+)?|1(?:\.0+)?)", line)
            if match:
                idx = int(match.group(1)) - 1
                prob = float(match.group(2))
                if 0.0 <= prob <= 1.0 and 0 <= idx < len(markets_batch):
                    results[idx] = prob

        return results

    except Exception as e:
        print(f"  batch_blind_screen error: {e}")
        return {}


# ── Kelly Criterion (Half-Kelly for safety) ────────────────
def kelly_fraction(p: float, price: float) -> float:
    if price <= 0.0 or price >= 1.0:
        return 0.0
    b = (1.0 / price) - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    if f < 0:
        f = 0.0
    f = f * 0.5  # HALF-KELLY: safer, survives estimation errors
    if f > KELLY_MAX:
        f = KELLY_MAX
    return f


# ── State Persistence ─────────────────────────────────────
def load_state(state_dir: str) -> dict:
    path = os.path.join(state_dir, "state.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def save_state(state_dir: str, state: dict):
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "state.json")
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2)


# ── Edge Calculation ──────────────────────────────────────
def calculate_edge(fair: float, yes_buy: float, yes_sell: float):
    """
    Calculate edges for both BUY YES and SELL YES sides.
    Returns: (side_str, net_edge, raw_edge, exec_price)
    """
    buy_edge = (fair - yes_buy) / yes_buy if yes_buy > 0 else -999
    buy_net = buy_edge - FEE_RATE

    sell_edge = (yes_sell - fair) / (1.0 - yes_sell) if yes_sell < 1.0 else -999
    sell_net = sell_edge - FEE_RATE

    if buy_net >= sell_net:
        return "BUY", buy_net, buy_edge, yes_buy
    else:
        return "SELL", sell_net, sell_edge, yes_sell


def should_trade_market(market):
    """
    Returns (should_trade: bool, reason: str).
    Filters based on ticker to prefer LLM-advantage markets.
    """
    # Extract ticker
    events = market.get("events", [])
    if not events:
        return False, "no events"

    ticker = events[0].get("ticker", "").lower()
    if not ticker:
        return True, "no ticker (neutral)"  # Allow if no ticker

    # Check AVOID patterns first (higher priority)
    for pattern in AVOID_TICKER_PATTERNS:
        if pattern.lower() in ticker:
            return False, f"avoid:{pattern}"

    # Check PREFER patterns
    for pattern in PREFER_TICKER_PATTERNS:
        if pattern.lower() in ticker:
            return True, f"prefer:{pattern}"

    # Neutral: allow but log
    return True, "neutral"


# ── Main ───────────────────────────────────────────────────
def main():
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    scan_markets = int(os.getenv("SCAN_MARKETS", "1000"))
    max_tokens = int(os.getenv("MAX_TOKENS", "500"))
    min_order_usd = float(os.getenv("MIN_USD_ORDER", "1.0"))
    max_orders = int(os.getenv("MAX_ORDERS_PER_RUN", "5"))
    state_dir = os.getenv("STATE_DIR", "state")

    # Secrets
    private_key = env("PM_PRIVATE_KEY").strip()
    funder = env("PM_FUNDER").strip()
    signature_type = int(env("PM_SIGNATURE_TYPE"))

    if not is_hex_bytes(private_key, 32):
        raise RuntimeError("PM_PRIVATE_KEY must be 32 bytes hex (0x + 64 hex chars)")
    if not is_hex_bytes(funder, 20):
        raise RuntimeError("PM_FUNDER must be 20 bytes address hex (0x + 40 hex chars)")

    api_key = env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    client = ClobClient(
        HOST,
        key=private_key,
        chain_id=CHAIN_ID,
        signature_type=signature_type,
        funder=funder,
    )
    client.set_api_creds(client.create_or_derive_api_creds())

    state = load_state(state_dir)

    # ── Phase 1: Scan markets ──
    print(f"\n{'='*60}")
    print(f"POLYMARKET AUTONOMOUS AGENT (v4.2 - Category Filtering)")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"DRY_RUN={dry_run}, SCAN={scan_markets}, MAX_EVAL={max_tokens}")
    print(f"SPREAD_MAX={SPREAD_MAX}, KELLY_MAX={KELLY_MAX} (half-Kelly)")
    print(f"MIN_VOLUME={MIN_VOLUME}, MIN_LIQUIDITY={MIN_LIQUIDITY}")
    print(f"BATCH_SIZE={BATCH_SIZE}, CANDIDATE_MIN_DIFF={CANDIDATE_MIN_DIFF}")
    print(f"MAX_ORDERS_PER_RUN={max_orders}")
    print(f"{'='*60}\n")

    markets = gamma_markets(scan_markets)
    tradable = extract_tradable_markets(markets, max_tokens)

    print(f"\nTradable markets: {len(tradable)}")

    if not tradable:
        msg = (
            f"No tradable markets found.\n"
            f"Gamma returned {len(markets)} markets but 0 passed filters.\n"
            f"Filters: enableOrderBook, clobTokenIds, price, spread<={SPREAD_MAX}, "
            f"volume>={MIN_VOLUME}, liquidity>={MIN_LIQUIDITY}"
        )
        print(msg)
        gh_issue("run: 0 tradable markets", msg)
        return

    # ── Phase 2: External context (best-effort) ──
    print("\n== Phase 2: Fetching external context ==")

    try:
        sports_data = fetch_sports_context()
        if sports_data:
            n_injuries = len(sports_data.get("injury_news_titles", []))
            print(f"* Sports context: {n_injuries} injury headlines")
        else:
            print("* Sports context: no relevant news")
    except Exception as e:
        sports_data = {"error": f"{type(e).__name__}: {e}"}
        print(f"* Sports context error: {e}")

    try:
        crypto_data = fetch_crypto_context()
        if crypto_data and "fear_greed" in crypto_data:
            fg = crypto_data["fear_greed"]
            if fg:
                print(
                    f"* Crypto context: Fear & Greed = {fg.get('value')} "
                    f"({fg.get('value_classification')})"
                )
        else:
            print("* Crypto context: partial data")
    except Exception as e:
        crypto_data = {"error": f"{type(e).__name__}: {e}"}
        print(f"* Crypto context error: {e}")

    # ── Phase 3: CLOB price refinement ──
    print("\n== Phase 3: CLOB price refinement ==")
    token_ids = [m["tid"] for m in tradable]
    try:
        clob_data = clob_prices(token_ids)
        print(f"* CLOB prices fetched for {len(clob_data)}/{len(token_ids)} markets")
    except Exception as e:
        clob_data = {}
        print(f"* CLOB prices error (using Gamma prices): {e}")

    for mkt in tradable:
        cd = clob_data.get(mkt["tid"])
        if cd:
            if "BUY" in cd:
                mkt["yes_buy"] = cd["BUY"]
            if "SELL" in cd:
                mkt["yes_sell"] = cd["SELL"]

    # ── Phase 4a: Weather markets (direct data, no LLM needed) ──
    print("\n== Phase 4a: Weather markets (direct data) ==")
    decisions = []
    weather_count = 0

    for mkt in tradable:
        if classify_market_type(mkt["title"]) != "weather":
            continue
        weather_count += 1

        wp, werr = fair_prob_weather(mkt["title"])
        if wp is None:
            print(f"  Weather skip ({werr}): {mkt['title'][:60]}")
            continue

        yes_buy = mkt["yes_buy"]
        yes_sell = mkt["yes_sell"]
        if not (0.0 < yes_buy < 1.0 and 0.0 < yes_sell < 1.0):
            continue

        side_str, net_edge, raw_edge, exec_price = calculate_edge(wp, yes_buy, yes_sell)
        th = dynamic_edge_threshold(yes_buy, yes_sell, has_data=True)

        print(f"  Weather: {mkt['title'][:60]}")
        print(f"    Data prob: {wp:.4f}, Market: {(yes_buy+yes_sell)/2:.4f}, "
              f"Edge: {net_edge:.4f} (threshold: {th:.4f})")

        if net_edge >= th:
            decisions.append({
                "net_edge": net_edge,
                "tid": mkt["tid"],
                "title": mkt["title"],
                "fair": wp,
                "exec_price": exec_price,
                "side": side_str,
                "source": "weather_data",
            })

    print(f"  Weather markets: {weather_count}, with edge: {len([d for d in decisions if d['source']=='weather_data'])}")

    # ── Phase 4b: Blind batch screening (quick pre-filter) ──
    print(f"\n== Phase 4b: Blind batch screening ==")

    # Sort by priority: crypto > sports > general
    non_weather = []
    for mkt in tradable:
        mtype = classify_market_type(mkt["title"])
        if mtype == "weather":
            continue
        if not (0.0 < mkt["yes_buy"] < 1.0 and 0.0 < mkt["yes_sell"] < 1.0):
            continue
        priority = 2
        if mtype == "crypto":
            priority = 0
        elif mtype == "sports":
            priority = 1
        non_weather.append((priority, mkt))

    non_weather.sort(key=lambda x: x[0])
    non_weather_markets = [m for _, m in non_weather]

    print(f"  Non-weather markets to screen: {len(non_weather_markets)}")

    blind_estimates = {}
    batch_count = 0
    batch_success = 0

    for i in range(0, len(non_weather_markets), BATCH_SIZE):
        batch = non_weather_markets[i:i + BATCH_SIZE]
        batch_count += 1

        results = batch_blind_screen(batch, api_key, model)
        if results:
            batch_success += 1

        for idx, prob in results.items():
            global_idx = i + idx
            if global_idx < len(non_weather_markets):
                blind_estimates[global_idx] = prob

        if i + BATCH_SIZE < len(non_weather_markets):
            time.sleep(0.3)

    print(f"  Batches: {batch_count}, successful: {batch_success}")
    print(f"  Blind estimates obtained: {len(blind_estimates)}/{len(non_weather_markets)}")

    # ── Phase 4c: Candidate selection ──
    print(f"\n== Phase 4c: Candidate selection ==")

    candidates = []
    for idx, blind_prob in blind_estimates.items():
        mkt = non_weather_markets[idx]
        midpoint = (mkt["yes_buy"] + mkt["yes_sell"]) / 2.0
        abs_diff = abs(blind_prob - midpoint)

        if abs_diff >= CANDIDATE_MIN_DIFF:
            candidates.append({
                "abs_diff": abs_diff,
                "idx": idx,
                "blind_prob": blind_prob,
                "mkt": mkt,
            })

    candidates.sort(key=lambda x: x["abs_diff"], reverse=True)
    candidates = candidates[:50]

    print(f"  Candidates with |blind - market| >= {CANDIDATE_MIN_DIFF}: {len(candidates)}")
    if candidates:
        top = candidates[0]
        print(f"  Top: diff={top['abs_diff']:.3f}, blind={top['blind_prob']:.3f}, "
              f"market={(top['mkt']['yes_buy']+top['mkt']['yes_sell'])/2:.3f}, "
              f"title={top['mkt']['title'][:50]}")

    # ── Phase 4d: Dual-pass batch verification ──
    # v4 KEY CHANGE: Use batch estimates DIRECTLY instead of individual eval.
    # Individual eval with step-by-step reasoning converges to market prices,
    # erasing the edge. Batch "gut" estimates are more useful for finding mispricings.
    # We run a SECOND independent batch pass on candidates for noise reduction.
    print(f"\n== Phase 4d: Dual-pass batch verification (NO individual eval) ==")

    verified = []
    diag_rows = []

    if not candidates:
        print("  No candidates to verify.")
    else:
        # Pass 2: Re-screen candidates in fresh batches
        cand_markets = [c["mkt"] for c in candidates]
        pass2_estimates = {}
        pass2_batches = 0

        for i in range(0, len(cand_markets), BATCH_SIZE):
            batch = cand_markets[i:i + BATCH_SIZE]
            pass2_batches += 1
            results = batch_blind_screen(batch, api_key, model)
            for idx, prob in results.items():
                global_idx = i + idx
                if global_idx < len(candidates):
                    pass2_estimates[global_idx] = prob
            if i + BATCH_SIZE < len(cand_markets):
                time.sleep(0.3)

        print(f"  Pass 2 batches: {pass2_batches}, estimates: {len(pass2_estimates)}/{len(candidates)}")

        # Combine pass 1 and pass 2, filter for directional agreement
        verified = []
        for ci, cand in enumerate(candidates):
            mkt = cand["mkt"]
            pass1 = cand["blind_prob"]
            pass2 = pass2_estimates.get(ci)
            if pass2 is None:
                continue

            midpoint = (mkt["yes_buy"] + mkt["yes_sell"]) / 2.0

            # Both passes must agree on direction vs market
            pass1_above = pass1 > midpoint
            pass2_above = pass2 > midpoint
            if pass1_above != pass2_above:
                print(f"  SKIP (disagreement): pass1={pass1:.3f}, pass2={pass2:.3f}, "
                      f"market={midpoint:.3f} | {mkt['title'][:50]}")
                continue

            # Use CONSERVATIVE estimate (closer to market) to reduce false signals.
            # If both say below market, use the higher one (closer to market).
            # If both say above market, use the lower one (closer to market).
            if pass1 < midpoint:
                conservative_est = max(pass1, pass2)
            else:
                conservative_est = min(pass1, pass2)

            verified.append({
                "mkt": mkt,
                "pass1": pass1,
                "pass2": pass2,
                "avg_est": conservative_est,
                "midpoint": midpoint,
            })

        print(f"  Directionally agreed: {len(verified)}/{len(candidates)}")

        # ── Phase 4e: Post-CLOB spread recheck ──
        print(f"\n== Phase 4e: Post-CLOB spread recheck ==")

        if verified:
            recheck_tids = [v["mkt"]["tid"] for v in verified]
            try:
                fresh_clob = clob_prices(recheck_tids)
                print(f"  Fresh CLOB prices for {len(fresh_clob)}/{len(recheck_tids)} candidates")
            except Exception as e:
                fresh_clob = {}
                print(f"  Fresh CLOB error (using earlier prices): {e}")

            for v in verified:
                mkt = v["mkt"]
                fc = fresh_clob.get(mkt["tid"])
                if fc:
                    if "BUY" in fc:
                        mkt["yes_buy"] = fc["BUY"]
                    if "SELL" in fc:
                        mkt["yes_sell"] = fc["SELL"]
                    v["midpoint"] = (mkt["yes_buy"] + mkt["yes_sell"]) / 2.0

        # Calculate edge and make decisions
        diag_rows = []
        for v in verified:
            mkt = v["mkt"]
            avg_est = v["avg_est"]
            yes_buy = mkt["yes_buy"]
            yes_sell = mkt["yes_sell"]
            tid = mkt["tid"]
            title = mkt["title"]

            # Re-check spread after CLOB refresh
            spread = yes_buy - yes_sell
            if spread > SPREAD_MAX or spread < 0:
                print(f"  SKIP (spread={spread:.3f}): {title[:50]}")
                continue

            if not (0.0 < yes_buy < 1.0 and 0.0 < yes_sell < 1.0):
                continue

            mtype = classify_market_type(title)
            has_data = mtype in ("crypto", "sports")

            side_str, net_edge, raw_edge, exec_price = calculate_edge(avg_est, yes_buy, yes_sell)
            th = dynamic_edge_threshold(yes_buy, yes_sell, has_data=has_data)

            midpoint = v["midpoint"]
            status = "PASS" if net_edge >= th else "FAIL"
            print(f"\n  {title[:65]}")
            print(f"    Pass1={v['pass1']:.3f}, Pass2={v['pass2']:.3f}, Avg={avg_est:.3f}")
            print(f"    Market={midpoint:.3f}, Side={side_str}, Edge={net_edge:.4f} "
                  f"(threshold={th:.4f}) -> {status}")

            diag_rows.append({
                "title": title[:60],
                "pass1": round(v["pass1"], 3),
                "pass2": round(v["pass2"], 3),
                "avg": round(avg_est, 3),
                "market": round(midpoint, 3),
                "side": side_str,
                "edge": round(net_edge, 4),
                "threshold": round(th, 4),
                "status": status,
            })

            if net_edge >= th:
                decisions.append({
                    "net_edge": net_edge,
                    "tid": tid,
                    "title": title,
                    "fair": avg_est,
                    "exec_price": exec_price,
                    "side": side_str,
                    "source": f"dual_pass_{'data' if has_data else 'llm'}",
                })

    # ── Summary ──
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tradable: {len(tradable)}")
    print(f"Weather evaluated: {weather_count}")
    print(f"Blind screened (pass 1): {len(blind_estimates)}")
    print(f"Candidates (|diff| >= {CANDIDATE_MIN_DIFF}): {len(candidates)}")
    print(f"Dual-pass verified: {len(verified) if candidates else 0}")
    print(f"Decisions (edge >= threshold): {len(decisions)}")

    if decisions:
        decisions.sort(key=lambda x: x["net_edge"], reverse=True)
        print(f"\nTop opportunities:")
        for i, d in enumerate(decisions[:10]):
            print(f"  {i+1}. [{d['side']}] edge={d['net_edge']:.4f} "
                  f"fair={d['fair']:.3f} price={d['exec_price']:.3f} "
                  f"src={d['source']} | {d['title'][:50]}")

    if not decisions:
        print(f"\n-- No profitable opportunities found this run")
        diag_text = ""
        if candidates and diag_rows:
            diag_text = "\n\n## Top Candidates (Diagnostics)\n"
            for i, row in enumerate(diag_rows[:10]):
                diag_text += (
                    f"\n{i+1}. **{row['title']}**\n"
                    f"   Pass1={row['pass1']}, Pass2={row['pass2']}, "
                    f"Avg={row['avg']}, Market={row['market']}\n"
                    f"   Side={row['side']}, Edge={row['edge']}, "
                    f"Threshold={row['threshold']}, Status={row['status']}\n"
                )
        n_verified = len(verified) if candidates else 0
        gh_issue(
            "run: no edge found",
            f"v4 Batch-direct pipeline found no opportunities.\n"
            f"Tradable: {len(tradable)}\n"
            f"Blind screened (pass 1): {len(blind_estimates)}\n"
            f"Candidates (|diff| >= {CANDIDATE_MIN_DIFF}): {len(candidates)}\n"
            f"Dual-pass verified: {n_verified}"
            f"{diag_text}",
        )
        return

    # ── Phase 5: Execution ──
    print(f"\n{'='*60}")
    print("EXECUTION PHASE")
    print(f"{'='*60}")

    bankroll = float(os.getenv("BANKROLL_USD", "50.0"))
    api_budget = float(os.getenv("API_BUDGET_USD", "0.0"))

    if bankroll - api_budget <= 0:
        gh_issue(
            "run: STOP (balance would hit $0)",
            f"bankroll={bankroll:.2f}, api_budget={api_budget:.2f}",
        )
        return

    executed = 0
    logs = []
    trade_records = state.get("trades", [])

    for d in decisions[:max_orders]:
        net_edge = d["net_edge"]
        tid = d["tid"]
        title = d["title"]
        fair = d["fair"]
        exec_price = d["exec_price"]
        side_str = d["side"]

        if side_str == "BUY":
            f = kelly_fraction(fair, exec_price)
        else:
            f = kelly_fraction(1.0 - fair, 1.0 - exec_price)

        usd = bankroll * f

        if usd < min_order_usd:
            logs.append(
                f"skip (too small): side={side_str} edge={net_edge:.3f} "
                f"kelly={f:.4f} usd={usd:.2f} title={title[:60]}"
            )
            continue

        if side_str == "BUY":
            size = usd / exec_price
            order_side = BUY
            order_price = exec_price
        else:
            size = usd / (1.0 - exec_price)
            order_side = SELL
            order_price = exec_price

        order = OrderArgs(
            price=order_price,
            size=size,
            side=order_side,
            token_id=tid,
        )

        if dry_run:
            try:
                signed = client.create_order(order)
                logs.append(
                    f"DRY_RUN OK: side={side_str} edge={net_edge:.3f} "
                    f"fair={fair:.3f} price={exec_price:.3f} kelly={f:.4f} "
                    f"usd={usd:.2f} size={size:.4f} src={d['source']} "
                    f"token={tid[:16]}... title={title[:50]}"
                )
                executed += 1
            except Exception as e:
                logs.append(
                    f"DRY_RUN FAIL: side={side_str} edge={net_edge:.3f} "
                    f"error={type(e).__name__}: {e} title={title[:50]}"
                )
        else:
            try:
                signed = client.create_order(order)
                resp = client.post_order(signed)
                logs.append(
                    f"LIVE OK: side={side_str} edge={net_edge:.3f} "
                    f"fair={fair:.3f} price={exec_price:.3f} kelly={f:.4f} "
                    f"usd={usd:.2f} size={size:.4f} src={d['source']} "
                    f"token={tid[:16]}... title={title[:50]}\nresp={resp}"
                )
                executed += 1
            except Exception as e:
                logs.append(
                    f"LIVE FAIL: side={side_str} edge={net_edge:.3f} "
                    f"error={type(e).__name__}: {e} title={title[:50]}"
                )

        trade_records.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "tid": tid[:20],
            "title": title[:100],
            "side": side_str,
            "fair": round(fair, 4),
            "price": round(exec_price, 4),
            "edge": round(net_edge, 4),
            "usd": round(usd, 2),
            "kelly": round(f, 4),
            "dry_run": dry_run,
            "source": d["source"],
        })

    print(f"\nOrders attempted: {min(len(decisions), max_orders)}")
    print(f"Executed: {executed}")
    for log in logs:
        print(f"  {log}")

    # Update state
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_tradable"] = len(tradable)
    state["last_blind_screened"] = len(blind_estimates)
    state["last_candidates"] = len(candidates)
    state["last_dual_pass_verified"] = len(verified) if candidates else 0
    state["last_decisions"] = len(decisions)
    state["last_executed"] = executed
    state["trades"] = trade_records[-200:]
    save_state(state_dir, state)

    title_issue = (
        f"run: {'DRY_RUN' if dry_run else 'LIVE'} "
        f"scanned={len(tradable)} candidates={len(candidates)} "
        f"decisions={len(decisions)} executed={executed}"
    )
    body = "\n".join(logs[:50])
    gh_issue(title_issue, body)


if __name__ == "__main__":
    main()
