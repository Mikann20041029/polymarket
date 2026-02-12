import os
import json
import time
import math
import re
from datetime import datetime, timezone
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

EDGE_MIN = float(os.getenv("EDGE_MIN", "0.08"))
KELLY_MAX = float(os.getenv("KELLY_MAX", "0.06"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.00"))
SLIPPAGE_MAX = float(os.getenv("SLIPPAGE_MAX", "0.01"))
SPREAD_MAX = float(os.getenv("SPREAD_MAX", "0.03"))


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


# ── GitHub Issue Reporter ──────────────────────────────────
def gh_issue(title: str, body: str):
    token = env("GITHUB_TOKEN")
    repo = env("GITHUB_REPOSITORY")
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
        print("crypto_features error:", e)
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


def extract_yes_token_ids(markets, max_tokens: int):
    token_ids = []
    picked = []
    for m in markets:
        if len(token_ids) >= max_tokens:
            break
        if not m.get("enableOrderBook", False):
            continue

        v = m.get("clobTokenIds")
        if v is None:
            continue

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
            picked.append(
                (tid, str(m.get("question") or m.get("title") or "unknown"))
            )
    return token_ids, picked


def clob_prices(token_ids):
    """
    Returns: {tid: {"BUY": best_ask, "SELL": best_bid}, ...}
    BUY  = price we pay to buy YES = best ask
    SELL = price we get selling YES = best bid
    """
    out = {}

    for tid in token_ids:
        tid = str(tid)
        try:
            r = requests.get(
                f"{HOST}/book",
                params={"token_id": tid},
                timeout=30,
            )
            r.raise_for_status()
            book = r.json() or {}
        except Exception as e:
            print(f"  clob_prices skip {tid[:12]}...: {e}")
            continue

        bids = book.get("bids") or []
        asks = book.get("asks") or []

        def _price(x):
            if isinstance(x, dict):
                return float(x.get("price", 0))
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


# ── Dynamic Edge Threshold ─────────────────────────────────
def dynamic_edge_threshold(yes_buy: float, yes_sell: float) -> float:
    if yes_buy <= 0.03 or yes_buy >= 0.97:
        return 0.10
    if yes_sell <= 0 or yes_sell >= 1:
        return 0.10

    spread = max(0.0, yes_buy - yes_sell)

    if spread <= 0.01:
        return 0.04
    if spread <= 0.03:
        return 0.06
    if spread <= 0.06:
        return 0.08
    return 0.10


# ── LLM Fair Probability ──────────────────────────────────
def openai_fair_prob(
    title: str,
    yes_buy: float,
    yes_sell: float,
    external_context: dict | None = None,
) -> float | None:
    """
    Returns fair probability [0,1] or None on failure.
    """
    api_key = env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    FAIR_MODE = os.getenv("FAIR_MODE", "FULL").strip().upper()

    ctx = external_context or {}

    if FAIR_MODE == "TITLE_ONLY":
        ctx_for_llm = {}
        title_for_llm = title
    elif FAIR_MODE == "EXTERNAL_ONLY":
        ctx_for_llm = ctx
        title_for_llm = title
    else:  # FULL
        ctx_for_llm = ctx
        title_for_llm = title

    ext = safe_json(ctx_for_llm)

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

    # FIX: Build USER prompt correctly with conditional lines
    user_lines = [
        "Estimate fair probability (YES) for this Polymarket event.",
        f"Title: {title_for_llm}",
        f"FAIR_MODE: {FAIR_MODE}",
    ]
    if FAIR_MODE == "EXTERNAL_ONLY":
        user_lines.append(
            "IMPORTANT: In EXTERNAL_ONLY mode, IGNORE the Title and "
            "rely ONLY on External context + prices."
        )
    user_lines.extend([
        f"Current YES BUY price: {yes_buy}",
        f"Current YES SELL price: {yes_sell}",
        "",
        "External context (may include weather/crypto/sports/etc):",
        ext,
        "",
        "Output format: just a decimal number in [0,1].",
    ])
    USER = "\n".join(user_lines)

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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

        # FIX: Extract number even if LLM returns extra text
        match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
        if not match:
            print(f"  OpenAI returned non-numeric: {text!r}")
            return None
        p = float(match.group(1))
        if not (0.0 <= p <= 1.0):
            print(f"  OpenAI returned out-of-range prob: {p}")
            return None
        return p

    except Exception as e:
        print(f"  OpenAI error: {type(e).__name__}: {e}")
        return None


# ── Kelly Criterion ────────────────────────────────────────
def kelly_fraction(p: float, price: float) -> float:
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


# ── Main ───────────────────────────────────────────────────
def main():
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    scan_markets = int(os.getenv("SCAN_MARKETS", "1000"))
    max_tokens = int(os.getenv("MAX_TOKENS", "500"))
    min_order_usd = float(os.getenv("MIN_USD_ORDER", "1.0"))
    max_orders = int(os.getenv("MAX_ORDERS_PER_RUN", "1"))
    state_dir = os.getenv("STATE_DIR", "state")

    # Secrets
    private_key = env("PM_PRIVATE_KEY").strip()
    funder = env("PM_FUNDER").strip()
    signature_type = int(env("PM_SIGNATURE_TYPE"))

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

    # Load persistent state
    state = load_state(state_dir)

    # ── Scan markets ──
    markets = gamma_markets(scan_markets)
    token_ids, picked = extract_yes_token_ids(markets, max_tokens)

    print(f"\n{'='*60}")
    print(f"Starting market scan: {len(picked)} markets to evaluate")
    print(f"DRY_RUN={dry_run}")
    print(f"{'='*60}\n")

    # ── External context (best-effort) ──
    try:
        sports_data = fetch_sports_context()
    except Exception as e:
        sports_data = {"error": f"{type(e).__name__}: {e}"}

    try:
        crypto_data = fetch_crypto_context()
        print("* Crypto context fetched successfully")
        if crypto_data and "fear_greed" in crypto_data:
            fg = crypto_data["fear_greed"]
            if fg:
                print(
                    f"  Fear & Greed: {fg.get('value')} "
                    f"({fg.get('value_classification')})"
                )
    except Exception as e:
        crypto_data = {"error": f"{type(e).__name__}: {e}"}
        print(f"* Crypto context error: {e}")

    # ── Prices ──
    try:
        prices = clob_prices(token_ids)
        print(f"* Prices fetched for {len(prices)} markets\n")
    except Exception as e:
        prices = {}
        print(f"* clob_prices error: {type(e).__name__}: {e}")

    decisions = []
    evaluated = 0
    best = None  # (edge, th, title, tid, fair, buy, sell)

    for tid, title in picked:
        crypto_features_data = None

        p = prices.get(tid)
        if not p:
            continue
        try:
            yes_buy = float(p.get("BUY", float("nan")))
            yes_sell = float(p.get("SELL", float("nan")))
        except (ValueError, TypeError):
            continue
        if not (0.0 < yes_buy < 1.0 and 0.0 < yes_sell < 1.0):
            continue

        # Spread check
        spread = yes_buy - yes_sell
        if spread > SPREAD_MAX:
            continue

        if evaluated < 10:
            print(f"\n--- Market #{evaluated+1} ---")
            print(f"Title: {title[:80]}")
            print(f"Token ID: {tid[:16]}...")
            print(f"YES BUY: {yes_buy:.4f}, YES SELL: {yes_sell:.4f}")

        if "bitcoin" in title.lower() or "btc" in title.lower():
            crypto_features_data = crypto_features("bitcoin")

        mtype = classify_market_type(title)

        if evaluated < 10:
            print(f"Market type: {mtype or 'general'}")

        # Weather data per-market
        weather_data = None
        if mtype == "weather":
            wp, werr = fair_prob_weather(title)
            if wp is not None:
                weather_data = {
                    "p_any_rain": wp,
                    "source": "open-meteo precipitation_probability -> any-rain",
                }
            else:
                weather_data = {"error": werr or "weather prob unavailable"}

        ctx = {
            "weather": weather_data,
            "sports": sports_data,
            "crypto": {"base": crypto_data, "features": crypto_features_data},
        }

        fair = openai_fair_prob(title, yes_buy, yes_sell, external_context=ctx)

        # FIX: Skip if LLM failed to return a valid probability
        if fair is None:
            if evaluated < 10:
                print("Fair prob: FAILED (skipping)")
            evaluated += 1
            continue

        # Edge: how much our fair value exceeds market price (for BUY YES)
        edge = (fair - yes_buy) / yes_buy

        # Account for fees in edge calculation
        net_edge = edge - FEE_RATE

        th = dynamic_edge_threshold(yes_buy, yes_sell)

        if evaluated < 10:
            print(f"Fair prob: {fair:.4f}")
            print(f"Edge: {edge:.4f}, Net edge: {net_edge:.4f} (threshold: {th:.4f})")
            print(f"Pass: {'YES' if net_edge >= th else 'NO'}")

        if best is None or edge > best[0]:
            best = (edge, th, title, tid, fair, yes_buy, yes_sell)

        if net_edge >= th:
            decisions.append((net_edge, tid, title, fair, yes_buy, yes_sell))

        evaluated += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total markets evaluated: {evaluated}")
    print(f"Decisions found (net_edge >= threshold): {len(decisions)}")

    if best:
        edge, th, title_b, tid_b, fair_b, buy_b, sell_b = best
        print(f"\nBest opportunity:")
        print(f"  Title: {title_b[:80]}")
        print(f"  Edge: {edge:.4f} (threshold: {th:.4f})")
        print(f"  Fair: {fair_b:.4f}, Market BUY: {buy_b:.4f}")
        print(f"  Token: {tid_b[:16]}...")

    if not decisions:
        print(f"\n-- No profitable opportunities found")
        print(f"  - All {evaluated} markets had edge < threshold")
        best_info = ""
        if best:
            print(f"  - Best edge was {best[0]:.4f} (needed {best[1]:.4f})")
            best_info = f"\nBest edge: {best[0]:.4f} (needed {best[1]:.4f})"
        gh_issue(
            "run: no edge (dynamic threshold)",
            f"No mispricing found.\nEvaluated: {evaluated} markets{best_info}",
        )
        return

    # ── Bankroll ──
    bankroll = float(os.getenv("BANKROLL_USD", "50.0"))
    api_budget = float(os.getenv("API_BUDGET_USD", "0.0"))
    if bankroll - api_budget <= 0:
        gh_issue(
            "run: STOP (balance would hit $0)",
            f"bankroll={bankroll:.2f}, api_budget={api_budget:.2f}",
        )
        return

    # Sort by edge descending
    decisions.sort(key=lambda x: x[0], reverse=True)

    executed = 0
    logs = []
    for net_edge, tid, title, fair, yes_buy, yes_sell in decisions[:max_orders]:
        f = kelly_fraction(fair, yes_buy)
        usd = bankroll * f

        if usd < min_order_usd:
            logs.append(
                f"skip (too small): edge={net_edge:.3f} kelly={f:.4f} "
                f"usd={usd:.2f} title={title}"
            )
            continue

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
                f"DRY_RUN build: edge={net_edge:.3f} fair={fair:.3f} "
                f"buy={yes_buy:.3f} kelly={f:.4f} usd={usd:.2f} "
                f"size={size:.4f} token={tid} title={title}"
            )
        else:
            signed = client.create_order(order)
            resp = client.post_order(signed)
            logs.append(
                f"LIVE post: edge={net_edge:.3f} fair={fair:.3f} "
                f"buy={yes_buy:.3f} kelly={f:.4f} usd={usd:.2f} "
                f"size={size:.4f} token={tid} title={title}\nresp={resp}"
            )
            executed += 1

    # Update state
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_evaluated"] = evaluated
    state["last_decisions"] = len(decisions)
    state["last_executed"] = executed
    save_state(state_dir, state)

    title_issue = (
        f"run: {'DRY_RUN' if dry_run else 'LIVE'} "
        f"decisions={len(decisions)} executed={executed}"
    )
    body = "\n".join(logs[:50])
    gh_issue(title_issue, body)


if __name__ == "__main__":
    main()
