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

EDGE_MIN = float(os.getenv("EDGE_MIN", "0.08"))
KELLY_MAX = float(os.getenv("KELLY_MAX", "0.06"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.00"))
SLIPPAGE_MAX = float(os.getenv("SLIPPAGE_MAX", "0.01"))
SPREAD_MAX = float(os.getenv("SPREAD_MAX", "0.05"))
MAX_DAYS_TO_RESOLUTION = int(os.getenv("MAX_DAYS_TO_RESOLUTION", "14"))
CLOB_WORKERS = int(os.getenv("CLOB_WORKERS", "10"))


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
    """
    Gamma API から CLOB対応のアクティブ市場を取得。
    ページネーション対応: 1回のリクエストで最大100件なので、ループで取得。
    """
    all_markets = []
    offset = 0
    page_size = min(limit, 100)  # Gamma API は1回100件が上限の場合がある

    while len(all_markets) < limit:
        try:
            r = requests.get(
                f"{GAMMA}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "enableOrderBook": "true",   # API側でフィルタ
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
            break  # 最終ページ
        offset += len(page)

    print(f"[Gamma] Fetched {len(all_markets)} markets (requested up to {limit})")

    # デバッグ: 最初の市場の構造を表示
    if all_markets:
        sample = all_markets[0]
        print(f"[Gamma] Sample market keys: {list(sample.keys())[:15]}")
        print(f"[Gamma] enableOrderBook = {sample.get('enableOrderBook')!r} (type={type(sample.get('enableOrderBook')).__name__})")
        print(f"[Gamma] clobTokenIds = {str(sample.get('clobTokenIds', 'MISSING'))[:80]}")
        end_date = sample.get("endDate") or sample.get("end_date_iso") or sample.get("endDateIso")
        print(f"[Gamma] endDate = {end_date}")

    return all_markets


def extract_yes_token_ids(markets, max_tokens: int):
    """
    市場リストから YES トークン ID を抽出。
    - enableOrderBook のチェックを堅牢化（string/bool両対応）
    - 短期市場を優先（MAX_DAYS_TO_RESOLUTION以内）
    - clobTokenIds は文字列JSON配列として返されることが多い
    """
    token_ids = []
    picked = []
    skipped_eob = 0
    skipped_no_clob = 0
    skipped_parse = 0
    skipped_end_date = 0
    now = datetime.now(timezone.utc)

    for m in markets:
        if len(token_ids) >= max_tokens:
            break

        # enableOrderBook チェック: bool True / string "true" 両対応
        eob = m.get("enableOrderBook")
        if eob is None:
            eob = m.get("enable_order_book")  # snake_case フォールバック
        if not (eob is True or str(eob).lower() == "true"):
            skipped_eob += 1
            continue

        # 短期市場フィルタ: 解決日が近い市場を優先
        end_date_str = (
            m.get("endDate")
            or m.get("end_date_iso")
            or m.get("endDateIso")
            or ""
        )
        if end_date_str and MAX_DAYS_TO_RESOLUTION > 0:
            try:
                end_dt = datetime.fromisoformat(
                    end_date_str.replace("Z", "+00:00")
                )
                days_left = (end_dt - now).total_seconds() / 86400
                if days_left < 0 or days_left > MAX_DAYS_TO_RESOLUTION:
                    skipped_end_date += 1
                    continue
            except Exception:
                pass  # パースできない場合はフィルタしない

        # clobTokenIds 取得
        v = m.get("clobTokenIds")
        if v is None:
            v = m.get("clob_token_ids")  # snake_case フォールバック
        if v is None:
            skipped_no_clob += 1
            continue

        # 文字列JSON → パース
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                skipped_parse += 1
                continue

        # YES トークン抽出
        yes = None
        if isinstance(v, dict):
            yes = v.get("YES") or v.get("Yes") or v.get("yes")
        elif isinstance(v, list) and len(v) > 0:
            yes = v[0]  # 配列の場合、index 0 = YES

        if yes:
            tid = str(yes)
            token_ids.append(tid)
            picked.append(
                (tid, str(m.get("question") or m.get("title") or "unknown"))
            )

    print(f"[Extract] Passed: {len(picked)}, "
          f"Skipped: eob={skipped_eob}, no_clob={skipped_no_clob}, "
          f"parse={skipped_parse}, end_date={skipped_end_date}")
    return token_ids, picked


def _fetch_one_book(tid: str):
    """1つのトークンのオーダーブックを取得（並列実行用）"""
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
    """
    並列でオーダーブックを取得。
    Returns: {tid: {"BUY": best_ask, "SELL": best_bid}, ...}
    """
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
    elif FAIR_MODE == "EXTERNAL_ONLY":
        ctx_for_llm = ctx
    else:  # FULL
        ctx_for_llm = ctx

    ext = safe_json(ctx_for_llm)

    try:
        ext_flags = (
            f"ext(weather={'Y' if ctx_for_llm.get('weather') else 'N'},"
            f" sports={'Y' if ctx_for_llm.get('sports') else 'N'},"
            f" crypto={'Y' if ctx_for_llm.get('crypto') else 'N'})"
        )
    except Exception:
        ext_flags = "ext(weather=N, sports=N, crypto=N)"

    # 改善されたシステムプロンプト: キャリブレーション重視
    SYSTEM = (
        "You are a calibrated prediction-market probability estimator.\n"
        "Your survival depends on accuracy - overconfidence kills.\n"
        "\n"
        "Rules:\n"
        "1. Base rates matter. Most things don't happen. Default toward base rates.\n"
        "2. Be skeptical of extreme probabilities. Rarely use values below 0.05 or above 0.95.\n"
        "3. If you lack information, stay close to the market price - the market aggregates many participants' knowledge.\n"
        "4. Only deviate significantly from market price when external data provides STRONG evidence of mispricing.\n"
        "5. Account for the market's wisdom - many participants have already priced in public information.\n"
        "6. Consider time to resolution - events further away have more uncertainty.\n"
        f"\n{ext_flags}\n"
        "Return ONLY a decimal number between 0 and 1. No explanation.\n"
    )

    user_lines = [
        "Estimate the TRUE fair probability (YES) for this Polymarket event.",
        f"Title: {title}",
    ]
    if FAIR_MODE == "EXTERNAL_ONLY":
        user_lines.append(
            "IMPORTANT: In EXTERNAL_ONLY mode, IGNORE the Title and "
            "rely ONLY on External context + prices."
        )
    user_lines.extend([
        f"Current YES BUY (best ask): {yes_buy:.4f}",
        f"Current YES SELL (best bid): {yes_sell:.4f}",
        f"Market midpoint: {(yes_buy + yes_sell) / 2:.4f}",
        "",
        "External context (may include weather/crypto/sports/etc):",
        ext,
        "",
        "Output: just a decimal number in [0,1].",
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

        # Extract number even if LLM returns extra text
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
    """
    Binary market Kelly: price = cost per share, payout = 1 if win.
    b = (1/price) - 1  (net payout ratio)
    f* = (b*p - q) / b, capped at KELLY_MAX
    """
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
    max_tokens = int(os.getenv("MAX_TOKENS", "200"))
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

    # ── Phase 1: Scan markets ──
    print(f"\n{'='*60}")
    print(f"POLYMARKET AUTONOMOUS AGENT")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"DRY_RUN={dry_run}, SCAN={scan_markets}, MAX_TOKENS={max_tokens}")
    print(f"SPREAD_MAX={SPREAD_MAX}, KELLY_MAX={KELLY_MAX}, FEE_RATE={FEE_RATE}")
    print(f"MAX_DAYS_TO_RESOLUTION={MAX_DAYS_TO_RESOLUTION}")
    print(f"{'='*60}\n")

    markets = gamma_markets(scan_markets)
    token_ids, picked = extract_yes_token_ids(markets, max_tokens)

    print(f"\nMarkets to evaluate: {len(picked)}")

    if not picked:
        msg = (
            f"No tradable markets found.\n"
            f"Gamma returned {len(markets)} markets but 0 passed filters.\n"
            f"Check enableOrderBook, clobTokenIds, endDate filters."
        )
        print(msg)
        gh_issue("run: 0 tradable markets", msg)
        return

    # ── Phase 2: External context (best-effort) ──
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

    # ── Phase 3: Prices (parallel) ──
    try:
        prices = clob_prices(token_ids)
        print(f"* Prices fetched for {len(prices)}/{len(token_ids)} markets\n")
    except Exception as e:
        prices = {}
        print(f"* clob_prices error: {type(e).__name__}: {e}")

    # ── Phase 4: Evaluate ──
    decisions = []
    evaluated = 0
    skipped_no_price = 0
    skipped_invalid_price = 0
    skipped_spread = 0
    skipped_llm = 0
    best = None  # (edge, th, title, tid, fair, price, side_str)

    for tid, title in picked:
        crypto_features_data = None

        p = prices.get(tid)
        if not p:
            skipped_no_price += 1
            continue

        try:
            yes_buy = float(p.get("BUY", float("nan")))
            yes_sell = float(p.get("SELL", float("nan")))
        except (ValueError, TypeError):
            skipped_invalid_price += 1
            continue

        if not (0.0 < yes_buy < 1.0 and 0.0 < yes_sell < 1.0):
            skipped_invalid_price += 1
            continue

        # Spread check
        spread = yes_buy - yes_sell
        if spread > SPREAD_MAX:
            skipped_spread += 1
            continue

        if evaluated < 10:
            print(f"\n--- Market #{evaluated+1} ---")
            print(f"Title: {title[:80]}")
            print(f"YES BUY(ask): {yes_buy:.4f}, YES SELL(bid): {yes_sell:.4f}, Spread: {spread:.4f}")

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

        if fair is None:
            skipped_llm += 1
            if evaluated < 10:
                print("  Fair prob: FAILED (skipping)")
            evaluated += 1
            continue

        # ── BUY YES edge ──
        buy_edge = (fair - yes_buy) / yes_buy if yes_buy > 0 else -999
        buy_net = buy_edge - FEE_RATE

        # ── SELL YES (= BUY NO) edge ──
        # fair_no = 1 - fair, price_no = 1 - yes_sell
        # edge_no = (fair_no - price_no) / price_no = (yes_sell - fair) / (1 - yes_sell)
        sell_edge = (yes_sell - fair) / (1.0 - yes_sell) if yes_sell < 1.0 else -999
        sell_net = sell_edge - FEE_RATE

        # Pick the better side
        if buy_net >= sell_net:
            side_str = "BUY"
            net_edge = buy_net
            edge = buy_edge
            exec_price = yes_buy
        else:
            side_str = "SELL"
            net_edge = sell_net
            edge = sell_edge
            exec_price = yes_sell

        th = dynamic_edge_threshold(yes_buy, yes_sell)

        if evaluated < 10:
            print(f"  Fair: {fair:.4f}, BUY edge: {buy_net:.4f}, SELL edge: {sell_net:.4f}")
            print(f"  Best side: {side_str}, Net edge: {net_edge:.4f} (threshold: {th:.4f})")
            print(f"  Pass: {'YES' if net_edge >= th else 'NO'}")

        if best is None or edge > best[0]:
            best = (edge, th, title, tid, fair, exec_price, side_str)

        if net_edge >= th:
            decisions.append((net_edge, tid, title, fair, exec_price, side_str))

        evaluated += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total markets evaluated: {evaluated}")
    print(f"Skipped: no_price={skipped_no_price}, invalid_price={skipped_invalid_price}, "
          f"spread={skipped_spread}, llm_fail={skipped_llm}")
    print(f"Decisions found (net_edge >= threshold): {len(decisions)}")

    if best:
        edge, th, title_b, tid_b, fair_b, price_b, side_b = best
        print(f"\nBest opportunity:")
        print(f"  Title: {title_b[:80]}")
        print(f"  Side: {side_b}, Edge: {edge:.4f} (threshold: {th:.4f})")
        print(f"  Fair: {fair_b:.4f}, Exec price: {price_b:.4f}")

    if not decisions:
        print(f"\n-- No profitable opportunities found")
        best_info = ""
        if best:
            print(f"  - Best edge was {best[0]:.4f} (needed {best[1]:.4f})")
            best_info = f"\nBest edge: {best[0]:.4f} (needed {best[1]:.4f}), side={best[6]}"
        gh_issue(
            "run: no edge (dynamic threshold)",
            f"No mispricing found.\n"
            f"Evaluated: {evaluated} markets\n"
            f"Skipped: no_price={skipped_no_price}, invalid={skipped_invalid_price}, "
            f"spread={skipped_spread}, llm={skipped_llm}"
            f"{best_info}",
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
    trade_records = state.get("trades", [])

    for net_edge, tid, title, fair, exec_price, side_str in decisions[:max_orders]:
        # Kelly fraction: for SELL side, use (1-fair) and (1-exec_price) as NO probability
        if side_str == "BUY":
            f = kelly_fraction(fair, exec_price)
        else:
            f = kelly_fraction(1.0 - fair, 1.0 - exec_price)

        usd = bankroll * f

        if usd < min_order_usd:
            logs.append(
                f"skip (too small): side={side_str} edge={net_edge:.3f} "
                f"kelly={f:.4f} usd={usd:.2f} title={title}"
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
                    f"DRY_RUN build: side={side_str} edge={net_edge:.3f} "
                    f"fair={fair:.3f} price={exec_price:.3f} kelly={f:.4f} "
                    f"usd={usd:.2f} size={size:.4f} token={tid[:16]}... "
                    f"title={title[:60]}"
                )
            except Exception as e:
                logs.append(
                    f"DRY_RUN build FAILED: side={side_str} edge={net_edge:.3f} "
                    f"error={type(e).__name__}: {e} title={title[:60]}"
                )
        else:
            try:
                signed = client.create_order(order)
                resp = client.post_order(signed)
                logs.append(
                    f"LIVE post: side={side_str} edge={net_edge:.3f} "
                    f"fair={fair:.3f} price={exec_price:.3f} kelly={f:.4f} "
                    f"usd={usd:.2f} size={size:.4f} token={tid[:16]}... "
                    f"title={title[:60]}\nresp={resp}"
                )
                executed += 1
            except Exception as e:
                logs.append(
                    f"LIVE post FAILED: side={side_str} edge={net_edge:.3f} "
                    f"error={type(e).__name__}: {e} title={title[:60]}"
                )

        # Trade record for history
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
        })

    # Update state
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_evaluated"] = evaluated
    state["last_decisions"] = len(decisions)
    state["last_executed"] = executed
    state["trades"] = trade_records[-200:]  # 直近200件を保持
    save_state(state_dir, state)

    title_issue = (
        f"run: {'DRY_RUN' if dry_run else 'LIVE'} "
        f"eval={evaluated} decisions={len(decisions)} executed={executed}"
    )
    body = "\n".join(logs[:50])
    gh_issue(title_issue, body)


if __name__ == "__main__":
    main()
