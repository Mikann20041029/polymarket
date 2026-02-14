#!/usr/bin/env python3
"""
Polymarket Arbitrage Scanner v5.0
=================================
Scans all active Polymarket markets for Yes/No arbitrage opportunities.

Strategy: If ask_yes + ask_no < 1.0 (after costs), buying both sides
guarantees profit since payout is always $1.00.

No LLM required. Pure arithmetic.
"""

import os
import json
import math
import time
import requests
import collections
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ──────────────────────────────────────────────
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA = "https://gamma-api.polymarket.com"

# ── Config (all from env) ──────────────────────────────────
FEE_BPS      = float(os.getenv("FEE_BPS", "0"))         # Polymarket fee in bps (0 for most markets)
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "50"))    # Slippage buffer in bps (0.5%)
FILL_RISK_BPS = float(os.getenv("FILL_RISK_BPS", "16"))  # One-leg fill risk buffer in bps
MIN_VOLUME    = float(os.getenv("MIN_VOLUME", "1000"))
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "500"))
CLOB_WORKERS  = int(os.getenv("CLOB_WORKERS", "15"))
SCAN_MARKETS  = int(os.getenv("SCAN_MARKETS", "2000"))   # More markets = better coverage
ARB_QUANTITY  = float(os.getenv("ARB_QUANTITY", "100"))   # Shares for VWAP calculation
TOP_N         = int(os.getenv("TOP_N", "10"))             # Top N candidates in Issue
DRY_RUN       = os.getenv("DRY_RUN", "1") == "1"


# ── Helpers ────────────────────────────────────────────────

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def gh_issue(title: str, body: str):
    """Post a GitHub Issue with the given title and body."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        print("[GH] GITHUB_TOKEN or GITHUB_REPOSITORY not set, skipping Issue.")
        return

    # Truncate body if too large
    if len(body) > 60000:
        body = body[:59000] + "\n\n... (truncated)"

    try:
        r = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"title": title[:256], "body": body},
            timeout=30,
        )
        r.raise_for_status()
        print(f"[GH] Issue created: {r.json().get('html_url', 'OK')}")
    except Exception as e:
        print(f"[GH] Failed to create issue: {e}")


# ── Gamma API ──────────────────────────────────────────────

def gamma_markets(limit: int = 2000):
    """Fetch active markets from Gamma API with pagination."""
    all_markets = []
    offset = 0
    page_size = 100  # Gamma max per page

    while len(all_markets) < limit:
        try:
            r = requests.get(
                f"{GAMMA}/markets",
                params={
                    "limit": page_size,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[Gamma] Error at offset={offset}: {e}")
            break

        if not data:
            break

        all_markets.extend(data)
        offset += len(data)

        if len(data) < page_size:
            break  # No more pages

    print(f"[Gamma] Fetched {len(all_markets)} markets")
    return all_markets[:limit]


# ── Market Extraction ──────────────────────────────────────

def extract_arb_markets(markets):
    """
    Extract markets that have both Yes and No token IDs.
    Returns list of dicts with yes_tid, no_tid, title, volume, liquidity.
    Also returns skip stats.
    """
    results = []
    skipped = collections.Counter()

    for m in markets:
        # Must have order book enabled
        eob = m.get("enableOrderBook") or m.get("enable_order_book")
        if not (eob is True or str(eob).lower() == "true"):
            skipped["no_orderbook"] += 1
            continue

        # Get clobTokenIds - need BOTH Yes and No
        v = m.get("clobTokenIds") or m.get("clob_token_ids")
        if v is None:
            skipped["no_clob_ids"] += 1
            continue

        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                skipped["parse_error"] += 1
                continue

        # Need at least 2 token IDs: [Yes, No]
        if not isinstance(v, list) or len(v) < 2:
            skipped["single_token"] += 1
            continue

        yes_tid = str(v[0])
        no_tid = str(v[1])

        if not yes_tid or not no_tid:
            skipped["empty_token"] += 1
            continue

        # Volume / liquidity check
        volume = safe_float(m.get("volume"))
        liquidity = safe_float(m.get("liquidity"))
        if volume < MIN_VOLUME:
            skipped["low_volume"] += 1
            continue
        if liquidity < MIN_LIQUIDITY:
            skipped["low_liquidity"] += 1
            continue

        title = str(m.get("question") or m.get("title") or "unknown")

        results.append({
            "yes_tid": yes_tid,
            "no_tid": no_tid,
            "title": title[:120],
            "volume": volume,
            "liquidity": liquidity,
            "condition_id": str(m.get("conditionId") or m.get("condition_id") or ""),
        })

    print(f"[Extract] Markets with both tokens: {len(results)}")
    print(f"[Extract] Skipped: {dict(skipped)}")
    return results, dict(skipped)


# ── Order Book Fetching ────────────────────────────────────

def fetch_book(tid: str):
    """
    Fetch full order book for a token.
    Returns dict with best_ask, best_bid, ask_levels, bid_levels.
    ask_levels/bid_levels: list of (price, size) tuples sorted by price.
    """
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
        return None

    asks_raw = book.get("asks") or []
    bids_raw = book.get("bids") or []

    def parse_levels(levels):
        parsed = []
        for lvl in levels:
            if isinstance(lvl, dict):
                p = safe_float(lvl.get("price"))
                s = safe_float(lvl.get("size"))
                if p > 0 and s > 0:
                    parsed.append((p, s))
        return parsed

    ask_levels = sorted(parse_levels(asks_raw), key=lambda x: x[0])   # lowest ask first
    bid_levels = sorted(parse_levels(bids_raw), key=lambda x: -x[0])  # highest bid first

    if not ask_levels and not bid_levels:
        return None

    return {
        "best_ask": ask_levels[0][0] if ask_levels else None,
        "best_bid": bid_levels[0][0] if bid_levels else None,
        "best_ask_size": ask_levels[0][1] if ask_levels else 0,
        "best_bid_size": bid_levels[0][1] if bid_levels else 0,
        "ask_levels": ask_levels[:10],  # Top 10 levels
        "bid_levels": bid_levels[:10],
    }


def fetch_all_books(markets):
    """
    Fetch order books for all Yes and No tokens in parallel.
    Returns dict: token_id -> book_data
    """
    all_tids = []
    for m in markets:
        all_tids.append(m["yes_tid"])
        all_tids.append(m["no_tid"])

    # Deduplicate
    unique_tids = list(set(all_tids))
    print(f"[CLOB] Fetching {len(unique_tids)} order books ({CLOB_WORKERS} workers)...")

    books = {}
    with ThreadPoolExecutor(max_workers=CLOB_WORKERS) as executor:
        futures = {executor.submit(fetch_book, tid): tid for tid in unique_tids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                result = future.result()
                if result:
                    books[tid] = result
            except Exception:
                pass

    print(f"[CLOB] Got {len(books)}/{len(unique_tids)} order books")
    return books


# ── VWAP Calculation ───────────────────────────────────────

def calc_vwap(ask_levels, quantity: float):
    """
    Calculate Volume-Weighted Average Price for buying `quantity` shares.
    Walks through ask levels from lowest to highest.
    Returns (vwap, fillable_qty) or (None, 0) if book is empty.
    """
    if not ask_levels or quantity <= 0:
        return None, 0.0

    total_cost = 0.0
    total_qty = 0.0

    for price, size in ask_levels:
        take = min(size, quantity - total_qty)
        total_cost += price * take
        total_qty += take
        if total_qty >= quantity:
            break

    if total_qty <= 0:
        return None, 0.0

    return total_cost / total_qty, total_qty


# ── Arbitrage Calculation ──────────────────────────────────

def calc_arb(market, books):
    """
    Calculate arbitrage opportunity for a single market.
    Returns diagnostic dict with all relevant info.
    """
    yes_tid = market["yes_tid"]
    no_tid = market["no_tid"]
    title = market["title"]

    diag = {
        "title": title,
        "yes_tid": yes_tid[:20],
        "no_tid": no_tid[:20],
        "volume": market["volume"],
        "liquidity": market["liquidity"],
        "best_ask_yes": None,
        "best_ask_no": None,
        "best_ask_yes_size": 0,
        "best_ask_no_size": 0,
        "sum_ask": None,
        "raw_edge": None,
        "cost_buffer": None,
        "net_edge": None,
        "vwap_yes": None,
        "vwap_no": None,
        "vwap_sum": None,
        "edge_at_q": None,
        "fillable_yes": 0,
        "fillable_no": 0,
        "two_leg_feasible": False,
        "reject_reason": None,
    }

    # Get books
    yes_book = books.get(yes_tid)
    no_book = books.get(no_tid)

    if not yes_book or yes_book.get("best_ask") is None:
        diag["reject_reason"] = "NO_YES_BOOK"
        return diag

    if not no_book or no_book.get("best_ask") is None:
        diag["reject_reason"] = "NO_NO_BOOK"
        return diag

    best_ask_yes = yes_book["best_ask"]
    best_ask_no = no_book["best_ask"]

    diag["best_ask_yes"] = round(best_ask_yes, 4)
    diag["best_ask_no"] = round(best_ask_no, 4)
    diag["best_ask_yes_size"] = round(yes_book["best_ask_size"], 1)
    diag["best_ask_no_size"] = round(no_book["best_ask_size"], 1)

    # Core calculation
    sum_ask = best_ask_yes + best_ask_no
    cost_buffer = (FEE_BPS + SLIPPAGE_BPS + FILL_RISK_BPS) / 10000.0
    raw_edge = 1.0 - sum_ask
    net_edge = raw_edge - cost_buffer

    diag["sum_ask"] = round(sum_ask, 4)
    diag["raw_edge"] = round(raw_edge, 4)
    diag["cost_buffer"] = round(cost_buffer, 4)
    diag["net_edge"] = round(net_edge, 4)

    # VWAP at target quantity
    vwap_yes, fill_yes = calc_vwap(yes_book["ask_levels"], ARB_QUANTITY)
    vwap_no, fill_no = calc_vwap(no_book["ask_levels"], ARB_QUANTITY)

    diag["fillable_yes"] = round(fill_yes, 1)
    diag["fillable_no"] = round(fill_no, 1)

    if vwap_yes is not None:
        diag["vwap_yes"] = round(vwap_yes, 4)
    if vwap_no is not None:
        diag["vwap_no"] = round(vwap_no, 4)

    if vwap_yes is not None and vwap_no is not None:
        vwap_sum = vwap_yes + vwap_no
        diag["vwap_sum"] = round(vwap_sum, 4)
        edge_at_q = 1.0 - vwap_sum - cost_buffer
        diag["edge_at_q"] = round(edge_at_q, 4)

    # Two-leg feasibility: can we buy equal quantities on both sides?
    min_fill = min(fill_yes, fill_no)
    diag["two_leg_feasible"] = min_fill >= ARB_QUANTITY

    # Reject reason
    if sum_ask >= 1.0:
        diag["reject_reason"] = "SUM_TOO_HIGH"
    elif net_edge <= 0:
        diag["reject_reason"] = "EDGE_AFTER_COST"
    elif not diag["two_leg_feasible"]:
        diag["reject_reason"] = "LOW_DEPTH"
    else:
        diag["reject_reason"] = None  # CANDIDATE!

    return diag


# ── Main ───────────────────────────────────────────────────

def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"{'='*60}")
    print(f"Polymarket Arbitrage Scanner v5.0")
    print(f"Time: {ts}")
    print(f"{'='*60}")
    print(f"FEE_BPS={FEE_BPS}, SLIPPAGE_BPS={SLIPPAGE_BPS}, "
          f"FILL_RISK_BPS={FILL_RISK_BPS}")
    print(f"MIN_VOLUME={MIN_VOLUME}, MIN_LIQUIDITY={MIN_LIQUIDITY}")
    print(f"ARB_QUANTITY={ARB_QUANTITY}, CLOB_WORKERS={CLOB_WORKERS}")
    print(f"SCAN_MARKETS={SCAN_MARKETS}, DRY_RUN={DRY_RUN}")
    print(f"{'='*60}\n")

    # ── Phase 1: Fetch all markets ──
    print("== Phase 1: Fetch markets ==")
    markets = gamma_markets(SCAN_MARKETS)

    if not markets:
        gh_issue(
            f"arb-scan: 0 markets from Gamma [{ts}]",
            "Gamma API returned 0 markets."
        )
        return

    # ── Phase 2: Extract markets with both Yes/No tokens ──
    print("\n== Phase 2: Extract markets with both tokens ==")
    arb_markets, skip_stats = extract_arb_markets(markets)

    if not arb_markets:
        skip_text = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v > 0)
        body = (
            f"# Arb Scanner v5.0 - {ts}\n\n"
            f"Gamma returned {len(markets)} markets but 0 had both Yes/No tokens.\n\n"
            f"## Skip Breakdown\n{skip_text}\n"
        )
        gh_issue(f"arb-scan: 0 dual-token markets [{ts}]", body)
        return

    # ── Phase 3: Fetch order books ──
    print("\n== Phase 3: Fetch order books ==")
    t0 = time.time()
    books = fetch_all_books(arb_markets)
    elapsed_books = time.time() - t0
    print(f"  Book fetch time: {elapsed_books:.1f}s")

    # ── Phase 4: Calculate arbitrage for each market ──
    print("\n== Phase 4: Calculate arbitrage ==")
    all_diags = []
    reject_counter = collections.Counter()

    for mkt in arb_markets:
        diag = calc_arb(mkt, books)
        all_diags.append(diag)

        if diag["reject_reason"]:
            reject_counter[diag["reject_reason"]] += 1

    # Sort by net_edge descending (best opportunities first)
    all_diags.sort(key=lambda d: d.get("net_edge") or -999, reverse=True)

    # Separate candidates from rejects
    candidates = [d for d in all_diags if d["reject_reason"] is None]
    has_opportunity = len(candidates) > 0

    # ── Phase 5: Build Issue ──
    print("\n== Phase 5: Build Issue ==")

    # --- Summary ---
    summary = (
        f"## Summary\n"
        f"- **Timestamp**: {ts}\n"
        f"- **Gamma markets**: {len(markets)}\n"
        f"- **With both tokens**: {len(arb_markets)}\n"
        f"- **Books fetched**: {len(books)} ({elapsed_books:.1f}s)\n"
        f"- **Arb candidates (edge>0)**: {len(candidates)}\n\n"
    )

    # --- Top N candidates ---
    top_n = all_diags[:TOP_N]
    top_text = f"## Top {TOP_N} Candidates\n\n"
    if top_n:
        for i, d in enumerate(top_n):
            status = "✅ CANDIDATE" if d["reject_reason"] is None else f"❌ {d['reject_reason']}"
            top_text += (
                f"### {i+1}. {d['title']}\n"
                f"- **status**: {status}\n"
                f"- **yes_token**: `{d['yes_tid']}...` | "
                f"**no_token**: `{d['no_tid']}...`\n"
                f"- **best_ask_yes**: {d['best_ask_yes']} "
                f"({d['best_ask_yes_size']} shares) | "
                f"**best_ask_no**: {d['best_ask_no']} "
                f"({d['best_ask_no_size']} shares)\n"
                f"- **sum_ask**: {d['sum_ask']} → "
                f"**raw_edge**: {d['raw_edge']}\n"
                f"- **cost_buffer**: {d['cost_buffer']} "
                f"(fee={FEE_BPS}bps + slip={SLIPPAGE_BPS}bps + "
                f"fill_risk={FILL_RISK_BPS}bps)\n"
                f"- **net_edge**: {d['net_edge']}\n"
                f"- **vwap_yes({int(ARB_QUANTITY)})**: {d['vwap_yes']} | "
                f"**vwap_no({int(ARB_QUANTITY)})**: {d['vwap_no']}\n"
                f"- **vwap_sum**: {d['vwap_sum']} → "
                f"**edge_at_{int(ARB_QUANTITY)}**: {d['edge_at_q']}\n"
                f"- **two_leg_feasible**: "
                f"{'YES' if d['two_leg_feasible'] else 'NO'} "
                f"(fillable: yes={d['fillable_yes']}, "
                f"no={d['fillable_no']})\n\n"
            )
    else:
        top_text += "_No candidates to display._\n\n"

    # --- Reject summary ---
    reject_text = "## Reject Counts\n"
    if reject_counter:
        reject_text += "```\n" + ", ".join(
            f"{k}={v}" for k, v in reject_counter.most_common()
        ) + "\n```\n\n"
    else:
        reject_text += "_No rejects._\n\n"

    # --- Config ---
    config_text = (
        "## Config\n```\n"
        f"FEE_BPS={FEE_BPS}, SLIPPAGE_BPS={SLIPPAGE_BPS}, "
        f"FILL_RISK_BPS={FILL_RISK_BPS}\n"
        f"MIN_VOLUME={MIN_VOLUME}, MIN_LIQUIDITY={MIN_LIQUIDITY}\n"
        f"ARB_QUANTITY={ARB_QUANTITY}, CLOB_WORKERS={CLOB_WORKERS}\n"
        f"SCAN_MARKETS={SCAN_MARKETS}, TOP_N={TOP_N}\n"
        f"```\n\n"
    )

    # Compose Issue
    status_emoji = "🟢" if has_opportunity else "🔴"
    issue_title = (
        f"arb-scan: {status_emoji} "
        f"candidates={len(candidates)} "
        f"scanned={len(arb_markets)} [{ts}]"
    )

    issue_body = (
        f"# Arb Scanner v5.0 {status_emoji}\n\n"
        f"{summary}"
        f"{top_text}"
        f"{reject_text}"
        f"{config_text}"
    )

    # Console summary
    print(f"\n{'='*60}")
    print("SCAN RESULTS")
    print(f"{'='*60}")
    print(f"Gamma markets: {len(markets)}")
    print(f"With both tokens: {len(arb_markets)}")
    print(f"Books fetched: {len(books)}")
    print(f"Arb candidates: {len(candidates)}")
    if reject_counter:
        print(f"Rejects: {', '.join(f'{k}={v}' for k, v in reject_counter.most_common())}")
    print()

    if candidates:
        print("🟢 ARBITRAGE OPPORTUNITIES FOUND:")
        for i, d in enumerate(candidates[:5]):
            print(f"  {i+1}. edge={d['net_edge']:.4f} "
                  f"sum={d['sum_ask']:.4f} "
                  f"yes={d['best_ask_yes']:.3f}+no={d['best_ask_no']:.3f} "
                  f"| {d['title'][:50]}")
    else:
        print("🔴 No arbitrage opportunities found this scan.")
        if top_n and top_n[0].get("net_edge") is not None:
            print(f"   Closest: edge={top_n[0]['net_edge']:.4f} "
                  f"sum={top_n[0]['sum_ask']} | {top_n[0]['title'][:50]}")

    gh_issue(issue_title, issue_body)


if __name__ == "__main__":
    main()
