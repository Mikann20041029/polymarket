import os
import json
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import requests
import sqlite3
from dataclasses import dataclass, asdict
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# ============================================================================

# CONSTANTS & CONFIGURATION

# ============================================================================

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA = “https://gamma-api.polymarket.com”

# Kelly & Risk

KELLY_MAX = 0.06 # Maximum Kelly fraction
KELLY_CONSERVATIVE = 0.5 # Kelly multiplier for safety

# Fees (Polymarket typical)

MAKER_FEE = 0.00 # 0% maker fee
TAKER_FEE = 0.02 # 2% taker fee

# Liquidity thresholds

MIN_LIQUIDITY_USD = 100.0 # Minimum total liquidity to consider
MAX_ORDERBOOK_DEPTH = 20 # Number of levels to fetch

# Edge requirements

BASE_EDGE_MIN = 0.08 # Base minimum edge
SPREAD_PENALTY_FACTOR = 2.0 # Multiply spread by this for edge requirement

# Volatility & uncertainty

VOLATILITY_LOOKBACK_HOURS = 24
UNCERTAINTY_PENALTY_MAX = 0.15 # Max adjustment for uncertainty

# Execution probability parameters

EXEC_PROB_BASE = 0.85 # Base execution probability for good liquidity
EXEC_PROB_MIN = 0.30 # Minimum execution probability to consider

# Historical feedback

PREDICTION_DECAY_DAYS = 30 # How long to keep prediction history
MIN_PREDICTIONS_FOR_CORRECTION = 5 # Minimum samples before applying correction

# ============================================================================

# DATA MODELS

# ============================================================================

@dataclass
class OrderbookLevel:
“”“Single level in the orderbook”””
price: float
size: float

@dataclass
class OrderbookSnapshot:
“”“Complete orderbook snapshot”””
token_id: str
timestamp: datetime
bids: List[OrderbookLevel]
asks: List[OrderbookLevel]

```
def total_bid_liquidity(self) -> float:
 """Total USD liquidity on bid side"""
 return sum(level.price * level.size for level in self.bids)

def total_ask_liquidity(self) -> float:
 """Total USD liquidity on ask side"""
 return sum(level.price * level.size for level in self.asks)

def best_bid(self) -> Optional[float]:
 return self.bids[0].price if self.bids else None

def best_ask(self) -> Optional[float]:
 return self.asks[0].price if self.asks else None

def spread(self) -> float:
 """Bid-ask spread"""
 bid = self.best_bid()
 ask = self.best_ask()
 if bid and ask:
 return ask - bid
 return float('inf')
```

@dataclass
class LiquidityProfile:
“”“Analyzed liquidity characteristics”””
total_bid_usd: float
total_ask_usd: float
spread: float
depth_score: float # 0-1, higher is better
avg_level_size: float

```
def is_tradeable(self, min_liquidity: float) -> bool:
 """Check if market has sufficient liquidity"""
 return (self.total_bid_usd >= min_liquidity and 
 self.total_ask_usd >= min_liquidity and
 self.spread < 0.5) # Spread shouldn't be too wide
```

@dataclass
class SlippageAnalysis:
“”“Slippage calculation for a given size”””
nominal_price: float # Best bid/ask
avg_execution_price: float # Average price after eating through book
total_slippage: float # Difference in percentage
max_executable_size: float # Max size before liquidity runs out
execution_probability: float # Probability of full execution

@dataclass
class VolatilityEstimate:
“”“Market volatility characteristics”””
recent_price_std: float # Standard deviation of recent prices
spread_volatility: float # Volatility of spread
uncertainty_score: float # 0-1, higher means more uncertain

@dataclass
class RiskAdjustedEdge:
“”“Edge calculation with all adjustments”””
raw_edge: float
slippage_adjusted: float
spread_penalty: float
fee_adjusted: float
volatility_adjusted: float
execution_probability: float
final_edge: float
required_threshold: float
is_tradeable: bool

@dataclass
class TradeDecision:
“”“Final trade decision”””
token_id: str
title: str
side: str # BUY or SELL
fair_prob: float
market_price: float
optimal_size_usd: float
optimal_size_shares: float
edge: RiskAdjustedEdge
expected_pnl: float

@dataclass
class PredictionRecord:
“”“Historical prediction for learning”””
timestamp: datetime
token_id: str
title: str
predicted_fair: float
market_price: float
edge: float
actual_outcome: Optional[float] = None # Set when resolved
prediction_error: Optional[float] = None

# ============================================================================

# DATABASE FOR HISTORICAL TRACKING

# ============================================================================

class PredictionDB:
“”“SQLite database for tracking predictions and learning from errors”””

```
def __init__(self, db_path: str = "/home/claude/predictions.db"):
 self.db_path = db_path
 self.init_db()

def init_db(self):
 """Initialize database schema"""
 conn = sqlite3.connect(self.db_path)
 c = conn.cursor()
 c.execute('''
 CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 timestamp TEXT NOT NULL,
 token_id TEXT NOT NULL,
 title TEXT,
 predicted_fair REAL NOT NULL,
 market_price REAL NOT NULL,
 edge REAL NOT NULL,
 actual_outcome REAL,
 prediction_error REAL,
 market_type TEXT,
 external_context TEXT
 )
 ''')
 c.execute('''
 CREATE INDEX IF NOT EXISTS idx_timestamp 
 ON predictions(timestamp)
 ''')
 c.execute('''
 CREATE INDEX IF NOT EXISTS idx_token 
 ON predictions(token_id)
 ''')
 conn.commit()
 conn.close()

def save_prediction(self, record: PredictionRecord, market_type: str = None, 
 context: dict = None):
 """Save a prediction"""
 conn = sqlite3.connect(self.db_path)
 c = conn.cursor()
 c.execute('''
 INSERT INTO predictions 
 (timestamp, token_id, title, predicted_fair, market_price, edge, 
 actual_outcome, prediction_error, market_type, external_context)
 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
 ''', (
 record.timestamp.isoformat(),
 record.token_id,
 record.title,
 record.predicted_fair,
 record.market_price,
 record.edge,
 record.actual_outcome,
 record.prediction_error,
 market_type,
 json.dumps(context) if context else None
 ))
 conn.commit()
 conn.close()

def get_recent_errors(self, market_type: str = None, days: int = 30) -> List[float]:
 """Get recent prediction errors for calibration"""
 conn = sqlite3.connect(self.db_path)
 c = conn.cursor()
 
 cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
 
 if market_type:
 c.execute('''
 SELECT prediction_error FROM predictions
 WHERE timestamp > ? AND market_type = ? AND prediction_error IS NOT NULL
 ORDER BY timestamp DESC
 ''', (cutoff, market_type))
 else:
 c.execute('''
 SELECT prediction_error FROM predictions
 WHERE timestamp > ? AND prediction_error IS NOT NULL
 ORDER BY timestamp DESC
 ''', (cutoff,))
 
 errors = [row[0] for row in c.fetchall()]
 conn.close()
 return errors

def calculate_bias_correction(self, market_type: str = None) -> float:
 """Calculate systematic bias in predictions"""
 errors = self.get_recent_errors(market_type)
 
 if len(errors) < MIN_PREDICTIONS_FOR_CORRECTION:
 return 0.0
 
 # Mean error (positive = we overestimate, negative = we underestimate)
 mean_error = sum(errors) / len(errors)
 
 # Apply conservative correction (only 50% of observed bias)
 correction = -0.5 * mean_error
 
 # Cap correction to avoid overcorrecting
 return max(-0.1, min(0.1, correction))
```

# ============================================================================

# EXTERNAL DATA SOURCES

# ============================================================================

def fetch_crypto_context() -> Optional[dict]:
“”“Fetch crypto market context”””
out = {“ts_utc”: datetime.now(timezone.utc).isoformat()}

```
try:
 r = requests.get(
 "https://api.coingecko.com/api/v3/simple/price",
 params={
 "ids": "bitcoin,ethereum,solana",
 "vs_currencies": "usd",
 "include_24hr_change": "true",
 "include_24hr_vol": "true"
 },
 timeout=20,
 )
 r.raise_for_status()
 out["coingecko"] = r.json()
except Exception as e:
 out["coingecko_error"] = f"{type(e).__name__}: {e}"

try:
 r = requests.get("https://api.alternative.me/fng/", params={"limit": "1"}, timeout=20)
 r.raise_for_status()
 data = r.json()
 out["fear_greed"] = data.get("data", [None])[0]
except Exception as e:
 out["fear_greed_error"] = f"{type(e).__name__}: {e}"

if "coingecko" not in out and "fear_greed" not in out:
 return None
return out
```

def fetch_sports_context() -> Optional[dict]:
“”“Fetch sports injury news context”””
out = {“ts_utc”: datetime.now(timezone.utc).isoformat()}

```
feeds = [
 ("nfl", "https://www.espn.com/espn/rss/nfl/news"),
 ("nba", "https://www.espn.com/espn/rss/nba/news"),
 ("mlb", "https://www.espn.com/espn/rss/mlb/news"),
 ("nhl", "https://www.espn.com/espn/rss/nhl/news"),
]

keywords = ["injury", "injured", "out", "questionable", "doubtful", "IL", 
 "concussion", "hamstring", "ankle", "knee"]

hits = []
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
 if any(k in t.lower() for k in keywords):
 hits.append({"league": league, "title": t})
 except Exception:
 continue

if hits:
 out["injury_news_titles"] = hits[:40]

return out if "injury_news_titles" in out else None
```

def parse_weather_question(q: str):
“”“Parse weather-related market question”””
import re
text = (q or “”).strip()
low = text.lower()

```
if "rain" not in low:
 return None

m_place = re.search(r"\bin\s+([A-Za-z .,'-]{2,60})", text)
m_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
if not m_place or not m_date:
 return None

place = m_place.group(1).strip(" ,.")
day = m_date.group(1)
return {"place": place, "date": day}
```

def fetch_weather_probability(place: str, date: str) -> Optional[float]:
“”“Fetch actual weather forecast probability”””
try:
# Geocode
r = requests.get(
“https://geocoding-api.open-meteo.com/v1/search”,
params={“name”: place, “count”: 1, “language”: “en”, “format”: “json”},
timeout=20
)
r.raise_for_status()
data = r.json()
if not data.get(“results”):
return None

```
 loc = data["results"][0]
 lat, lon = loc["latitude"], loc["longitude"]
 
 # Get forecast
 r = requests.get(
 "https://api.open-meteo.com/v1/forecast",
 params={
 "latitude": lat,
 "longitude": lon,
 "hourly": "precipitation_probability",
 "timezone": "UTC",
 "start_date": date,
 "end_date": date,
 },
 timeout=20
 )
 r.raise_for_status()
 data = r.json()
 
 probs = data.get("hourly", {}).get("precipitation_probability", [])
 if not probs:
 return None
 
 # Calculate "any rain" probability: P(any) = 1 - Π(1 - p_i)
 ps = [max(0.0, min(1.0, float(x) / 100.0)) for x in probs if x is not None]
 if not ps:
 return None
 
 prod = 1.0
 for p in ps:
 prod *= (1.0 - p)
 
 return 1.0 - prod
 
except Exception:
 return None
```

def classify_market_type(title: str) -> Optional[str]:
“”“Classify market into type”””
import re
t = (title or “”).lower()

```
if "rain" in t and re.search(r"\b\d{4}-\d{2}-\d{2}\b", t) and " in " in t:
 return "weather"

if (" vs " in t) or (" v " in t) or (" at " in t):
 return "sports"

crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "sol", "solana", 
 "etf", "sec", "cpi", "fed", "rate cut", "rate hike"]
if any(k in t for k in crypto_kw):
 return "crypto"

return "general"
```

# ============================================================================

# ORDERBOOK & LIQUIDITY ANALYSIS

# ============================================================================

def fetch_full_orderbook(token_id: str) -> Optional[OrderbookSnapshot]:
“”“Fetch complete orderbook with depth”””
try:
r = requests.get(
f”{HOST}/book”,
params={“token_id”: token_id},
timeout=30,
)
r.raise_for_status()
book = r.json() or {}

```
 def parse_levels(levels_raw) -> List[OrderbookLevel]:
 """Parse orderbook levels (handles both dict and list formats)"""
 levels = []
 for item in levels_raw[:MAX_ORDERBOOK_DEPTH]:
 if isinstance(item, dict):
 price = float(item.get("price", 0))
 size = float(item.get("size", 0))
 elif isinstance(item, (list, tuple)) and len(item) >= 2:
 price = float(item[0])
 size = float(item[1])
 else:
 continue
 
 if price > 0 and size > 0:
 levels.append(OrderbookLevel(price=price, size=size))
 
 return levels
 
 bids = parse_levels(book.get("bids", []))
 asks = parse_levels(book.get("asks", []))
 
 if not bids and not asks:
 return None
 
 return OrderbookSnapshot(
 token_id=token_id,
 timestamp=datetime.now(timezone.utc),
 bids=bids,
 asks=asks
 )
 
except Exception as e:
 print(f"Error fetching orderbook for {token_id}: {e}")
 return None
```

def analyze_liquidity(orderbook: OrderbookSnapshot) -> LiquidityProfile:
“”“Analyze orderbook liquidity characteristics”””
bid_usd = orderbook.total_bid_liquidity()
ask_usd = orderbook.total_ask_liquidity()
spread = orderbook.spread()

```
# Depth score: how many meaningful levels exist
bid_depth = sum(1 for level in orderbook.bids if level.price * level.size > 10)
ask_depth = sum(1 for level in orderbook.asks if level.price * level.size > 10)
depth_score = min(1.0, (bid_depth + ask_depth) / 10.0)

# Average level size
all_levels = orderbook.bids + orderbook.asks
avg_size = sum(l.price * l.size for l in all_levels) / len(all_levels) if all_levels else 0

return LiquidityProfile(
 total_bid_usd=bid_usd,
 total_ask_usd=ask_usd,
 spread=spread,
 depth_score=depth_score,
 avg_level_size=avg_size
)
```

def calculate_slippage(orderbook: OrderbookSnapshot, size_usd: float,
side: str) -> SlippageAnalysis:
“”“Calculate slippage for a given trade size”””

```
levels = orderbook.asks if side == BUY else orderbook.bids

if not levels:
 return SlippageAnalysis(
 nominal_price=0.0,
 avg_execution_price=0.0,
 total_slippage=1.0,
 max_executable_size=0.0,
 execution_probability=0.0
 )

nominal_price = levels[0].price

# Walk through orderbook to fill the order
remaining_usd = size_usd
total_cost = 0.0
total_shares = 0.0

for level in levels:
 level_usd = level.price * level.size
 
 if remaining_usd <= level_usd:
 # Can fill remainder at this level
 shares_needed = remaining_usd / level.price
 total_cost += remaining_usd
 total_shares += shares_needed
 remaining_usd = 0
 break
 else:
 # Consume entire level
 total_cost += level_usd
 total_shares += level.size
 remaining_usd -= level_usd

# Calculate results
if total_shares > 0:
 avg_price = total_cost / total_shares
 slippage = abs(avg_price - nominal_price) / nominal_price
 
 # Max executable is what we could actually fill
 max_executable = size_usd - remaining_usd
 
 # Execution probability based on how much we can fill
 if remaining_usd > 0:
 fill_ratio = (size_usd - remaining_usd) / size_usd
 exec_prob = EXEC_PROB_BASE * fill_ratio
 else:
 # Full fill possible - probability based on depth
 exec_prob = EXEC_PROB_BASE * (1.0 - slippage * 2.0) # More slippage = lower prob
 exec_prob = max(EXEC_PROB_MIN, min(1.0, exec_prob))
else:
 avg_price = nominal_price
 slippage = 1.0 # Maximum slippage
 max_executable = 0.0
 exec_prob = 0.0

return SlippageAnalysis(
 nominal_price=nominal_price,
 avg_execution_price=avg_price,
 total_slippage=slippage,
 max_executable_size=max_executable,
 execution_probability=exec_prob
)
```

def estimate_volatility(token_id: str, orderbook: OrderbookSnapshot) -> VolatilityEstimate:
“”“Estimate market volatility and uncertainty”””

```
# Use spread as proxy for instantaneous volatility
spread = orderbook.spread()
best_bid = orderbook.best_bid() or 0.5
best_ask = orderbook.best_ask() or 0.5
mid = (best_bid + best_ask) / 2.0

# Spread volatility (normalized)
spread_vol = spread / mid if mid > 0 else 1.0

# Recent price std (simplified - using spread as proxy)
# In production, you'd fetch historical trades
price_std = spread_vol * 0.5

# Uncertainty score combines factors
# - Wide spread = high uncertainty
# - Thin liquidity = high uncertainty
# - Extreme prices = high uncertainty
liquidity = analyze_liquidity(orderbook)

liquidity_factor = 1.0 / (1.0 + math.log(1 + liquidity.total_bid_usd + liquidity.total_ask_usd))
spread_factor = min(1.0, spread_vol * 5.0)
extreme_factor = 1.0 if (mid < 0.05 or mid > 0.95) else 0.0

uncertainty = (liquidity_factor + spread_factor + extreme_factor) / 3.0

return VolatilityEstimate(
 recent_price_std=price_std,
 spread_volatility=spread_vol,
 uncertainty_score=min(1.0, uncertainty)
)
```

# ============================================================================

# FAIR PROBABILITY ESTIMATION

# ============================================================================

def estimate_fair_probability(title: str, market_price: float,
external_context: Optional[dict],
prediction_db: PredictionDB) -> float:
“”“Estimate fair probability using LLM with historical bias correction”””

```
api_key = env("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Get market type for bias correction
market_type = classify_market_type(title)
bias_correction = prediction_db.calculate_bias_correction(market_type)

# Build context string
ctx = external_context or {}
context_str = json.dumps(ctx, ensure_ascii=False, indent=2)

try:
 ext_flags = (
 f"ext(weather={'Y' if ctx.get('weather') else 'N'}, "
 f"sports={'Y' if ctx.get('sports') else 'N'}, "
 f"crypto={'Y' if ctx.get('crypto') else 'N'})"
 )
except Exception:
 ext_flags = "ext(weather=N, sports=N, crypto=N)"

SYSTEM = (
 "You are an autonomous prediction-market trading agent.\n"
 "You must survive by making profitable trades.\n"
 "Do not blindly follow market prices.\n"
 "Use external data when available and relevant.\n"
 f"{ext_flags}\n"
 f"Historical bias correction: {bias_correction:+.3f}\n"
 "Return ONLY a decimal number between 0 and 1.\n"
)

USER = (
 f"Estimate fair probability (YES) for:\n\n"
 f"Title: {title}\n"
 f"Current market price: {market_price:.4f}\n"
 f"Market type: {market_type}\n\n"
 f"External context:\n{context_str}\n\n"
 "Output: decimal in [0,1]"
)

try:
 r = requests.post(
 "https://api.openai.com/v1/chat/completions",
 headers={
 "Authorization": f"Bearer {api_key}",
 "Content-Type": "application/json"
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
 raw_prob = float(text)
 
 # Apply bias correction
 corrected_prob = raw_prob + bias_correction
 
 # Clamp to valid range
 final_prob = max(0.01, min(0.99, corrected_prob))
 
 return final_prob
 
except Exception as e:
 print(f"Error in fair probability estimation: {e}")
 # Fallback to market price with slight uncertainty
 return max(0.01, min(0.99, market_price))
```

# ============================================================================

# RISK-ADJUSTED EDGE CALCULATION

# ============================================================================

def calculate_risk_adjusted_edge(
fair_prob: float,
orderbook: OrderbookSnapshot,
liquidity: LiquidityProfile,
size_usd: float,
side: str = BUY
) -> RiskAdjustedEdge:
“”“Calculate comprehensive risk-adjusted edge”””

```
# 1. Raw edge
best_ask = orderbook.best_ask()
best_bid = orderbook.best_bid()

if side == BUY:
 nominal_price = best_ask
 raw_edge = (fair_prob - nominal_price) / nominal_price if nominal_price else 0
else:
 nominal_price = best_bid
 raw_edge = (nominal_price - fair_prob) / fair_prob if fair_prob else 0

if not nominal_price or nominal_price <= 0:
 return RiskAdjustedEdge(
 raw_edge=0,
 slippage_adjusted=0,
 spread_penalty=0,
 fee_adjusted=0,
 volatility_adjusted=0,
 execution_probability=0,
 final_edge=0,
 required_threshold=1.0,
 is_tradeable=False
 )

# 2. Slippage adjustment
slippage = calculate_slippage(orderbook, size_usd, side)
slippage_cost = slippage.total_slippage
slippage_adjusted = raw_edge - slippage_cost

# 3. Spread penalty
spread = liquidity.spread
mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else nominal_price
spread_pct = spread / mid if mid > 0 else 1.0
spread_penalty = spread_pct * SPREAD_PENALTY_FACTOR

# 4. Fee adjustment
fee = TAKER_FEE # Assuming we're taking liquidity
fee_adjusted = slippage_adjusted - fee - spread_penalty

# 5. Volatility adjustment
volatility = estimate_volatility(orderbook.token_id, orderbook)
vol_penalty = volatility.uncertainty_score * UNCERTAINTY_PENALTY_MAX
volatility_adjusted = fee_adjusted - vol_penalty

# 6. Execution probability
exec_prob = slippage.execution_probability

# 7. Final edge (expected value)
final_edge = volatility_adjusted * exec_prob

# 8. Dynamic threshold based on market conditions
base_threshold = BASE_EDGE_MIN

# Increase threshold for:
# - Wide spreads (illiquid)
# - High volatility (uncertain)
# - Low execution probability
threshold_multiplier = 1.0
threshold_multiplier *= (1.0 + spread_pct * 2.0)
threshold_multiplier *= (1.0 + volatility.uncertainty_score)
threshold_multiplier *= (1.0 + (1.0 - exec_prob))

required_threshold = base_threshold * threshold_multiplier

# Trade is viable if final edge exceeds dynamic threshold
is_tradeable = (
 final_edge >= required_threshold and
 exec_prob >= EXEC_PROB_MIN and
 liquidity.is_tradeable(MIN_LIQUIDITY_USD)
)

return RiskAdjustedEdge(
 raw_edge=raw_edge,
 slippage_adjusted=slippage_adjusted,
 spread_penalty=spread_penalty,
 fee_adjusted=fee_adjusted,
 volatility_adjusted=volatility_adjusted,
 execution_probability=exec_prob,
 final_edge=final_edge,
 required_threshold=required_threshold,
 is_tradeable=is_tradeable
)
```

# ============================================================================

# KELLY SIZING WITH LIQUIDITY CONSTRAINTS

# ============================================================================

def calculate_optimal_size(
fair_prob: float,
edge: RiskAdjustedEdge,
orderbook: OrderbookSnapshot,
liquidity: LiquidityProfile,
bankroll: float
) -> Tuple[float, float]:
“”“Calculate optimal position size using Kelly with constraints”””

```
if not edge.is_tradeable or edge.final_edge <= 0:
 return 0.0, 0.0

# Kelly fraction for binary outcome
nominal_price = orderbook.best_ask() or 0.5

if nominal_price <= 0 or nominal_price >= 1:
 return 0.0, 0.0

# Binary Kelly: f = (bp - q) / b where b = (1/price - 1)
b = (1.0 / nominal_price) - 1.0
q = 1.0 - fair_prob

kelly_fraction = (b * fair_prob - q) / b if b > 0 else 0

# Apply conservative multiplier
kelly_fraction *= KELLY_CONSERVATIVE

# Cap at maximum
kelly_fraction = max(0, min(KELLY_MAX, kelly_fraction))

# Size in USD based on Kelly
kelly_size_usd = bankroll * kelly_fraction

# Constraint 1: Available liquidity
# Only use a fraction of available liquidity to avoid moving market too much
max_liquidity_usd = liquidity.total_ask_usd * 0.3 # Use max 30% of book

# Constraint 2: Slippage tolerance
# Iterate to find size where slippage is acceptable
test_sizes = [kelly_size_usd * mult for mult in [1.0, 0.75, 0.5, 0.25, 0.1]]

best_size_usd = 0.0
best_expected_pnl = 0.0

for size in test_sizes:
 if size <= 0:
 continue
 
 if size > max_liquidity_usd:
 continue
 
 # Calculate slippage for this size
 slippage = calculate_slippage(orderbook, size, BUY)
 
 # Recalculate edge for this specific size
 edge_at_size = calculate_risk_adjusted_edge(
 fair_prob, orderbook, liquidity, size, BUY
 )
 
 # Expected PnL = edge * size * execution_probability
 expected_pnl = edge_at_size.final_edge * size * edge_at_size.execution_probability
 
 if expected_pnl > best_expected_pnl:
 best_expected_pnl = expected_pnl
 best_size_usd = size

# Convert to shares
if best_size_usd > 0:
 # Use average execution price for sizing
 slippage = calculate_slippage(orderbook, best_size_usd, BUY)
 size_shares = best_size_usd / slippage.avg_execution_price
else:
 size_shares = 0.0

return best_size_usd, size_shares
```

# ============================================================================

# MARKET SCANNING & EVALUATION

# ============================================================================

def fetch_markets(limit: int) -> List[dict]:
“”“Fetch active markets from Gamma API”””
try:
r = requests.get(
f”{GAMMA}/markets”,
params={
“active”: “true”,
“closed”: “false”,
“archived”: “false”,
“limit”: str(limit)
},
timeout=30,
)
r.raise_for_status()
return r.json()
except Exception as e:
print(f”Error fetching markets: {e}”)
return []

def extract_token_ids(markets: List[dict], max_tokens: int) -> List[Tuple[str, str]]:
“”“Extract YES token IDs from markets”””
results = []

```
for m in markets:
 if len(results) >= max_tokens:
 break
 
 if not m.get("enableOrderBook", False):
 continue
 
 v = m.get("clobTokenIds")
 if v is None:
 continue
 
 # Handle string JSON
 if isinstance(v, str):
 try:
 v = json.loads(v)
 except Exception:
 continue
 
 yes_token = None
 if isinstance(v, dict):
 yes_token = v.get("YES") or v.get("Yes") or v.get("yes")
 elif isinstance(v, list) and len(v) > 0:
 yes_token = v[0]
 
 if yes_token:
 title = str(m.get("question") or m.get("title") or "Unknown")
 results.append((str(yes_token), title))

return results
```

def evaluate_market(
token_id: str,
title: str,
external_context: dict,
prediction_db: PredictionDB,
bankroll: float
) -> Optional[TradeDecision]:
“”“Comprehensive market evaluation”””

```
# 1. Fetch orderbook
orderbook = fetch_full_orderbook(token_id)
if not orderbook:
 return None

# 2. Analyze liquidity
liquidity = analyze_liquidity(orderbook)
if not liquidity.is_tradeable(MIN_LIQUIDITY_USD):
 return None

# 3. Get market price
best_ask = orderbook.best_ask()
if not best_ask or best_ask <= 0 or best_ask >= 1:
 return None

# 4. Estimate fair probability
fair_prob = estimate_fair_probability(
 title, best_ask, external_context, prediction_db
)

# 5. Calculate initial edge with small size for screening
initial_size = min(50.0, bankroll * 0.01) # Small probe size
edge = calculate_risk_adjusted_edge(
 fair_prob, orderbook, liquidity, initial_size, BUY
)

if not edge.is_tradeable:
 return None

# 6. Calculate optimal size
optimal_usd, optimal_shares = calculate_optimal_size(
 fair_prob, edge, orderbook, liquidity, bankroll
)

if optimal_usd < float(os.getenv("MIN_ORDER_USD", "1.0")):
 return None

# 7. Recalculate edge at optimal size
final_edge = calculate_risk_adjusted_edge(
 fair_prob, orderbook, liquidity, optimal_usd, BUY
)

if not final_edge.is_tradeable:
 return None

# 8. Calculate expected PnL
expected_pnl = final_edge.final_edge * optimal_usd * final_edge.execution_probability

return TradeDecision(
 token_id=token_id,
 title=title,
 side=BUY,
 fair_prob=fair_prob,
 market_price=best_ask,
 optimal_size_usd=optimal_usd,
 optimal_size_shares=optimal_shares,
 edge=final_edge,
 expected_pnl=expected_pnl
)
```

# ============================================================================

# UTILITIES

# ============================================================================

def env(k: str) -> str:
“”“Get required environment variable”””
v = os.getenv(k)
if not v:
raise RuntimeError(f”Missing environment variable: {k}”)
return v

def is_hex_bytes(s: str, n_bytes: int) -> bool:
“”“Validate hex string”””
s = s.strip()
if s.startswith(“0x”):
s = s[2:]
if len(s) != n_bytes * 2:
return False
try:
int(s, 16)
return True
except Exception:
return False

def create_github_issue(title: str, body: str):
“”“Create GitHub issue for logging”””
try:
token = env(“GITHUB_TOKEN”)
repo = env(“GITHUB_REPOSITORY”)
r = requests.post(
f”https://api.github.com/repos/{repo}/issues”,
headers={
“Authorization”: f”Bearer {token}”,
“Accept”: “application/vnd.github+json”
},
json={“title”: title, “body”: body},
timeout=30,
)
r.raise_for_status()
except Exception as e:
print(f”Error creating GitHub issue: {e}”)

# ============================================================================

# MAIN EXECUTION

# ============================================================================

def main():
“”“Main trading loop”””

```
print(f"\n{'='*80}")
print(f"POLYMARKET AUTONOMOUS AGENT - COMPLETE SYSTEM")
print(f"{'='*80}\n")

# Configuration
dry_run = os.getenv("DRY_RUN", "1") == "1"
scan_markets = int(os.getenv("SCAN_MARKETS", "1000"))
max_tokens = int(os.getenv("MAX_TOKENS", "500"))
max_orders = int(os.getenv("MAX_ORDERS_PER_RUN", "1"))
bankroll = float(os.getenv("BANKROLL_USD", "50.0"))
api_budget = float(os.getenv("API_BUDGET_USD", "0.0"))

print(f"Configuration:")
print(f" Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
print(f" Bankroll: ${bankroll:.2f}")
print(f" API Budget: ${api_budget:.2f}")
print(f" Scanning: {scan_markets} markets ({max_tokens} tokens)")
print(f" Max orders: {max_orders}")
print()

# Safety check
if bankroll - api_budget <= 0:
 msg = f"INSUFFICIENT FUNDS: bankroll=${bankroll:.2f}, api_budget=${api_budget:.2f}"
 print(f" {msg}")
 create_github_issue("STOP: Insufficient funds", msg)
 return

# Initialize
prediction_db = PredictionDB()

# Client setup
private_key = env("PM_PRIVATE_KEY").strip()
funder = env("PM_FUNDER").strip()
signature_type = int(env("PM_SIGNATURE_TYPE"))

if not is_hex_bytes(private_key, 32):
 raise RuntimeError("PM_PRIVATE_KEY must be 32 bytes hex")
if not is_hex_bytes(funder, 20):
 raise RuntimeError("PM_FUNDER must be 20 bytes address hex")

client = ClobClient(
 HOST,
 key=private_key,
 chain_id=CHAIN_ID,
 signature_type=signature_type,
 funder=funder,
)
client.set_api_creds(client.create_or_derive_api_creds())

# Fetch external context
print("Fetching external context...")
crypto_context = fetch_crypto_context()
sports_context = fetch_sports_context()

if crypto_context and "fear_greed" in crypto_context:
 fg = crypto_context["fear_greed"]
 print(f" ✓ Crypto: Fear & Greed = {fg.get('value')} ({fg.get('value_classification')})")
else:
 print(f" Crypto context unavailable")

if sports_context:
 injuries = len(sports_context.get("injury_news_titles", []))
 print(f" ✓ Sports: {injuries} injury news items")
else:
 print(f" Sports context unavailable")

print()

# Fetch markets
print("Fetching markets...")
markets = fetch_markets(scan_markets)
token_pairs = extract_token_ids(markets, max_tokens)
print(f" ✓ Found {len(token_pairs)} tradeable markets\n")

# Evaluate markets
print(f"{'='*80}")
print(f"MARKET EVALUATION")
print(f"{'='*80}\n")

decisions = []

for idx, (token_id, title) in enumerate(token_pairs):
 # Enhanced external context per market
 market_type = classify_market_type(title)
 
 context = {
 "crypto": crypto_context,
 "sports": sports_context,
 "weather": None
 }
 
 # Weather-specific data
 if market_type == "weather":
 parsed = parse_weather_question(title)
 if parsed:
 rain_prob = fetch_weather_probability(parsed["place"], parsed["date"])
 if rain_prob is not None:
 context["weather"] = {
 "place": parsed["place"],
 "date": parsed["date"],
 "rain_probability": rain_prob,
 "source": "open-meteo"
 }
 
 # Evaluate
 decision = evaluate_market(
 token_id, title, context, prediction_db, bankroll
 )
 
 if decision:
 decisions.append(decision)
 
 # Log first few for debugging
 if idx < 10:
 print(f"✓ Opportunity #{len(decisions)}: {title[:60]}")
 print(f" Fair: {decision.fair_prob:.3f} | Market: {decision.market_price:.3f}")
 print(f" Edge: {decision.edge.final_edge:.3f} (threshold: {decision.edge.required_threshold:.3f})")
 print(f" Size: ${decision.optimal_size_usd:.2f} ({decision.optimal_size_shares:.2f} shares)")
 print(f" Expected PnL: ${decision.expected_pnl:.2f}")
 print()

# Sort by expected PnL
decisions.sort(key=lambda d: d.expected_pnl, reverse=True)

print(f"\n{'='*80}")
print(f"EXECUTION SUMMARY")
print(f"{'='*80}\n")
print(f"Opportunities found: {len(decisions)}")

if not decisions:
 print(" No profitable opportunities found")
 create_github_issue(
 "Run complete: No opportunities",
 f"Scanned {len(token_pairs)} markets, found no profitable trades"
 )
 return

print(f"\nTop opportunities:")
for i, d in enumerate(decisions[:5], 1):
 print(f"{i}. {d.title[:60]}")
 print(f" Expected PnL: ${d.expected_pnl:.2f} | Edge: {d.edge.final_edge:.3f}")

# Execute trades
print(f"\n{'='*80}")
print(f"TRADE EXECUTION")
print(f"{'='*80}\n")

executed = 0
execution_log = []

for decision in decisions[:max_orders]:
 try:
 order = OrderArgs(
 price=decision.market_price,
 size=decision.optimal_size_shares,
 side=decision.side,
 token_id=decision.token_id,
 )
 
 if dry_run:
 signed = client.create_order(order)
 log_entry = (
 f"DRY RUN:\n"
 f" Market: {decision.title}\n"
 f" Size: ${decision.optimal_size_usd:.2f} ({decision.optimal_size_shares:.4f} shares)\n"
 f" Price: {decision.market_price:.4f}\n"
 f" Fair: {decision.fair_prob:.4f}\n"
 f" Edge: {decision.edge.final_edge:.4f}\n"
 f" Expected PnL: ${decision.expected_pnl:.2f}"
 )
 print(f"✓ {log_entry}\n")
 execution_log.append(log_entry)
 executed += 1
 
 else:
 signed = client.create_order(order)
 resp = client.post_order(signed)
 
 log_entry = (
 f"LIVE TRADE EXECUTED:\n"
 f" Market: {decision.title}\n"
 f" Size: ${decision.optimal_size_usd:.2f} ({decision.optimal_size_shares:.4f} shares)\n"
 f" Price: {decision.market_price:.4f}\n"
 f" Fair: {decision.fair_prob:.4f}\n"
 f" Edge: {decision.edge.final_edge:.4f}\n"
 f" Expected PnL: ${decision.expected_pnl:.2f}\n"
 f" Response: {json.dumps(resp, indent=2)}"
 )
 print(f"✓ {log_entry}\n")
 execution_log.append(log_entry)
 executed += 1
 
 # Save prediction for learning
 market_type = classify_market_type(decision.title)
 record = PredictionRecord(
 timestamp=datetime.now(timezone.utc),
 token_id=decision.token_id,
 title=decision.title,
 predicted_fair=decision.fair_prob,
 market_price=decision.market_price,
 edge=decision.edge.final_edge
 )
 prediction_db.save_prediction(
 record,
 market_type=market_type,
 context={"size_usd": decision.optimal_size_usd}
 )
 
 except Exception as e:
 error_log = f"ERROR executing {decision.title}: {e}"
 print(f" {error_log}\n")
 execution_log.append(error_log)

# Final report
print(f"{'='*80}")
print(f"RUN COMPLETE")
print(f"{'='*80}\n")
print(f"Markets scanned: {len(token_pairs)}")
print(f"Opportunities found: {len(decisions)}")
print(f"Trades executed: {executed}")

# GitHub issue
issue_title = f"Run: {'DRY' if dry_run else 'LIVE'} | Scanned: {len(token_pairs)} | Executed: {executed}"
issue_body = "\n\n".join(execution_log[:20]) # Limit size
create_github_issue(issue_title, issue_body)

print("\n✓ Run complete\n")
```

if **name** == “**main**”:
main()
