"""
Market Intelligence Agent — v1
Layer 1: Price-based regime detection from Alpaca data
Layer 2: Economic calendar (Fed, CPI, Jobs)

Output: structured MarketBrief passed to Hypothesis Generator
which then feeds the Strategy Creation Agent.

Schedule: Run daily at 7 AM ET before market open (GitHub Actions)
"""

import os
import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional
import json

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ─── Layer 2: Economic Calendar ───────────────────────────────────────────────
# Hardcoded 2026 high-impact dates. Update annually.
# Format: (date_string, event_name, expected_impact)

ECONOMIC_CALENDAR_2026 = [
    # Fed Meeting dates (FOMC)
    ("2026-01-28", "FOMC Rate Decision",        "very_high"),
    ("2026-03-18", "FOMC Rate Decision",        "very_high"),
    ("2026-04-29", "FOMC Rate Decision",        "very_high"),
    ("2026-06-17", "FOMC Rate Decision",        "very_high"),
    ("2026-07-29", "FOMC Rate Decision",        "very_high"),
    ("2026-09-16", "FOMC Rate Decision",        "very_high"),
    ("2026-11-04", "FOMC Rate Decision",        "very_high"),
    ("2026-12-16", "FOMC Rate Decision",        "very_high"),

    # CPI releases (approximate — 2nd or 3rd week of month)
    ("2026-01-14", "CPI Inflation Report",      "high"),
    ("2026-02-11", "CPI Inflation Report",      "high"),
    ("2026-03-11", "CPI Inflation Report",      "high"),
    ("2026-04-10", "CPI Inflation Report",      "high"),
    ("2026-05-13", "CPI Inflation Report",      "high"),
    ("2026-06-10", "CPI Inflation Report",      "high"),
    ("2026-07-14", "CPI Inflation Report",      "high"),
    ("2026-08-12", "CPI Inflation Report",      "high"),
    ("2026-09-09", "CPI Inflation Report",      "high"),
    ("2026-10-14", "CPI Inflation Report",      "high"),
    ("2026-11-12", "CPI Inflation Report",      "high"),
    ("2026-12-09", "CPI Inflation Report",      "high"),

    # Jobs Report (Non-Farm Payrolls — first Friday of month)
    ("2026-01-09", "Non-Farm Payrolls",         "high"),
    ("2026-02-06", "Non-Farm Payrolls",         "high"),
    ("2026-03-06", "Non-Farm Payrolls",         "high"),
    ("2026-04-03", "Non-Farm Payrolls",         "high"),
    ("2026-05-01", "Non-Farm Payrolls",         "high"),
    ("2026-06-05", "Non-Farm Payrolls",         "high"),
    ("2026-07-10", "Non-Farm Payrolls",         "high"),
    ("2026-08-07", "Non-Farm Payrolls",         "high"),
    ("2026-09-04", "Non-Farm Payrolls",         "high"),
    ("2026-10-02", "Non-Farm Payrolls",         "high"),
    ("2026-11-06", "Non-Farm Payrolls",         "high"),
    ("2026-12-04", "Non-Farm Payrolls",         "high"),

    # Earnings seasons (approximate — peak weeks)
    ("2026-01-12", "Earnings Season Start",     "medium"),
    ("2026-04-13", "Earnings Season Start",     "medium"),
    ("2026-07-13", "Earnings Season Start",     "medium"),
    ("2026-10-12", "Earnings Season Start",     "medium"),
]


def get_upcoming_events(days_ahead: int = 3) -> list[dict]:
    """Return high-impact events in the next N days."""
    today     = date.today()
    lookahead = today + timedelta(days=days_ahead)
    upcoming  = []

    for date_str, event, impact in ECONOMIC_CALENDAR_2026:
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if today <= event_date <= lookahead:
            days_until = (event_date - today).days
            upcoming.append({
                "date":       date_str,
                "event":      event,
                "impact":     impact,
                "days_until": days_until,
            })

    return sorted(upcoming, key=lambda x: x["date"])


# ─── Layer 1: Price Regime Detection ─────────────────────────────────────────

def fetch_market_data() -> dict:
    """
    Fetch recent daily bars for key market indicators from Alpaca.
    Returns OHLCV data for SPY, QQQ, VIX proxy (VIXY), and sector ETFs.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    import pandas as pd

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    client = StockHistoricalDataClient(api_key, secret_key)

    # Symbols to fetch — market + sectors
    symbols = [
        "SPY",   # S&P 500 — broad market
        "QQQ",   # Nasdaq — tech/growth
        "VIXY",  # VIX proxy ETF (short-term volatility)
        "XLK",   # Tech
        "XLE",   # Energy
        "XLF",   # Financials
        "XLV",   # Healthcare
        "XLI",   # Industrials
        "XLP",   # Consumer Staples (defensive)
        "XLU",   # Utilities (defensive)
    ]

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=300)  # enough for SMA200

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start,
        end=end,
        adjustment="all",
        feed="iex",
    )

    bars  = client.get_stock_bars(request)
    df    = bars.df

    result = {}
    for sym in symbols:
        try:
            if isinstance(df.index, pd.MultiIndex):
                sym_df = df.xs(sym, level="symbol")
            else:
                sym_df = df[df.index.get_level_values("symbol") == sym]

            sym_df = sym_df.sort_index()
            result[sym] = {
                "close":  list(sym_df["close"]),
                "volume": list(sym_df["volume"]),
                "dates":  [str(d) for d in sym_df.index],
            }
        except Exception as e:
            logger.warning(f"Could not fetch {sym}: {e}")

    logger.info(f"Fetched market data for {len(result)} symbols")
    return result


def compute_sma(prices: list, period: int) -> Optional[float]:
    """Simple moving average of last N prices."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def compute_rsi(prices: list, period: int = 14) -> Optional[float]:
    """RSI from closing prices."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    gains  = [d for d in recent if d > 0]
    losses = [abs(d) for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs  = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def sector_performance(data: dict, days: int = 20) -> dict:
    """
    Compute N-day performance for each sector ETF.
    Returns dict of {symbol: pct_change}
    """
    sectors = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLU"]
    perf    = {}
    for sym in sectors:
        if sym not in data:
            continue
        prices = data[sym]["close"]
        if len(prices) >= days + 1:
            start_price = prices[-(days+1)]
            end_price   = prices[-1]
            if start_price > 0:
                perf[sym] = round((end_price / start_price - 1) * 100, 2)
    return perf


def classify_regime(data: dict) -> dict:
    """
    Classify current market regime from price data.

    Returns:
        regime:            bull_quiet | bull_volatile | bear | sideways
        confidence:        0.0 - 1.0
        spy_vs_sma200:     above | below | at
        spy_vs_sma50:      above | below | at
        vix_level:         low | medium | high | extreme
        vix_value:         float
        trend_strength:    strong | moderate | weak
        breadth:           broad | mixed | narrow
    """
    result = {}

    # ── SPY analysis ─────────────────────────────────────────────────────────
    if "SPY" not in data:
        return {"regime": "unknown", "confidence": 0.0}

    spy_prices  = data["SPY"]["close"]
    spy_current = spy_prices[-1]

    sma50  = compute_sma(spy_prices, 50)
    sma200 = compute_sma(spy_prices, 200)
    sma20  = compute_sma(spy_prices, 20)
    rsi    = compute_rsi(spy_prices)

    # SPY vs moving averages
    if sma200:
        pct_from_200 = (spy_current - sma200) / sma200 * 100
        if spy_current > sma200 * 1.01:
            result["spy_vs_sma200"] = "above"
        elif spy_current < sma200 * 0.99:
            result["spy_vs_sma200"] = "below"
        else:
            result["spy_vs_sma200"] = "at"
    else:
        result["spy_vs_sma200"] = "unknown"
        pct_from_200 = 0

    if sma50:
        if spy_current > sma50 * 1.005:
            result["spy_vs_sma50"] = "above"
        elif spy_current < sma50 * 0.995:
            result["spy_vs_sma50"] = "below"
        else:
            result["spy_vs_sma50"] = "at"
    else:
        result["spy_vs_sma50"] = "unknown"

    # Trend strength (20-day momentum)
    if sma20 and len(spy_prices) >= 21:
        momentum_20d = (spy_prices[-1] / spy_prices[-21] - 1) * 100
        if abs(momentum_20d) > 5:
            result["trend_strength"] = "strong"
        elif abs(momentum_20d) > 2:
            result["trend_strength"] = "moderate"
        else:
            result["trend_strength"] = "weak"
        result["momentum_20d"] = round(momentum_20d, 2)
    else:
        result["trend_strength"] = "unknown"
        result["momentum_20d"]   = 0

    result["spy_price"]  = round(spy_current, 2)
    result["spy_rsi"]    = rsi
    result["spy_sma50"]  = round(sma50, 2) if sma50 else None
    result["spy_sma200"] = round(sma200, 2) if sma200 else None

    # ── VIX proxy (VIXY) ─────────────────────────────────────────────────────
    vix_value = None
    if "VIXY" in data:
        vixy_prices = data["VIXY"]["close"]
        vixy_current = vixy_prices[-1]
        vixy_sma20   = compute_sma(vixy_prices, 20)

        # VIXY is a VIX ETF — price roughly correlates with VIX
        # Typical ranges: <15 low, 15-25 medium, 25-35 high, >35 extreme
        # VIXY trades at different levels so we use relative comparison
        if vixy_sma20:
            vix_ratio = vixy_current / vixy_sma20
        else:
            vix_ratio = 1.0

        vix_value = vixy_current
        result["vixy_price"]  = round(vixy_current, 2)
        result["vixy_sma20"]  = round(vixy_sma20, 2) if vixy_sma20 else None
        result["vixy_ratio"]  = round(vix_ratio, 2)

        # Classify volatility
        if vix_ratio > 1.3 or vixy_current > 25:
            result["vix_level"] = "extreme"
        elif vix_ratio > 1.1 or vixy_current > 18:
            result["vix_level"] = "high"
        elif vix_ratio < 0.9 and vixy_current < 14:
            result["vix_level"] = "low"
        else:
            result["vix_level"] = "medium"
    else:
        result["vix_level"]  = "unknown"
        result["vixy_price"] = None

    # ── Sector breadth ────────────────────────────────────────────────────────
    sector_perf = sector_performance(data, days=20)
    result["sector_performance"] = sector_perf

    positive_sectors = [s for s, p in sector_perf.items() if p > 0]
    negative_sectors = [s for s, p in sector_perf.items() if p < 0]

    if len(positive_sectors) >= 6:
        result["breadth"] = "broad"
    elif len(positive_sectors) >= 4:
        result["breadth"] = "mixed"
    else:
        result["breadth"] = "narrow"

    # Leading/lagging sectors
    sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
    result["leading_sectors"]  = [s for s, _ in sorted_sectors[:3]]
    result["lagging_sectors"]  = [s for s, _ in sorted_sectors[-3:]]

    # ── QQQ vs SPY (growth vs value) ─────────────────────────────────────────
    if "QQQ" in data and "SPY" in data:
        qqq_perf = (data["QQQ"]["close"][-1] / data["QQQ"]["close"][-21] - 1) * 100
        spy_perf = (data["SPY"]["close"][-1] / data["SPY"]["close"][-21] - 1) * 100
        result["qqq_vs_spy_20d"] = round(qqq_perf - spy_perf, 2)
        result["growth_leading"] = qqq_perf > spy_perf
    else:
        result["qqq_vs_spy_20d"] = 0
        result["growth_leading"] = None

    # ── Regime classification ─────────────────────────────────────────────────
    above_200 = result["spy_vs_sma200"] == "above"
    above_50  = result["spy_vs_sma50"]  == "above"
    vix_high  = result["vix_level"] in ("high", "extreme")
    vix_low   = result["vix_level"] == "low"

    confidence = 0.5

    if above_200 and above_50:
        if vix_low:
            regime     = "bull_quiet"
            confidence = 0.85
        elif vix_high:
            regime     = "bull_volatile"
            confidence = 0.75
        else:
            regime     = "bull_quiet"
            confidence = 0.65
    elif not above_200:
        regime     = "bear"
        confidence = 0.80
    elif above_200 and not above_50:
        regime     = "sideways"
        confidence = 0.65
    else:
        regime     = "sideways"
        confidence = 0.50

    # Reduce confidence if trend is weak or breadth is narrow
    if result["trend_strength"] == "weak":
        confidence -= 0.10
    if result["breadth"] == "narrow":
        confidence -= 0.10

    result["regime"]     = regime
    result["confidence"] = round(max(0.1, min(1.0, confidence)), 2)

    return result


# ─── Strategy hints from regime ───────────────────────────────────────────────

REGIME_STRATEGY_MAP = {
    "bull_quiet": {
        "recommended": ["trend_following", "momentum_breakout", "sma_crossover"],
        "avoid":       ["mean_reversion", "counter_trend", "short_bias"],
        "hints": [
            "Trend-following strategies work best in low-volatility bull markets",
            "Momentum breakouts above key levels likely to sustain",
            "Full position sizing appropriate — low risk environment",
        ],
        "position_size_modifier": 1.0,
        "stop_loss_modifier":     1.0,
    },
    "bull_volatile": {
        "recommended": ["pullback_entry", "rsi_oversold_recovery", "mean_reversion"],
        "avoid":       ["breakout_without_confirmation", "wide_stops"],
        "hints": [
            "Bull trend intact but volatility elevated — wait for pullbacks",
            "RSI-based entries at oversold levels work well",
            "Reduce position size by 30-50% due to volatility",
        ],
        "position_size_modifier": 0.65,
        "stop_loss_modifier":     1.3,
    },
    "bear": {
        "recommended": ["cash", "short_term_mean_reversion", "defensive_rotation"],
        "avoid":       ["trend_following", "momentum_long", "breakout_long"],
        "hints": [
            "Bear market — avoid new long trend entries",
            "Short-term mean reversion only on extreme oversold readings",
            "Significantly reduce position sizes or stay flat",
        ],
        "position_size_modifier": 0.3,
        "stop_loss_modifier":     0.8,
    },
    "sideways": {
        "recommended": ["range_bound", "rsi_oscillator", "mean_reversion"],
        "avoid":       ["trend_following", "breakout_long"],
        "hints": [
            "Sideways market — trend-following strategies will whipsaw",
            "RSI oscillator strategies work well in range-bound conditions",
            "Keep positions small — direction unclear",
        ],
        "position_size_modifier": 0.5,
        "stop_loss_modifier":     0.9,
    },
}


def get_strategy_hints(regime: str, upcoming_events: list) -> dict:
    """Generate strategy recommendations based on regime and upcoming events."""
    hints_data = REGIME_STRATEGY_MAP.get(regime, REGIME_STRATEGY_MAP["sideways"]).copy()

    # Adjust for upcoming events
    if upcoming_events:
        next_event = upcoming_events[0]
        if next_event["impact"] == "very_high" and next_event["days_until"] <= 2:
            # FOMC within 2 days — skip new entries
            hints_data["hints"].insert(0,
                f"⚠ {next_event['event']} in {next_event['days_until']} day(s) — "
                f"avoid new entries, high volatility expected"
            )
            hints_data["position_size_modifier"] = 0.0  # no new trades
            hints_data["event_blackout"] = True
        elif next_event["impact"] in ("very_high", "high") and next_event["days_until"] <= 3:
            hints_data["hints"].insert(0,
                f"⚠ {next_event['event']} in {next_event['days_until']} day(s) — "
                f"reduce position size and widen stops"
            )
            hints_data["position_size_modifier"] *= 0.6
            hints_data["stop_loss_modifier"]     *= 1.2
        hints_data["event_blackout"] = hints_data.get("event_blackout", False)
    else:
        hints_data["event_blackout"] = False

    return hints_data


# ─── Main: Build Market Brief ─────────────────────────────────────────────────

def run_market_intelligence_agent() -> dict:
    """
    Main entrypoint. Runs full market intelligence analysis.

    Returns structured MarketBrief ready for Hypothesis Generator.
    """
    logger.info("Market Intelligence Agent starting...")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        # Layer 1: Fetch and classify price data
        logger.info("Fetching market data from Alpaca...")
        market_data = fetch_market_data()

        logger.info("Classifying market regime...")
        regime_data = classify_regime(market_data)

        # Layer 2: Economic calendar
        logger.info("Checking economic calendar...")
        upcoming_events = get_upcoming_events(days_ahead=3)

        # Strategy hints
        strategy_hints = get_strategy_hints(
            regime_data["regime"],
            upcoming_events,
        )

        brief = {
            "date":             today,
            "generated_at":     datetime.now(timezone.utc).isoformat(),

            # Regime
            "regime":           regime_data["regime"],
            "regime_confidence": regime_data["confidence"],

            # SPY
            "spy_price":        regime_data.get("spy_price"),
            "spy_vs_sma50":     regime_data.get("spy_vs_sma50"),
            "spy_vs_sma200":    regime_data.get("spy_vs_sma200"),
            "spy_rsi":          regime_data.get("spy_rsi"),
            "spy_sma50":        regime_data.get("spy_sma50"),
            "spy_sma200":       regime_data.get("spy_sma200"),
            "momentum_20d":     regime_data.get("momentum_20d"),

            # Volatility
            "vix_level":        regime_data.get("vix_level"),
            "vixy_price":       regime_data.get("vixy_price"),

            # Breadth and sectors
            "breadth":          regime_data.get("breadth"),
            "trend_strength":   regime_data.get("trend_strength"),
            "leading_sectors":  regime_data.get("leading_sectors", []),
            "lagging_sectors":  regime_data.get("lagging_sectors", []),
            "sector_performance": regime_data.get("sector_performance", {}),
            "growth_leading":   regime_data.get("growth_leading"),

            # Calendar
            "upcoming_events":  upcoming_events,
            "event_blackout":   strategy_hints.get("event_blackout", False),

            # Strategy recommendations
            "recommended_strategy_types": strategy_hints["recommended"],
            "avoid_strategy_types":       strategy_hints["avoid"],
            "strategy_hints":             strategy_hints["hints"],
            "position_size_modifier":     strategy_hints["position_size_modifier"],
            "stop_loss_modifier":         strategy_hints["stop_loss_modifier"],
        }

        logger.info(
            f"Market brief complete — regime={brief['regime']} "
            f"confidence={brief['regime_confidence']} "
            f"vix={brief['vix_level']} "
            f"events={len(upcoming_events)}"
        )

        return {"success": True, "brief": brief}

    except Exception as e:
        logger.error(f"Market Intelligence Agent failed: {e}")
        return {
            "success": False,
            "error":   str(e),
            "brief":   None,
        }


# ─── Run standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_market_intelligence_agent()
    print(json.dumps(result, indent=2, default=str))
