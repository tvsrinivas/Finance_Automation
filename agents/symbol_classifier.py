"""
Symbol Classifier Agent
Fetches price data for all symbols in symbol_universe and classifies
them by trading characteristics into strategy types.

Computes:
  trend_score        — how often price is above SMA200 (0-1)
  mean_reversion_score — how frequently RSI oscillates (0-1)
  momentum_score     — combination of trend + volatility
  avg_atr_pct_30d    — average ATR as % of price (volatility)
  avg_volume_30d     — average daily volume (liquidity)

Writes results to:
  symbol_universe         — computed characteristics
  symbol_strategy_mapping — strategy type assignments

Schedule: Run weekly (Sunday) via GitHub Actions
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ─── Strategy type thresholds ─────────────────────────────────────────────────

STRATEGY_THRESHOLDS = {
    "trend_following": {
        "min_trend_score":    0.6,   # above SMA200 > 60% of time
        "max_atr_pct":        4.0,   # not too volatile
        "min_volume":         1_000_000,
    },
    "momentum_breakout": {
        "min_trend_score":    0.55,
        "min_atr_pct":        2.0,   # needs volatility
        "min_momentum_score": 0.6,
        "min_volume":         500_000,
    },
    "mean_reversion": {
        "max_trend_score":    0.75,  # not strongly trending
        "min_mr_score":       0.5,   # oscillates frequently
        "min_volume":         500_000,
    },
    "pullback_entry": {
        "min_trend_score":    0.65,  # in uptrend
        "min_volume":         1_000_000,
        "max_atr_pct":        5.0,
    },
    "rsi_oversold_recovery": {
        "min_volume":         1_000_000,
        "min_atr_pct":        1.0,   # needs some volatility to get oversold
    },
    "sma_crossover": {
        "min_trend_score":    0.5,
        "min_volume":         1_000_000,
        "max_atr_pct":        4.0,
    },
}


# ─── Fetch price data ─────────────────────────────────────────────────────────

def fetch_symbol_data(symbol: str, days: int = 260) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV bars for a symbol from Alpaca."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
        )

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
            adjustment="all",
            feed="iex",
        )

        bars = client.get_stock_bars(request)
        df   = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        if df.empty or len(df) < 50:
            return None

        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return df

    except Exception as e:
        logger.warning(f"Could not fetch {symbol}: {e}")
        return None


# ─── Compute characteristics ──────────────────────────────────────────────────

def compute_characteristics(df: pd.DataFrame) -> dict:
    """Compute trading characteristics from OHLCV data."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    n      = len(df)

    # ── Trend score: % of days price > SMA200 ───────────────────────────────
    if n >= 200:
        sma200     = close.rolling(200).mean()
        above_200  = (close > sma200).iloc[200:].mean()
        trend_score = float(above_200)
    elif n >= 50:
        sma50      = close.rolling(50).mean()
        above_50   = (close > sma50).iloc[50:].mean()
        trend_score = float(above_50) * 0.8  # discount for shorter window
    else:
        trend_score = 0.5

    # ── SMA50 trend ──────────────────────────────────────────────────────────
    if n >= 50:
        sma50_val = close.rolling(50).mean().iloc[-1]
        sma50_trend = float(close.iloc[-1] / sma50_val - 1) * 100
    else:
        sma50_trend = 0.0

    # ── ATR as % of price (volatility) ──────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-30:]
    price_30 = close.iloc[-30:]
    atr_pct = float((atr_14 / price_30).mean() * 100) if len(atr_14) > 0 else 0.0

    # ── Mean reversion score: RSI oscillation frequency ─────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))

    # Count crosses through 50 (oscillation frequency)
    if len(rsi.dropna()) > 20:
        rsi_clean    = rsi.dropna().iloc[-100:]  # last 100 days
        crosses      = ((rsi_clean.shift(1) < 50) & (rsi_clean >= 50)).sum() + \
                       ((rsi_clean.shift(1) > 50) & (rsi_clean <= 50)).sum()
        mr_score     = float(min(crosses / 20, 1.0))  # normalise: 20 crosses = perfect MR
    else:
        mr_score = 0.5

    # ── Momentum score: trend + volatility combo ─────────────────────────────
    momentum_score = float(trend_score * 0.5 + min(atr_pct / 4, 1.0) * 0.5)

    # ── Average volume ───────────────────────────────────────────────────────
    avg_volume = float(volume.iloc[-30:].mean()) if len(volume) >= 30 else float(volume.mean())

    return {
        "trend_score":         round(trend_score, 4),
        "mean_reversion_score": round(mr_score, 4),
        "momentum_score":      round(momentum_score, 4),
        "avg_atr_pct_30d":     round(atr_pct, 4),
        "avg_volume_30d":      round(avg_volume, 2),
        "sma50_trend_pct":     round(sma50_trend, 2),
        "bars_analysed":       n,
    }


# ─── Classify into strategy types ────────────────────────────────────────────

def classify_symbol(chars: dict) -> list[dict]:
    """
    Given computed characteristics, return list of strategy type assignments.
    Each item has strategy_type and confidence score.
    """
    trend  = chars["trend_score"]
    mr     = chars["mean_reversion_score"]
    mom    = chars["momentum_score"]
    atr    = chars["avg_atr_pct_30d"]
    vol    = chars["avg_volume_30d"]

    assignments = []
    thresh      = STRATEGY_THRESHOLDS

    # Trend following
    t = thresh["trend_following"]
    if trend >= t["min_trend_score"] and atr <= t["max_atr_pct"] and vol >= t["min_volume"]:
        confidence = trend * 0.7 + (1 - min(atr / 5, 1)) * 0.3
        assignments.append({"strategy_type": "trend_following", "confidence": round(confidence, 3)})

    # Momentum breakout
    t = thresh["momentum_breakout"]
    if trend >= t["min_trend_score"] and atr >= t["min_atr_pct"] and mom >= t["min_momentum_score"] and vol >= t["min_volume"]:
        confidence = mom * 0.6 + min(atr / 4, 1) * 0.4
        assignments.append({"strategy_type": "momentum_breakout", "confidence": round(confidence, 3)})

    # Mean reversion
    t = thresh["mean_reversion"]
    if trend <= t["max_trend_score"] and mr >= t["min_mr_score"] and vol >= t["min_volume"]:
        confidence = mr * 0.6 + (1 - trend) * 0.4
        assignments.append({"strategy_type": "mean_reversion", "confidence": round(confidence, 3)})

    # Pullback entry
    t = thresh["pullback_entry"]
    if trend >= t["min_trend_score"] and vol >= t["min_volume"] and atr <= t["max_atr_pct"]:
        confidence = trend * 0.6 + (1 - min(atr / 5, 1)) * 0.4
        assignments.append({"strategy_type": "pullback_entry", "confidence": round(confidence, 3)})

    # RSI oversold recovery
    t = thresh["rsi_oversold_recovery"]
    if vol >= t["min_volume"] and atr >= t["min_atr_pct"]:
        confidence = min(atr / 3, 1) * 0.4 + mr * 0.3 + trend * 0.3
        assignments.append({"strategy_type": "rsi_oversold_recovery", "confidence": round(confidence, 3)})

    # SMA crossover
    t = thresh["sma_crossover"]
    if trend >= t["min_trend_score"] and vol >= t["min_volume"] and atr <= t["max_atr_pct"]:
        confidence = trend * 0.8 + (1 - min(atr / 5, 1)) * 0.2
        assignments.append({"strategy_type": "sma_crossover", "confidence": round(confidence, 3)})

    return sorted(assignments, key=lambda x: x["confidence"], reverse=True)


# ─── Update DB ────────────────────────────────────────────────────────────────

def update_symbol_in_db(symbol: str, chars: dict, assignments: list, db) -> None:
    """Write computed characteristics and strategy mappings to Neon."""
    from sqlalchemy import text

    # Update symbol_universe with computed characteristics
    db.execute(text("""
        UPDATE symbol_universe SET
            avg_volume_30d       = :vol,
            avg_atr_pct_30d      = :atr,
            trend_score          = :trend,
            mean_reversion_score = :mr,
            momentum_score       = :mom,
            last_classified      = NOW()
        WHERE symbol = :symbol
    """), {
        "symbol": symbol,
        "vol":    chars["avg_volume_30d"],
        "atr":    chars["avg_atr_pct_30d"],
        "trend":  chars["trend_score"],
        "mr":     chars["mean_reversion_score"],
        "mom":    chars["momentum_score"],
    })

    # Deactivate old classifier assignments (keep manual ones)
    db.execute(text("""
        UPDATE symbol_strategy_mapping
        SET is_active = FALSE
        WHERE symbol = :symbol AND assigned_by = 'classifier_agent'
    """), {"symbol": symbol})

    # Insert new assignments
    for a in assignments:
        db.execute(text("""
            INSERT INTO symbol_strategy_mapping
                (symbol, strategy_type, confidence, assigned_by, assigned_at, is_active)
            VALUES
                (:symbol, :strategy_type, :confidence, 'classifier_agent', NOW(), TRUE)
            ON CONFLICT (symbol, strategy_type) DO UPDATE SET
                confidence   = EXCLUDED.confidence,
                assigned_by  = 'classifier_agent',
                assigned_at  = NOW(),
                is_active    = TRUE
        """), {
            "symbol":        symbol,
            "strategy_type": a["strategy_type"],
            "confidence":    a["confidence"],
        })

    db.commit()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_symbol_classifier(max_symbols: int = 100) -> dict:
    """
    Classify all active symbols in symbol_universe.
    Fetches price data, computes characteristics, updates DB.
    """
    from db.connection import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    results = {
        "classified":  0,
        "failed":      0,
        "symbols":     [],
        "errors":      [],
    }

    try:
        rows = db.execute(text("""
            SELECT symbol FROM symbol_universe
            WHERE is_active = TRUE
            ORDER BY market_cap_rank ASC NULLS LAST
            LIMIT :limit
        """), {"limit": max_symbols}).fetchall()

        symbols = [r[0] for r in rows]
        logger.info(f"Classifying {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"  [{i}/{len(symbols)}] {symbol}")
            try:
                df = fetch_symbol_data(symbol)
                if df is None:
                    logger.warning(f"  Skipping {symbol} — no data")
                    results["failed"] += 1
                    continue

                chars       = compute_characteristics(df)
                assignments = classify_symbol(chars)

                update_symbol_in_db(symbol, chars, assignments, db)

                results["classified"] += 1
                results["symbols"].append({
                    "symbol":          symbol,
                    "trend_score":     chars["trend_score"],
                    "atr_pct":         chars["avg_atr_pct_30d"],
                    "momentum_score":  chars["momentum_score"],
                    "strategy_types":  [a["strategy_type"] for a in assignments[:3]],
                    "top_confidence":  assignments[0]["confidence"] if assignments else 0,
                })
                logger.info(f"    -> {[a['strategy_type'] for a in assignments[:2]]}")

            except Exception as e:
                logger.error(f"  Error classifying {symbol}: {e}")
                results["errors"].append({"symbol": symbol, "error": str(e)})
                results["failed"] += 1

    finally:
        db.close()

    logger.info(f"Classification complete: {results['classified']} classified, {results['failed']} failed")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    result = run_symbol_classifier()
    print(json.dumps(result, indent=2, default=str))
