"""
Data Layer — fetches OHLCV price data.
Primary source: Alpaca Market Data API (hourly bars)
Fallback:       yfinance (for testing without Alpaca keys)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─── Alpaca ───────────────────────────────────────────────────────────────────

def fetch_alpaca(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Hour",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca Market Data API.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in environment.

    timeframe options: "1Hour", "1Day", "15Min", "5Min"
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment")

        client = StockHistoricalDataClient(api_key, secret_key)

        tf_map = {
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day":  TimeFrame(1, TimeFrameUnit.Day),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
            adjustment="all",  # split + dividend adjusted
            feed="iex",
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df.rename(columns={
            "open":   "open",
            "high":   "high",
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        df = df.sort_index()

        logger.info(f"Alpaca: fetched {len(df)} bars for {symbol}")
        return df

    except ImportError:
        raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
    except Exception as e:
        logger.error(f"Alpaca fetch failed: {e}")
        raise


# ─── yfinance fallback ────────────────────────────────────────────────────────

def fetch_yfinance(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Hour",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from yfinance.
    Used as fallback when Alpaca keys are not available.
    Note: yfinance hourly data only available for last 730 days.
    """
    try:
        import yfinance as yf

        interval_map = {
            "1Hour": "1h",
            "1Day":  "1d",
            "15Min": "15m",
            "5Min":  "5m",
        }
        interval = interval_map.get(timeframe, "1h")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({start} to {end})")

        df = df.rename(columns={
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        df = df.sort_index()

        logger.info(f"yfinance: fetched {len(df)} bars for {symbol}")
        return df

    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")


# ─── Public interface ─────────────────────────────────────────────────────────

def fetch_price_data(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Hour",
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data. Tries Alpaca first, falls back to yfinance.

    Args:
        symbol:    Ticker symbol e.g. "AAPL"
        start:     Start date "YYYY-MM-DD"
        end:       End date "YYYY-MM-DD"
        timeframe: "1Hour" | "1Day" | "15Min" | "5Min"
        source:    Force "alpaca" or "yfinance" (auto-detect if None)

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: timestamp (UTC timezone-aware)
    """
    symbol = symbol.upper().strip()

    has_alpaca_keys = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))

    if source == "alpaca" or (source is None and has_alpaca_keys):
        try:
            return fetch_alpaca(symbol, start, end, timeframe)
        except Exception as e:
            if source == "alpaca":
                raise
            logger.warning(f"Alpaca failed, falling back to yfinance: {e}")

    return fetch_yfinance(symbol, start, end, timeframe)


def validate_symbol(symbol: str) -> dict:
    """Quick check that a symbol exists and has data."""
    try:
        from datetime import datetime, timedelta
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        df = fetch_price_data(symbol, start, end, "1Day")
        return {"valid": True, "name": symbol, "bars": len(df)}
    except Exception as e:
        return {"valid": False, "error": str(e)}