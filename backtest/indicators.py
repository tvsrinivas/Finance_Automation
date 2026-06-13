"""
Indicator Engine — computes all technical indicators from OHLCV data.
All functions are pure: DataFrame in, Series/dict out.
No look-ahead bias: each value at index i uses only data up to i.
"""

import pandas as pd
import numpy as np
from typing import Union


# ─── Core indicator functions ─────────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, period: int = 20, source: str = "close") -> pd.Series:
    return df[source].rolling(window=period, min_periods=period).mean()


def compute_ema(df: pd.DataFrame, period: int = 20, source: str = "close") -> pd.Series:
    return df[source].ewm(span=period, adjust=False, min_periods=period).mean()


def compute_rsi(df: pd.DataFrame, period: int = 14, source: str = "close") -> pd.Series:
    delta = df[source].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("RSI")


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    ema_fast   = df["close"].ewm(span=fast,   adjust=False, min_periods=fast).mean()
    ema_slow   = df["close"].ewm(span=slow,   adjust=False, min_periods=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram  = macd_line - signal_line
    return {
        "macd_line":   macd_line.rename("MACD_line"),
        "signal_line": signal_line.rename("MACD_signal"),
        "histogram":   histogram.rename("MACD_hist"),
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False, min_periods=period).mean().rename("ATR")


def compute_price(df: pd.DataFrame, source: str = "close") -> pd.Series:
    return df[source].rename(f"PRICE_{source}")


def compute_volume(df: pd.DataFrame, avg_period: int = None) -> pd.Series:
    if avg_period:
        return df["volume"].rolling(window=avg_period, min_periods=avg_period).mean().rename("VOLUME_avg")
    return df["volume"].rename("VOLUME")


def compute_52w_high(df: pd.DataFrame) -> pd.Series:
    # 52 weeks = 252 trading days. For hourly: 252 * 6.5 ≈ 1638 bars
    # We use 252-day rolling on daily-equivalent; for hourly we use 1638 bars
    bars = 1638
    return df["high"].rolling(window=bars, min_periods=1).max().rename("WEEK52_HIGH")


def compute_52w_low(df: pd.DataFrame) -> pd.Series:
    bars = 1638
    return df["low"].rolling(window=bars, min_periods=1).min().rename("WEEK52_LOW")


# ─── Indicator dispatch map ───────────────────────────────────────────────────

INDICATOR_FUNCTIONS = {
    "SMA":         compute_sma,
    "EMA":         compute_ema,
    "RSI":         compute_rsi,
    "MACD":        compute_macd,
    "ATR":         compute_atr,
    "PRICE":       compute_price,
    "VOLUME":      compute_volume,
    "WEEK52_HIGH": compute_52w_high,
    "WEEK52_LOW":  compute_52w_low,
}


# ─── Public: compute all indicators from strategy JSON ───────────────────────

def compute_all_indicators(
    df: pd.DataFrame,
    indicators_spec: dict,
) -> dict[str, Union[pd.Series, dict]]:
    """
    Compute all indicators defined in the strategy's indicators block.

    Args:
        df:              OHLCV DataFrame
        indicators_spec: strategy["indicators"] dict

    Returns:
        Dict mapping indicator_id → Series (or dict of Series for MACD)
    """
    computed = {}

    for ind_id, ind_def in indicators_spec.items():
        ind_type = ind_def.get("type")
        params   = ind_def.get("params", {})

        if ind_type not in INDICATOR_FUNCTIONS:
            raise ValueError(f"Unknown indicator type: {ind_type}")

        fn = INDICATOR_FUNCTIONS[ind_type]

        try:
            result = fn(df, **{k: v for k, v in params.items() if v is not None})
        except TypeError as e:
            raise ValueError(f"Error computing {ind_type} with params {params}: {e}")

        # For composite indicators (MACD), store by component
        if isinstance(result, dict):
            component = ind_def.get("component", "macd_line")
            computed[ind_id] = result[component]
        else:
            computed[ind_id] = result

    return computed


def get_indicator_value(
    computed: dict,
    ind_id: str,
    bar_idx: int,
) -> float:
    """Get indicator value at a specific bar index. Returns NaN if not available."""
    series = computed.get(ind_id)
    if series is None:
        return np.nan
    if bar_idx < 0 or bar_idx >= len(series):
        return np.nan
    return series.iloc[bar_idx]
