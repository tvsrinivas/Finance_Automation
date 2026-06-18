"""
Pivot Point Indicators — Standard Floor Trader Pivots
======================================================
Computes daily/weekly/monthly pivot levels from prior-period OHLC data.

Formulas (from your image):
    PIVOT = (High + Low + Close) / 3
    R1    = 2 × Pivot − Low
    R2    = Pivot + (High − Low)
    R3    = R1 + (High − Low)
    S1    = 2 × Pivot − High
    S2    = Pivot − (High − Low)
    S3    = S1 − (High − Low)

How this integrates with your existing engine
----------------------------------------------
Each pivot function returns a pd.Series aligned to the INTRADAY df index.
The value at each bar is the pivot level computed from the *prior* day/week/month
OHLC — so there is zero look-ahead.

Usage in strategy JSON:
    "indicators": {
        "pivot":  {"type": "PIVOT",    "params": {"period": "daily"}},
        "r1":     {"type": "PIVOT_R1", "params": {"period": "daily"}},
        "r2":     {"type": "PIVOT_R2", "params": {"period": "daily"}},
        "r3":     {"type": "PIVOT_R3", "params": {"period": "daily"}},
        "s1":     {"type": "PIVOT_S1", "params": {"period": "daily"}},
        "s2":     {"type": "PIVOT_S2", "params": {"period": "daily"}},
        "s3":     {"type": "PIVOT_S3", "params": {"period": "daily"}}
    }

Supported period values: "daily" | "weekly" | "monthly"
"""

import pandas as pd
import numpy as np


# ─── Core pivot math ──────────────────────────────────────────────────────────

def _pivot(h: float, l: float, c: float) -> float:
    return (h + l + c) / 3.0

def _r1(pivot: float, l: float) -> float:
    return 2 * pivot - l

def _r2(pivot: float, h: float, l: float) -> float:
    return pivot + (h - l)

def _r3(r1: float, h: float, l: float) -> float:
    return r1 + (h - l)

def _s1(pivot: float, h: float) -> float:
    return 2 * pivot - h

def _s2(pivot: float, h: float, l: float) -> float:
    return pivot - (h - l)

def _s3(s1: float, h: float, l: float) -> float:
    return s1 - (h - l)


# ─── Prior-period OHLC aggregation ────────────────────────────────────────────

def _get_prior_period_ohlc(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Resample intraday OHLC to prior-period bars.
    Returns a DataFrame indexed by the START of each period, containing
    the H/L/C of the *completed* prior period.

    Args:
        df:     Intraday OHLCV DataFrame with DatetimeIndex (UTC or tz-aware)
        period: "daily" | "weekly" | "monthly"

    Returns:
        DataFrame with columns [prior_high, prior_low, prior_close] and the same
        intraday DatetimeIndex as df, forward-filled to every intraday bar.
    """
    if period == "daily":
        freq = "D"
    elif period == "weekly":
        freq = "W-MON"   # week ending Sunday, label on Monday
    elif period == "monthly":
        freq = "MS"       # month start
    else:
        raise ValueError(f"Unsupported pivot period: '{period}'. Use daily/weekly/monthly.")

    # Resample to the chosen period
    ohlc = df["close"].resample(freq).ohlc()
    highs  = df["high"].resample(freq).max()
    lows   = df["low"].resample(freq).min()
    closes = df["close"].resample(freq).last()

    period_df = pd.DataFrame({
        "prior_high":  highs,
        "prior_low":   lows,
        "prior_close": closes,
    }).dropna()

    # Shift by 1 so we use PRIOR period data (no look-ahead)
    period_df = period_df.shift(1).dropna()

    # Reindex to intraday and forward-fill so every bar has a value
    prior = period_df.reindex(df.index, method="ffill")
    return prior


# ─── Public compute functions (match existing indicator function signature) ────

def compute_pivot(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Daily/weekly/monthly pivot point: (H + L + C) / 3 of prior period."""
    prior = _get_prior_period_ohlc(df, period)
    result = prior.apply(
        lambda row: _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
        axis=1
    )
    return result.rename(f"PIVOT_{period}")


def compute_pivot_r1(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Resistance 1: 2 × Pivot − Prior Low"""
    prior = _get_prior_period_ohlc(df, period)
    pivot = prior.apply(
        lambda row: _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
        axis=1
    )
    result = prior.apply(
        lambda row: _r1(
            _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
            row["prior_low"]
        ),
        axis=1
    )
    return result.rename(f"PIVOT_R1_{period}")


def compute_pivot_r2(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Resistance 2: Pivot + (Prior High − Prior Low)"""
    prior = _get_prior_period_ohlc(df, period)
    result = prior.apply(
        lambda row: _r2(
            _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
            row["prior_high"], row["prior_low"]
        ),
        axis=1
    )
    return result.rename(f"PIVOT_R2_{period}")


def compute_pivot_r3(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Resistance 3: R1 + (Prior High − Prior Low)"""
    prior = _get_prior_period_ohlc(df, period)
    result = prior.apply(
        lambda row: _r3(
            _r1(
                _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
                row["prior_low"]
            ),
            row["prior_high"], row["prior_low"]
        ),
        axis=1
    )
    return result.rename(f"PIVOT_R3_{period}")


def compute_pivot_s1(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Support 1: 2 × Pivot − Prior High"""
    prior = _get_prior_period_ohlc(df, period)
    result = prior.apply(
        lambda row: _s1(
            _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
            row["prior_high"]
        ),
        axis=1
    )
    return result.rename(f"PIVOT_S1_{period}")


def compute_pivot_s2(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Support 2: Pivot − (Prior High − Prior Low)"""
    prior = _get_prior_period_ohlc(df, period)
    result = prior.apply(
        lambda row: _s2(
            _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
            row["prior_high"], row["prior_low"]
        ),
        axis=1
    )
    return result.rename(f"PIVOT_S2_{period}")


def compute_pivot_s3(df: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Support 3: S1 − (Prior High − Prior Low)"""
    prior = _get_prior_period_ohlc(df, period)
    result = prior.apply(
        lambda row: _s3(
            _s1(
                _pivot(row["prior_high"], row["prior_low"], row["prior_close"]),
                row["prior_high"]
            ),
            row["prior_high"], row["prior_low"]
        ),
        axis=1
    )
    return result.rename(f"PIVOT_S3_{period}")


# ─── Convenience: compute all 7 levels at once ────────────────────────────────

def compute_all_pivot_levels(df: pd.DataFrame, period: str = "daily") -> dict:
    """
    Compute all 7 pivot levels in one pass (more efficient than calling each
    function separately since prior-period OHLC is fetched only once).

    Returns:
        dict with keys: pivot, r1, r2, r3, s1, s2, s3 — each a pd.Series
    """
    prior = _get_prior_period_ohlc(df, period)

    h = prior["prior_high"]
    l = prior["prior_low"]
    c = prior["prior_close"]
    rng = h - l

    pvt = (h + l + c) / 3.0
    r1  = 2 * pvt - l
    r2  = pvt + rng
    r3  = r1 + rng
    s1  = 2 * pvt - h
    s2  = pvt - rng
    s3  = s1 - rng

    return {
        "pivot": pvt.rename(f"PIVOT_{period}"),
        "r1":    r1.rename(f"PIVOT_R1_{period}"),
        "r2":    r2.rename(f"PIVOT_R2_{period}"),
        "r3":    r3.rename(f"PIVOT_R3_{period}"),
        "s1":    s1.rename(f"PIVOT_S1_{period}"),
        "s2":    s2.rename(f"PIVOT_S2_{period}"),
        "s3":    s3.rename(f"PIVOT_S3_{period}"),
    }


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("Downloading SPY 60-min data for pivot test...")
    raw = yf.download("SPY", period="30d", interval="1h", auto_adjust=True)
    raw.columns = [c.lower() for c in raw.columns]

    levels = compute_all_pivot_levels(raw, period="daily")

    print("\nLast 5 bars — daily pivot levels:")
    result_df = pd.DataFrame(levels).tail(5)
    print(result_df.round(2).to_string())

    print("\nFormula verification on most recent complete day:")
    # Get yesterday's OHLC from daily bars
    daily = yf.download("SPY", period="5d", interval="1d", auto_adjust=True)
    daily.columns = [c.lower() for c in daily.columns]
    prev = daily.iloc[-2]
    h, l, c_px = prev["high"], prev["low"], prev["close"]
    pvt = (h + l + c_px) / 3
    print(f"  Prior day  H={h:.2f}  L={l:.2f}  C={c_px:.2f}")
    print(f"  Pivot={pvt:.2f}  R1={2*pvt-l:.2f}  R2={pvt+(h-l):.2f}  R3={2*pvt-l+(h-l):.2f}")
    print(f"  S1={2*pvt-h:.2f}  S2={pvt-(h-l):.2f}  S3={2*pvt-h-(h-l):.2f}")
    print(f"\n  From series (last bar): Pivot={levels['pivot'].iloc[-1]:.2f}")
    print("  ✓ Match" if abs(levels['pivot'].iloc[-1] - pvt) < 0.01 else "  ✗ Mismatch — check timezone alignment")
