"""
Metrics Engine — calculates all performance statistics from trade log + equity curve.
Annualisation factor for hourly bars on US equities: sqrt(252 * 6.5) ≈ 40.47
"""

import math
import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

HOURLY_ANNUAL_FACTOR = np.sqrt(252 * 6.5)
DAILY_ANNUAL_FACTOR  = np.sqrt(252)


def _safe(val, fallback=0):
    """Convert NaN/Inf/numpy types to JSON-safe Python values."""
    if val is None:
        return fallback
    if isinstance(val, (np.floating, np.float64, np.float32)):
        val = float(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return fallback
    return val


def _clean(d: dict) -> dict:
    """Recursively make a metrics dict JSON-safe."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _clean(v)
        elif isinstance(v, list):
            result[k] = [_clean(i) if isinstance(i, dict) else _safe(i) for i in v]
        else:
            result[k] = _safe(v)
    return result


def calculate_metrics(
    equity_curve: list[dict],
    trade_log: list[dict],
    initial_capital: float,
    timeframe: str = "1Hour",
) -> dict:
    """
    Calculate full suite of performance metrics.

    Args:
        equity_curve:    [{timestamp, equity}]
        trade_log:       [{entry_time, exit_time, pnl, pnl_pct, exit_reason, ...}]
        initial_capital: starting capital
        timeframe:       "1Hour" | "1Day" | "15Min"

    Returns:
        Dict of all metrics — all values JSON-safe (no NaN/Inf/numpy types)
    """
    if not equity_curve:
        return _empty_metrics()

    eq_series = pd.Series(
        [e["equity"] for e in equity_curve],
        index=pd.to_datetime([e["timestamp"] for e in equity_curve]),
    )

    annual_factor = HOURLY_ANNUAL_FACTOR if timeframe != "1Day" else DAILY_ANNUAL_FACTOR

    # ── Returns ───────────────────────────────────────────────────────────────
    bar_returns      = eq_series.pct_change().dropna()
    total_return_pct = (eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100

    # ── CAGR ─────────────────────────────────────────────────────────────────
    start = eq_series.index[0]
    end   = eq_series.index[-1]
    years = max((end - start).days / 365.25, 0.01)
    try:
        ratio = float(eq_series.iloc[-1]) / float(initial_capital)
        cagr_pct = (ratio ** (1.0 / years) - 1) * 100 if ratio > 0 else 0.0
    except Exception:
        cagr_pct = 0.0

    # ── Sharpe ────────────────────────────────────────────────────────────────
    sharpe = 0.0
    std = bar_returns.std()
    if _safe(std, 0) > 0:
        sharpe = float((bar_returns.mean() / std) * annual_factor)

    # ── Sortino ───────────────────────────────────────────────────────────────
    sortino  = 0.0
    downside = bar_returns[bar_returns < 0]
    if len(downside) > 0:
        dstd = downside.std()
        if _safe(dstd, 0) > 0:
            sortino = float((bar_returns.mean() / dstd) * annual_factor)

    # ── Max Drawdown ─────────────────────────────────────────────────────────
    rolling_max = eq_series.cummax()
    drawdown    = (eq_series - rolling_max) / rolling_max * 100
    max_dd_pct  = float(drawdown.min())

    drawdown_curve = [
        {"timestamp": str(ts), "drawdown_pct": _safe(round(dd, 4))}
        for ts, dd in drawdown.items()
    ]

    # ── No trades case ────────────────────────────────────────────────────────
    if not trade_log:
        return _clean({
            "total_return_pct":      round(total_return_pct, 4),
            "cagr_pct":              round(cagr_pct, 4),
            "sharpe_ratio":          round(sharpe, 4),
            "sortino_ratio":         round(sortino, 4),
            "max_drawdown_pct":      round(max_dd_pct, 4),
            "win_rate_pct":          0,
            "profit_factor":         0,
            "total_trades":          0,
            "avg_holding_bars":      0,
            "avg_holding_hours":     0,
            "gross_profit":          0,
            "gross_loss":            0,
            "largest_win":           0,
            "largest_loss":          0,
            "exit_reason_breakdown": {},
            "drawdown_curve":        drawdown_curve,
            "monthly_returns":       {},
        })

    # ── Trade-level stats ─────────────────────────────────────────────────────
    pnls      = [t["pnl"] for t in trade_log]
    winners   = [p for p in pnls if p > 0]
    losers    = [p for p in pnls if p <= 0]
    bars_held = [t["bars_held"] for t in trade_log]

    win_rate      = len(winners) / len(pnls) * 100
    gross_profit  = sum(winners) if winners else 0.0
    gross_loss    = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    avg_bars  = float(np.mean(bars_held)) if bars_held else 0.0
    avg_hours = avg_bars

    exit_reasons: dict[str, int] = {}
    for t in trade_log:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # ── Monthly returns ───────────────────────────────────────────────────────
    try:
        monthly = (
            eq_series
            .resample("ME")
            .last()
            .pct_change()
            .dropna()
            * 100
        )
        monthly_returns = {
            str(ts.strftime("%Y-%m")): _safe(round(ret, 4))
            for ts, ret in monthly.items()
        }
    except Exception:
        monthly_returns = {}

    raw = {
        "total_return_pct":      round(total_return_pct, 4),
        "cagr_pct":              round(cagr_pct, 4),
        "sharpe_ratio":          round(sharpe, 4),
        "sortino_ratio":         round(sortino, 4),
        "max_drawdown_pct":      round(max_dd_pct, 4),
        "win_rate_pct":          round(win_rate, 4),
        "profit_factor":         round(profit_factor, 4) if profit_factor is not None else None,
        "total_trades":          len(trade_log),
        "avg_holding_bars":      round(avg_bars, 1),
        "avg_holding_hours":     round(avg_hours, 1),
        "gross_profit":          round(gross_profit, 2),
        "gross_loss":            round(gross_loss, 2),
        "largest_win":           round(max(winners), 2) if winners else 0,
        "largest_loss":          round(min(losers), 2)  if losers  else 0,
        "exit_reason_breakdown": exit_reasons,
        "monthly_returns":       monthly_returns,
        "drawdown_curve":        drawdown_curve,
    }

    return _clean(raw)


def _empty_metrics() -> dict:
    return {
        "total_return_pct": 0, "cagr_pct": 0, "sharpe_ratio": 0,
        "sortino_ratio": 0, "max_drawdown_pct": 0, "win_rate_pct": 0,
        "profit_factor": 0, "total_trades": 0, "avg_holding_bars": 0,
        "avg_holding_hours": 0, "gross_profit": 0, "gross_loss": 0,
        "largest_win": 0, "largest_loss": 0,
        "exit_reason_breakdown": {}, "monthly_returns": {}, "drawdown_curve": [],
    }
