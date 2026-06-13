"""
Metrics Engine — calculates all performance statistics from trade log + equity curve.
Annualisation factor for hourly bars on US equities: sqrt(252 * 6.5) ≈ 40.47
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Hourly bars annualisation: 252 trading days × 6.5 trading hours
HOURLY_ANNUAL_FACTOR = np.sqrt(252 * 6.5)
DAILY_ANNUAL_FACTOR  = np.sqrt(252)


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
        Dict of all metrics
    """
    if not equity_curve:
        return _empty_metrics()

    eq_series = pd.Series(
        [e["equity"] for e in equity_curve],
        index=pd.to_datetime([e["timestamp"] for e in equity_curve]),
    )

    annual_factor = HOURLY_ANNUAL_FACTOR if timeframe != "1Day" else DAILY_ANNUAL_FACTOR

    # ── Returns ──────────────────────────────────────────────────────────────
    bar_returns = eq_series.pct_change().dropna()
    total_return_pct = (eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100

    # ── CAGR ─────────────────────────────────────────────────────────────────
    start = eq_series.index[0]
    end   = eq_series.index[-1]
    years = max((end - start).days / 365.25, 0.01)
    cagr_pct = ((eq_series.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100

    # ── Sharpe Ratio ─────────────────────────────────────────────────────────
    sharpe = 0.0
    if bar_returns.std() > 0:
        sharpe = (bar_returns.mean() / bar_returns.std()) * annual_factor

    # ── Sortino Ratio ─────────────────────────────────────────────────────────
    sortino = 0.0
    downside = bar_returns[bar_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (bar_returns.mean() / downside.std()) * annual_factor

    # ── Max Drawdown ─────────────────────────────────────────────────────────
    rolling_max = eq_series.cummax()
    drawdown    = (eq_series - rolling_max) / rolling_max * 100
    max_dd_pct  = drawdown.min()

    drawdown_curve = [
        {"timestamp": str(ts), "drawdown_pct": round(dd, 4)}
        for ts, dd in drawdown.items()
    ]

    # ── Trade-level stats ─────────────────────────────────────────────────────
    if not trade_log:
        return {
            "total_return_pct":  round(total_return_pct, 4),
            "cagr_pct":          round(cagr_pct, 4),
            "sharpe_ratio":      round(sharpe, 4),
            "sortino_ratio":     round(sortino, 4),
            "max_drawdown_pct":  round(max_dd_pct, 4),
            "win_rate_pct":      0,
            "profit_factor":     0,
            "total_trades":      0,
            "avg_holding_bars":  0,
            "avg_holding_hours": 0,
            "gross_profit":      0,
            "gross_loss":        0,
            "largest_win":       0,
            "largest_loss":      0,
            "exit_reason_breakdown": {},
            "drawdown_curve":    drawdown_curve,
        }

    pnls       = [t["pnl"] for t in trade_log]
    winners    = [p for p in pnls if p > 0]
    losers     = [p for p in pnls if p <= 0]
    bars_held  = [t["bars_held"] for t in trade_log]

    win_rate      = len(winners) / len(pnls) * 100
    gross_profit  = sum(winners) if winners else 0
    gross_loss    = abs(sum(losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_bars  = np.mean(bars_held) if bars_held else 0
    avg_hours = avg_bars  # for hourly timeframe, bars = hours

    exit_reasons = {}
    for t in trade_log:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # ── Monthly returns ───────────────────────────────────────────────────────
    monthly = (
        eq_series
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
        * 100
    )
    monthly_returns = {
        str(ts.strftime("%Y-%m")): round(ret, 4)
        for ts, ret in monthly.items()
    }

    return {
        "total_return_pct":       round(total_return_pct, 4),
        "cagr_pct":               round(cagr_pct, 4),
        "sharpe_ratio":           round(sharpe, 4),
        "sortino_ratio":          round(sortino, 4),
        "max_drawdown_pct":       round(max_dd_pct, 4),
        "win_rate_pct":           round(win_rate, 4),
        "profit_factor":          round(profit_factor, 4) if profit_factor != float("inf") else None,
        "total_trades":           len(trade_log),
        "avg_holding_bars":       round(avg_bars, 1),
        "avg_holding_hours":      round(avg_hours, 1),
        "gross_profit":           round(gross_profit, 2),
        "gross_loss":             round(gross_loss, 2),
        "largest_win":            round(max(winners), 2) if winners else 0,
        "largest_loss":           round(min(losers), 2) if losers else 0,
        "exit_reason_breakdown":  exit_reasons,
        "monthly_returns":        monthly_returns,
        "drawdown_curve":         drawdown_curve,
    }


def _empty_metrics() -> dict:
    return {
        "total_return_pct": 0, "cagr_pct": 0, "sharpe_ratio": 0,
        "sortino_ratio": 0, "max_drawdown_pct": 0, "win_rate_pct": 0,
        "profit_factor": 0, "total_trades": 0, "avg_holding_bars": 0,
        "avg_holding_hours": 0, "gross_profit": 0, "gross_loss": 0,
        "largest_win": 0, "largest_loss": 0,
        "exit_reason_breakdown": {}, "monthly_returns": {}, "drawdown_curve": [],
    }
