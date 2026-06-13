"""
Backtest Engine — orchestrates the full backtest pipeline.
Accepts validated strategy JSON + backtest params, returns full results.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
import os

from .data       import fetch_price_data
from .indicators import compute_all_indicators
from .signals    import generate_signals, evaluate_group
from .positions  import PositionManager
from .metrics    import calculate_metrics
from dotenv import load_dotenv

load_dotenv()  # ← this line loads the .env file

logger = logging.getLogger(__name__)


def run_backtest(
    strategy: dict,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    timeframe: str = "1Hour",
    data_source: Optional[str] = None,
    position_sizing: str = "fixed",
) -> dict:
    """
    Run a full backtest for a validated strategy against a symbol.

    Args:
        strategy:        Validated strategy JSON (from Strategy Builder)
        symbol:          Ticker e.g. "AAPL"
        start_date:      "YYYY-MM-DD"
        end_date:        "YYYY-MM-DD"
        initial_capital: Starting capital in USD
        commission_pct:  Commission per trade side (%)
        slippage_pct:    Execution slippage (%)
        timeframe:       "1Hour" | "1Day" | "15Min"
        data_source:     "alpaca" | "yfinance" | None (auto)

    Returns:
        Full backtest result dict
    """
    logger.info(f"Starting backtest: {symbol} {start_date}→{end_date} capital=${initial_capital}")

    # ── Step 1: Fetch price data ──────────────────────────────────────────────
    try:
        df = fetch_price_data(symbol, start_date, end_date, timeframe, data_source)
    except Exception as e:
        return _error_result(f"Data fetch failed: {e}", symbol, start_date, end_date)

    if len(df) < 50:
        return _error_result(
            f"Insufficient data: only {len(df)} bars returned. Try a longer date range.",
            symbol, start_date, end_date
        )

    logger.info(f"Fetched {len(df)} bars for {symbol}")

    # ── Step 2: Compute all indicators ───────────────────────────────────────
    indicators_spec = strategy.get("indicators", {})
    try:
        computed = compute_all_indicators(df, indicators_spec)
    except Exception as e:
        return _error_result(f"Indicator computation failed: {e}", symbol, start_date, end_date)

    # Determine warmup: max lookback period across all indicators
    warmup_bars = _calculate_warmup(indicators_spec)
    logger.info(f"Warmup period: {warmup_bars} bars")

    # ── Step 3: Extract strategy rules ───────────────────────────────────────
    conditions  = strategy.get("conditions", {})
    entry_rules = strategy.get("entry_rules", {})
    exit_rules  = strategy.get("exit_rules", {})
    risk_rules  = strategy.get("risk_rules", {})

    stop_loss_pct    = risk_rules.get("stop_loss", {}).get("value") if risk_rules.get("stop_loss") else None
    take_profit_pct  = risk_rules.get("take_profit", {}).get("value") if risk_rules.get("take_profit") else None
    position_size_pct = risk_rules.get("position_size", {}).get("value", 10)

    # ── Step 4: Run bar-by-bar simulation ────────────────────────────────────
    pm = PositionManager(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        position_sizing=position_sizing,
    )

    # Pre-generate signals for all bars
    bar_signals: dict[int, list[str]] = {}

    for i in range(warmup_bars, len(df)):
        entry_sig = evaluate_group(entry_rules, conditions, computed, i)
        exit_sig  = evaluate_group(exit_rules,  conditions, computed, i)
        sigs = []
        if entry_sig: sigs.append("entry")
        if exit_sig:  sigs.append("exit")
        if sigs:
            bar_signals[i] = sigs

    # Walk bars — signals from bar i execute at open of bar i+1
    for i, (idx, row) in enumerate(df.iterrows()):
        # Signals queued from previous bar
        prev_signals = bar_signals.get(i - 1, [])
        pm.process_bar(row, i, prev_signals)

    # ── Step 5: Calculate metrics ─────────────────────────────────────────────
    summary = pm.get_summary()
    metrics = calculate_metrics(
        equity_curve=summary["equity_curve"],
        trade_log=summary["trade_log"],
        initial_capital=initial_capital,
        timeframe=timeframe,
    )

    # ── Step 6: Equity curve (sampled for large datasets) ────────────────────
    equity_curve = summary["equity_curve"]
    if len(equity_curve) > 500:
        # Downsample to ~500 points for the chart
        step = len(equity_curve) // 500
        equity_curve = equity_curve[::step]

    # ── Step 7: Build result ──────────────────────────────────────────────────
    result = {
        "success":         True,
        "symbol":          symbol,
        "start_date":      start_date,
        "end_date":        end_date,
        "timeframe":       timeframe,
        "initial_capital": initial_capital,
        "final_capital":   summary["final_capital"],
        "strategy_name":   strategy.get("strategy_name", "Unnamed"),
        "total_bars":      len(df),
        "warmup_bars":     warmup_bars,
        "metrics":         metrics,
        "equity_curve":    equity_curve,
        "drawdown_curve":  metrics.pop("drawdown_curve", []),
        "monthly_returns": metrics.pop("monthly_returns", {}),
        "trade_log":       summary["trade_log"],
        "exit_reason_breakdown": metrics.pop("exit_reason_breakdown", {}),
        "data_source": "alpaca" if os.getenv("ALPACA_API_KEY") else "yfinance",
    }

    logger.info(
        f"Backtest complete: {len(summary['trade_log'])} trades, "
        f"return={metrics['total_return_pct']:.2f}%, "
        f"sharpe={metrics['sharpe_ratio']:.2f}"
    )

    return result


def _calculate_warmup(indicators_spec: dict) -> int:
    """Calculate minimum bars needed before signals are valid."""
    max_period = 0
    for ind in indicators_spec.values():
        params = ind.get("params", {})
        # For MACD, warmup is fast + slow + signal
        if ind.get("type") == "MACD":
            p = params.get("fast", 12) + params.get("slow", 26) + params.get("signal", 9)
        else:
            p = params.get("period", 1)
            if ind.get("type") in ("WEEK52_HIGH", "WEEK52_LOW"):
                p = 1638
        max_period = max(max_period, p)
    return min(max_period + 10, 200)  # cap at 200 for 52w indicators


def _error_result(message: str, symbol: str, start: str, end: str) -> dict:
    return {
        "success":    False,
        "error":      message,
        "symbol":     symbol,
        "start_date": start,
        "end_date":   end,
    }
