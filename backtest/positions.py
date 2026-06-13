"""
Position Manager — tracks position state, applies risk rules, builds trade log.
Enforces stop loss > take profit > signal exit priority.
Executes all orders at next bar open (no look-ahead).

position_sizing modes:
  "fixed"       — always uses initial_capital × position_size_pct
                  Gives honest picture of strategy performance.
  "compounding" — uses current capital × position_size_pct
                  Shows theoretical max but distorts results.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PositionManager:

    def __init__(
        self,
        initial_capital: float,
        position_size_pct: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
        max_open_positions: int = 1,
        position_sizing: str = "fixed",
    ):
        self.initial_capital     = initial_capital
        self.capital             = initial_capital
        self.position_size_pct   = position_size_pct / 100
        self.stop_loss_pct       = stop_loss_pct / 100   if stop_loss_pct   else None
        self.take_profit_pct     = take_profit_pct / 100 if take_profit_pct else None
        self.commission_pct      = commission_pct / 100
        self.slippage_pct        = slippage_pct / 100
        self.position_sizing     = position_sizing  # "fixed" | "compounding"

        # Current position state
        self.in_position         = False
        self.entry_price         = None
        self.entry_time          = None
        self.entry_bar           = None
        self.shares              = 0
        self.stop_loss_price     = None
        self.take_profit_price   = None
        self.entry_capital       = None

        # Pending signals (execute next bar)
        self.pending_entry       = False
        self.pending_exit        = False

        # Output
        self.trade_log           = []
        self.equity_curve        = []

    def _apply_slippage(self, price: float, direction: str) -> float:
        if direction == "buy":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _apply_commission(self, trade_value: float) -> float:
        return trade_value * self.commission_pct

    def _enter_position(self, bar: pd.Series, bar_idx: int):
        exec_price = self._apply_slippage(bar["open"], "buy")

        # ── Position sizing mode ──────────────────────────────────────────────
        # fixed:       always use initial capital as the base
        #              → each trade risks the same dollar amount
        #              → honest measure of strategy edge
        # compounding: use current capital as the base
        #              → position sizes grow with wins, shrink with losses
        #              → distorts results when wins cluster early
        if self.position_sizing == "fixed":
            base = self.initial_capital
        else:
            base = self.capital

        position_value = base * self.position_size_pct
        commission     = self._apply_commission(position_value)
        shares         = position_value / exec_price

        self.capital       -= position_value  # deduct position cost
        self.capital       -= commission      # deduct commission
        self.in_position    = True
        self.entry_price    = exec_price
        self.entry_time     = bar.name
        self.entry_bar      = bar_idx
        self.shares         = shares
        self.entry_capital  = self.capital

        if self.stop_loss_pct:
            self.stop_loss_price   = exec_price * (1 - self.stop_loss_pct)
        if self.take_profit_pct:
            self.take_profit_price = exec_price * (1 + self.take_profit_pct)

        logger.debug(f"ENTRY  {bar.name} price={exec_price:.2f} shares={shares:.2f} "
                     f"sl={self.stop_loss_price} tp={self.take_profit_price} "
                     f"sizing={self.position_sizing} base=${base:.2f}")

    def _exit_position(
        self,
        bar: pd.Series,
        bar_idx: int,
        exit_price: float,
        exit_reason: str,
    ):
        trade_value = self.shares * exit_price
        commission  = self._apply_commission(trade_value)
        pnl         = (exit_price - self.entry_price) * self.shares - commission
        pnl_pct     = (exit_price / self.entry_price - 1) * 100
        bars_held   = bar_idx - self.entry_bar

        self.capital += trade_value - commission

        trade = {
            "entry_time":    str(self.entry_time),
            "exit_time":     str(bar.name),
            "entry_price":   round(self.entry_price, 4),
            "exit_price":    round(exit_price, 4),
            "shares":        round(self.shares, 4),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round(pnl_pct, 4),
            "exit_reason":   exit_reason,
            "bars_held":     bars_held,
            "entry_capital": round(self.entry_capital, 2),
        }
        self.trade_log.append(trade)

        logger.debug(f"EXIT   {bar.name} price={exit_price:.2f} reason={exit_reason} pnl={pnl:.2f}")

        # Reset state
        self.in_position       = False
        self.entry_price       = None
        self.entry_time        = None
        self.entry_bar         = None
        self.shares            = 0
        self.stop_loss_price   = None
        self.take_profit_price = None
        self.entry_capital     = None
        self.pending_exit      = False

    def process_bar(self, bar: pd.Series, bar_idx: int, signals: list[str]):
        """
        Process one bar.
        signals = list of signals that fired on the PREVIOUS bar.
        Priority: pending_entry → check SL/TP → check pending_exit
        """
        # Execute pending entry from previous bar's signal
        if self.pending_entry and not self.in_position:
            self._enter_position(bar, bar_idx)
            self.pending_entry = False

        # While in position: check stop loss and take profit (intrabar)
        if self.in_position:
            exited = False

            # Stop loss: bar low touched SL price
            if self.stop_loss_price and bar["low"] <= self.stop_loss_price:
                sl_exec = self._apply_slippage(self.stop_loss_price, "sell")
                self._exit_position(bar, bar_idx, sl_exec, "stop_loss")
                exited = True

            # Take profit: bar high touched TP price
            if not exited and self.take_profit_price and bar["high"] >= self.take_profit_price:
                tp_exec = self._apply_slippage(self.take_profit_price, "sell")
                self._exit_position(bar, bar_idx, tp_exec, "take_profit")
                exited = True

            # Signal exit: execute at this bar's open if pending from last bar
            if not exited and self.pending_exit:
                exec_price = self._apply_slippage(bar["open"], "sell")
                self._exit_position(bar, bar_idx, exec_price, "signal")

        # Queue signals for next bar execution
        for sig in signals:
            if sig == "entry" and not self.in_position:
                self.pending_entry = True
            elif sig == "exit" and self.in_position:
                self.pending_exit = True

        # Record equity (mark-to-market)
        if self.in_position:
            position_value = self.shares * bar["close"]
            equity = self.capital + position_value
        else:
            equity = self.capital

        self.equity_curve.append({
            "timestamp": str(bar.name),
            "equity":    round(equity, 2),
        })

    def get_summary(self) -> dict:
        final_equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.capital
        return {
            "initial_capital": self.initial_capital,
            "final_capital":   round(final_equity, 2),
            "trade_log":       self.trade_log,
            "equity_curve":    self.equity_curve,
        }
