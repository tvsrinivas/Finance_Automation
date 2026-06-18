"""
live_trader.py
==============
Liquidity Candle + Hammer / Bullish Engulfing — Paper Trading via Alpaca
Runs all symbols in one pass. Called by GitHub Actions on a schedule.

Modes (set via --mode flag):
  open    : Run at 9:35 ET. Fetch ATR, check liquidity candle, cache state.
  scan    : Run every 5 min 9:40–11:00 ET. Look for signals, place orders.
  exit    : Run at 11:05 ET. Close all open positions for these symbols.

State is persisted in state/<SYMBOL>.json so each GitHub Actions run picks up
where the previous one left off (committed back to repo, or stored as artifact).

Setup:
    pip install alpaca-py pandas numpy python-dotenv

    GitHub Secrets required:
        ALPACA_API_KEY
        ALPACA_SECRET_KEY

Usage (local test):
    python live_trader.py --mode open
    python live_trader.py --mode scan
    python live_trader.py --mode exit
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, date, timezone
from datetime import time as dtime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical        import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests          import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe         import TimeFrame
from alpaca.data.enums             import DataFeed, OptionsFeed
from alpaca.trading.client         import TradingClient
from alpaca.trading.requests       import (MarketOrderRequest,
                                           LimitOrderRequest,
                                           GetOptionContractsRequest,
                                           GetOrdersRequest,
                                           ClosePositionRequest)
from alpaca.trading.enums          import (OrderSide, TimeInForce,
                                           ContractType, QueryOrderStatus)
from alpaca.common.exceptions      import APIError

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOLS           = ["AAPL", "SPY", "WMT", "ORCL", "TSLA"]
ATR_PERIOD        = 14
LIQUIDITY_PCT     = 0.25
HAMMER_BODY_PCT   = 0.35
HAMMER_WICK_RATIO = 2.0
EXIT_TIME         = dtime(11, 5)
OTM_STEPS         = 0          # 0 = ATM, 1 = 1-strike OTM, etc.
MIN_DTE           = 1          # minimum days to expiry
STATE_DIR         = Path("state")

# ── Position sizing ───────────────────────────────────────────────────────────
# RISK_PER_TRADE_PCT : % of account equity to risk on each signal
#                      e.g. 2.0 means risk at most 2% of account per trade
# MAX_SIMULTANEOUS   : hard cap on how many open option positions at once
#                      across ALL symbols combined
# MIN_CONTRACTS      : never go below this many contracts (floor)
# MAX_CONTRACTS      : never exceed this many contracts (ceiling)
RISK_PER_TRADE_PCT = 2.0    # % of account equity per trade
MAX_SIMULTANEOUS   = 2       # max open positions across all symbols at once
MIN_CONTRACTS      = 1       # floor
MAX_CONTRACTS      = 5       # ceiling
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
# ═══════════════════════════════════════════════════════════════════════════════

# ── Clients ───────────────────────────────────────────────────────────────────
stock_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
option_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
trade_client  = TradingClient(API_KEY, SECRET_KEY, paper=True)

STATE_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE  (persisted as JSON per symbol)
# ═══════════════════════════════════════════════════════════════════════════════
def _state_path(symbol: str) -> Path:
    return STATE_DIR / f"{symbol}.json"


def load_state(symbol: str) -> dict:
    p = _state_path(symbol)
    if p.exists():
        with open(p) as f:
            s = json.load(f)
        # Only use state from today
        if s.get("date") == str(date.today()):
            return s
    return {"date": str(date.today()), "symbol": symbol}


def save_state(symbol: str, state: dict):
    state["date"] = str(date.today())
    with open(_state_path(symbol), "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


def _fetch_daily(symbol: str) -> pd.DataFrame:
    end   = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=ATR_PERIOD * 3 + 30)
    req   = StockBarsRequest(symbol_or_symbols=symbol,
                             timeframe=TimeFrame.Day,
                             feed=DataFeed.IEX,
                             start=start, end=end)
    df = stock_client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def _fetch_today_intraday(symbol: str) -> pd.DataFrame:
    today = date.today()
    start = datetime.combine(today, dtime(9, 30))
    end   = datetime.combine(today, dtime(11, 10))
    req   = StockBarsRequest(symbol_or_symbols=symbol,
                             timeframe=TimeFrame.Minute,
                             feed=DataFeed.IEX,
                             start=start, end=end)
    df = stock_client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df.sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
#  ATR
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_atr(df: pd.DataFrame) -> float:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
    return float(atr.dropna().iloc[-1])


# ═══════════════════════════════════════════════════════════════════════════════
#  PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def _is_hammer(o, h, l, c) -> bool:
    total = h - l
    if total == 0 or abs(c - o) == 0:
        return False
    body       = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (body / total      <= HAMMER_BODY_PCT   and
            lower_wick / body >= HAMMER_WICK_RATIO and
            upper_wick        <= body)


def _is_bullish_engulfing(po, pc, co, cc) -> bool:
    return pc < po and cc > co and cc >= po and co <= pc


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION CONTRACT LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
def _strike_increment(price: float) -> float:
    if price < 25:   return 0.50
    if price < 200:  return 1.00
    return 5.00


def _atm_strike(price: float, increment: float) -> float:
    return round(round(price / increment) * increment, 2)


def _next_weekly_expiry(signal_date: date, min_dte: int) -> date:
    candidate  = signal_date + timedelta(days=min_dte)
    days_ahead = (4 - candidate.weekday()) % 7
    return candidate + timedelta(days=days_ahead)


def _lookup_contract(symbol: str, stock_price: float) -> tuple[str | None, float, date]:
    today         = date.today()
    increment     = _strike_increment(stock_price)
    target_strike = round(_atm_strike(stock_price, increment) + OTM_STEPS * increment, 2)
    earliest_exp  = today + timedelta(days=MIN_DTE)
    exp_window    = today + timedelta(days=60)
    lo_strike     = round(stock_price * 0.85, 2)
    hi_strike     = round(stock_price * 1.15, 2)

    try:
        resp = trade_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[symbol],
                type=ContractType.CALL,
                expiration_date_gte=str(earliest_exp),
                expiration_date_lte=str(exp_window),
                strike_price_gte=str(lo_strike),
                strike_price_lte=str(hi_strike),
                limit=200,
            )
        )
        contracts = resp.option_contracts
    except Exception as e:
        log.error(f"  Contract lookup failed for {symbol}: {e}")
        return None, target_strike, _next_weekly_expiry(today, MIN_DTE)

    if not contracts:
        log.warning(f"  No active contracts for {symbol} near {target_strike}")
        return None, target_strike, _next_weekly_expiry(today, MIN_DTE)

    expiries = sorted({c.expiration_date for c in contracts})
    nearest  = expiries[0]
    cands    = [c for c in contracts if c.expiration_date == nearest]
    best     = min(cands, key=lambda c: abs(float(c.strike_price) - target_strike))
    return best.symbol, float(best.strike_price), best.expiration_date


# ═══════════════════════════════════════════════════════════════════════════════
#  ORDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _count_open_positions() -> int:
    """Count how many of today's symbols currently have an open option position."""
    count = 0
    for sym in SYMBOLS:
        s = load_state(sym)
        if s.get("in_trade"):
            count += 1
    return count


def _calc_contracts(ask_price: float) -> int:
    """
    Size position based on account equity and RISK_PER_TRADE_PCT.

    Logic:
      risk_dollars  = account_equity * RISK_PER_TRADE_PCT / 100
      contracts     = floor(risk_dollars / (ask_price * 100))
      Then clamp to [MIN_CONTRACTS, MAX_CONTRACTS].

    1 contract = 100 shares, so cost = ask * 100.
    We treat the full premium as the risk (no stop on options here —
    the 11:05 hard exit handles the downside).
    """
    try:
        account  = trade_client.get_account()
        equity   = float(account.equity)
    except Exception as e:
        log.warning(f"  Could not fetch account equity: {e} — defaulting to {MIN_CONTRACTS} contract")
        return MIN_CONTRACTS

    risk_dollars = equity * RISK_PER_TRADE_PCT / 100.0
    cost_per_contract = ask_price * 100.0

    if cost_per_contract <= 0:
        return MIN_CONTRACTS

    raw = int(risk_dollars / cost_per_contract)
    qty = max(MIN_CONTRACTS, min(MAX_CONTRACTS, raw))

    log.info(f"  Sizing: equity=${equity:,.0f}  risk={RISK_PER_TRADE_PCT}%"
             f"  risk$=${risk_dollars:.0f}  ask=${ask_price:.2f}/contract"
             f"  raw={raw}  final_qty={qty}  "
             f"[clamped to {MIN_CONTRACTS}–{MAX_CONTRACTS}]")
    return qty


def _place_option_order(occ_symbol: str, entry_price: float, qty: int) -> str | None:
    """Place a limit buy order for `qty` call contracts. Returns order ID."""
    try:
        order = trade_client.submit_order(
            LimitOrderRequest(
                symbol=occ_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(entry_price * 1.02, 2),  # 2% above ask for fill
            )
        )
        log.info(f"  Order placed: {order.id}  {occ_symbol}  "
                 f"qty={qty}  limit={entry_price:.2f}  "
                 f"total_cost=${entry_price * qty * 100:,.0f}")
        return str(order.id)
    except APIError as e:
        log.error(f"  Order failed for {occ_symbol}: {e}")
        return None


def _close_position(symbol: str):
    """Close any open position for this symbol."""
    try:
        trade_client.close_position(symbol)
        log.info(f"  Closed position: {symbol}")
    except APIError as e:
        # 404 = no position, which is fine
        if "position does not exist" not in str(e).lower():
            log.warning(f"  Close {symbol}: {e}")


def _cancel_open_orders(occ_symbol: str):
    """Cancel any unfilled orders for this symbol."""
    try:
        orders = trade_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[occ_symbol])
        )
        for o in orders:
            trade_client.cancel_order_by_id(str(o.id))
            log.info(f"  Cancelled order {o.id}")
    except Exception as e:
        log.warning(f"  Cancel orders for {occ_symbol}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE: OPEN  (run at 9:35 ET)
# ═══════════════════════════════════════════════════════════════════════════════
def run_open():
    """Fetch ATR, check first 5-min candle for liquidity, cache state."""
    log.info("═" * 56)
    log.info(f"MODE: OPEN  |  {date.today()}  |  {', '.join(SYMBOLS)}")
    log.info("═" * 56)

    for symbol in SYMBOLS:
        state = {"date": str(date.today()), "symbol": symbol}
        try:
            # Step 1 — ATR
            daily  = _fetch_daily(symbol)
            prev   = daily[daily.index.date < date.today()]
            if len(prev) < ATR_PERIOD:
                log.warning(f"  {symbol}: insufficient daily bars for ATR")
                state["skip"] = "insufficient ATR data"
                save_state(symbol, state)
                continue
            atr_val = _compute_atr(prev)
            state["atr"] = round(atr_val, 4)
            log.info(f"  {symbol}  ATR={atr_val:.4f}")

            # Step 2 — First 5-min candle
            intraday = _fetch_today_intraday(symbol)
            if intraday.empty:
                log.warning(f"  {symbol}: no intraday data yet")
                state["skip"] = "no intraday data"
                save_state(symbol, state)
                continue

            day_5m = intraday.resample("5min").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()

            if day_5m.empty:
                state["skip"] = "no 5-min bars"
                save_state(symbol, state)
                continue

            first      = day_5m.iloc[0]
            o1, c1     = float(first["open"]), float(first["close"])
            body1      = abs(o1 - c1)
            threshold  = (LIQUIDITY_PCT / 100.0) * atr_val

            if body1 < threshold:
                log.info(f"  {symbol}: NOT a liquidity candle "
                         f"(body={body1:.4f} < threshold={threshold:.4f}) — skipping today")
                state["skip"]    = "not a liquidity candle"
                state["body1"]   = round(body1, 4)
                state["threshold"] = round(threshold, 4)
                save_state(symbol, state)
                continue

            state["a_high"]    = round(max(o1, c1), 4)
            state["b_low"]     = round(min(o1, c1), 4)
            state["liquidity"] = True
            state["in_trade"]  = False
            log.info(f"  {symbol}: Liquidity candle ✓  "
                     f"a_high={state['a_high']}  b_low={state['b_low']}")

        except Exception as e:
            log.error(f"  {symbol} OPEN error: {e}")
            state["skip"] = f"error: {e}"

        save_state(symbol, state)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE: SCAN  (run every 5 min 9:40–11:00 ET)
# ═══════════════════════════════════════════════════════════════════════════════
def run_scan():
    """Check each symbol for Hammer / Bullish Engulfing signal below b_low."""
    now_et = _now_et()
    log.info(f"MODE: SCAN  |  ET={now_et.strftime('%H:%M')}  |  {date.today()}")

    for symbol in SYMBOLS:
        state = load_state(symbol)

        if state.get("skip"):
            log.info(f"  {symbol}: skipped ({state['skip']})")
            continue
        if not state.get("liquidity"):
            log.info(f"  {symbol}: no liquidity candle today")
            continue
        if state.get("in_trade"):
            log.info(f"  {symbol}: already in trade ({state.get('occ_symbol')})")
            continue

        a_high = state["a_high"]
        b_low  = state["b_low"]

        try:
            intraday = _fetch_today_intraday(symbol)
            if intraday.empty:
                continue

            day_5m = intraday.resample("5min").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()

            # Skip first candle (liquidity candle), scan the rest
            rest   = day_5m.iloc[1:]
            prev_o = prev_c = None

            for ts, row in rest.iterrows():
                t = ts.time() if hasattr(ts, "time") else ts
                if t >= EXIT_TIME:
                    break

                o = float(row["open"]); h = float(row["high"])
                l = float(row["low"]);  c = float(row["close"])

                if c >= b_low:
                    prev_o, prev_c = o, c
                    continue

                signal = None
                if _is_hammer(o, h, l, c):
                    signal = "Hammer"
                elif prev_o is not None and _is_bullish_engulfing(prev_o, prev_c, o, c):
                    signal = "Bullish Engulfing"

                if signal:
                    stock_price = h  # entry = high of signal candle
                    log.info(f"  {symbol}: SIGNAL {signal} @ {t}  "
                             f"entry={stock_price:.2f}  sl={l:.2f}  target={a_high:.2f}")

                    # ── Simultaneous position cap ──────────────────────────
                    open_now = _count_open_positions()
                    if open_now >= MAX_SIMULTANEOUS:
                        log.info(f"  {symbol}: MAX_SIMULTANEOUS={MAX_SIMULTANEOUS} reached "
                                 f"({open_now} open) — skipping signal")
                        break

                    # Look up option contract
                    occ_symbol, strike, expiry = _lookup_contract(symbol, stock_price)
                    if occ_symbol is None:
                        log.warning(f"  {symbol}: no contract found — skipping")
                        break

                    log.info(f"  {symbol}: contract={occ_symbol}  "
                             f"strike={strike}  expiry={expiry}")

                    # ── Get current option ask price ───────────────────────
                    try:
                        from alpaca.data.requests import OptionLatestQuoteRequest
                        q_req  = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
                        quotes = option_client.get_option_latest_quote(q_req)
                        ask    = float(quotes[occ_symbol].ask_price)
                        if ask <= 0:
                            ask = float(quotes[occ_symbol].bid_price) * 1.05
                    except Exception as e:
                        log.warning(f"  {symbol}: could not get option quote: {e} — using fallback")
                        ask = round(max(0.50, stock_price * 0.01), 2)

                    # ── Calculate position size ────────────────────────────
                    qty = _calc_contracts(ask)

                    # ── Place order ────────────────────────────────────────
                    order_id = _place_option_order(occ_symbol, ask, qty)
                    if order_id:
                        state["in_trade"]    = True
                        state["occ_symbol"]  = occ_symbol
                        state["order_id"]    = order_id
                        state["qty"]         = qty
                        state["signal"]      = signal
                        state["signal_time"] = str(t)
                        state["stock_entry"] = round(stock_price, 4)
                        state["stock_sl"]    = round(l, 4)
                        state["opt_entry"]   = round(ask, 4)
                        state["strike"]      = strike
                        state["expiry"]      = str(expiry)
                    save_state(symbol, state)
                    break  # one signal per symbol per day

                prev_o, prev_c = o, c

        except Exception as e:
            log.error(f"  {symbol} SCAN error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE: EXIT  (run at 11:05 ET)
# ═══════════════════════════════════════════════════════════════════════════════
def run_exit():
    """Close all open option positions for today's symbols."""
    log.info(f"MODE: EXIT  |  {date.today()}")

    for symbol in SYMBOLS:
        state = load_state(symbol)

        if not state.get("in_trade"):
            log.info(f"  {symbol}: no open trade")
            continue

        occ_symbol = state.get("occ_symbol")
        if not occ_symbol:
            continue

        log.info(f"  {symbol}: closing {occ_symbol}")
        _cancel_open_orders(occ_symbol)
        _close_position(occ_symbol)

        # Fetch exit price for logging
        try:
            from alpaca.data.requests import OptionLatestTradeRequest
            t_req  = OptionLatestTradeRequest(symbol_or_symbols=occ_symbol)
            trades = option_client.get_option_latest_trade(t_req)
            exit_px = float(trades[occ_symbol].price)
        except Exception:
            exit_px = None

        entry_px = state.get("opt_entry")
        if entry_px and exit_px:
            pnl = round((exit_px - entry_px) * 100, 2)
            outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "SCRATCH"
            log.info(f"  {symbol}: entry={entry_px}  exit={exit_px}  "
                     f"PnL=${pnl:+.2f}  [{outcome}]")
        else:
            log.info(f"  {symbol}: position closed (could not fetch exit price)")

        state["in_trade"]   = False
        state["exit_time"]  = str(EXIT_TIME)
        state["opt_exit"]   = exit_px
        save_state(symbol, state)

    log.info("EXIT complete.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Liquidity Candle Live Trader")
    parser.add_argument("--mode", required=True, choices=["open", "scan", "exit"],
                        help="open | scan | exit")
    args = parser.parse_args()

    if   args.mode == "open": run_open()
    elif args.mode == "scan":  run_scan()
    elif args.mode == "exit":  run_exit()


if __name__ == "__main__":
    main()
