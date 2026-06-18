"""
options_backtest.py  —  STANDALONE
====================================
Options backtest for the Liquidity Candle + Hammer / Bullish Engulfing strategy.
Buys a CALL option when the stock signal fires, exits at the same stock exit time.

Strike selection  : ATM (0) or OTM by N strikes via --otm  (default: 0 = ATM)
Expiry selection  : nearest weekly expiry ≥ --min-dte days away (default: 1)

Setup:
    pip install alpaca-py pandas numpy python-dotenv

    .env file:
        ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
        ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Usage:
    python options_backtest.py --symbol AAPL
    python options_backtest.py --symbol MSFT --days 60 --otm 0          # ATM call
    python options_backtest.py --symbol MSFT --days 60 --otm 1          # 1-strike OTM
    python options_backtest.py --symbol MSFT --days 60 --otm 2          # 2-strikes OTM
    python options_backtest.py --symbol TSLA --days 90 --otm 1 --min-dte 2 --save

Notes:
    - Option PnL is per CONTRACT (1 contract = 100 shares).
    - Alpaca historical options data available from Feb 2024 onward.
    - Uses OptionsFeed.INDICATIVE (free tier, no OPRA subscription needed).
    - Contract symbol is looked up via Alpaca's contracts API (not constructed manually).
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, date, timezone
from datetime import time as dtime

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical        import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests          import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe         import TimeFrame
from alpaca.data.enums             import DataFeed, OptionsFeed
from alpaca.trading.client         import TradingClient
from alpaca.trading.requests       import GetOptionContractsRequest
from alpaca.trading.enums          import ContractType

# ── Load .env before reading env vars ─────────────────────────────────────────
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
ATR_PERIOD        = 14
LIQUIDITY_PCT     = 0.25
HAMMER_BODY_PCT   = 0.35
HAMMER_WICK_RATIO = 2.0
EXIT_TIME         = dtime(11, 5)

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
# ═══════════════════════════════════════════════════════════════════════════════

# ── Alpaca clients ─────────────────────────────────────────────────────────────
stock_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
option_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
trade_client  = TradingClient(API_KEY, SECRET_KEY, paper=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _fetch_daily(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(symbol_or_symbols=symbol,
                           timeframe=TimeFrame.Day,
                           feed=DataFeed.IEX,
                           start=start, end=end)
    df  = stock_client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def _fetch_intraday(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(symbol_or_symbols=symbol,
                           timeframe=TimeFrame.Minute,
                           feed=DataFeed.IEX,
                           start=start, end=end)
    df  = stock_client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df.sort_index()


def _atr(df: pd.DataFrame) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  PATTERN HELPERS
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
#  OPTION CONTRACT LOOKUP  (via Alpaca's own contracts API)
# ═══════════════════════════════════════════════════════════════════════════════
def _strike_increment(price: float) -> float:
    """Standard strike increment by underlying price."""
    if price < 25:
        return 0.50
    elif price < 200:
        return 1.00
    else:
        return 5.00


def _atm_strike(price: float, increment: float) -> float:
    """Round price to nearest standard strike."""
    return round(round(price / increment) * increment, 2)


def _next_weekly_expiry(signal_date: date, min_dte: int) -> date:
    """Nearest Friday on or after signal_date + min_dte."""
    candidate  = signal_date + timedelta(days=min_dte)
    days_ahead = (4 - candidate.weekday()) % 7
    return candidate + timedelta(days=days_ahead)


def _lookup_contract(symbol: str, signal_date: date,
                     stock_price: float, otm_steps: int,
                     min_dte: int) -> tuple[str | None, float, date]:
    """
    Look up the best matching CALL contract from Alpaca.

    Strategy:
      1. Query Alpaca with a 60-day expiry window and wide strike range.
         Alpaca only returns ACTIVE contracts — expired ones are purged.
      2. From all returned contracts, keep those whose expiry is at least
         min_dte days after signal_date, then pick:
           - Nearest expiry (smallest DTE >= min_dte)
           - Within that expiry, closest strike to target_strike
      3. If no active contract is found (all expired), build the OCC symbol
         manually so the bar-fetch can still attempt to pull historical data.

    This avoids the "tight Friday window" bug where the target expiry was
    correct but no contracts were listed until a later date on Alpaca.
    """
    increment     = _strike_increment(stock_price)
    target_strike = round(_atm_strike(stock_price, increment) + otm_steps * increment, 2)
    earliest_exp  = signal_date + timedelta(days=min_dte)

    # Wide strike + long expiry window — let Alpaca return everything active
    lo_strike  = round(stock_price * 0.85, 2)
    hi_strike  = round(stock_price * 1.15, 2)
    exp_window = signal_date + timedelta(days=60)

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
        print(f"           -> Contract lookup API error: {e}")
        contracts = []

    if contracts:
        # Step 1: find the nearest available expiry >= min_dte
        expiries = sorted({c.expiration_date for c in contracts})
        nearest_exp = expiries[0]

        # Step 2: among contracts at that expiry, pick closest strike
        candidates = [c for c in contracts if c.expiration_date == nearest_exp]
        best = min(candidates,
                   key=lambda c: abs(float(c.strike_price) - target_strike))
        print(f"           -> Found: {best.symbol}  "
              f"strike={best.strike_price}  expiry={best.expiration_date}")
        return best.symbol, float(best.strike_price), best.expiration_date

    # No active contracts — all have expired (common for old signal dates).
    # Build compact OCC symbol manually; bar-fetch may still work for recent history.
    expiry     = _next_weekly_expiry(signal_date, min_dte)
    strike_int = int(round(target_strike * 1000))
    occ_symbol = f"{symbol}{expiry.strftime('%y%m%d')}C{strike_int:08d}"
    print(f"           -> No active contracts found (all expired). "
          f"Manual symbol: {occ_symbol}")
    return occ_symbol, target_strike, expiry


def _fetch_option_bars(occ_symbol: str, trade_date: date) -> pd.DataFrame:
    """Fetch 1-min option bars for the contract on trade_date, resample to 5-min."""
    start = datetime.combine(trade_date, dtime(9, 30))
    end   = datetime.combine(trade_date, dtime(16, 0))
    try:
        req = OptionBarsRequest(
            symbol_or_symbols=occ_symbol,
            timeframe=TimeFrame.Minute,
            feed=OptionsFeed.INDICATIVE,
            start=start,
            end=end,
        )
        bars = option_client.get_option_bars(req)
        df   = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            # xs may fail if symbol not present — use loc
            try:
                df = df.xs(occ_symbol, level="symbol")
            except KeyError:
                df = df.xs(df.index.get_level_values("symbol")[0], level="symbol")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        df5 = df.resample("5min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()
        return df5
    except Exception as e:
        print(f"           ↳ Bar fetch error [{occ_symbol}]: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def _simulate_option(opt_bars: pd.DataFrame,
                     signal_time: dtime,
                     stock_exit_time: str) -> dict:
    """
    Buy option at open of first bar ≥ signal_time.
    Sell at open of bar that matches stock exit time (or 11:05 at latest).
    PnL × 100 per contract.
    """
    # ── Find entry bar ────────────────────────────────────────────
    entry_row  = None
    entry_time = None
    for ts, row in opt_bars.iterrows():
        t = ts.time() if hasattr(ts, "time") else ts
        if t >= signal_time:
            entry_row  = row
            entry_time = t
            break

    # Reject entry if signal itself is at or after EXIT_TIME
    if entry_row is None or float(entry_row["open"]) == 0 or entry_time >= EXIT_TIME:
        return {"opt_entry": None, "opt_exit": None,
                "opt_entry_time": None, "opt_exit_time": None,
                "opt_pnl_per_contract": None, "opt_outcome": "NO DATA"}

    opt_entry = float(entry_row["open"])

    # ── Parse stock exit time ─────────────────────────────────────
    try:
        # stock_exit_time may be "HH:MM:SS" or "HH:MM:SS.ffffff"
        sex_t = dtime.fromisoformat(stock_exit_time[:8])
    except Exception:
        sex_t = EXIT_TIME

    force_exit_t = min(sex_t, EXIT_TIME)

    # ── Walk forward from entry to find exit ──────────────────────
    past_entry = False
    for ts, row in opt_bars.iterrows():
        t = ts.time() if hasattr(ts, "time") else ts

        if not past_entry:
            past_entry = (t >= entry_time)
            continue          # skip up to and including entry bar

        if t >= force_exit_t:
            opt_exit = float(row["open"])
            pnl      = round((opt_exit - opt_entry) * 100, 2)
            return {
                "opt_entry":            round(opt_entry, 4),
                "opt_exit":             round(opt_exit, 4),
                "opt_entry_time":       str(entry_time)[:8],
                "opt_exit_time":        str(t)[:8],
                "opt_pnl_per_contract": pnl,
                "opt_outcome":          "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "SCRATCH",
            }

    # ── Fallback: close at last available bar ─────────────────────
    last_ts, last_row = list(opt_bars.iterrows())[-1]
    opt_exit = float(last_row["close"])
    pnl      = round((opt_exit - opt_entry) * 100, 2)
    return {
        "opt_entry":            round(opt_entry, 4),
        "opt_exit":             round(opt_exit, 4),
        "opt_entry_time":       str(entry_time)[:8],
        "opt_exit_time":        str(last_ts.time())[:8],
        "opt_pnl_per_contract": pnl,
        "opt_outcome":          "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "SCRATCH",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE-DAY STOCK SIGNAL LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def _simulate_stock(remaining: pd.DataFrame, entry, sl, target) -> dict:
    """Walk-forward stock sim — only used to capture exit time for option exit."""
    for ts, row in remaining.iterrows():
        t     = ts.time() if hasattr(ts, "time") else ts
        t_str = str(t)
        if t >= EXIT_TIME:
            return {"exit_time": str(EXIT_TIME), "exit_reason": "Time Exit 11:05"}
        if float(row["low"])  <= sl:
            return {"exit_time": t_str, "exit_reason": "Stop Loss"}
        if float(row["high"]) >= target:
            return {"exit_time": t_str, "exit_reason": "Target Hit"}
    return {"exit_time": str(EXIT_TIME), "exit_reason": "Time Exit 11:05"}


def _process_day(day: date, daily_df: pd.DataFrame,
                 intraday_df: pd.DataFrame,
                 symbol: str, otm_steps: int, min_dte: int) -> dict | None:

    # Step 1 — ATR
    prev = daily_df[daily_df.index.date < day]
    if len(prev) < ATR_PERIOD:
        return None
    atr_val = float(_atr(prev).dropna().iloc[-1])

    # Resample to 5-min
    day_1m = intraday_df[intraday_df.index.date == day]
    if len(day_1m) < 3:
        return None
    day_5m = day_1m.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()
    if len(day_5m) < 2:
        return None

    # Step 2 — Liquidity candle
    first      = day_5m.iloc[0]
    o1, c1     = float(first["open"]), float(first["close"])
    body1      = abs(o1 - c1)
    threshold  = (LIQUIDITY_PCT / 100.0) * atr_val
    if body1 < threshold:
        return None

    a_high = max(o1, c1)
    b_low  = min(o1, c1)

    # Step 3 — Signal scan
    rest   = day_5m.iloc[1:]
    prev_o = prev_c = None

    for i, (ts, row) in enumerate(rest.iterrows()):
        t = ts.time() if hasattr(ts, "time") else ts
        # Skip any candle at or after EXIT_TIME — no new entries allowed
        if t >= EXIT_TIME:
            break

        o, h, l, c = (float(row["open"]),  float(row["high"]),
                      float(row["low"]),   float(row["close"]))

        if c >= b_low:
            prev_o, prev_c = o, c
            continue

        signal = None
        if _is_hammer(o, h, l, c):
            signal = "Hammer"
        elif prev_o is not None and _is_bullish_engulfing(prev_o, prev_c, o, c):
            signal = "Bullish Engulfing"

        if signal:
            stock_exit = _simulate_stock(rest.iloc[i + 1:], h, l, a_high)
            return {
                "date":              str(day),
                "signal":            signal,
                "signal_time":       str(t),
                "atr":               round(atr_val, 4),
                "a_high":            round(a_high, 4),
                "b_low":             round(b_low, 4),
                "stock_entry":       round(h, 4),
                "stock_sl":          round(l, 4),
                "stock_target":      round(a_high, 4),
                "stock_exit_time":   stock_exit.get("exit_time"),
                "stock_exit_reason": stock_exit.get("exit_reason"),
                # Private — used in main, stripped before final record
                "_signal_time_obj":  t,
                "_stock_price":      h,
            }

        prev_o, prev_c = o, c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════
def _print_summary(df: pd.DataFrame, symbol: str, days: int,
                   otm_steps: int, min_dte: int):
    total   = len(df)
    no_data = (df["opt_outcome"] == "NO DATA").sum()
    valid_n = total - no_data
    wins    = (df["opt_outcome"] == "WIN").sum()
    losses  = (df["opt_outcome"] == "LOSS").sum()
    scratch = (df["opt_outcome"] == "SCRATCH").sum()
    winrate = wins / valid_n * 100 if valid_n > 0 else 0

    valid     = df[df["opt_pnl_per_contract"].notna()]["opt_pnl_per_contract"]
    total_pnl = float(valid.sum())
    avg_win   = float(valid[valid > 0].mean()) if (valid > 0).any() else 0.0
    avg_loss  = float(valid[valid < 0].mean()) if (valid < 0).any() else 0.0
    rr        = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    strike_label = "ATM" if otm_steps == 0 else f"{otm_steps}-strike OTM"
    W = 130
    print(f"\n  {'═'*W}")
    print(f"  OPTIONS BACKTEST  |  {symbol}  |  CALL  |  {strike_label}  "
          f"|  Min DTE: {min_dte}  |  Last {days} days")
    print(f"  {'─'*W}")
    print(f"  {'Signals found':<30}: {total}  (No option data: {no_data})")
    print(f"  {'Wins / Losses / Scratch':<30}: {wins} / {losses} / {scratch}")
    print(f"  {'Win Rate':<30}: {winrate:.1f}%")
    print(f"  {'Avg Win  ($ per contract)':<30}: {avg_win:+.2f}")
    print(f"  {'Avg Loss ($ per contract)':<30}: {avg_loss:+.2f}")
    print(f"  {'Risk / Reward ratio':<30}: {rr:.2f}")
    print(f"  {'Total PnL ($ per contract)':<30}: {total_pnl:+.2f}")
    print(f"  {'─'*W}")

    print(f"\n  {'DATE':<12} {'SIGNAL':<20} {'ATR':>8} "
          f"{'STRIKE':>7} {'EXPIRY':<12} {'OCC SYMBOL':<26} "
          f"{'OPT BUY':>8} {'OPT SELL':>9} "
          f"{'IN':>8} {'OUT':>8} {'PNL/CTR':>10}  REASON")
    print(f"  {'─'*W}")

    for _, r in df.iterrows():
        is_no_data = r["opt_outcome"] == "NO DATA"
        pnl_val = r["opt_pnl_per_contract"]
        pnl_s   = f"{pnl_val:+.2f}" if (pnl_val is not None and str(pnl_val) != "nan"
                                         and pnl_val == pnl_val) else "    N/A"
        opt_buy = f"{r['opt_entry']:.4f}"  if (r["opt_entry"]  is not None
                                               and str(r["opt_entry"])  != "nan") else "   N/A"
        opt_sel = f"{r['opt_exit']:.4f}"   if (r["opt_exit"]   is not None
                                               and str(r["opt_exit"])   != "nan") else "   N/A"
        ent_t   = str(r.get("opt_entry_time", "") or "")[:8]
        ext_t   = str(r.get("opt_exit_time",  "") or "")[:8]
        # For NO DATA rows show why — distinguish "expired contract" from "no bar at signal time"
        if is_no_data:
            reason = "No option bar at signal time (thin market / signal too late)"
        else:
            reason = r.get("stock_exit_reason", "")
        expiry  = str(r.get("expiry", ""))
        occ     = str(r.get("occ_symbol", ""))
        strike  = r.get("strike", 0)
        print(f"  {r['date']:<12} {r['signal']:<20} {r['atr']:>8.4f} "
              f"{strike:>7.2f} {expiry:<12} {occ:<26} "
              f"{opt_buy:>8} {opt_sel:>9} "
              f"{ent_t:>8} {ext_t:>8} {pnl_s:>10}  {reason}")

    verdict = ("✓  PROFITABLE — consider paper trading options"
               if total_pnl > 0 and winrate >= 40
               else "✗  NOT PROFITABLE — review strike / expiry parameters")
    print(f"\n  {verdict}")
    print(f"  {'═'*W}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Options Backtest: Liquidity Candle + Hammer / Bullish Engulfing — CALL"
    )
    parser.add_argument("--symbol",  default="AAPL",
                        help="Underlying ticker (default: AAPL)")
    parser.add_argument("--days",    default=60, type=int,
                        help="Look-back trading days (default: 60)")
    parser.add_argument("--otm",     default=0, type=int,
                        help="OTM steps above ATM (0=ATM, 1=1-strike OTM, …) (default: 0)")
    parser.add_argument("--min-dte", default=1, type=int,
                        help="Min days-to-expiry for contract selection (default: 1)")
    parser.add_argument("--save",    action="store_true",
                        help="Save results to CSV")
    args = parser.parse_args()

    symbol    = args.symbol.upper()
    days      = args.days
    otm_steps = args.otm
    min_dte   = args.min_dte

    strike_label = "ATM" if otm_steps == 0 else f"{otm_steps}-strike OTM"
    print(f"\n  Liquidity Candle Strategy — Options Backtest (CALL)")
    print(f"  Symbol: {symbol}  |  Strike: {strike_label}  "
          f"|  Min DTE: {min_dte}  |  Look-back: {days} days")

    end   = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days + 45)

    print("  Fetching daily stock bars ...")
    daily    = _fetch_daily(symbol, start, end)

    print("  Fetching 1-min stock bars ...")
    intraday = _fetch_intraday(symbol, start, end)

    cutoff     = (end - timedelta(days=days)).date()
    trade_days = sorted({d for d in intraday.index.date if d >= cutoff})
    print(f"  Processing {len(trade_days)} trading days ...\n")

    trades = []
    for d in trade_days:
        result = _process_day(d, daily, intraday, symbol, otm_steps, min_dte)
        if result is None:
            continue

        # Pop internal fields
        sig_t        = result.pop("_signal_time_obj")
        stock_price  = result.pop("_stock_price")
        stk_exit_t   = result.get("stock_exit_time") or str(EXIT_TIME)

        # ── Look up real contract from Alpaca ──────────────────────
        print(f"  {d}  {result['signal']:<20}  stock_price={stock_price:.2f}")
        occ_symbol, actual_strike, actual_expiry = _lookup_contract(
            symbol, d, stock_price, otm_steps, min_dte
        )

        if occ_symbol is None:
            print(f"           ↳ No matching contract found — skipping")
            trades.append({
                **result,
                "strike": actual_strike, "expiry": str(actual_expiry),
                "occ_symbol": "NOT FOUND",
                "opt_entry": None, "opt_exit": None,
                "opt_entry_time": None, "opt_exit_time": None,
                "opt_pnl_per_contract": None, "opt_outcome": "NO DATA",
            })
            continue

        print(f"           ↳ Contract: {occ_symbol}  strike={actual_strike:.2f}  expiry={actual_expiry}")

        # ── Fetch option bars and simulate ─────────────────────────
        opt_bars = _fetch_option_bars(occ_symbol, d)

        if opt_bars.empty:
            print(f"           ↳ No bar data for {occ_symbol} on {d}")
            opt_result = {
                "opt_entry": None, "opt_exit": None,
                "opt_entry_time": None, "opt_exit_time": None,
                "opt_pnl_per_contract": None, "opt_outcome": "NO DATA",
            }
        else:
            opt_result = _simulate_option(opt_bars, sig_t, stk_exit_t)
            pnl_str = (f"${opt_result['opt_pnl_per_contract']:+.2f}"
                       if opt_result["opt_pnl_per_contract"] is not None else "N/A")
            print(f"           ↳ Buy @ {opt_result['opt_entry']}  "
                  f"Sell @ {opt_result['opt_exit']}  "
                  f"PnL/contract = {pnl_str}  [{opt_result['opt_outcome']}]")

        trades.append({
            **result,
            "strike":     actual_strike,
            "expiry":     str(actual_expiry),
            "occ_symbol": occ_symbol,
            **opt_result,
        })

    if not trades:
        print("\n  No signals found. Try --days 90 or a different symbol.\n")
        sys.exit(0)

    df = pd.DataFrame(trades)
    _print_summary(df, symbol, days, otm_steps, min_dte)

    if args.save:
        path = f"options_backtest_{symbol}_{days}d_otm{otm_steps}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved → {path}\n")


if __name__ == "__main__":
    main()
