"""
backtest_engine.py  —  STANDALONE
===================================
Backtests the Liquidity Candle + Hammer / Bullish Engulfing BUY strategy
using Alpaca's free historical data API.

Setup:
    pip install alpaca-py pandas numpy python-dotenv

    Create a .env file in the same directory:
        ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
        ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Usage:
    python backtest_engine.py
    python backtest_engine.py --symbol TSLA
    python backtest_engine.py --symbol MSFT --days 90 --save
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, date, timezone
from datetime import time as dtime

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockBarsRequest
from alpaca.data.timeframe  import TimeFrame
from alpaca.data.enums      import DataFeed

# Load .env file before reading any env vars
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
ATR_PERIOD        = 14
LIQUIDITY_PCT     = 0.25    # first-candle body must be ≥ this % of ATR
HAMMER_BODY_PCT   = 0.35    # body ≤ 35% of total range
HAMMER_WICK_RATIO = 2.0     # lower wick ≥ 2× body
EXIT_TIME         = dtime(11, 5)

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
#  ALPACA DATA
# ───────────────────────────────────────────────────────────────────────────────
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def _fetch_daily(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(symbol_or_symbols=symbol,
                           timeframe=TimeFrame.Day,
                           feed=DataFeed.IEX,
                           start=start, end=end)
    df  = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def _fetch_intraday(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(symbol_or_symbols=symbol,
                           timeframe=TimeFrame.Minute,
                           feed=DataFeed.IEX,
                           start=start, end=end)
    df  = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df.sort_index()


# ───────────────────────────────────────────────────────────────────────────────
#  ATR
# ───────────────────────────────────────────────────────────────────────────────
def _atr(df: pd.DataFrame) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()


# ───────────────────────────────────────────────────────────────────────────────
#  PATTERN DETECTION
# ───────────────────────────────────────────────────────────────────────────────
def _is_hammer(o, h, l, c) -> bool:
    total = h - l
    if total == 0 or abs(c - o) == 0:
        return False
    body       = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (body / total       <= HAMMER_BODY_PCT   and
            lower_wick / body  >= HAMMER_WICK_RATIO and
            upper_wick         <= body)


def _is_bullish_engulfing(po, pc, co, cc) -> bool:
    return pc < po and cc > co and cc >= po and co <= pc


# ───────────────────────────────────────────────────────────────────────────────
#  TRADE SIMULATION
# ───────────────────────────────────────────────────────────────────────────────
def _simulate(remaining: pd.DataFrame, entry, sl, target) -> dict:
    for ts, row in remaining.iterrows():
        t     = ts.time() if hasattr(ts, "time") else ts
        t_str = str(t)
        if t >= EXIT_TIME:
            ep  = float(row["open"])
            pnl = round(ep - entry, 4)
            return {"exit_price": round(ep, 4), "exit_time": t_str,
                    "exit_reason": "Time Exit 11:05",
                    "pnl": pnl,
                    "outcome": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "SCRATCH"}
        if float(row["low"]) <= sl:
            return {"exit_price": round(sl, 4),      "exit_time": t_str,
                    "exit_reason": "Stop Loss",
                    "pnl": round(sl - entry, 4),      "outcome": "LOSS"}
        if float(row["high"]) >= target:
            return {"exit_price": round(target, 4),  "exit_time": t_str,
                    "exit_reason": "Target Hit",
                    "pnl": round(target - entry, 4),  "outcome": "WIN"}
    return {"exit_price": None, "exit_time": None, "exit_reason": "Data ended",
            "pnl": None, "outcome": "OPEN"}


# ───────────────────────────────────────────────────────────────────────────────
#  SINGLE-DAY LOGIC
# ───────────────────────────────────────────────────────────────────────────────
def _process_day(day: date, daily_df: pd.DataFrame,
                 intraday_df: pd.DataFrame) -> dict | None:

    # Step 1 — ATR as of previous trading day
    prev = daily_df[daily_df.index.date < day]
    if len(prev) < ATR_PERIOD:
        return None
    atr_val = float(_atr(prev).dropna().iloc[-1])

    # Filter to this day, resample to 5-min
    day_1m = intraday_df[intraday_df.index.date == day]
    if len(day_1m) < 3:
        return None
    day_5m = day_1m.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last",  "volume": "sum"}
    ).dropna()
    if len(day_5m) < 2:
        return None

    # Step 2 — First candle liquidity check
    first     = day_5m.iloc[0]
    o1, c1    = float(first["open"]), float(first["close"])
    body1     = abs(o1 - c1)
    threshold = (LIQUIDITY_PCT / 100.0) * atr_val
    if body1 < threshold:
        return None

    a_high = max(o1, c1)
    b_low  = min(o1, c1)

    # Step 3 — Scan subsequent candles for signal below b_low
    rest   = day_5m.iloc[1:]
    prev_o = prev_c = None

    for i, (ts, row) in enumerate(rest.iterrows()):
        t = ts.time() if hasattr(ts, "time") else ts
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
            outcome = _simulate(rest.iloc[i + 1:], h, l, a_high)
            return {
                "date": str(day),          "signal": signal,
                "signal_time": str(t),     "atr": round(atr_val, 4),
                "first_body": round(body1, 4),
                "a_high": round(a_high, 4), "b_low": round(b_low, 4),
                "entry": round(h, 4),      "stop_loss": round(l, 4),
                "target": round(a_high, 4),
                "risk": round(h - l, 4),   "reward": round(a_high - h, 4),
                **outcome,
            }

        prev_o, prev_c = o, c
    return None


# ───────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ───────────────────────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame, symbol: str, days: int):
    total   = len(df)
    wins    = (df["outcome"] == "WIN").sum()
    losses  = (df["outcome"] == "LOSS").sum()
    scratch = (df["outcome"] == "SCRATCH").sum()
    winrate = wins / total * 100 if total else 0

    valid     = df[df["pnl"].notna()]["pnl"]
    total_pnl = float(valid.sum())
    avg_win   = float(valid[valid > 0].mean()) if (valid > 0).any() else 0.0
    avg_loss  = float(valid[valid < 0].mean()) if (valid < 0).any() else 0.0
    rr        = abs(avg_win / avg_loss)         if avg_loss != 0 else float("inf")

    W = 100
    print(f"\n  {'═'*W}")
    print(f"  BACKTEST RESULTS  |  {symbol}  |  Last {days} days")
    print(f"  {'─'*W}")
    print(f"  {'Signals found':<28}: {total}")
    print(f"  {'Wins / Losses / Scratch':<28}: {wins} / {losses} / {scratch}")
    print(f"  {'Win Rate':<28}: {winrate:.1f}%")
    print(f"  {'Avg Win  ($ per share)':<28}: {avg_win:+.4f}")
    print(f"  {'Avg Loss ($ per share)':<28}: {avg_loss:+.4f}")
    print(f"  {'Risk / Reward ratio':<28}: {rr:.2f}")
    print(f"  {'Total PnL ($ per share)':<28}: {total_pnl:+.4f}")
    print(f"  {'─'*W}")

    # Trade-by-trade table
    H = (f"  {'DATE':<12} {'SIGNAL':<20} {'ATR':>7} {'ENTRY':>7} {'SL':>7} "
         f"{'TARGET':>7} {'ENTRY TIME':>10} {'EXIT TIME':>10} {'PNL':>8}  REASON")
    print(f"\n{H}")
    print(f"  {'─'*W}")
    for _, r in df.iterrows():
        pnl_s    = f"{r['pnl']:+.4f}" if r["pnl"] is not None else "   N/A"
        ent_time = str(r.get("signal_time", ""))[:8]
        ext_time = str(r.get("exit_time",   ""))[:8] if r.get("exit_time") else "  N/A  "
        print(f"  {r['date']:<12} {r['signal']:<20} {r['atr']:>7.4f} "
              f"{r['entry']:>7.4f} {r['stop_loss']:>7.4f} {r['target']:>7.4f} "
              f"{ent_time:>10} {ext_time:>10} {pnl_s:>8}  {r.get('exit_reason','')}")

    verdict = ("✓  PROFITABLE — consider paper trading"
               if total_pnl > 0 and winrate >= 40
               else "✗  NOT PROFITABLE — review parameters")
    print(f"\n  {verdict}")
    print(f"  {'═'*W}\n")


# ───────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backtest: Liquidity Candle + Hammer / Bullish Engulfing"
    )
    parser.add_argument("--symbol", default="AAPL",  help="Ticker (default: AAPL)")
    parser.add_argument("--days",   default=60, type=int,
                        help="Look-back trading days (default: 60)")
    parser.add_argument("--save",   action="store_true",
                        help="Save trade-by-trade results to CSV")
    args   = parser.parse_args()
    symbol = args.symbol.upper()
    days   = args.days

    print(f"\n  Liquidity Candle Strategy — Backtest")
    print(f"  Symbol: {symbol}  |  Look-back: {days} days")

    end   = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days + 45)

    print("  Fetching daily bars ...")
    daily    = _fetch_daily(symbol, start, end)

    print("  Fetching 1-min bars ...")
    intraday = _fetch_intraday(symbol, start, end)

    cutoff     = (end - timedelta(days=days)).date()
    trade_days = sorted({d for d in intraday.index.date if d >= cutoff})
    print(f"  Processing {len(trade_days)} trading days ...\n")

    trades = [r for d in trade_days
              if (r := _process_day(d, daily, intraday)) is not None]

    if not trades:
        print("  No signals found. Try a longer --days window or different symbol.\n")
        sys.exit(0)

    df = pd.DataFrame(trades)
    _print_summary(df, symbol, days)

    if args.save:
        path = f"backtest_{symbol}_{days}d.csv"
        df.to_csv(path, index=False)
        print(f"  Saved → {path}\n")


if __name__ == "__main__":
    main()
