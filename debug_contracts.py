"""
debug_contracts.py
==================
Run this locally to diagnose NOT FOUND contracts.
It prints exactly what Alpaca returns for each signal date so you can
see the real symbol format, available strikes, and expiry dates.

Usage:
    python debug_contracts.py --symbol AAPL
    python debug_contracts.py --symbol MSFT
"""
import os
import argparse
from datetime import date, timedelta
from dotenv import load_dotenv
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums    import ContractType

load_dotenv()
API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def probe(symbol: str, sig_date: date, stock_price: float):
    """Wide probe — find ANY contracts near signal date to see what exists."""
    print(f"\n{'═'*70}")
    print(f"  Signal: {sig_date}  |  {symbol}  |  stock_price ≈ {stock_price:.2f}")
    print(f"{'═'*70}")

    # 1. What expiries exist in the next 30 days?
    exp_start = sig_date
    exp_end   = sig_date + timedelta(days=30)
    lo_strike = stock_price * 0.95
    hi_strike = stock_price * 1.05

    print(f"\n  [A] Contracts expiring {exp_start} → {exp_end}, "
          f"strike {lo_strike:.1f}→{hi_strike:.1f}:")
    try:
        resp = trade_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[symbol],
                type=ContractType.CALL,
                expiration_date_gte=str(exp_start),
                expiration_date_lte=str(exp_end),
                strike_price_gte=str(round(lo_strike, 2)),
                strike_price_lte=str(round(hi_strike, 2)),
                limit=20,
            )
        )
        contracts = resp.option_contracts
        if not contracts:
            print("    → NONE FOUND")
        for c in contracts:
            print(f"    {c.symbol:<32}  strike={c.strike_price:<8}  "
                  f"expiry={c.expiration_date}  status={c.status}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # 2. Any status? Try including expired/inactive
    print(f"\n  [B] Same but wider strike (±20%) — check if symbol exists at all:")
    try:
        resp = trade_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[symbol],
                type=ContractType.CALL,
                expiration_date_gte=str(exp_start),
                expiration_date_lte=str(exp_start + timedelta(days=7)),
                strike_price_gte=str(round(stock_price * 0.80, 2)),
                strike_price_lte=str(round(stock_price * 1.20, 2)),
                limit=20,
            )
        )
        contracts = resp.option_contracts
        if not contracts:
            print("    → NONE FOUND — contracts may have expired/been purged")
        for c in contracts:
            print(f"    {c.symbol:<32}  strike={c.strike_price:<8}  "
                  f"expiry={c.expiration_date}  status={c.status}")
    except Exception as e:
        print(f"    ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    args = parser.parse_args()

    # Paste your NOT FOUND cases here:
    # (signal_date, approximate_stock_price)
    not_found_cases = {
        "AAPL": [
            (date(2026, 4, 23), 275.0),
            (date(2026, 4, 28), 270.0),
            (date(2026, 5, 14), 300.0),
        ],
        "MSFT": [
            (date(2026, 4, 22), 425.0),
            (date(2026, 5, 12), 410.0),
            (date(2026, 5, 21), 415.0),
            (date(2026, 6,  3), 435.0),
        ],
    }

    symbol = args.symbol.upper()
    cases  = not_found_cases.get(symbol, [])
    if not cases:
        print(f"No cases defined for {symbol}. Edit the not_found_cases dict.")
    for sig_date, price in cases:
        probe(symbol, sig_date, price)
