"""
Paper Trading Engine.
Reads active deployments from Neon, evaluates strategy signals
on live Alpaca data, places paper orders, tracks positions.

Runs on a schedule — every hour during US market hours.
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.orm import Session

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import SessionLocal
from db.models import DeploymentStatus, StrategyMaster, PaperPosition
from backtest.indicators import compute_all_indicators
from backtest.signals import evaluate_group

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('paper_trading.log'),
    ]
)


# ─── Market hours check ───────────────────────────────────────────────────────

def is_market_open() -> bool:
    """Check if US market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et <= market_close


# ─── Alpaca client ────────────────────────────────────────────────────────────

def get_alpaca_trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key    = os.getenv("ALPACA_API_KEY"),
        secret_key = os.getenv("ALPACA_SECRET_KEY"),
        paper      = True,
    )

def get_alpaca_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        api_key    = os.getenv("ALPACA_API_KEY"),
        secret_key = os.getenv("ALPACA_SECRET_KEY"),
    )


# ─── Fetch recent bars ────────────────────────────────────────────────────────

def fetch_recent_bars(symbol: str, timeframe: str = "1Hour", lookback_bars: int = 300) -> pd.DataFrame:
    """
    Fetch recent OHLCV bars for signal evaluation.
    Needs enough bars to compute all indicators (warmup).
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = get_alpaca_data_client()

    tf_map = {
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    }
    tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))

    # Fetch enough bars for indicator warmup
    end   = datetime.now(timezone.utc)
    # For hourly: 300 bars ≈ 46 trading days
    start = end - timedelta(days=90)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        adjustment="all",
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.sort_index()

    logger.info(f"Fetched {len(df)} bars for {symbol}")
    return df


# ─── Evaluate signal at current bar ──────────────────────────────────────────

def evaluate_current_signal(
    df: pd.DataFrame,
    strategy_json: dict,
) -> dict:
    """
    Compute indicators and evaluate entry/exit signal at the latest bar.
    Returns {"entry": bool, "exit": bool, "bar_time": timestamp}
    """
    indicators_spec = strategy_json.get("indicators", {})
    conditions      = strategy_json.get("conditions", {})
    entry_rules     = strategy_json.get("entry_rules", {})
    exit_rules      = strategy_json.get("exit_rules", {})

    computed = compute_all_indicators(df, indicators_spec)
    last_bar = len(df) - 1

    entry_signal = evaluate_group(entry_rules, conditions, computed, last_bar)
    exit_signal  = evaluate_group(exit_rules,  conditions, computed, last_bar)

    return {
        "entry":    entry_signal,
        "exit":     exit_signal,
        "bar_time": str(df.index[-1]),
        "close":    float(df["close"].iloc[-1]),
    }


# ─── Alpaca order management ──────────────────────────────────────────────────

def place_market_order(symbol: str, qty: float, side: str) -> Optional[str]:
    """Place a market order. Returns order_id or None."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    try:
        client = get_alpaca_trading_client()
        order  = client.submit_order(MarketOrderRequest(
            symbol        = symbol,
            qty           = round(qty, 2),
            side          = OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        ))
        logger.info(f"Placed {side} order: {symbol} qty={qty} order_id={order.id}")
        return str(order.id)
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return None


def get_account_info() -> dict:
    """Get Alpaca paper account details."""
    try:
        client  = get_alpaca_trading_client()
        account = client.get_account()
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
        }
    except Exception as e:
        logger.error(f"Could not get account info: {e}")
        return {}


def get_open_position(symbol: str) -> Optional[dict]:
    """Check if we have an open position in Alpaca for this symbol."""
    try:
        client   = get_alpaca_trading_client()
        position = client.get_open_position(symbol)
        return {
            "symbol":   symbol,
            "qty":      float(position.qty),
            "avg_cost": float(position.avg_entry_price),
            "market_value": float(position.market_value),
            "unrealized_pnl": float(position.unrealized_pl),
        }
    except Exception:
        return None  # No open position


# ─── Main execution loop ──────────────────────────────────────────────────────

def process_deployment(db: Session, deployment: DeploymentStatus):
    """
    Process one active deployment:
    1. Fetch latest bars
    2. Evaluate signal
    3. If entry signal + no position → buy
    4. If exit signal + open position → sell
    5. Record to paper_positions
    """
    symbol       = deployment.symbol
    timeframe    = deployment.timeframe or "1Hour"
    strategy_json = deployment.strategy.strategy_json
    risk_rules   = strategy_json.get("risk_rules", {})
    position_pct = risk_rules.get("position_size", {}).get("value", 10) / 100
    stop_loss_pct    = risk_rules.get("stop_loss", {}).get("value")
    take_profit_pct  = risk_rules.get("take_profit", {}).get("value")

    logger.info(f"Processing deployment {deployment.deployment_id}: {symbol}")

    try:
        # Fetch bars
        df = fetch_recent_bars(symbol, timeframe)
        if len(df) < 60:
            logger.warning(f"Not enough bars for {symbol} ({len(df)}), skipping")
            return

        # Evaluate signal
        signal = evaluate_current_signal(df, strategy_json)
        logger.info(f"{symbol} signal: entry={signal['entry']} exit={signal['exit']} close=${signal['close']:.2f}")

        # Check Alpaca for open position
        alpaca_position = get_open_position(symbol)
        has_position    = alpaca_position is not None

        # Check DB for open paper position
        open_db_position = db.query(PaperPosition).filter(
            PaperPosition.deployment_id == deployment.deployment_id,
            PaperPosition.status        == "open",
        ).first()

        # ── Entry logic ──────────────────────────────────────────────────────
        if signal["entry"] and not has_position:
            account = get_account_info()
            capital = float(deployment.capital_allocated or 10000)
            position_value = capital * position_pct
            shares = position_value / signal["close"]
            current_price = signal["close"]

            order_id = place_market_order(symbol, shares, "buy")

            if order_id:
                sl_price = current_price * (1 - stop_loss_pct/100) if stop_loss_pct else None
                tp_price = current_price * (1 + take_profit_pct/100) if take_profit_pct else None

                pos = PaperPosition(
                    deployment_id     = deployment.deployment_id,
                    strategy_id       = deployment.strategy_id,
                    symbol            = symbol,
                    entry_order_id    = order_id,
                    shares            = round(shares, 4),
                    entry_price       = round(current_price, 4),
                    stop_loss_price   = round(sl_price, 4) if sl_price else None,
                    take_profit_price = round(tp_price, 4) if tp_price else None,
                    status            = "open",
                    entry_time        = datetime.now(timezone.utc),
                )
                db.add(pos)
                db.commit()
                logger.info(f"ENTRY: {symbol} {shares:.2f} shares @ ${current_price:.2f} SL=${sl_price} TP=${tp_price}")

        # ── Exit logic ───────────────────────────────────────────────────────
        elif signal["exit"] and has_position and open_db_position:
            current_price = signal["close"]
            shares        = abs(alpaca_position["qty"])
            order_id      = place_market_order(symbol, shares, "sell")

            if order_id and open_db_position:
                entry_price = float(open_db_position.entry_price or current_price)
                pnl         = (current_price - entry_price) * float(open_db_position.shares or shares)
                pnl_pct     = (current_price / entry_price - 1) * 100

                open_db_position.exit_order_id = order_id
                open_db_position.exit_price     = round(current_price, 4)
                open_db_position.pnl            = round(pnl, 2)
                open_db_position.pnl_pct        = round(pnl_pct, 4)
                open_db_position.status         = "closed"
                open_db_position.exit_reason    = "signal"
                open_db_position.exit_time      = datetime.now(timezone.utc)
                db.commit()
                logger.info(f"EXIT: {symbol} @ ${current_price:.2f} PnL=${pnl:.2f} ({pnl_pct:.2f}%)")

        else:
            logger.info(f"{symbol}: no action (entry={signal['entry']} has_position={has_position})")

    except Exception as e:
        logger.error(f"Error processing deployment {deployment.deployment_id}: {e}")
        db.rollback()


def run_paper_trading_cycle():
    """
    Main cycle — called by the scheduler every hour.
    Loads all active paper trading deployments and processes each one.
    """
    if not is_market_open():
        logger.info("Market is closed — skipping cycle")
        return

    logger.info("=" * 60)
    logger.info(f"Paper trading cycle started: {datetime.now(timezone.utc)}")

    db = SessionLocal()
    try:
        # Load all active paper trading deployments with their strategy
        deployments = (
            db.query(DeploymentStatus)
            .join(StrategyMaster, StrategyMaster.strategy_sk == DeploymentStatus.strategy_sk)
            .filter(
                DeploymentStatus.is_active       == True,
                DeploymentStatus.deployment_stage == "paper_trading",
            )
            .all()
        )

        logger.info(f"Found {len(deployments)} active deployments")

        for deployment in deployments:
            process_deployment(db, deployment)

    except Exception as e:
        logger.error(f"Cycle error: {e}")
    finally:
        db.close()

    logger.info("Paper trading cycle complete")
    logger.info("=" * 60)
