"""
Backtest Repository — save backtest runs and handle approval.
Approving a backtest:
  1. Marks run as approved
  2. Promotes strategy to 'approved' status (SCD2)
  3. Creates a deployment_status row for strategy + ticker
"""

import uuid
import numpy as np
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from .models import BacktestResult, StrategyMaster, DeploymentStatus
from .strategy_repo import update_strategy, get_current_strategy
import logging

logger = logging.getLogger(__name__)


def _clean(val):
    """Convert numpy scalar types to plain Python types for PostgreSQL."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
        return float(val)
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _clean_json(obj):
    """Recursively clean numpy types from nested dict/list for JSONB storage."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    return _clean(obj)


def save_backtest_result(
    db: Session,
    strategy_id: str,
    strategy_sk: int,
    backtest_result: dict,
    request_params: dict,
) -> BacktestResult:
    """
    Save a backtest run to the database.
    Auto-promotes strategy from draft → backtested.
    """
    sid = uuid.UUID(strategy_id)
    m   = backtest_result.get("metrics", {})

    result = BacktestResult(
        strategy_sk       = strategy_sk,
        strategy_id       = sid,
        symbol            = request_params.get("symbol"),
        start_date        = request_params.get("start_date"),
        end_date          = request_params.get("end_date"),
        timeframe         = request_params.get("timeframe", "1Hour"),
        initial_capital   = request_params.get("initial_capital", 10000),
        position_sizing   = request_params.get("position_sizing", "fixed"),
        commission_pct    = request_params.get("commission_pct", 0.1),
        slippage_pct      = request_params.get("slippage_pct", 0.05),
        final_capital     = _clean(backtest_result.get("final_capital")),
        total_return_pct  = _clean(m.get("total_return_pct")),
        cagr_pct          = _clean(m.get("cagr_pct")),
        sharpe_ratio      = _clean(m.get("sharpe_ratio")),
        sortino_ratio     = _clean(m.get("sortino_ratio")),
        max_drawdown_pct  = _clean(m.get("max_drawdown_pct")),
        win_rate_pct      = _clean(m.get("win_rate_pct")),
        profit_factor     = _clean(m.get("profit_factor")),
        total_trades      = _clean(m.get("total_trades")),
        avg_holding_hours = _clean(m.get("avg_holding_hours")),
        gross_profit      = _clean(m.get("gross_profit")),
        gross_loss        = _clean(m.get("gross_loss")),
        full_result_json  = _clean_json({
            "trade_log":             backtest_result.get("trade_log", []),
            "monthly_returns":       backtest_result.get("monthly_returns", {}),
            "exit_reason_breakdown": backtest_result.get("exit_reason_breakdown", {}),
            "equity_curve":          backtest_result.get("equity_curve", [])[:100],
        }),
        data_source       = backtest_result.get("data_source"),
        approved          = False,
    )

    db.add(result)

    # Auto-promote draft → backtested
    strategy = get_current_strategy(db, strategy_id)
    if strategy and strategy.status == "draft":
        strategy.is_current   = False
        strategy.effective_to = datetime.now(timezone.utc)
        new_version = StrategyMaster(
            strategy_id    = strategy.strategy_id,
            version        = strategy.version + 1,
            is_current     = True,
            effective_from = datetime.now(timezone.utc),
            effective_to   = None,
            strategy_name  = strategy.strategy_name,
            strategy_json  = strategy.strategy_json,
            status         = "backtested",
            notes          = strategy.notes,
            created_by     = strategy.created_by,
        )
        db.add(new_version)
        logger.info(f"Strategy {strategy_id} promoted draft → backtested")

    db.commit()
    db.refresh(result)
    logger.info(f"Saved backtest run_id={result.run_id} strategy={strategy_id} symbol={result.symbol}")
    return result


def approve_backtest(
    db: Session,
    run_id: int,
    approval_notes: str = None,
) -> dict:
    """
    Approve a backtest run:
    1. Mark run as approved
    2. Promote strategy → approved (SCD2)
    3. Create deployment_status row for this strategy + ticker combination
    """
    result = db.query(BacktestResult).filter(
        BacktestResult.run_id == run_id
    ).first()

    if not result:
        raise ValueError(f"Backtest run {run_id} not found")

    if result.approved:
        raise ValueError(f"Backtest run {run_id} is already approved")

    # Step 1 — mark run as approved
    result.approved       = True
    result.approved_at    = datetime.now(timezone.utc)
    result.approval_notes = approval_notes

    # Step 2 — promote strategy to approved via SCD2
    strategy_id  = str(result.strategy_id)
    new_strategy = update_strategy(
        db,
        strategy_id,
        status="approved",
        notes=f"Approved based on backtest run #{run_id} on {result.symbol}. {approval_notes or ''}".strip()
    )

    # Step 3 — create deployment_status row (strategy + ticker)
    # Check if a deployment already exists for this strategy + symbol
    existing = db.query(DeploymentStatus).filter(
        DeploymentStatus.strategy_id == result.strategy_id,
        DeploymentStatus.symbol      == result.symbol,
        DeploymentStatus.is_active   == True,
    ).first()

    if existing:
        # Update existing deployment to point to new approved backtest
        existing.approved_backtest_run_id = run_id
        existing.strategy_sk              = new_strategy.strategy_sk
        logger.info(f"Updated existing deployment {existing.deployment_id} for {result.symbol}")
        deployment_id = existing.deployment_id
    else:
        # Create new deployment row
        deployment = DeploymentStatus(
            strategy_id              = result.strategy_id,
            strategy_sk              = new_strategy.strategy_sk,
            approved_backtest_run_id = run_id,
            symbol                   = result.symbol,
            timeframe                = result.timeframe,
            capital_allocated        = float(result.initial_capital) if result.initial_capital else 10000,
            position_sizing          = result.position_sizing or "fixed",
            deployment_stage         = "paper_trading",
            is_active                = True,
        )
        db.add(deployment)
        db.flush()
        deployment_id = deployment.deployment_id
        logger.info(f"Created deployment {deployment_id} for {result.symbol}")

    db.commit()
    logger.info(f"Approved run_id={run_id} strategy={strategy_id} → v{new_strategy.version} symbol={result.symbol}")

    return {
        "run_id":         run_id,
        "strategy_id":    strategy_id,
        "new_version":    new_strategy.version,
        "status":         "approved",
        "deployment_id":  deployment_id,
        "symbol":         result.symbol,
        "deployment_stage": "paper_trading",
    }


def get_backtest_results(
    db: Session,
    strategy_id: str = None,
    symbol: str = None,
    approved_only: bool = False,
    limit: int = 50,
) -> list[BacktestResult]:
    q = db.query(BacktestResult)
    if strategy_id:
        q = q.filter(BacktestResult.strategy_id == uuid.UUID(strategy_id))
    if symbol:
        q = q.filter(BacktestResult.symbol == symbol.upper())
    if approved_only:
        q = q.filter(BacktestResult.approved == True)
    return q.order_by(BacktestResult.run_at.desc()).limit(limit).all()


def get_backtest_result(db: Session, run_id: int) -> BacktestResult:
    return db.query(BacktestResult).filter(
        BacktestResult.run_id == run_id
    ).first()
