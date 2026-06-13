"""
Strategy Repository — SCD2 insert/update/query operations.
All strategy changes create new version rows, never UPDATE existing ones.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_
from .models import StrategyMaster
import logging

logger = logging.getLogger(__name__)


def create_strategy(
    db: Session,
    strategy_name: str,
    strategy_json: dict,
    status: str = "draft",
    notes: str = None,
    created_by: str = "user",
) -> StrategyMaster:
    """
    Create a brand new strategy (version 1).
    Assigns a new strategy_id UUID.
    """
    strategy = StrategyMaster(
        strategy_id    = uuid.uuid4(),
        version        = 1,
        is_current     = True,
        effective_from = datetime.now(timezone.utc),
        effective_to   = None,
        strategy_name  = strategy_name,
        strategy_json  = strategy_json,
        status         = status,
        notes          = notes,
        created_by     = created_by,
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    logger.info(f"Created strategy {strategy.strategy_id} v{strategy.version} — {strategy_name}")
    return strategy


def update_strategy(
    db: Session,
    strategy_id: str,
    strategy_json: dict = None,
    strategy_name: str = None,
    status: str = None,
    notes: str = None,
) -> StrategyMaster:
    """
    SCD2 update — closes current version, inserts new version.
    Always call this when anything about the strategy changes.
    """
    sid = uuid.UUID(strategy_id)

    # Step 1 — find current version
    current = db.query(StrategyMaster).filter(
        and_(
            StrategyMaster.strategy_id == sid,
            StrategyMaster.is_current == True,
        )
    ).first()

    if not current:
        raise ValueError(f"Strategy {strategy_id} not found")

    now = datetime.now(timezone.utc)

    # Step 2 — close current version
    current.is_current   = False
    current.effective_to = now

    # Step 3 — insert new version with changes applied
    new_version = StrategyMaster(
        strategy_id    = sid,
        version        = current.version + 1,
        is_current     = True,
        effective_from = now,
        effective_to   = None,
        strategy_name  = strategy_name  or current.strategy_name,
        strategy_json  = strategy_json  or current.strategy_json,
        status         = status         or current.status,
        notes          = notes          or current.notes,
        created_by     = current.created_by,
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)
    logger.info(f"Updated strategy {strategy_id} → v{new_version.version} status={new_version.status}")
    return new_version


def get_current_strategy(db: Session, strategy_id: str) -> StrategyMaster:
    """Get the current version of a strategy."""
    sid = uuid.UUID(strategy_id)
    return db.query(StrategyMaster).filter(
        and_(
            StrategyMaster.strategy_id == sid,
            StrategyMaster.is_current == True,
        )
    ).first()


def get_strategy_history(db: Session, strategy_id: str) -> list[StrategyMaster]:
    """Get all versions of a strategy — full SCD2 audit trail."""
    sid = uuid.UUID(strategy_id)
    return db.query(StrategyMaster).filter(
        StrategyMaster.strategy_id == sid
    ).order_by(StrategyMaster.version).all()


def list_strategies(
    db: Session,
    status: str = None,
    current_only: bool = True,
) -> list[StrategyMaster]:
    """List strategies. Defaults to current versions only."""
    q = db.query(StrategyMaster)
    if current_only:
        q = q.filter(StrategyMaster.is_current == True)
    if status:
        q = q.filter(StrategyMaster.status == status)
    return q.order_by(StrategyMaster.created_at.desc()).all()


def list_approved_strategies(db: Session) -> list[StrategyMaster]:
    """
    Returns only current, approved strategies.
    This is what the trading screen reads.
    """
    return db.query(StrategyMaster).filter(
        and_(
            StrategyMaster.is_current == True,
            StrategyMaster.status == "approved",
        )
    ).order_by(StrategyMaster.effective_from.desc()).all()


def change_status(
    db: Session,
    strategy_id: str,
    new_status: str,
    notes: str = None,
) -> StrategyMaster:
    """
    Change strategy status — creates new SCD2 version.
    Valid transitions:
      draft → backtested → approved → paper_trading → live
      any   → paused
      any   → retired
    """
    valid_statuses = ["draft","backtested","approved","paper_trading","live","paused","retired"]
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid status: {new_status}")

    return update_strategy(db, strategy_id, status=new_status, notes=notes)
