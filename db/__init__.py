"""
DB package — Neon PostgreSQL integration.
Call create_tables() on startup to ensure schema exists.
"""

from .connection import engine, get_db, test_connection
from .models import Base, StrategyMaster, BacktestResult, DeploymentStatus
from .strategy_repo import (
    create_strategy, update_strategy, get_current_strategy,
    get_strategy_history, list_strategies, list_approved_strategies, change_status
)
from .backtest_repo import (
    save_backtest_result, approve_backtest,
    get_backtest_results, get_backtest_result
)


def create_tables():
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)


__all__ = [
    "engine", "get_db", "test_connection", "create_tables",
    "StrategyMaster", "BacktestResult", "DeploymentStatus",
    "create_strategy", "update_strategy", "get_current_strategy",
    "get_strategy_history", "list_strategies", "list_approved_strategies",
    "change_status", "save_backtest_result", "approve_backtest",
    "get_backtest_results", "get_backtest_result",
]
