"""
SQLAlchemy ORM models.
Four tables:
  strategy_master    — SCD2 strategy registry
  backtest_results   — one row per backtest run
  deployment_status  — active paper/live deployments (strategy + ticker)
  paper_positions    — open and closed paper trading positions
"""

from sqlalchemy import (
    Column, Integer, String, Boolean, Numeric,
    DateTime, Date, Text, ForeignKey, CheckConstraint
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base
import uuid


class StrategyMaster(Base):
    __tablename__ = "strategy_master"

    strategy_sk     = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id     = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4, index=True)
    version         = Column(Integer, nullable=False, default=1)
    is_current      = Column(Boolean, nullable=False, default=True)
    effective_from  = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    effective_to    = Column(DateTime(timezone=True), nullable=True)
    strategy_name   = Column(String(200), nullable=False)
    strategy_json   = Column(JSONB, nullable=False)
    status          = Column(String(50), nullable=False, default="draft")
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    created_by      = Column(String(100), default="user")
    notes           = Column(Text, nullable=True)

    backtest_results = relationship("BacktestResult", back_populates="strategy")
    deployments      = relationship("DeploymentStatus", back_populates="strategy")

    __table_args__ = (
        CheckConstraint(
            "status IN ('draft','backtested','approved','paper_trading','live','paused','retired')",
            name="valid_status"
        ),
    )

    def to_dict(self) -> dict:
        return {
            "strategy_sk":    self.strategy_sk,
            "strategy_id":    str(self.strategy_id),
            "version":        self.version,
            "is_current":     self.is_current,
            "effective_from": str(self.effective_from) if self.effective_from else None,
            "effective_to":   str(self.effective_to)   if self.effective_to   else None,
            "strategy_name":  self.strategy_name,
            "strategy_json":  self.strategy_json,
            "status":         self.status,
            "created_at":     str(self.created_at) if self.created_at else None,
            "created_by":     self.created_by,
            "notes":          self.notes,
        }


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    run_id              = Column(Integer, primary_key=True, autoincrement=True)
    strategy_sk         = Column(Integer, ForeignKey("strategy_master.strategy_sk"), nullable=False)
    strategy_id         = Column(UUID(as_uuid=True), nullable=False, index=True)
    symbol              = Column(String(20), nullable=False)
    start_date          = Column(Date, nullable=False)
    end_date            = Column(Date, nullable=False)
    timeframe           = Column(String(20), nullable=False)
    initial_capital     = Column(Numeric(12, 2), nullable=False)
    position_sizing     = Column(String(20), nullable=False, default="fixed")
    commission_pct      = Column(Numeric(5, 3), nullable=False)
    slippage_pct        = Column(Numeric(5, 3), nullable=False)
    final_capital       = Column(Numeric(12, 2))
    total_return_pct    = Column(Numeric(10, 4))
    cagr_pct            = Column(Numeric(10, 4))
    sharpe_ratio        = Column(Numeric(8, 4))
    sortino_ratio       = Column(Numeric(8, 4))
    max_drawdown_pct    = Column(Numeric(8, 4))
    win_rate_pct        = Column(Numeric(8, 4))
    profit_factor       = Column(Numeric(8, 4))
    total_trades        = Column(Integer)
    avg_holding_hours   = Column(Numeric(8, 2))
    gross_profit        = Column(Numeric(12, 2))
    gross_loss          = Column(Numeric(12, 2))
    full_result_json    = Column(JSONB)
    data_source         = Column(String(20))
    approved            = Column(Boolean, default=False, index=True)
    approved_at         = Column(DateTime(timezone=True), nullable=True)
    approval_notes      = Column(Text, nullable=True)
    run_at              = Column(DateTime(timezone=True), server_default=func.now())

    strategy    = relationship("StrategyMaster", back_populates="backtest_results")
    deployments = relationship("DeploymentStatus", back_populates="approved_backtest")

    def to_dict(self, include_full_result: bool = False) -> dict:
        d = {
            "run_id":           self.run_id,
            "strategy_sk":      self.strategy_sk,
            "strategy_id":      str(self.strategy_id),
            "symbol":           self.symbol,
            "start_date":       str(self.start_date),
            "end_date":         str(self.end_date),
            "timeframe":        self.timeframe,
            "initial_capital":  float(self.initial_capital)   if self.initial_capital   else None,
            "position_sizing":  self.position_sizing,
            "commission_pct":   float(self.commission_pct)    if self.commission_pct    else None,
            "slippage_pct":     float(self.slippage_pct)      if self.slippage_pct      else None,
            "final_capital":    float(self.final_capital)     if self.final_capital     else None,
            "total_return_pct": float(self.total_return_pct)  if self.total_return_pct  else None,
            "cagr_pct":         float(self.cagr_pct)          if self.cagr_pct          else None,
            "sharpe_ratio":     float(self.sharpe_ratio)      if self.sharpe_ratio      else None,
            "sortino_ratio":    float(self.sortino_ratio)     if self.sortino_ratio     else None,
            "max_drawdown_pct": float(self.max_drawdown_pct)  if self.max_drawdown_pct  else None,
            "win_rate_pct":     float(self.win_rate_pct)      if self.win_rate_pct      else None,
            "profit_factor":    float(self.profit_factor)     if self.profit_factor     else None,
            "total_trades":     self.total_trades,
            "avg_holding_hours":float(self.avg_holding_hours) if self.avg_holding_hours else None,
            "gross_profit":     float(self.gross_profit)      if self.gross_profit      else None,
            "gross_loss":       float(self.gross_loss)        if self.gross_loss        else None,
            "data_source":      self.data_source,
            "approved":         self.approved,
            "approved_at":      str(self.approved_at)  if self.approved_at  else None,
            "approval_notes":   self.approval_notes,
            "run_at":           str(self.run_at)        if self.run_at        else None,
        }
        if include_full_result:
            d["full_result_json"] = self.full_result_json
        return d


class DeploymentStatus(Base):
    """
    One row per approved strategy+ticker combination.
    approved_backtest_run_id links to the specific backtest that
    proved this strategy works on this ticker.
    """
    __tablename__ = "deployment_status"

    deployment_id            = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id              = Column(UUID(as_uuid=True), nullable=False, index=True)
    strategy_sk              = Column(Integer, ForeignKey("strategy_master.strategy_sk"), nullable=False)
    approved_backtest_run_id = Column(Integer, ForeignKey("backtest_results.run_id"), nullable=True)

    # What to trade
    symbol                   = Column(String(20), nullable=False)
    timeframe                = Column(String(20), nullable=False)
    capital_allocated        = Column(Numeric(12, 2))
    position_sizing          = Column(String(20))

    # Deployment stage and state
    deployment_stage         = Column(String(50), nullable=False)
    is_active                = Column(Boolean, nullable=False, default=True, index=True)
    started_at               = Column(DateTime(timezone=True), server_default=func.now())
    stopped_at               = Column(DateTime(timezone=True), nullable=True)
    stop_reason              = Column(Text, nullable=True)

    strategy         = relationship("StrategyMaster", back_populates="deployments")
    approved_backtest = relationship("BacktestResult", back_populates="deployments")
    positions        = relationship("PaperPosition", back_populates="deployment")

    __table_args__ = (
        CheckConstraint(
            "deployment_stage IN ('paper_trading','live')",
            name="valid_stage"
        ),
    )

    def to_dict(self) -> dict:
        return {
            "deployment_id":             self.deployment_id,
            "strategy_id":               str(self.strategy_id),
            "strategy_sk":               self.strategy_sk,
            "approved_backtest_run_id":  self.approved_backtest_run_id,
            "symbol":                    self.symbol,
            "timeframe":                 self.timeframe,
            "capital_allocated":         float(self.capital_allocated) if self.capital_allocated else None,
            "position_sizing":           self.position_sizing,
            "deployment_stage":          self.deployment_stage,
            "is_active":                 self.is_active,
            "started_at":                str(self.started_at)  if self.started_at  else None,
            "stopped_at":                str(self.stopped_at)  if self.stopped_at  else None,
            "stop_reason":               self.stop_reason,
        }


class PaperPosition(Base):
    """
    Tracks every paper trade — open and closed.
    One row per trade entry. Updated on exit.
    """
    __tablename__ = "paper_positions"

    position_id      = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id    = Column(Integer, ForeignKey("deployment_status.deployment_id"), nullable=False, index=True)
    strategy_id      = Column(UUID(as_uuid=True), nullable=False)
    symbol           = Column(String(20), nullable=False)

    # Alpaca order IDs
    entry_order_id   = Column(String(100), nullable=True)
    exit_order_id    = Column(String(100), nullable=True)
    sl_order_id      = Column(String(100), nullable=True)
    tp_order_id      = Column(String(100), nullable=True)

    # Trade details
    shares           = Column(Numeric(12, 4), nullable=True)
    entry_price      = Column(Numeric(12, 4), nullable=True)
    exit_price       = Column(Numeric(12, 4), nullable=True)
    stop_loss_price  = Column(Numeric(12, 4), nullable=True)
    take_profit_price = Column(Numeric(12, 4), nullable=True)

    # P&L
    pnl              = Column(Numeric(12, 2), nullable=True)
    pnl_pct          = Column(Numeric(8, 4), nullable=True)

    # Status
    status           = Column(String(20), nullable=False, default="open")
    exit_reason      = Column(String(50), nullable=True)

    # Timestamps
    entry_time       = Column(DateTime(timezone=True), nullable=True)
    exit_time        = Column(DateTime(timezone=True), nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    deployment = relationship("DeploymentStatus", back_populates="positions")

    __table_args__ = (
        CheckConstraint(
            "status IN ('open','closed','cancelled')",
            name="valid_position_status"
        ),
    )

    def to_dict(self) -> dict:
        return {
            "position_id":       self.position_id,
            "deployment_id":     self.deployment_id,
            "strategy_id":       str(self.strategy_id),
            "symbol":            self.symbol,
            "entry_order_id":    self.entry_order_id,
            "exit_order_id":     self.exit_order_id,
            "shares":            float(self.shares)        if self.shares        else None,
            "entry_price":       float(self.entry_price)   if self.entry_price   else None,
            "exit_price":        float(self.exit_price)    if self.exit_price    else None,
            "stop_loss_price":   float(self.stop_loss_price)  if self.stop_loss_price  else None,
            "take_profit_price": float(self.take_profit_price) if self.take_profit_price else None,
            "pnl":               float(self.pnl)           if self.pnl           else None,
            "pnl_pct":           float(self.pnl_pct)       if self.pnl_pct       else None,
            "status":            self.status,
            "exit_reason":       self.exit_reason,
            "entry_time":        str(self.entry_time) if self.entry_time else None,
            "exit_time":         str(self.exit_time)  if self.exit_time  else None,
        }
