from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
import uuid
import json
import os
import re
import sys
from datetime import datetime

app = FastAPI(title="Strategy Builder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Indicator Library Spec ───────────────────────────────────────────────────

INDICATOR_SPEC = {
    "SMA": {
        "label": "Simple Moving Average",
        "group": "Trend",
        "params": {
            "period": {"type": "int", "default": 20, "min": 2, "max": 500},
            "source": {"type": "enum", "options": ["open", "high", "low", "close"], "default": "close"},
        },
        "output_type": "line",
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["line", "scalar", "level"],
    },
    "EMA": {
        "label": "Exponential Moving Average",
        "group": "Trend",
        "params": {
            "period": {"type": "int", "default": 20, "min": 2, "max": 500},
            "source": {"type": "enum", "options": ["open", "high", "low", "close"], "default": "close"},
        },
        "output_type": "line",
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["line", "scalar", "level"],
    },
    "RSI": {
        "label": "Relative Strength Index",
        "group": "Momentum",
        "params": {
            "period": {"type": "int", "default": 14, "min": 2, "max": 100},
            "source": {"type": "enum", "options": ["open", "high", "low", "close"], "default": "close"},
        },
        "output_type": "line",
        "output_range": {"min": 0, "max": 100},
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["scalar"],
        "rhs_constraints": {"min": 0, "max": 100},
    },
    "MACD": {
        "label": "MACD",
        "group": "Momentum",
        "params": {
            "fast":   {"type": "int", "default": 12, "min": 2, "max": 100},
            "slow":   {"type": "int", "default": 26, "min": 2, "max": 200},
            "signal": {"type": "int", "default": 9,  "min": 2, "max": 50},
        },
        "output_type": "composite",
        "components": {
            "macd_line":   {"output_type": "crossable_line", "label": "MACD Line"},
            "signal_line": {"output_type": "crossable_line", "label": "Signal Line"},
            "histogram":   {"output_type": "histogram",      "label": "Histogram"},
        },
        "default_component": "macd_line",
        "valid_operators": {
            "macd_line":   [">", "<", "cross_above", "cross_below"],
            "signal_line": [">", "<", "cross_above", "cross_below"],
            "histogram":   [">", "<", ">=", "<="],
        },
        "valid_rhs": {
            "macd_line":   ["scalar", "crossable_line"],
            "signal_line": ["scalar", "crossable_line"],
            "histogram":   ["scalar"],
        },
        "param_constraints": [
            {"rule": "fast < slow", "error": "MACD fast period must be less than slow period"}
        ],
    },
    "ATR": {
        "label": "Average True Range",
        "group": "Volatility",
        "params": {
            "period": {"type": "int", "default": 14, "min": 2, "max": 100},
        },
        "output_type": "line",
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar", "line"],
        "usage_note": "Primarily used in risk rules as ATR multiple",
    },
    "PRICE": {
        "label": "Price",
        "group": "Price & Volume",
        "params": {
            "source": {"type": "enum", "options": ["open", "high", "low", "close"], "default": "close"},
        },
        "output_type": "scalar",
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["scalar", "line", "level"],
    },
    "VOLUME": {
        "label": "Volume",
        "group": "Price & Volume",
        "params": {
            "avg_period": {"type": "int", "default": None, "min": 2, "max": 200, "optional": True},
        },
        "output_type": "scalar",
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar"],
    },
    "WEEK52_HIGH": {
        "label": "52-Week High",
        "group": "Price & Volume",
        "params": {},
        "output_type": "level",
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar", "line"],
    },
    "WEEK52_LOW": {
        "label": "52-Week Low",
        "group": "Price & Volume",
        "params": {},
        "output_type": "level",
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar", "line"],
    },
}

OPERATORS = {
    ">":           {"label": "is greater than",   "valid_for": ["line", "scalar", "level", "crossable_line"]},
    "<":           {"label": "is less than",       "valid_for": ["line", "scalar", "level", "crossable_line"]},
    ">=":          {"label": "≥",                  "valid_for": ["line", "scalar", "level", "crossable_line"]},
    "<=":          {"label": "≤",                  "valid_for": ["line", "scalar", "level", "crossable_line"]},
    "==":          {"label": "equals",             "valid_for": ["line", "scalar", "level"]},
    "cross_above": {"label": "crosses above",      "valid_for": ["line", "crossable_line"]},
    "cross_below": {"label": "crosses below",      "valid_for": ["line", "crossable_line"]},
}

# ─── Models ───────────────────────────────────────────────────────────────────

class IndicatorRef(BaseModel):
    type: str
    params: dict[str, Any] = {}
    component: Optional[str] = None

class ConditionDef(BaseModel):
    left: IndicatorRef
    operator: str
    right: dict[str, Any]  # {"type": "scalar", "value": 40} or {"type": "indicator", ...}

class RuleGroup(BaseModel):
    logic: str  # AND | OR
    conditions: list[str] = []
    groups: list[Any] = []

class RiskRules(BaseModel):
    stop_loss: Optional[dict] = None
    take_profit: Optional[dict] = None
    position_size: Optional[dict] = None
    max_open_positions: Optional[int] = 5

class StrategyPayload(BaseModel):
    strategy_name: str
    indicators: dict[str, IndicatorRef]
    conditions: dict[str, ConditionDef]
    entry_rules: RuleGroup
    exit_rules: RuleGroup
    risk_rules: RiskRules

# ─── Validation Engine ────────────────────────────────────────────────────────

def validate_strategy(payload: dict) -> dict:
    errors = []
    warnings = []

    indicators = payload.get("indicators", {})
    conditions = payload.get("conditions", {})
    entry_rules = payload.get("entry_rules", {})
    exit_rules  = payload.get("exit_rules", {})
    risk_rules  = payload.get("risk_rules", {})

    # Layer 1 — Schema Integrity
    if not payload.get("strategy_name", "").strip():
        errors.append({"layer": 1, "severity": "error", "condition_ids": [], "field": "strategy_name", "message": "Strategy name is required", "ui_target": "strategy_name"})

    if not indicators:
        errors.append({"layer": 1, "severity": "error", "condition_ids": [], "field": "indicators", "message": "At least one indicator must be defined", "ui_target": "entry_rules"})

    if not conditions:
        errors.append({"layer": 1, "severity": "error", "condition_ids": [], "field": "conditions", "message": "At least one condition must be defined", "ui_target": "entry_rules"})

    # Collect all referenced condition IDs
    def collect_condition_ids(group):
        ids = list(group.get("conditions", []))
        for g in group.get("groups", []):
            ids.extend(collect_condition_ids(g))
        return ids

    entry_cids = collect_condition_ids(entry_rules)
    exit_cids  = collect_condition_ids(exit_rules)
    all_referenced = set(entry_cids + exit_cids)

    for cid in all_referenced:
        if cid not in conditions:
            errors.append({"layer": 1, "severity": "error", "condition_ids": [cid], "field": f"conditions.{cid}", "message": f"Condition '{cid}' is referenced but not defined", "ui_target": "entry_rules"})

    # Orphaned indicators
    # Note: left is a string key referencing indicators dict
    referenced_indicators = set()
    for cond in conditions.values():
        left = cond.get("left")
        right = cond.get("right", {})
        if isinstance(left, str):
            referenced_indicators.add(left)
        elif isinstance(left, dict) and left.get("id"):
            referenced_indicators.add(left["id"])
        if isinstance(right, dict):
            if right.get("type") == "indicator" and right.get("ref"):
                referenced_indicators.add(right["ref"])
            elif right.get("type") == "indicator" and right.get("id"):
                referenced_indicators.add(right["id"])

    for iid in indicators:
        if iid not in referenced_indicators:
            warnings.append({"layer": 1, "severity": "warning", "condition_ids": [], "field": f"indicators.{iid}", "message": f"Indicator '{iid}' is defined but never used in any condition", "ui_target": "entry_rules"})

    # Layer 2 — Indicator Validity
    for iid, ind in indicators.items():
        itype = ind.get("type")
        if itype not in INDICATOR_SPEC:
            errors.append({"layer": 2, "severity": "error", "condition_ids": [], "field": f"indicators.{iid}", "message": f"'{itype}' is not a supported indicator", "ui_target": "entry_rules"})
            continue

        spec = INDICATOR_SPEC[itype]
        params = ind.get("params", {})

        for pname, pspec in spec["params"].items():
            if pspec.get("optional") and pname not in params:
                continue
            if pname not in params and not pspec.get("optional"):
                errors.append({"layer": 2, "severity": "error", "condition_ids": [], "field": f"indicators.{iid}.params.{pname}", "message": f"{itype} requires parameter '{pname}'", "ui_target": "entry_rules"})
                continue

            val = params.get(pname)
            if val is None:
                continue

            if pspec["type"] == "int":
                if not isinstance(val, int):
                    errors.append({"layer": 2, "severity": "error", "condition_ids": [], "field": f"indicators.{iid}.params.{pname}", "message": f"{itype} '{pname}' must be an integer", "ui_target": "entry_rules"})
                elif val < pspec["min"] or val > pspec["max"]:
                    errors.append({"layer": 2, "severity": "error", "condition_ids": [], "field": f"indicators.{iid}.params.{pname}", "message": f"{itype} '{pname}' must be between {pspec['min']} and {pspec['max']}", "ui_target": "entry_rules"})

            if pspec["type"] == "enum" and val not in pspec["options"]:
                errors.append({"layer": 2, "severity": "error", "condition_ids": [], "field": f"indicators.{iid}.params.{pname}", "message": f"{itype} '{pname}' must be one of: {', '.join(pspec['options'])}", "ui_target": "entry_rules"})

        # Cross-param constraints (MACD)
        for constraint in spec.get("param_constraints", []):
            if spec["params"].get("fast") and spec["params"].get("slow"):
                fast = params.get("fast", 0)
                slow = params.get("slow", 0)
                if fast and slow and fast >= slow:
                    errors.append({"layer": 2, "severity": "error", "condition_ids": [], "field": f"indicators.{iid}", "message": constraint["error"], "ui_target": "entry_rules"})

    # Layer 3 — Condition Validity
    for cid, cond in conditions.items():
        left_raw = cond.get("left")
        operator = cond.get("operator", "")
        right = cond.get("right", {})

        # left may be a string key referencing the indicators dict
        if isinstance(left_raw, str):
            left = indicators.get(left_raw, {})
        else:
            left = left_raw or {}

        ltype = left.get("type") if isinstance(left, dict) else None
        if ltype not in INDICATOR_SPEC:
            continue  # already caught in layer 2

        spec = INDICATOR_SPEC[ltype]
        component = left.get("component") if isinstance(left, dict) else None

        # Get effective output type
        if spec["output_type"] == "composite" and component:
            if component not in spec["components"]:
                errors.append({"layer": 3, "severity": "error", "condition_ids": [cid], "field": f"conditions.{cid}", "message": f"'{component}' is not a valid component of {ltype}", "ui_target": cid})
                continue
            output_type = spec["components"][component]["output_type"]
            valid_ops = spec["valid_operators"].get(component, [])
        else:
            output_type = spec["output_type"]
            valid_ops = spec.get("valid_operators", [])

        # Operator validity
        if operator not in valid_ops:
            errors.append({"layer": 3, "severity": "error", "condition_ids": [cid], "field": f"conditions.{cid}.operator", "message": f"'{operator}' is not valid for {ltype} — valid operators: {', '.join(valid_ops)}", "ui_target": cid})

        # RHS validity
        rhs_type = right.get("type")
        if rhs_type == "scalar":
            val = right.get("value")
            rhs_constraints = spec.get("rhs_constraints")
            if rhs_constraints and val is not None:
                if val < rhs_constraints["min"] or val > rhs_constraints["max"]:
                    errors.append({"layer": 3, "severity": "error", "condition_ids": [cid], "field": f"conditions.{cid}.right", "message": f"{ltype} values range from {rhs_constraints['min']} to {rhs_constraints['max']} — got {val}", "ui_target": cid})

    # Layer 4 — Rule Group Validity
    if not entry_cids:
        errors.append({"layer": 4, "severity": "error", "condition_ids": [], "field": "entry_rules", "message": "Entry rules are empty — add at least one condition", "ui_target": "entry_rules"})

    if not exit_cids:
        errors.append({"layer": 4, "severity": "error", "condition_ids": [], "field": "exit_rules", "message": "Exit rules are empty — add at least one condition", "ui_target": "exit_rules"})

    # Layer 5 — Strategy Completeness
    if risk_rules:
        sl = risk_rules.get("stop_loss")
        tp = risk_rules.get("take_profit")
        ps = risk_rules.get("position_size")

        if not sl:
            warnings.append({"layer": 5, "severity": "warning", "condition_ids": [], "field": "risk_rules.stop_loss", "message": "No stop loss defined — positions have unlimited downside risk", "ui_target": "risk_panel"})

        if sl and tp:
            sl_val = sl.get("value", 0)
            tp_val = tp.get("value", 0)
            if tp_val <= sl_val:
                errors.append({"layer": 5, "severity": "error", "condition_ids": [], "field": "risk_rules", "message": f"Take profit ({tp_val}%) must be greater than stop loss ({sl_val}%)", "ui_target": "risk_panel"})

        if ps:
            ps_val = ps.get("value", 0)
            if ps_val <= 0 or ps_val > 100:
                errors.append({"layer": 5, "severity": "error", "condition_ids": [], "field": "risk_rules.position_size", "message": "Position size must be between 1% and 100%", "ui_target": "risk_panel"})

    # Layer 6 — Semantic Coherence (lightweight rule-based)
    def get_ind_type(cond_obj):
        left_raw = cond_obj.get("left")
        if isinstance(left_raw, str):
            return indicators.get(left_raw, {}).get("type")
        elif isinstance(left_raw, dict):
            return left_raw.get("type")
        return None

    entry_rsi = [cid for cid in entry_cids if get_ind_type(conditions.get(cid, {})) == "RSI"]
    exit_rsi  = [cid for cid in exit_cids  if get_ind_type(conditions.get(cid, {})) == "RSI"]

    for ecid in entry_rsi:
        for xcid in exit_rsi:
            ec = conditions[ecid]
            xc = conditions[xcid]
            eop = ec.get("operator")
            xop = xc.get("operator")
            eval_ = ec.get("right", {}).get("value", 50)
            xval  = xc.get("right", {}).get("value", 50)

            if eop in ["<", "<="] and xop in ["<", "<="]:
                if xval <= eval_:
                    errors.append({"layer": 6, "severity": "error", "condition_ids": [ecid, xcid], "field": "semantic", "message": f"Entry requires RSI {eop} {eval_} but exit triggers at RSI {xop} {xval} — exit may fire immediately after entry", "ui_target": ecid})

    has_errors = len(errors) > 0
    return {
        "valid": not has_errors,
        "can_proceed_to_backtest": not has_errors,
        "errors": errors,
        "warnings": warnings,
    }

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/indicators")
def get_indicators():
    return {"indicators": INDICATOR_SPEC, "operators": OPERATORS}

@app.post("/api/validate")
def validate(payload: dict):
    result = validate_strategy(payload)
    return result

@app.post("/api/strategy")
def save_strategy(payload: dict):
    validation = validate_strategy(payload)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"validation": validation})

    strategy_id = str(uuid.uuid4())
    strategy = {
        "schema_version": "1.0",
        "strategy_id": strategy_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "draft",
        **payload,
    }
    return {"strategy_id": strategy_id, "strategy": strategy, "validation": validation}

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ─── Backtest Routes ──────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engine import run_backtest
from backtest.data import validate_symbol as _validate_symbol

class BacktestRequest(BaseModel):
    strategy: dict
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 10_000
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    timeframe: str = "1Hour"

@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    # Validate strategy first
    validation = validate_strategy(req.strategy)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"validation": validation})

    result = run_backtest(
        strategy=req.strategy,
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        initial_capital=req.initial_capital,
        commission_pct=req.commission_pct,
        slippage_pct=req.slippage_pct,
        timeframe=req.timeframe,
    )
    return result

@app.get("/api/symbol/{symbol}/validate")
def validate_symbol_route(symbol: str):
    return _validate_symbol(symbol)


# ─── Database Integration ─────────────────────────────────────────────────────

from db import (
    create_tables, test_connection, get_db,
    create_strategy, update_strategy, get_current_strategy,
    get_strategy_history, list_strategies, list_approved_strategies,
    change_status, save_backtest_result, approve_backtest,
    get_backtest_results, get_backtest_result,
)
from db.models import StrategyMaster, DeploymentStatus, PaperPosition

# Create tables on startup
@app.on_event("startup")
def startup():
    ok = test_connection()
    if ok:
        create_tables()
        print("✓ Neon database connected and tables ready")
    else:
        print("✗ Neon database connection failed — check NEON_DATABASE_URL in .env")


# ── Strategy Registry Routes ──────────────────────────────────────────────────

class SaveStrategyRequest(BaseModel):
    strategy: dict
    notes: Optional[str] = None
    created_by: str = "user"

class UpdateStrategyRequest(BaseModel):
    strategy: Optional[dict] = None
    strategy_name: Optional[str] = None
    notes: Optional[str] = None

class StatusChangeRequest(BaseModel):
    status: str
    notes: Optional[str] = None

class ApproveRequest(BaseModel):
    notes: Optional[str] = None


@app.post("/api/registry/strategies")
def registry_save_strategy(req: SaveStrategyRequest, db: Session = Depends(get_db)):
    """Save a new strategy to the registry (draft status)."""
    validation = validate_strategy(req.strategy)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"validation": validation})

    strategy = create_strategy(
        db=db,
        strategy_name=req.strategy.get("strategy_name", "Unnamed"),
        strategy_json=req.strategy,
        status="draft",
        notes=req.notes,
        created_by=req.created_by,
    )
    return {"success": True, "strategy": strategy.to_dict()}


@app.get("/api/registry/strategies")
def registry_list_strategies(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all current strategy versions."""
    strategies = list_strategies(db, status=status)
    return {"strategies": [s.to_dict() for s in strategies]}


@app.get("/api/registry/strategies/approved")
def registry_approved_strategies(db: Session = Depends(get_db)):
    """
    List only approved strategies.
    This is what the trading screen reads.
    """
    strategies = list_approved_strategies(db)
    return {"strategies": [s.to_dict() for s in strategies]}


@app.get("/api/registry/strategies/{strategy_id}")
def registry_get_strategy(strategy_id: str, db: Session = Depends(get_db)):
    """Get current version of a specific strategy."""
    strategy = get_current_strategy(db, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return strategy.to_dict()


@app.get("/api/registry/strategies/{strategy_id}/history")
def registry_strategy_history(strategy_id: str, db: Session = Depends(get_db)):
    """Get full SCD2 version history of a strategy."""
    history = get_strategy_history(db, strategy_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return {"strategy_id": strategy_id, "versions": [s.to_dict() for s in history]}


@app.put("/api/registry/strategies/{strategy_id}")
def registry_update_strategy(
    strategy_id: str,
    req: UpdateStrategyRequest,
    db: Session = Depends(get_db)
):
    """Update a strategy — creates new SCD2 version."""
    if req.strategy:
        validation = validate_strategy(req.strategy)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail={"validation": validation})

    strategy = update_strategy(
        db=db,
        strategy_id=strategy_id,
        strategy_json=req.strategy,
        strategy_name=req.strategy_name,
        notes=req.notes,
    )
    return {"success": True, "strategy": strategy.to_dict()}


@app.put("/api/registry/strategies/{strategy_id}/status")
def registry_change_status(
    strategy_id: str,
    req: StatusChangeRequest,
    db: Session = Depends(get_db)
):
    """Change strategy status — creates new SCD2 version."""
    try:
        strategy = change_status(db, strategy_id, req.status, req.notes)
        return {"success": True, "strategy": strategy.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Backtest + Registry Integration ──────────────────────────────────────────

class BacktestAndSaveRequest(BaseModel):
    strategy_id: Optional[str] = None   # if None, creates new strategy first
    strategy: dict
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 10_000
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    timeframe: str = "1Hour"
    position_sizing: str = "fixed"
    notes: Optional[str] = None


@app.post("/api/registry/backtest")
def registry_run_and_save_backtest(
    req: BacktestAndSaveRequest,
    db: Session = Depends(get_db)
):
    """
    Run a backtest AND save the result to the registry in one call.
    If strategy_id provided, links result to existing strategy.
    If not, creates a new strategy first.
    """
    # Validate strategy
    validation = validate_strategy(req.strategy)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"validation": validation})

    # Get or create strategy in registry
    if req.strategy_id:
        strategy = get_current_strategy(db, req.strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {req.strategy_id} not found")
    else:
        strategy = create_strategy(
            db=db,
            strategy_name=req.strategy.get("strategy_name", "Unnamed"),
            strategy_json=req.strategy,
            status="draft",
            notes=req.notes,
        )

    # Run the backtest
    backtest_result = run_backtest(
        strategy=req.strategy,
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        initial_capital=req.initial_capital,
        commission_pct=req.commission_pct,
        slippage_pct=req.slippage_pct,
        timeframe=req.timeframe,
        position_sizing=req.position_sizing,
    )

    if not backtest_result.get("success"):
        raise HTTPException(status_code=400, detail=backtest_result.get("error"))

    # Save result to DB
    saved = save_backtest_result(
        db=db,
        strategy_id=str(strategy.strategy_id),
        strategy_sk=strategy.strategy_sk,
        backtest_result=backtest_result,
        request_params={
            "symbol":          req.symbol,
            "start_date":      req.start_date,
            "end_date":        req.end_date,
            "timeframe":       req.timeframe,
            "initial_capital": req.initial_capital,
            "position_sizing": req.position_sizing,
            "commission_pct":  req.commission_pct,
            "slippage_pct":    req.slippage_pct,
        },
    )

    return {
        "success":      True,
        "strategy_id":  str(strategy.strategy_id),
        "strategy_sk":  strategy.strategy_sk,
        "run_id":       saved.run_id,
        "backtest":     backtest_result,
    }


@app.post("/api/registry/backtest/{run_id}/approve")
def registry_approve_backtest(
    run_id: int,
    req: ApproveRequest,
    db: Session = Depends(get_db)
):
    """
    Approve a backtest run.
    Marks the run as approved and promotes strategy status to 'approved'.
    """
    try:
        result = approve_backtest(db, run_id, req.notes)
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/registry/backtest")
def registry_list_backtests(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    approved_only: bool = False,
    db: Session = Depends(get_db)
):
    """List backtest results with optional filters."""
    results = get_backtest_results(db, strategy_id, symbol, approved_only)
    return {"results": [r.to_dict() for r in results]}


@app.get("/api/registry/backtest/{run_id}")
def registry_get_backtest(run_id: int, db: Session = Depends(get_db)):
    """Get a single backtest result including full trade log."""
    result = get_backtest_result(db, run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return result.to_dict(include_full_result=True)


# ─── Paper Trading Routes ─────────────────────────────────────────────────────


@app.get("/api/paper/deployments")
def list_deployments(db: Session = Depends(get_db)):
    """List all active paper trading deployments."""
    deployments = (
        db.query(DeploymentStatus)
        .filter(
            DeploymentStatus.is_active       == True,
            DeploymentStatus.deployment_stage == "paper_trading",
        )
        .all()
    )
    return {"deployments": [d.to_dict() for d in deployments]}


@app.get("/api/paper/positions")
def list_paper_positions(
    deployment_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List paper positions — open or closed."""
    q = db.query(PaperPosition)
    if deployment_id:
        q = q.filter(PaperPosition.deployment_id == deployment_id)
    if status:
        q = q.filter(PaperPosition.status == status)
    positions = q.order_by(PaperPosition.created_at.desc()).limit(100).all()
    return {"positions": [p.to_dict() for p in positions]}


@app.get("/api/paper/summary")
def paper_trading_summary(db: Session = Depends(get_db)):
    """
    Summary of all paper trading activity.
    Used by the paper trading UI dashboard.
    """
    deployments = (
        db.query(DeploymentStatus)
        .filter(
            DeploymentStatus.is_active       == True,
            DeploymentStatus.deployment_stage == "paper_trading",
        ).all()
    )

    summary = []
    for d in deployments:
        positions = db.query(PaperPosition).filter(
            PaperPosition.deployment_id == d.deployment_id
        ).all()

        closed   = [p for p in positions if p.status == "closed"]
        open_pos = [p for p in positions if p.status == "open"]
        total_pnl = sum(float(p.pnl or 0) for p in closed)
        wins      = [p for p in closed if (p.pnl or 0) > 0]
        win_rate  = len(wins) / len(closed) * 100 if closed else 0

        strategy = db.query(StrategyMaster).filter(
            StrategyMaster.strategy_sk == d.strategy_sk
        ).first()

        summary.append({
            "deployment_id":   d.deployment_id,
            "strategy_name":   strategy.strategy_name if strategy else "Unknown",
            "symbol":          d.symbol,
            "timeframe":       d.timeframe,
            "capital":         float(d.capital_allocated or 0),
            "started_at":      str(d.started_at) if d.started_at else None,
            "total_trades":    len(closed),
            "open_positions":  len(open_pos),
            "total_pnl":       round(total_pnl, 2),
            "win_rate":        round(win_rate, 1),
            "is_active":       d.is_active,
        })

    return {"summary": summary}


@app.put("/api/paper/deployments/{deployment_id}/stop")
def stop_deployment(
    deployment_id: int,
    db: Session = Depends(get_db)
):
    """Stop an active paper trading deployment."""
    deployment = db.query(DeploymentStatus).filter(
        DeploymentStatus.deployment_id == deployment_id
    ).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment.is_active  = False
    deployment.stopped_at = datetime.utcnow()
    deployment.stop_reason = "Stopped by user"
    db.commit()
    return {"success": True, "deployment_id": deployment_id}


# ─── AI Review Routes ─────────────────────────────────────────────────────────

class AIReviewRequest(BaseModel):
    backtest_result: dict
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"


@app.post("/api/ai/review")
def ai_review(req: AIReviewRequest):
    """
    AI Backtest Interpretation Agent.
    Routes to correct LLM provider. API keys stored in .env, never in browser.
    Supported providers: anthropic | openai | openrouter | groq
    """
    d = req.backtest_result
    m = d.get("metrics", {})
    exits   = d.get("exit_reason_breakdown", {})
    monthly = d.get("monthly_returns", {})
    trades  = d.get("trade_log", [])

    pos_months = len([v for v in monthly.values() if v > 0])
    neg_months = len([v for v in monthly.values() if v < 0])

    prompt = f"""You are an expert quantitative trading analyst.
Analyse this backtest result and return ONLY a valid JSON object — no markdown, no text outside JSON.

Strategy: {d.get('strategy_name')}
Symbol: {d.get('symbol')}  |  Period: {d.get('start_date')} to {d.get('end_date')}  |  Timeframe: {d.get('timeframe')}
Capital: ${d.get('initial_capital')} to ${d.get('final_capital')}  |  Position sizing: {d.get('position_sizing')}

Metrics:
- Total return: {m.get('total_return_pct', 0):.2f}%
- CAGR: {m.get('cagr_pct', 0):.2f}%
- Sharpe: {m.get('sharpe_ratio', 0):.2f}  |  Sortino: {m.get('sortino_ratio', 0):.2f}
- Max drawdown: {m.get('max_drawdown_pct', 0):.2f}%
- Win rate: {m.get('win_rate_pct', 0):.1f}%  |  Profit factor: {m.get('profit_factor', 0)}
- Total trades: {m.get('total_trades', 0)}  |  Avg hold: {m.get('avg_holding_hours', 0):.1f}h
- Gross profit: ${m.get('gross_profit', 0):.2f}  |  Gross loss: ${m.get('gross_loss', 0):.2f}

Exit breakdown: {exits}
Profitable months: {pos_months} / {len(monthly)}  |  Losing months: {neg_months}
Sample trades first 2: {trades[:2]}
Sample trades last 2: {trades[-2:]}

Return ONLY this JSON:
{{
  "verdict": "strong or borderline or weak or failing",
  "verdict_text": "one direct sentence on overall performance",
  "score": 0 to 100,
  "strengths": ["up to 3 specific strengths citing actual numbers"],
  "issues": ["up to 3 specific problems citing actual numbers"],
  "suggestions": ["up to 3 concrete actionable improvements"],
  "regime_warning": "one sentence about conditions this may fail in or null",
  "ready_for_paper_trading": true or false,
  "ready_reason": "one sentence explaining paper trading readiness"
}}

Be direct. Reference actual numbers. Do not be encouraging if the strategy is poor."""

    try:
        text = _call_llm_provider(req.provider, req.model, prompt)

        clean  = re.sub(r'```json|```', '', text).strip()
        result = json.loads(clean)
        return {"success": True, "review": result}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _call_llm_provider(provider: str, model: str, prompt: str) -> str:
    """
    Routes prompt to the correct LLM provider.
    API keys read from environment variables:
      ANTHROPIC_API_KEY  — for anthropic
      OPENAI_API_KEY     — for openai
      OPENROUTER_API_KEY — for openrouter
      GROQ_API_KEY       — for groq
    """
    if provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    elif provider in ("openai", "openrouter", "groq"):
        import httpx
        base_urls = {
            "openai":     "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "groq":       "https://api.groq.com/openai/v1",
        }
        env_keys = {
            "openai":     "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq":       "GROQ_API_KEY",
        }
        api_key = os.getenv(env_keys[provider])
        if not api_key:
            raise ValueError(f"{env_keys[provider]} not set in .env")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://strategy-builder"

        response = httpx.post(
            f"{base_urls[provider]}/chat/completions",
            headers=headers,
            json={
                "model":      model,
                "max_tokens": 1200,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── Strategy Creation Agent Routes ───────────────────────────────────────────

from agents.strategy_creation_agent import run_strategy_creation_agent

class StrategyCreationRequest(BaseModel):
    goal: str
    timeframe: str = "1Hour"
    market_context: Optional[dict] = {}
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    source: str = "human"   # "human" | "hypothesis_agent" (Phase 2)


@app.post("/api/agent/create-strategy")
def agent_create_strategy(req: StrategyCreationRequest):
    """
    Strategy Creation Agent — converts plain English to validated DSL.

    Phase 1: user describes idea in natural language
    Phase 2: Hypothesis Agent passes structured context (same endpoint)

    Returns strategy JSON ready to load into Strategy Builder.
    """
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal is required")

    result = run_strategy_creation_agent(
        goal=req.goal,
        timeframe=req.timeframe,
        market_context=req.market_context or {},
        provider=req.provider,
        model=req.model,
        source=req.source,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail={
                "error":           result.get("error"),
                "steps_completed": result.get("steps_completed", []),
            }
        )

    return result


# ─── Market Intelligence Agent Routes ────────────────────────────────────────

from agents.market_intelligence_agent import run_market_intelligence_agent
from agents.hypothesis_generator      import run_hypothesis_generator
from agents.morning_pipeline          import run_morning_pipeline


class MorningPipelineRequest(BaseModel):
    provider: str = "anthropic"
    model:    str = "claude-sonnet-4-6"
    symbol:   str = "SPY"
    dry_run:  bool = False


@app.get("/api/agent/market-brief")
def get_market_brief():
    """
    Run Market Intelligence Agent — returns current regime classification.
    Layer 1: Price-based regime from Alpaca
    Layer 2: Economic calendar check
    """
    result = run_market_intelligence_agent()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@app.post("/api/agent/hypotheses")
def generate_hypotheses(req: MorningPipelineRequest):
    """
    Run Hypothesis Generator on current market conditions.
    Returns 2-3 strategy hypotheses ready for the Creation Agent.
    """
    mi_result = run_market_intelligence_agent()
    if not mi_result["success"]:
        raise HTTPException(status_code=500, detail=mi_result.get("error"))

    hyp_result = run_hypothesis_generator(
        brief=mi_result["brief"],
        provider=req.provider,
        model=req.model,
    )
    return {
        "brief":      mi_result["brief"],
        "hypotheses": hyp_result,
    }


@app.post("/api/agent/morning-pipeline")
def morning_pipeline(req: MorningPipelineRequest):
    """
    Run full morning pipeline:
    Market Intelligence → Hypotheses → Strategy Creation → Backtest → Registry
    """
    result = run_morning_pipeline(
        provider=req.provider,
        model=req.model,
        symbol=req.symbol,
        dry_run=req.dry_run,
    )
    return result


# ─── Symbol Universe Routes ───────────────────────────────────────────────────

@app.get("/api/universe/symbols")
def list_symbols(
    strategy_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List symbols in universe, optionally filtered by strategy type."""
    from sqlalchemy import text
    if strategy_type:
        rows = db.execute(text("""
            SELECT su.symbol, su.company_name, su.sector,
                   su.trend_score, su.avg_atr_pct_30d, su.momentum_score,
                   su.avg_volume_30d, su.last_classified,
                   ssm.confidence, ssm.assigned_by
            FROM symbol_universe su
            JOIN symbol_strategy_mapping ssm ON ssm.symbol = su.symbol
            WHERE ssm.strategy_type = :st AND ssm.is_active = TRUE AND su.is_active = TRUE
            ORDER BY ssm.confidence DESC
        """), {"st": strategy_type}).fetchall()
    else:
        rows = db.execute(text("""
            SELECT symbol, company_name, sector,
                   trend_score, avg_atr_pct_30d, momentum_score,
                   avg_volume_30d, last_classified, NULL, NULL
            FROM symbol_universe WHERE is_active = TRUE
            ORDER BY market_cap_rank ASC NULLS LAST
        """)).fetchall()

    return {"symbols": [
        {
            "symbol":       r[0], "company_name": r[1], "sector": r[2],
            "trend_score":  float(r[3]) if r[3] else None,
            "atr_pct":      float(r[4]) if r[4] else None,
            "momentum":     float(r[5]) if r[5] else None,
            "volume":       float(r[6]) if r[6] else None,
            "last_classified": str(r[7]) if r[7] else None,
            "confidence":   float(r[8]) if r[8] else None,
            "assigned_by":  r[9],
        }
        for r in rows
    ]}


@app.post("/api/universe/classify")
def classify_symbols(max_symbols: int = 30):
    """
    Run the Symbol Classifier Agent.
    Computes trading characteristics for all symbols and updates strategy mappings.
    """
    from agents.symbol_classifier import run_symbol_classifier
    result = run_symbol_classifier(max_symbols=max_symbols)
    return result


@app.post("/api/universe/symbols")
def add_symbol(symbol: str, company_name: str = "", sector: str = "", db: Session = Depends(get_db)):
    """Add a new symbol to the universe."""
    from sqlalchemy import text
    db.execute(text("""
        INSERT INTO symbol_universe (symbol, company_name, sector)
        VALUES (:symbol, :company_name, :sector)
        ON CONFLICT (symbol) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector       = EXCLUDED.sector,
            is_active    = TRUE
    """), {"symbol": symbol.upper(), "company_name": company_name, "sector": sector})
    db.commit()
    return {"success": True, "symbol": symbol.upper()}
