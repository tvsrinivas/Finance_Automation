"""
Signal Engine — evaluates entry and exit conditions at each bar.
Handles AND/OR groups, cross operators, scalar vs indicator RHS.
No look-ahead: signals at bar i use only data from bar i and earlier.
Orders execute at open of bar i+1.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


# ─── Operator evaluation ──────────────────────────────────────────────────────

def evaluate_operator(
    operator: str,
    left_val: float,
    right_val: float,
    left_prev: float = None,
    right_prev: float = None,
) -> bool:
    """Evaluate a single comparison between two values."""
    if np.isnan(left_val) or np.isnan(right_val):
        return False

    if operator == ">":   return left_val >  right_val
    if operator == "<":   return left_val <  right_val
    if operator == ">=":  return left_val >= right_val
    if operator == "<=":  return left_val <= right_val
    if operator == "==":  return abs(left_val - right_val) < 1e-9

    # Cross operators need previous bar values
    if operator in ("cross_above", "cross_below"):
        if left_prev is None or right_prev is None:
            return False
        if np.isnan(left_prev) or np.isnan(right_prev):
            return False
        if operator == "cross_above":
            return left_prev <= right_prev and left_val > right_val
        if operator == "cross_below":
            return left_prev >= right_prev and left_val < right_val

    logger.warning(f"Unknown operator: {operator}")
    return False


# ─── Condition evaluation ─────────────────────────────────────────────────────

def evaluate_condition(
    condition: dict,
    computed_indicators: dict,
    bar_idx: int,
) -> bool:
    """
    Evaluate a single condition at bar_idx.
    condition format:
        {
            "left": "indicator_id",
            "operator": ">",
            "right": {"type": "scalar", "value": 40}
                   or {"type": "indicator", "ref": "other_id"}
        }
    """
    from .indicators import get_indicator_value

    left_id  = condition.get("left")
    operator = condition.get("operator", "")
    right    = condition.get("right", {})

    # Get left value
    left_val  = get_indicator_value(computed_indicators, left_id, bar_idx)
    left_prev = get_indicator_value(computed_indicators, left_id, bar_idx - 1) if bar_idx > 0 else np.nan

    # Get right value
    if right.get("type") == "scalar":
        right_val  = float(right.get("value", np.nan))
        right_prev = right_val  # scalar doesn't change bar to bar

    elif right.get("type") == "indicator":
        right_id   = right.get("ref") or right.get("id")
        right_val  = get_indicator_value(computed_indicators, right_id, bar_idx)
        right_prev = get_indicator_value(computed_indicators, right_id, bar_idx - 1) if bar_idx > 0 else np.nan

    else:
        logger.warning(f"Unknown RHS type: {right.get('type')}")
        return False

    return evaluate_operator(operator, left_val, right_val, left_prev, right_prev)


# ─── Rule group evaluation ────────────────────────────────────────────────────

def evaluate_group(
    group: dict,
    conditions: dict,
    computed_indicators: dict,
    bar_idx: int,
) -> bool:
    """
    Recursively evaluate a rule group (AND/OR) at bar_idx.
    Supports nested groups up to 2 levels deep.
    """
    logic      = group.get("logic", "AND").upper()
    cond_ids   = group.get("conditions", [])
    sub_groups = group.get("groups", [])

    results = []

    # Evaluate direct conditions
    for cid in cond_ids:
        cond = conditions.get(cid)
        if cond is None:
            logger.warning(f"Condition '{cid}' not found")
            results.append(False)
            continue
        results.append(evaluate_condition(cond, computed_indicators, bar_idx))

    # Evaluate nested groups recursively
    for sub_group in sub_groups:
        results.append(evaluate_group(sub_group, conditions, computed_indicators, bar_idx))

    if not results:
        return False

    if logic == "AND":
        return all(results)
    elif logic == "OR":
        return any(results)
    else:
        logger.warning(f"Unknown logic operator: {logic}")
        return False


# ─── Full signal scan ─────────────────────────────────────────────────────────

def generate_signals(
    df_len: int,
    entry_rules: dict,
    exit_rules: dict,
    conditions: dict,
    computed_indicators: dict,
    warmup_bars: int = 0,
) -> list[dict]:
    """
    Walk all bars and generate entry/exit signals.
    Returns list of {bar_idx, signal: "entry"|"exit"}.

    Signals are generated at bar i.
    Execution happens at open of bar i+1 (handled by position manager).
    """
    signals = []

    for i in range(warmup_bars, df_len):
        entry_signal = evaluate_group(entry_rules, conditions, computed_indicators, i)
        exit_signal  = evaluate_group(exit_rules,  conditions, computed_indicators, i)

        if entry_signal:
            signals.append({"bar_idx": i, "signal": "entry"})
        if exit_signal:
            signals.append({"bar_idx": i, "signal": "exit"})

    return signals
