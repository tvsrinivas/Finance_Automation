"""
Strategy Creation Agent — Phase 1
Converts plain English trading ideas into validated strategy DSL.

Flow:
  1. Intent Parser     — extract structured intent from free text
  2. Indicator Mapper  — map intent to available library indicators
  3. DSL Generator     — build complete strategy JSON
  4. Validator         — run existing validation engine
  5. Self-Correction   — fix errors and retry (max 3 attempts)
  6. Explainer         — produce plain English explanation

Phase 2 compatible: accepts structured StrategyIntent directly
from the Strategy Hypothesis Agent without changes.
"""

import json
import os
import re
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─── Indicator Library Spec (source of truth for the agent) ──────────────────

INDICATOR_LIBRARY = {
    "SMA": {
        "description": "Simple Moving Average — smooths price over N bars. Use for trend direction.",
        "params": {"period": {"type": "int", "min": 2, "max": 500, "default": 20}, "source": {"type": "enum", "options": ["close","open","high","low"], "default": "close"}},
        "output": "line",
        "use_cases": ["trend direction", "support/resistance", "crossover signals"],
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["SMA", "EMA", "PRICE", "WEEK52_HIGH", "WEEK52_LOW", "scalar"],
    },
    "EMA": {
        "description": "Exponential Moving Average — like SMA but weights recent prices more. Faster to react.",
        "params": {"period": {"type": "int", "min": 2, "max": 500, "default": 20}, "source": {"type": "enum", "options": ["close","open","high","low"], "default": "close"}},
        "output": "line",
        "use_cases": ["trend direction", "faster crossover signals", "dynamic support"],
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["SMA", "EMA", "PRICE", "WEEK52_HIGH", "WEEK52_LOW", "scalar"],
    },
    "RSI": {
        "description": "Relative Strength Index — momentum oscillator 0-100. Below 30=oversold, above 70=overbought.",
        "params": {"period": {"type": "int", "min": 2, "max": 100, "default": 14}, "source": {"type": "enum", "options": ["close","open","high","low"], "default": "close"}},
        "output": "line",
        "output_range": {"min": 0, "max": 100},
        "use_cases": ["oversold/overbought detection", "momentum recovery", "divergence"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["scalar"],
        "rhs_range": {"min": 0, "max": 100},
    },
    "MACD": {
        "description": "MACD — trend-following momentum. Components: macd_line, signal_line, histogram.",
        "params": {"fast": {"type": "int", "default": 12}, "slow": {"type": "int", "default": 26}, "signal": {"type": "int", "default": 9}},
        "output": "composite",
        "components": {"macd_line": "crossable_line", "signal_line": "crossable_line", "histogram": "histogram"},
        "use_cases": ["momentum direction", "trend reversals", "crossover signals"],
        "valid_operators": {"macd_line": [">", "<", "cross_above", "cross_below"], "signal_line": [">", "<", "cross_above", "cross_below"], "histogram": [">", "<", ">=", "<="]},
    },
    "ATR": {
        "description": "Average True Range — measures volatility. Use for stop loss sizing.",
        "params": {"period": {"type": "int", "min": 2, "max": 100, "default": 14}},
        "output": "line",
        "use_cases": ["volatility measurement", "stop loss sizing", "position sizing"],
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar", "line"],
    },
    "PRICE": {
        "description": "Raw price (open/high/low/close). Use to compare price to moving averages.",
        "params": {"source": {"type": "enum", "options": ["close","open","high","low"], "default": "close"}},
        "output": "scalar",
        "use_cases": ["price vs moving average", "breakout detection", "trend confirmation"],
        "valid_operators": [">", "<", ">=", "<=", "==", "cross_above", "cross_below"],
        "valid_rhs": ["SMA", "EMA", "WEEK52_HIGH", "WEEK52_LOW", "scalar"],
    },
    "VOLUME": {
        "description": "Trading volume. Use avg_period to compare current volume to average.",
        "params": {"avg_period": {"type": "int", "optional": True, "default": None}},
        "output": "scalar",
        "use_cases": ["volume confirmation", "unusual activity detection", "breakout confirmation"],
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar"],
    },
    "WEEK52_HIGH": {
        "description": "52-week highest price. Price breaking above = strong breakout signal.",
        "params": {},
        "output": "level",
        "use_cases": ["breakout detection", "resistance levels", "momentum confirmation"],
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar", "line"],
    },
    "WEEK52_LOW": {
        "description": "52-week lowest price. Price breaking below = breakdown signal.",
        "params": {},
        "output": "level",
        "use_cases": ["breakdown detection", "support levels", "mean reversion"],
        "valid_operators": [">", "<", ">=", "<="],
        "valid_rhs": ["scalar", "line"],
    },
    
    "PIVOT": {
        "description": "Floor trader pivot point: (Prior High + Prior Low + Prior Close) / 3. The central support/resistance level for the session.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": [
            "intraday support/resistance",
            "mean reversion targets",
            "trend bias (price above pivot = bullish bias)",
        ],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar", "PIVOT_R1", "PIVOT_R2", "PIVOT_R3", "PIVOT_S1", "PIVOT_S2", "PIVOT_S3"],
        "notes": "Recalculated each session using prior session's H/L/C. Zero look-ahead.",
    },
    "PIVOT_R1": {
        "description": "Resistance 1: 2 × Pivot − Prior Low. First upside target above pivot.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": ["first profit target for longs", "short entry fade level"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar", "PIVOT", "PIVOT_R2"],
        "notes": "Price rarely sustains above R3. Fade moves at R3.",
    },
    "PIVOT_R2": {
        "description": "Resistance 2: Pivot + (Prior High − Prior Low). Second upside target.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": ["second profit target for longs", "strong resistance"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar", "PIVOT", "PIVOT_R1", "PIVOT_R3"],
    },
    "PIVOT_R3": {
        "description": "Resistance 3: R1 + (Prior High − Prior Low). Extreme upside — markets rarely sustain above here; fade at this level.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": ["extreme resistance", "fade (counter-trend short) trigger"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar"],
    },
    "PIVOT_S1": {
        "description": "Support 1: 2 × Pivot − Prior High. First downside support below pivot.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": ["first buy-the-dip target", "first profit target for shorts"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar", "PIVOT", "PIVOT_S2"],
    },
    "PIVOT_S2": {
        "description": "Support 2: Pivot − (Prior High − Prior Low). Second downside support.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": ["deeper support level", "second profit target for shorts"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar", "PIVOT", "PIVOT_S1", "PIVOT_S3"],
    },
    "PIVOT_S3": {
        "description": "Support 3: S1 − (Prior High − Prior Low). Extreme downside — markets rarely sustain below here; fade at this level.",
        "params": {
            "period": {"type": "enum", "options": ["daily", "weekly", "monthly"], "default": "daily"}
        },
        "output": "line",
        "use_cases": ["extreme support", "fade (counter-trend long) trigger"],
        "valid_operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
        "valid_rhs": ["PRICE", "scalar"],
    },

}

# Common concepts that need library gap warnings
LIBRARY_GAPS = {
    "institutional": "Institutional buying/selling requires Level 2 order flow data not available. Approximated with volume anomaly.",
    "earnings": "Earnings momentum requires fundamental data not available. Consider using price breakout as proxy.",
    "sentiment": "Sentiment data not available. Consider using price momentum indicators instead.",
    "options flow": "Options flow data not available in this system.",
    "fundamental": "Fundamental metrics (P/E, EPS, revenue) not available. System uses technical indicators only.",
    "sector": "Sector relative strength requires multi-symbol data. Use 52-week high/low as proxy for relative strength.",
    "implied volatility": "Implied volatility (VIX-based) not available. Use ATR as a proxy for realized volatility.",
    "order flow": "Order flow data not available. Use volume as a proxy.",
}


# ─── LLM caller ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str, provider: str = "anthropic", model: str = "claude-sonnet-4-6") -> str:
    """Call LLM provider. Reads API keys from environment."""
    if provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=2000,
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
        resp = httpx.post(
            f"{base_urls[provider]}/chat/completions",
            headers=headers,
            json={"model": model, "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response."""
    clean = re.sub(r'```json|```', '', text).strip()
    # Try to find JSON object if there's surrounding text
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    return json.loads(clean)


# ─── Step 1: Intent Parser ────────────────────────────────────────────────────

def step1_parse_intent(
    goal: str,
    timeframe: str,
    market_context: dict,
    provider: str,
    model: str,
) -> dict:
    """
    Extract structured trading intent from free text.
    Identifies: entry conditions, exit conditions, risk preferences,
    any concepts that may fall outside the indicator library.
    """
    library_summary = "\n".join([
        f"- {name}: {spec['description']} | Use cases: {', '.join(spec['use_cases'])}"
        for name, spec in INDICATOR_LIBRARY.items()
    ])

    prompt = f"""You are a trading strategy analyst. Extract the structured intent from this trading idea.

AVAILABLE INDICATOR LIBRARY:
{library_summary}

USER'S TRADING IDEA:
"{goal}"

TIMEFRAME: {timeframe}
MARKET CONTEXT: {json.dumps(market_context) if market_context else "Not specified"}

Extract and return ONLY this JSON (no other text):
{{
  "strategy_name": "short descriptive name for this strategy",
  "core_thesis": "one sentence explaining the trading logic",
  "entry_concepts": [
    {{"concept": "what triggers entry", "indicator_mapping": "which indicator from library maps to this or null", "confidence": "high/medium/low"}}
  ],
  "exit_concepts": [
    {{"concept": "what triggers exit", "indicator_mapping": "which indicator maps to this or null", "confidence": "high/medium/low"}}
  ],
  "risk_preferences": {{
    "stop_loss_pct": number or null,
    "take_profit_pct": number or null,
    "position_size_pct": number or null
  }},
  "library_gaps": ["list any concepts that cannot be expressed with available indicators"],
  "suggested_proxies": {{"concept": "proxy explanation"}}
}}"""

    logger.info("Step 1: Parsing intent...")
    text   = _call_llm(prompt, provider, model)
    result = _parse_json(text)
    logger.info(f"Step 1 complete: {result.get('strategy_name')} — {len(result.get('entry_concepts', []))} entry concepts")
    return result


# ─── Step 2: Indicator Mapper ─────────────────────────────────────────────────

def step2_map_indicators(intent: dict, provider: str, model: str) -> dict:
    """
    Map parsed intent to concrete indicator configurations.
    Handles library gaps with negotiated proxies.
    Returns indicator definitions ready for DSL.
    """
    library_detail = json.dumps(INDICATOR_LIBRARY, indent=2)

    prompt = f"""You are a technical indicator specialist. Map this trading intent to concrete indicator configurations.

INDICATOR LIBRARY (full spec):
{library_detail}

PARSED INTENT:
{json.dumps(intent, indent=2)}

Rules:
1. Only use indicators from the library above — no others
2. Each indicator needs a unique ID (e.g. "sma200", "rsi14", "price_close")
3. For MACD specify the component: "macd_line", "signal_line", or "histogram"
4. For library gaps use the best available proxy and document it
5. Every condition needs: left indicator ID, operator, right (indicator ID or scalar value)
6. Operators must be valid for the indicator type
7. RSI right-hand side must be scalar between 0-100
8. MACD fast period must be less than slow period

Return ONLY this JSON:
{{
  "indicators": {{
    "indicator_id": {{
      "type": "SMA|EMA|RSI|MACD|ATR|PRICE|VOLUME|WEEK52_HIGH|WEEK52_LOW",
      "params": {{}},
      "component": "macd_line|signal_line|histogram (only for MACD)"
    }}
  }},
  "entry_conditions": [
    {{
      "id": "e1",
      "left": "indicator_id",
      "operator": ">|<|>=|<=|==|cross_above|cross_below",
      "right": {{"type": "scalar", "value": number}} or {{"type": "indicator", "ref": "indicator_id"}},
      "rationale": "why this condition captures the intent"
    }}
  ],
  "exit_conditions": [
    {{
      "id": "x1",
      "left": "indicator_id",
      "operator": "operator",
      "right": {{"type": "scalar", "value": number}} or {{"type": "indicator", "ref": "indicator_id"}},
      "rationale": "why this condition captures exit intent"
    }}
  ],
  "entry_logic": "AND|OR",
  "exit_logic": "AND|OR",
  "library_gaps_handled": {{
    "original_concept": "proxy used and why"
  }}
}}"""

    logger.info("Step 2: Mapping indicators...")
    text   = _call_llm(prompt, provider, model)
    result = _parse_json(text)
    logger.info(f"Step 2 complete: {len(result.get('indicators', {}))} indicators, {len(result.get('entry_conditions', []))} entry conditions")
    return result


# ─── Step 3: DSL Generator ────────────────────────────────────────────────────

def step3_build_dsl(intent: dict, mapping: dict, timeframe: str) -> dict:
    """
    Assemble the complete strategy DSL from intent + mapping.
    Pure Python — no LLM needed.
    """
    risk = intent.get("risk_preferences", {})

    # Build indicators block
    indicators = {}
    for ind_id, ind_def in mapping.get("indicators", {}).items():
        entry = {"type": ind_def["type"], "params": ind_def.get("params", {})}
        if ind_def.get("component"):
            entry["component"] = ind_def["component"]
        indicators[ind_id] = entry

    # Build conditions block
    conditions = {}
    for cond in mapping.get("entry_conditions", []):
        conditions[cond["id"]] = {
            "left":     cond["left"],
            "operator": cond["operator"],
            "right":    cond["right"],
        }
    for cond in mapping.get("exit_conditions", []):
        conditions[cond["id"]] = {
            "left":     cond["left"],
            "operator": cond["operator"],
            "right":    cond["right"],
        }

    entry_ids = [c["id"] for c in mapping.get("entry_conditions", [])]
    exit_ids  = [c["id"] for c in mapping.get("exit_conditions", [])]

    dsl = {
        "schema_version": "1.0",
        "strategy_name":  intent.get("strategy_name", "Agent Generated Strategy"),
        "indicators":     indicators,
        "conditions":     conditions,
        "entry_rules":    {"logic": mapping.get("entry_logic", "AND"), "conditions": entry_ids},
        "exit_rules":     {"logic": mapping.get("exit_logic", "OR"),  "conditions": exit_ids},
        "risk_rules": {
            "stop_loss":     {"type": "percent", "value": risk.get("stop_loss_pct")  or 5},
            "take_profit":   {"type": "percent", "value": risk.get("take_profit_pct") or 15},
            "position_size": {"type": "percent_of_capital", "value": risk.get("position_size_pct") or 95},
            "max_open_positions": 1,
        },
    }

    logger.info(f"Step 3: DSL built — {len(indicators)} indicators, {len(conditions)} conditions")
    return dsl


# ─── Step 4 + 5: Validator + Self-Correction ─────────────────────────────────

def step4_validate_and_correct(
    dsl: dict,
    mapping: dict,
    intent: dict,
    provider: str,
    model: str,
    max_attempts: int = 3,
) -> tuple[dict, dict, int]:
    """
    Validate DSL and self-correct up to max_attempts times.
    Returns: (corrected_dsl, validation_result, attempts_used)
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend_main import validate_strategy

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Step 4: Validation attempt {attempt}...")
        validation = validate_strategy(dsl)

        if validation["valid"]:
            logger.info(f"Step 4: Valid on attempt {attempt}")
            return dsl, validation, attempt

        errors = validation.get("errors", [])
        logger.info(f"Step 4: {len(errors)} errors — attempting self-correction")

        if attempt == max_attempts:
            break

        # Step 5: Self-correction
        error_summary = "\n".join([f"- {e['message']}" for e in errors])

        prompt = f"""You are fixing a trading strategy DSL that has validation errors.

CURRENT DSL:
{json.dumps(dsl, indent=2)}

VALIDATION ERRORS:
{error_summary}

INDICATOR LIBRARY RULES:
- RSI right-hand side must be scalar between 0 and 100
- MACD fast period must be less than slow period
- cross_above/cross_below only valid for line output type indicators
- histogram output type only supports >, <, >=, <=
- All condition IDs in entry_rules/exit_rules must exist in conditions block

Fix ALL errors and return the complete corrected DSL as ONLY valid JSON.
Do not change the strategy logic — only fix the structural/type errors."""

        text = _call_llm(prompt, provider, model)
        try:
            dsl = _parse_json(text)
            logger.info(f"Step 5: Self-correction attempt {attempt} applied")
        except Exception as e:
            logger.error(f"Step 5: Could not parse corrected DSL: {e}")
            break

    return dsl, validation, attempt


# ─── Step 6: Explainer ────────────────────────────────────────────────────────

def step6_explain(
    intent: dict,
    mapping: dict,
    dsl: dict,
    validation: dict,
    provider: str,
    model: str,
) -> dict:
    """
    Generate plain English explanation of the generated strategy.
    Explains each condition, identifies library gaps, suggests improvements.
    """
    entry_conditions = mapping.get("entry_conditions", [])
    exit_conditions  = mapping.get("exit_conditions", [])
    gaps_handled     = mapping.get("library_gaps_handled", {})
    gaps_original    = intent.get("library_gaps", [])

    prompt = f"""You are explaining a generated trading strategy to a trader.

ORIGINAL GOAL: "{intent.get('core_thesis', '')}"

ENTRY CONDITIONS CHOSEN:
{json.dumps([{"condition": c["id"], "rationale": c.get("rationale", "")} for c in entry_conditions], indent=2)}

EXIT CONDITIONS CHOSEN:
{json.dumps([{"condition": c["id"], "rationale": c.get("rationale", "")} for c in exit_conditions], indent=2)}

LIBRARY GAPS AND PROXIES USED:
{json.dumps(gaps_handled, indent=2) if gaps_handled else "None — all concepts mapped directly"}

Return ONLY this JSON:
{{
  "summary": "2-3 sentence plain English description of what this strategy does",
  "entry_explanation": [
    {{"condition_id": "e1", "plain_english": "what this condition checks in simple terms"}}
  ],
  "exit_explanation": [
    {{"condition_id": "x1", "plain_english": "what this exit condition checks"}}
  ],
  "library_gap_warnings": [
    {{"original": "what was asked", "proxy": "what was used", "limitation": "what this proxy misses"}}
  ],
  "improvement_suggestions": ["2-3 suggestions to make this strategy more robust"],
  "estimated_signal_frequency": "rough estimate of how often this strategy will trigger"
}}"""

    logger.info("Step 6: Generating explanation...")
    text   = _call_llm(prompt, provider, model)
    result = _parse_json(text)
    logger.info("Step 6: Explanation complete")
    return result


# ─── Main Agent Entrypoint ────────────────────────────────────────────────────

def run_strategy_creation_agent(
    goal: str,
    timeframe: str = "1Hour",
    market_context: dict = None,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    source: str = "human",
) -> dict:
    """
    Main entrypoint for the Strategy Creation Agent.

    Phase 1: accepts plain English from human user
    Phase 2: accepts structured context from Hypothesis Agent (same interface)

    Args:
        goal:            Plain English trading idea
        timeframe:       "1Hour" | "1Day" | "15Min"
        market_context:  Optional dict with regime, VIX etc (populated by Phase 2)
        provider:        LLM provider
        model:           Model name
        source:          "human" | "hypothesis_agent"

    Returns:
        {
            success:           bool,
            strategy_json:     dict,    ← ready to load into builder
            explanation:       dict,    ← plain English explanation
            library_gaps:      list,    ← concepts outside the library
            validation_result: dict,    ← validation outcome
            attempts:          int,     ← how many correction cycles used
            steps_completed:   list,    ← audit trail
            error:             str      ← only if success=False
        }
    """
    steps_completed = []
    market_context  = market_context or {}

    logger.info(f"Strategy Creation Agent started — source={source} goal='{goal[:60]}...'")

    try:
        # ── Step 1: Parse intent ──────────────────────────────────────────────
        intent = step1_parse_intent(goal, timeframe, market_context, provider, model)
        steps_completed.append("intent_parsed")

        # ── Step 2: Map indicators ────────────────────────────────────────────
        mapping = step2_map_indicators(intent, provider, model)
        steps_completed.append("indicators_mapped")

        # ── Step 3: Build DSL ─────────────────────────────────────────────────
        dsl = step3_build_dsl(intent, mapping, timeframe)
        steps_completed.append("dsl_built")

        # ── Steps 4+5: Validate + self-correct ───────────────────────────────
        dsl, validation, attempts = step4_validate_and_correct(
            dsl, mapping, intent, provider, model
        )
        steps_completed.append(f"validated_in_{attempts}_attempt{'s' if attempts>1 else ''}")

        # ── Step 6: Explain ───────────────────────────────────────────────────
        explanation = step6_explain(intent, mapping, dsl, validation, provider, model)
        steps_completed.append("explained")

        # ── Collect library gaps ──────────────────────────────────────────────
        all_gaps = []
        for gap in intent.get("library_gaps", []):
            proxy = mapping.get("library_gaps_handled", {}).get(gap)
            all_gaps.append({
                "concept":    gap,
                "available":  False,
                "proxy_used": proxy,
            })
        for warn in explanation.get("library_gap_warnings", []):
            all_gaps.append({
                "concept":    warn.get("original"),
                "available":  False,
                "proxy_used": warn.get("proxy"),
                "limitation": warn.get("limitation"),
            })

        logger.info(f"Agent complete — {len(steps_completed)} steps, valid={validation['valid']}, gaps={len(all_gaps)}")

        return {
            "success":           True,
            "strategy_json":     dsl,
            "explanation":       explanation,
            "library_gaps":      all_gaps,
            "validation_result": validation,
            "attempts":          attempts,
            "steps_completed":   steps_completed,
            "intent":            intent,
        }

    except Exception as e:
        logger.error(f"Agent failed at step {steps_completed[-1] if steps_completed else 'init'}: {e}")
        return {
            "success":         False,
            "error":           str(e),
            "steps_completed": steps_completed,
        }
