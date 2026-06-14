"""
Hypothesis Generator
Takes a MarketBrief from the Market Intelligence Agent and generates
2-3 structured StrategyIntent objects for the Strategy Creation Agent.

This is the bridge between market conditions and tradeable strategies.
"""

import os
import json
import logging
import re
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _call_llm(prompt: str, provider: str = "anthropic", model: str = "claude-sonnet-4-6") -> str:
    """Call LLM — same pattern as other agents."""
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
    else:
        import httpx
        base_urls = {"openai":"https://api.openai.com/v1","openrouter":"https://openrouter.ai/api/v1","groq":"https://api.groq.com/openai/v1"}
        env_keys  = {"openai":"OPENAI_API_KEY","openrouter":"OPENROUTER_API_KEY","groq":"GROQ_API_KEY"}
        api_key   = os.getenv(env_keys.get(provider,"OPENAI_API_KEY"))
        headers   = {"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
        resp = httpx.post(f"{base_urls[provider]}/chat/completions",headers=headers,
            json={"model":model,"max_tokens":2000,"messages":[{"role":"user","content":prompt}]},timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


AVAILABLE_INDICATORS = [
    "SMA (Simple Moving Average) — trend direction, crossover signals",
    "EMA (Exponential Moving Average) — faster trend signals",
    "RSI (Relative Strength Index, 0-100) — momentum, oversold/overbought",
    "MACD (macd_line, signal_line, histogram) — momentum direction, crossovers",
    "ATR (Average True Range) — volatility measurement, stop sizing",
    "PRICE (close/open/high/low) — compare to moving averages or levels",
    "VOLUME — unusual activity detection",
    "WEEK52_HIGH — 52-week high breakout detection",
    "WEEK52_LOW — 52-week low breakdown detection",
]


def generate_hypotheses(
    brief: dict,
    num_hypotheses: int = 2,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """
    Generate strategy hypotheses from a market brief.

    Each hypothesis is a StrategyIntent object compatible with
    the Strategy Creation Agent's input format.

    Returns list of hypothesis dicts each containing:
      - goal: plain English strategy description
      - rationale: why this makes sense given current conditions
      - strategy_type: what kind of strategy this is
      - timeframe: recommended timeframe
      - market_context: the regime data for the creation agent
    """
    # Don't generate if in event blackout
    if brief.get("event_blackout"):
        logger.info("Event blackout — no hypotheses generated")
        return []

    regime         = brief.get("regime", "unknown")
    confidence     = brief.get("regime_confidence", 0.5)
    recommended    = brief.get("recommended_strategy_types", [])
    avoid          = brief.get("avoid_strategy_types", [])
    hints          = brief.get("strategy_hints", [])
    leading        = brief.get("leading_sectors", [])
    lagging        = brief.get("lagging_sectors", [])
    vix_level      = brief.get("vix_level", "medium")
    spy_vs_sma200  = brief.get("spy_vs_sma200", "unknown")
    spy_vs_sma50   = brief.get("spy_vs_sma50", "unknown")
    spy_rsi        = brief.get("spy_rsi")
    momentum_20d   = brief.get("momentum_20d", 0)
    growth_leading = brief.get("growth_leading")
    breadth        = brief.get("breadth", "mixed")
    upcoming       = brief.get("upcoming_events", [])
    pos_modifier   = brief.get("position_size_modifier", 1.0)
    sl_modifier    = brief.get("stop_loss_modifier", 1.0)

    sector_map = {
        "XLK": "Technology", "XLE": "Energy", "XLF": "Financials",
        "XLV": "Healthcare", "XLI": "Industrials",
        "XLP": "Consumer Staples", "XLU": "Utilities",
    }
    leading_names  = [sector_map.get(s, s) for s in leading]
    lagging_names  = [sector_map.get(s, s) for s in lagging]

    prompt = f"""You are a systematic trading strategist. Based on current market conditions,
generate {num_hypotheses} specific trading strategy hypotheses for SPY on a 1-hour timeframe.

CURRENT MARKET CONDITIONS:
- Date: {brief.get('date')}
- Market Regime: {regime} (confidence: {confidence:.0%})
- SPY Price: ${brief.get('spy_price')}
- SPY vs SMA50: {spy_vs_sma50}
- SPY vs SMA200: {spy_vs_sma200}
- SPY RSI(14): {spy_rsi}
- 20-day momentum: {momentum_20d:+.1f}%
- Volatility (VIX proxy): {vix_level}
- Market breadth: {breadth}
- Trend strength: {brief.get('trend_strength')}
- Leading sectors: {', '.join(leading_names) or 'None'}
- Lagging sectors: {', '.join(lagging_names) or 'None'}
- Growth vs Value: {'Growth leading' if growth_leading else 'Value leading' if growth_leading is False else 'Neutral'}
- Upcoming events: {', '.join([e['event'] + ' in ' + str(e['days_until']) + 'd' for e in upcoming]) or 'None in next 3 days'}

REGIME GUIDANCE:
- Recommended strategy types: {', '.join(recommended)}
- Avoid: {', '.join(avoid)}
- Context hints: {'; '.join(hints)}

AVAILABLE INDICATORS (only these can be used):
{chr(10).join('- ' + ind for ind in AVAILABLE_INDICATORS)}

RISK ADJUSTMENTS FOR THIS REGIME:
- Position size: {pos_modifier:.0%} of normal
- Stop loss: {sl_modifier:.0%} of normal width

Generate {num_hypotheses} hypotheses. Each must:
1. Be appropriate for the current regime
2. Only reference indicators from the available list
3. Be specific about entry and exit logic
4. Include realistic risk parameters adjusted for current conditions
5. Explain WHY this works in the current regime

Return ONLY this JSON (no other text):
{{
  "hypotheses": [
    {{
      "goal": "Plain English description of the complete strategy — entry conditions, exit conditions, and why",
      "strategy_type": "one of: {', '.join(recommended)}",
      "rationale": "Why this strategy fits today's market conditions specifically",
      "timeframe": "1Hour",
      "risk_notes": "Stop loss %, take profit %, position sizing note",
      "confidence": 0.0-1.0,
      "market_context": {{
        "regime": "{regime}",
        "regime_confidence": {confidence},
        "spy_vs_sma200": "{spy_vs_sma200}",
        "spy_vs_sma50": "{spy_vs_sma50}",
        "vix_level": "{vix_level}",
        "leading_sectors": {json.dumps(leading_names)},
        "position_size_modifier": {pos_modifier},
        "stop_loss_modifier": {sl_modifier}
      }}
    }}
  ]
}}"""

    logger.info(f"Generating {num_hypotheses} hypotheses for {regime} regime...")
    text  = _call_llm(prompt, provider, model)
    clean = re.sub(r'```json|```', '', text).strip()

    # Find JSON object
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)

    parsed     = json.loads(clean)
    hypotheses = parsed.get("hypotheses", [])

    logger.info(f"Generated {len(hypotheses)} hypotheses")
    return hypotheses


def run_hypothesis_generator(
    brief: dict,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Main entrypoint. Takes market brief, returns strategy hypotheses.
    """
    try:
        hypotheses = generate_hypotheses(brief, num_hypotheses=2, provider=provider, model=model)
        return {
            "success":    True,
            "hypotheses": hypotheses,
            "count":      len(hypotheses),
            "regime":     brief.get("regime"),
            "date":       brief.get("date"),
        }
    except Exception as e:
        logger.error(f"Hypothesis Generator failed: {e}")
        return {"success": False, "error": str(e), "hypotheses": []}
