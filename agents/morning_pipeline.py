"""
Morning Pipeline Orchestrator v2
Full autonomous strategy generation cycle with:
- Regime-aware strategy reuse (skip generation if recent approved strategies exist)
- Multi-symbol backtesting from symbol_universe DB table
- 180-day backtest window for sufficient trade sample
- ASCII-only log messages (Windows cp1252 compatible)

Schedule: GitHub Actions 7 AM ET daily Mon-Fri
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, date, timedelta

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_intelligence_agent import run_market_intelligence_agent
from agents.hypothesis_generator      import run_hypothesis_generator
from agents.strategy_creation_agent   import run_strategy_creation_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('morning_pipeline.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")

# Fallback universe if DB lookup fails
DEFAULT_UNIVERSE = {
    "trend_following":       ["SPY", "QQQ", "MSFT", "AAPL"],
    "momentum_breakout":     ["NVDA", "TSLA", "META", "AMD"],
    "mean_reversion":        ["SPY", "QQQ", "XLK"],
    "pullback_entry":        ["SPY", "QQQ", "AAPL", "MSFT"],
    "rsi_oversold_recovery": ["SPY", "QQQ", "AAPL"],
    "sma_crossover":         ["SPY", "QQQ", "MSFT"],
    "default":               ["SPY", "QQQ"],
}


# ─── Symbol universe from DB ──────────────────────────────────────────────────

def get_symbols_for_strategy(strategy_type: str, limit: int = 5) -> list[str]:
    """
    Fetch symbols for a strategy type from symbol_strategy_mapping DB table.
    Falls back to DEFAULT_UNIVERSE if DB not available.
    """
    try:
        from db.connection import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            rows = db.execute(text("""
                SELECT ssm.symbol
                FROM symbol_strategy_mapping ssm
                JOIN symbol_universe su ON su.symbol = ssm.symbol
                WHERE ssm.strategy_type = :strategy_type
                  AND ssm.is_active = TRUE
                  AND su.is_active  = TRUE
                ORDER BY ssm.confidence DESC, su.avg_volume_30d DESC NULLS LAST
                LIMIT :limit
            """), {"strategy_type": strategy_type, "limit": limit}).fetchall()

            symbols = [r[0] for r in rows]
            if symbols:
                logger.info(f"DB universe for {strategy_type}: {symbols}")
                return symbols
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"DB symbol lookup failed, using defaults: {e}")

    fallback = DEFAULT_UNIVERSE.get(strategy_type, DEFAULT_UNIVERSE["default"])
    logger.info(f"Fallback universe for {strategy_type}: {fallback}")
    return fallback


# ─── Regime reuse check ───────────────────────────────────────────────────────

def get_existing_strategies_for_regime(regime: str, days_fresh: int = 30) -> list[dict]:
    """
    Check if approved strategies exist for current regime backtested recently.
    Returns list of strategy dicts if found, empty list if generation needed.
    """
    try:
        from db.connection import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_fresh)
            rows = db.execute(text("""
                SELECT sm.strategy_id, sm.strategy_name, sm.status,
                       sm.regime_tag, sm.last_backtested_at
                FROM strategy_master sm
                WHERE sm.is_current   = TRUE
                  AND sm.status       IN ('approved', 'paper_trading')
                  AND sm.regime_tag   = :regime
                  AND (sm.last_backtested_at IS NULL OR sm.last_backtested_at > :cutoff)
                ORDER BY sm.created_at DESC
                LIMIT 5
            """), {"regime": regime, "cutoff": cutoff}).fetchall()

            return [
                {
                    "strategy_id":       str(r[0]),
                    "strategy_name":     r[1],
                    "status":            r[2],
                    "regime_tag":        r[3],
                    "last_backtested_at": str(r[4]) if r[4] else None,
                }
                for r in rows
            ]
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Regime check failed: {e}")
        return []


def tag_strategy_with_regime(strategy_id: str, regime: str) -> None:
    """Tag a strategy with the market regime it was designed for."""
    try:
        from db.connection import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            db.execute(text("""
                UPDATE strategy_master
                SET regime_tag = :regime, last_backtested_at = NOW()
                WHERE strategy_id = :sid AND is_current = TRUE
            """), {"regime": regime, "sid": strategy_id})
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not tag strategy with regime: {e}")


# ─── Backtest ─────────────────────────────────────────────────────────────────

def run_quick_backtest(
    strategy_json: dict,
    strategy_id:   str,
    symbol:        str = "SPY",
) -> dict:
    """
    Run 180-day backtest via registry endpoint so result is
    linked to the strategy in Neon — metrics show in Registry panel.
    """
    import requests

    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")

    payload = {
        "strategy_id":     strategy_id,
        "strategy":        strategy_json,
        "symbol":          symbol,
        "start_date":      start,
        "end_date":        end,
        "initial_capital": 10000,
        "commission_pct":  0.1,
        "slippage_pct":    0.05,
        "timeframe":       "1Hour",
        "position_sizing": "fixed",
    }

    try:
        res = requests.post(f"{API_BASE}/api/registry/backtest", json=payload, timeout=120)
        if res.ok:
            return res.json()
        return {"success": False, "error": res.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_multi_symbol_backtest(
    strategy_json: dict,
    strategy_id:   str,
    strategy_type: str,
    max_symbols:   int = 4,
) -> dict:
    """
    Test strategy across symbol universe for this strategy type.
    Returns best result by Sharpe + all results for comparison.
    """
    symbols = get_symbols_for_strategy(strategy_type, limit=max_symbols)
    all_results = []

    for symbol in symbols:
        logger.info(f"    Backtesting on {symbol}...")
        bt = run_quick_backtest(strategy_json, strategy_id, symbol=symbol)

        if bt.get("success"):
            m = bt.get("metrics", {})
            all_results.append({
                "symbol":        symbol,
                "sharpe":        m.get("sharpe_ratio", 0) or 0,
                "total_return":  m.get("total_return_pct", 0) or 0,
                "profit_factor": m.get("profit_factor", 0) or 0,
                "total_trades":  m.get("total_trades", 0) or 0,
                "max_drawdown":  m.get("max_drawdown_pct", 0) or 0,
                "backtest":      bt,
            })
            logger.info(
                f"    {symbol}: trades={m.get('total_trades',0)} "
                f"sharpe={m.get('sharpe_ratio',0):.2f} "
                f"return={m.get('total_return_pct',0):.1f}%"
            )
        else:
            logger.warning(f"    {symbol}: backtest failed — {bt.get('error','unknown')[:60]}")

    if not all_results:
        return {"best": None, "all_results": [], "symbols_tested": symbols}

    # Rank: primary by Sharpe, secondary by profit factor
    all_results.sort(key=lambda x: (x["sharpe"], x["profit_factor"]), reverse=True)
    best = all_results[0]

    logger.info(
        f"    Best symbol: {best['symbol']} "
        f"(Sharpe={best['sharpe']:.2f}, Return={best['total_return']:.1f}%)"
    )

    return {
        "best":           best,
        "all_results":    all_results,
        "symbols_tested": symbols,
    }


def save_to_registry(strategy_json: dict, notes: str) -> dict:
    """Save generated strategy to registry as draft."""
    import requests
    payload = {
        "strategy":   strategy_json,
        "notes":      notes,
        "created_by": "market_intelligence_agent",
    }
    try:
        res = requests.post(f"{API_BASE}/api/registry/strategies", json=payload, timeout=30)
        if res.ok:
            return res.json()
        return {"success": False, "error": res.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def is_strategy_viable(
    best_result: dict,
    min_sharpe: float = 0.3,
    min_profit_factor: float = 1.0,
) -> bool:
    """Filter out obviously weak strategies."""
    if not best_result:
        return False
    trades = best_result.get("total_trades", 0)
    if trades < 3:
        return True   # too few trades — keep for human review
    sharpe = best_result.get("sharpe", 0) or 0
    pf     = best_result.get("profit_factor", 0) or 0
    return sharpe >= min_sharpe or pf >= min_profit_factor


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_morning_pipeline(
    provider:    str  = "anthropic",
    model:       str  = "claude-sonnet-4-6",
    symbol:      str  = "SPY",
    dry_run:     bool = False,
    force_generate: bool = False,
) -> dict:
    """
    Full morning pipeline v2.

    Args:
        provider:       LLM provider
        model:          model name
        symbol:         fallback symbol if DB universe unavailable
        dry_run:        skip backtest and registry save
        force_generate: skip reuse check, always generate fresh strategies
    """
    start_time = datetime.now(timezone.utc)
    results = {
        "date":                 start_time.strftime("%Y-%m-%d"),
        "started_at":           start_time.isoformat(),
        "regime":               None,
        "reused_existing":      False,
        "existing_strategies":  [],
        "hypotheses_generated": 0,
        "strategies_created":   0,
        "strategies_viable":    0,
        "strategies_saved":     0,
        "strategies":           [],
        "errors":               [],
        "event_blackout":       False,
    }

    logger.info("=" * 70)
    logger.info(f"Morning Pipeline v2 started — {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 70)

    # ── Step 1: Market Intelligence ───────────────────────────────────────────
    logger.info("Step 1: Running Market Intelligence Agent...")
    mi_result = run_market_intelligence_agent()

    if not mi_result["success"]:
        results["errors"].append(f"Market Intelligence failed: {mi_result['error']}")
        return results

    brief                  = mi_result["brief"]
    regime                 = brief["regime"]
    results["regime"]      = regime
    results["event_blackout"] = brief.get("event_blackout", False)

    logger.info(f"Regime: {regime} (confidence: {brief['regime_confidence']:.0%})")
    logger.info(f"VIX: {brief['vix_level']} | Breadth: {brief['breadth']}")

    if brief.get("upcoming_events"):
        for ev in brief["upcoming_events"]:
            logger.info(f"Event: {ev['event']} in {ev['days_until']} days")

    if brief.get("event_blackout"):
        logger.info("Event blackout — pipeline paused")
        results["message"] = "Paused: high-impact event within 2 days"
        return results

    # ── Step 2: Reuse check ───────────────────────────────────────────────────
    if not force_generate:
        logger.info(f"Step 2: Checking existing strategies for regime '{regime}'...")
        existing = get_existing_strategies_for_regime(regime, days_fresh=30)

        if existing:
            logger.info(f"Found {len(existing)} existing approved strategies for {regime}:")
            for s in existing:
                logger.info(f"  - {s['strategy_name']} ({s['status']})")
            results["reused_existing"]     = True
            results["existing_strategies"] = existing
            logger.info("Reusing existing strategies — skipping generation")
            logger.info("(Use force_generate=True to override)")
            results["message"] = f"Reused {len(existing)} existing strategies for {regime} regime"
            return results
        else:
            logger.info(f"No recent approved strategies for {regime} — generating new ones")
    else:
        logger.info("Step 2: Skipped (force_generate=True)")

    # ── Step 3: Generate hypotheses ───────────────────────────────────────────
    logger.info("Step 3: Generating strategy hypotheses...")
    hyp_result = run_hypothesis_generator(brief, provider=provider, model=model)

    if not hyp_result["success"]:
        results["errors"].append(f"Hypothesis generation failed: {hyp_result['error']}")
        return results

    hypotheses = hyp_result["hypotheses"]
    results["hypotheses_generated"] = len(hypotheses)
    logger.info(f"Generated {len(hypotheses)} hypotheses")

    # ── Step 4-7: Create -> Save -> Multi-symbol backtest -> Viability ────────
    for i, hyp in enumerate(hypotheses, 1):
        logger.info(f"\nProcessing hypothesis {i}/{len(hypotheses)}: {hyp['strategy_type']}")
        logger.info(f"Goal: {hyp['goal'][:80]}...")

        strategy_result = {
            "hypothesis":    hyp,
            "strategy_json": None,
            "best_symbol":   None,
            "best_sharpe":   None,
            "all_symbols":   [],
            "viable":        False,
            "saved":         False,
            "registry_id":   None,
            "errors":        [],
        }

        # Step 4a: Strategy Creation Agent
        logger.info("  Creating strategy DSL...")
        try:
            creation = run_strategy_creation_agent(
                goal           = hyp["goal"],
                timeframe      = hyp.get("timeframe", "1Hour"),
                market_context = hyp.get("market_context", {}),
                provider       = provider,
                model          = model,
                source         = "hypothesis_agent",
            )
            if not creation["success"]:
                strategy_result["errors"].append(f"Creation failed: {creation.get('error')}")
                results["strategies"].append(strategy_result)
                continue

            strategy_json = creation["strategy_json"]
            strategy_result["strategy_json"] = strategy_json
            results["strategies_created"] += 1
            logger.info(f"  Strategy created: {strategy_json.get('strategy_name')}")

        except Exception as e:
            strategy_result["errors"].append(f"Creation exception: {e}")
            results["strategies"].append(strategy_result)
            continue

        # Step 4b: Save to registry FIRST (need strategy_id for backtest)
        notes_draft = (
            f"Auto-generated by Market Intelligence Agent on {results['date']}. "
            f"Regime: {regime} ({brief['regime_confidence']:.0%} confidence). "
            f"Type: {hyp['strategy_type']}. "
            f"Rationale: {hyp['rationale']}"
        )
        registry_id = ""
        try:
            save_result = save_to_registry(strategy_json, notes=notes_draft)
            if save_result.get("success"):
                registry_id                    = save_result["strategy"]["strategy_id"]
                strategy_result["saved"]       = True
                strategy_result["registry_id"] = registry_id
                results["strategies_saved"]   += 1
                logger.info(f"  Saved to registry: {registry_id}")
                # Tag with regime immediately
                tag_strategy_with_regime(registry_id, regime)
            else:
                strategy_result["errors"].append(f"Save failed: {save_result.get('error')}")
        except Exception as e:
            strategy_result["errors"].append(f"Save exception: {e}")

        if dry_run:
            strategy_result["viable"] = True
            results["strategies"].append(strategy_result)
            continue

        # Step 5: Multi-symbol backtest
        logger.info(f"  Running multi-symbol backtest (strategy type: {hyp['strategy_type']})...")
        try:
            mt = run_multi_symbol_backtest(
                strategy_json  = strategy_json,
                strategy_id    = registry_id,
                strategy_type  = hyp["strategy_type"],
                max_symbols    = 4,
            )

            strategy_result["all_symbols"] = [
                {
                    "symbol":       r["symbol"],
                    "sharpe":       r["sharpe"],
                    "total_return": r["total_return"],
                    "trades":       r["total_trades"],
                }
                for r in mt.get("all_results", [])
            ]

            if mt.get("best"):
                best = mt["best"]
                strategy_result["best_symbol"] = best["symbol"]
                strategy_result["best_sharpe"] = best["sharpe"]

                # Step 6: Viability check on best result
                viable = is_strategy_viable(best)
                strategy_result["viable"] = viable

                if viable:
                    results["strategies_viable"] += 1
                    logger.info(
                        f"  Viable on {best['symbol']}: "
                        f"Sharpe={best['sharpe']:.2f} "
                        f"Return={best['total_return']:.1f}% "
                        f"Trades={best['total_trades']}"
                    )
                    # Update registry notes with best symbol info
                    other_symbols = [
                        f"{r['symbol']}(s={r['sharpe']:.2f})"
                        for r in mt["all_results"][1:3]
                        if r["sharpe"] > 0.2
                    ]
                    tag_strategy_with_regime(registry_id, regime)
                else:
                    logger.info(
                        f"  Filtered: best={best['symbol']} "
                        f"sharpe={best['sharpe']:.2f} "
                        f"trades={best['total_trades']} "
                        f"(below viability threshold)"
                    )
            else:
                logger.warning("  No successful backtests across symbol universe")

        except Exception as e:
            strategy_result["errors"].append(f"Backtest exception: {e}")
            logger.error(f"  Backtest error: {e}")

        results["strategies"].append(strategy_result)

    # ── Summary ───────────────────────────────────────────────────────────────
    duration = (datetime.now(timezone.utc) - start_time).seconds
    results["completed_at"]      = datetime.now(timezone.utc).isoformat()
    results["duration_seconds"]  = duration

    logger.info("\n" + "=" * 70)
    logger.info("Morning Pipeline Complete")
    logger.info(f"  Regime:         {results['regime']}")
    logger.info(f"  Hypotheses:     {results['hypotheses_generated']}")
    logger.info(f"  Created:        {results['strategies_created']}")
    logger.info(f"  Saved:          {results['strategies_saved']}")
    logger.info(f"  Viable:         {results['strategies_viable']}")
    logger.info(f"  Duration:       {duration}s")
    logger.info("=" * 70)

    if results["strategies_saved"] > 0:
        logger.info("-> Check the Registry panel in app.html to review and approve strategies")
        for s in results["strategies"]:
            if s.get("best_symbol") and s.get("best_sharpe") is not None:
                others = [
                    f"{r['symbol']}({r['sharpe']:.2f})"
                    for r in s.get("all_symbols", [])[1:]
                    if r["sharpe"] > 0.1
                ]
                logger.info(
                    f"  - {s['strategy_json'].get('strategy_name','?')}: "
                    f"best={s['best_symbol']} sharpe={s['best_sharpe']:.2f}"
                    + (f", also works on: {', '.join(others)}" if others else "")
                )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Morning Pipeline v2")
    parser.add_argument("--provider",       default="anthropic")
    parser.add_argument("--model",          default="claude-sonnet-4-6")
    parser.add_argument("--symbol",         default="SPY")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--force-generate", action="store_true",
                        help="Skip reuse check, always generate fresh strategies")
    args = parser.parse_args()

    result = run_morning_pipeline(
        provider       = args.provider,
        model          = args.model,
        symbol         = args.symbol,
        dry_run        = args.dry_run,
        force_generate = args.force_generate,
    )
    print(json.dumps(result, indent=2, default=str))
