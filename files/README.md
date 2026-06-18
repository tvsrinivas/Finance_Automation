# Liquidity Candle Options Paper Trader

Automated options paper trading via GitHub Actions + Alpaca.

## Strategy
- **Symbols**: AAPL, SPY, WMT, ORCL, TSLA
- **Signal**: Hammer or Bullish Engulfing candle below `b_low` (liquidity candle bottom)
- **Entry**: Buy ATM call option at signal
- **Exit**: 11:05 AM ET hard stop (no overnight)

## Schedule (all times ET, weekdays only)
| Time  | Action |
|-------|--------|
| 9:35  | Check ATR + first candle liquidity |
| 9:40–11:00 | Scan every 5 min for signals |
| 11:05 | Force-close all open positions |

## Setup

### 1. Fork / clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Add GitHub Secrets
Go to **Settings → Secrets and variables → Actions → New repository secret**:
- `ALPACA_API_KEY` — your Alpaca Paper Trading API key
- `ALPACA_SECRET_KEY` — your Alpaca Paper Trading secret key

### 3. Enable GitHub Actions
Go to **Actions** tab → click **"I understand my workflows, go ahead and enable them"**

### 4. Test manually
Trigger any workflow manually via Actions → select workflow → "Run workflow"

### 5. Local testing
```bash
pip install -r requirements.txt
cp .env.example .env      # fill in your keys
mkdir state

python live_trader.py --mode open   # simulate 9:35
python live_trader.py --mode scan   # simulate signal scan
python live_trader.py --mode exit   # simulate 11:05 exit
```

## Configuration (live_trader.py top section)
```python
SYMBOLS    = ["AAPL", "SPY", "WMT", "ORCL", "TSLA"]  # add/remove symbols
QTY        = 1       # contracts per trade
OTM_STEPS  = 0       # 0=ATM, 1=1-strike OTM, 2=2-strikes OTM
MIN_DTE    = 1       # minimum days to expiry
EXIT_TIME  = 11:05   # force-exit time
```

## State files
Each symbol writes a `state/<SYMBOL>.json` after each run.
These are committed back to the repo so the next workflow run picks up the state.

```json
{
  "date": "2026-06-17",
  "symbol": "AAPL",
  "atr": 5.234,
  "a_high": 212.45,
  "b_low": 211.10,
  "liquidity": true,
  "in_trade": true,
  "occ_symbol": "AAPL260619C00212000",
  "opt_entry": 2.35,
  "strike": 212.0,
  "expiry": "2026-06-19"
}
```

## Important notes
- GitHub Actions cron has ~1 min scheduling jitter — acceptable for 5-min bars
- Free tier GitHub Actions: 2,000 minutes/month — this strategy uses ~17 runs/day × ~1 min = ~17 min/day, well within limits
- DST change: adjust UTC offsets in YAML files when clocks change (EDT=UTC-4, EST=UTC-5)
