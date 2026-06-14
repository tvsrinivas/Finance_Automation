-- ─────────────────────────────────────────────────────────────────────────────
-- Symbol Universe Tables
-- Run this in Neon SQL Editor
-- ─────────────────────────────────────────────────────────────────────────────

-- Strategy master additions
ALTER TABLE strategy_master ADD COLUMN IF NOT EXISTS regime_tag VARCHAR(50);
ALTER TABLE strategy_master ADD COLUMN IF NOT EXISTS last_backtested_at TIMESTAMPTZ;

-- Symbol universe — one row per tradeable symbol
CREATE TABLE IF NOT EXISTS symbol_universe (
    symbol              VARCHAR(20)  PRIMARY KEY,
    company_name        VARCHAR(200),
    sector              VARCHAR(100),
    market_cap_rank     INTEGER,         -- rank in S&P 500 (1=largest)
    avg_volume_30d      NUMERIC(20,2),   -- avg daily volume
    avg_atr_pct_30d     NUMERIC(8,4),    -- ATR as % of price (volatility)
    trend_score         NUMERIC(5,4),    -- 0-1: how often above SMA200
    mean_reversion_score NUMERIC(5,4),   -- 0-1: RSI oscillation frequency
    momentum_score      NUMERIC(5,4),    -- 0-1: trend + vol combo
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    last_classified     TIMESTAMPTZ,
    added_at            TIMESTAMPTZ DEFAULT NOW(),
    notes               TEXT
);

-- Symbol to strategy type mapping — many-to-many
CREATE TABLE IF NOT EXISTS symbol_strategy_mapping (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL REFERENCES symbol_universe(symbol),
    strategy_type   VARCHAR(50) NOT NULL,
    confidence      NUMERIC(5,4) NOT NULL DEFAULT 0.5,  -- 0-1
    assigned_by     VARCHAR(50) NOT NULL DEFAULT 'classifier_agent',
    assigned_at     TIMESTAMPTZ DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE(symbol, strategy_type)
);

-- Index for fast lookup by strategy type
CREATE INDEX IF NOT EXISTS idx_ssm_strategy_type 
    ON symbol_strategy_mapping(strategy_type) 
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_ssm_symbol
    ON symbol_strategy_mapping(symbol)
    WHERE is_active = TRUE;

-- Seed: Top 30 S&P 500 symbols with manual classification
-- (classifier agent will override these with computed values)
INSERT INTO symbol_universe (symbol, company_name, sector, market_cap_rank) VALUES
    ('SPY',  'SPDR S&P 500 ETF',              'ETF',          1),
    ('QQQ',  'Invesco QQQ Trust',              'ETF',          2),
    ('AAPL', 'Apple Inc',                      'Technology',   3),
    ('MSFT', 'Microsoft Corporation',           'Technology',   4),
    ('NVDA', 'NVIDIA Corporation',              'Technology',   5),
    ('AMZN', 'Amazon.com Inc',                 'Consumer',     6),
    ('META', 'Meta Platforms Inc',             'Technology',   7),
    ('GOOGL','Alphabet Inc',                   'Technology',   8),
    ('TSLA', 'Tesla Inc',                      'Automotive',   9),
    ('BRK.B','Berkshire Hathaway',             'Financials',   10),
    ('JPM',  'JPMorgan Chase',                 'Financials',   11),
    ('LLY',  'Eli Lilly',                      'Healthcare',   12),
    ('V',    'Visa Inc',                       'Financials',   13),
    ('UNH',  'UnitedHealth Group',             'Healthcare',   14),
    ('XOM',  'Exxon Mobil',                    'Energy',       15),
    ('MA',   'Mastercard Inc',                 'Financials',   16),
    ('JNJ',  'Johnson & Johnson',              'Healthcare',   17),
    ('HD',   'Home Depot',                     'Consumer',     18),
    ('PG',   'Procter & Gamble',               'Consumer',     19),
    ('COST', 'Costco Wholesale',               'Consumer',     20),
    ('ABBV', 'AbbVie Inc',                     'Healthcare',   21),
    ('MRK',  'Merck & Co',                     'Healthcare',   22),
    ('CVX',  'Chevron Corporation',            'Energy',       23),
    ('KO',   'Coca-Cola Company',              'Consumer',     24),
    ('BAC',  'Bank of America',                'Financials',   25),
    ('AMD',  'Advanced Micro Devices',         'Technology',   26),
    ('NFLX', 'Netflix Inc',                    'Communication',27),
    ('WMT',  'Walmart Inc',                    'Consumer',     28),
    ('CRM',  'Salesforce Inc',                 'Technology',   29),
    ('PEP',  'PepsiCo Inc',                    'Consumer',     30),
    -- Sector ETFs for regime context
    ('XLK',  'Technology Select SPDR',         'ETF-Sector',   NULL),
    ('XLE',  'Energy Select SPDR',             'ETF-Sector',   NULL),
    ('XLF',  'Financial Select SPDR',          'ETF-Sector',   NULL),
    ('XLV',  'Health Care Select SPDR',        'ETF-Sector',   NULL),
    ('XLI',  'Industrial Select SPDR',         'ETF-Sector',   NULL),
    ('XLP',  'Consumer Staples SPDR',          'ETF-Sector',   NULL),
    ('XLU',  'Utilities Select SPDR',          'ETF-Sector',   NULL)
ON CONFLICT (symbol) DO NOTHING;

-- Default strategy mappings (will be refined by classifier agent)
INSERT INTO symbol_strategy_mapping (symbol, strategy_type, confidence, assigned_by) VALUES
    -- Trend following: liquid ETFs and mega-caps
    ('SPY',  'trend_following',   0.9, 'default'),
    ('QQQ',  'trend_following',   0.9, 'default'),
    ('MSFT', 'trend_following',   0.8, 'default'),
    ('AAPL', 'trend_following',   0.8, 'default'),
    ('GOOGL','trend_following',   0.75,'default'),
    -- Momentum breakout: high volatility growth
    ('NVDA', 'momentum_breakout', 0.9, 'default'),
    ('TSLA', 'momentum_breakout', 0.9, 'default'),
    ('META', 'momentum_breakout', 0.85,'default'),
    ('AMD',  'momentum_breakout', 0.85,'default'),
    ('NFLX', 'momentum_breakout', 0.8, 'default'),
    ('AMZN', 'momentum_breakout', 0.75,'default'),
    -- Mean reversion: stable large caps and ETFs
    ('SPY',  'mean_reversion',    0.8, 'default'),
    ('QQQ',  'mean_reversion',    0.75,'default'),
    ('XLK',  'mean_reversion',    0.75,'default'),
    ('KO',   'mean_reversion',    0.8, 'default'),
    ('PG',   'mean_reversion',    0.8, 'default'),
    ('JNJ',  'mean_reversion',    0.75,'default'),
    -- Pullback entry: quality uptrend stocks
    ('SPY',  'pullback_entry',    0.9, 'default'),
    ('QQQ',  'pullback_entry',    0.85,'default'),
    ('AAPL', 'pullback_entry',    0.85,'default'),
    ('MSFT', 'pullback_entry',    0.85,'default'),
    ('COST', 'pullback_entry',    0.8, 'default'),
    ('V',    'pullback_entry',    0.8, 'default'),
    -- RSI oversold recovery: high-volume liquid names
    ('SPY',  'rsi_oversold_recovery', 0.85,'default'),
    ('QQQ',  'rsi_oversold_recovery', 0.8, 'default'),
    ('AAPL', 'rsi_oversold_recovery', 0.8, 'default'),
    ('NVDA', 'rsi_oversold_recovery', 0.75,'default'),
    -- SMA crossover: trending ETFs and stable large caps
    ('SPY',  'sma_crossover',    0.9, 'default'),
    ('QQQ',  'sma_crossover',    0.85,'default'),
    ('MSFT', 'sma_crossover',    0.8, 'default'),
    ('AAPL', 'sma_crossover',    0.75,'default')
ON CONFLICT (symbol, strategy_type) DO NOTHING;
