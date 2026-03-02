-- Migration 001: Initial schema for bayesbot
-- Requires: PostgreSQL with TimescaleDB extension

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";     -- for gen_random_uuid()

-- Raw source bars (1-second or whatever resolution we import)
CREATE TABLE IF NOT EXISTS raw_bars (
    timestamp    TIMESTAMPTZ NOT NULL,
    symbol       VARCHAR(20) NOT NULL,
    bar_type     VARCHAR(20) NOT NULL,
    open         NUMERIC(12,4) NOT NULL,
    high         NUMERIC(12,4) NOT NULL,
    low          NUMERIC(12,4) NOT NULL,
    close        NUMERIC(12,4) NOT NULL,
    volume       BIGINT NOT NULL,
    vwap         NUMERIC(12,4),
    metadata     JSONB DEFAULT '{}'
);
SELECT create_hypertable('raw_bars', 'timestamp',
                         chunk_time_interval => INTERVAL '1 week',
                         if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_bars_symbol_time
    ON raw_bars (symbol, timestamp DESC);

-- Constructed dollar bars — the primary data the model operates on
CREATE TABLE IF NOT EXISTS dollar_bars (
    timestamp       TIMESTAMPTZ NOT NULL,
    bar_start       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    bar_index       INTEGER NOT NULL,
    open            NUMERIC(12,4) NOT NULL,
    high            NUMERIC(12,4) NOT NULL,
    low             NUMERIC(12,4) NOT NULL,
    close           NUMERIC(12,4) NOT NULL,
    volume          BIGINT NOT NULL,
    vwap            NUMERIC(12,4),
    dollar_volume   NUMERIC(16,2) NOT NULL,
    tick_count      INTEGER,
    buy_volume      BIGINT DEFAULT 0,
    sell_volume     BIGINT DEFAULT 0,
    threshold_used  NUMERIC(16,2),
    metadata        JSONB DEFAULT '{}'
);
SELECT create_hypertable('dollar_bars', 'timestamp',
                         chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_dollar_bars_symbol_time
    ON dollar_bars (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_dollar_bars_symbol_index
    ON dollar_bars (symbol, bar_index DESC);

-- Feature vectors computed from dollar bars
CREATE TABLE IF NOT EXISTS feature_vectors (
    timestamp            TIMESTAMPTZ NOT NULL,
    symbol               VARCHAR(20) NOT NULL,
    bar_index            INTEGER NOT NULL,
    features             JSONB NOT NULL,
    normalized_features  JSONB NOT NULL,
    model_version        VARCHAR(50)
);
SELECT create_hypertable('feature_vectors', 'timestamp',
                         chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_symbol_time
    ON feature_vectors (symbol, timestamp DESC);

-- Regime labels from the HMM
CREATE TABLE IF NOT EXISTS regime_labels (
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    regime_id           INTEGER NOT NULL,
    regime_name         VARCHAR(50) NOT NULL,
    probability         NUMERIC(6,4) NOT NULL,
    state_probabilities JSONB NOT NULL,
    model_version       VARCHAR(50)
);
SELECT create_hypertable('regime_labels', 'timestamp',
                         chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_regime_symbol_time
    ON regime_labels (symbol, timestamp DESC);

-- Completed trades
CREATE TABLE IF NOT EXISTS trades (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol         VARCHAR(20) NOT NULL,
    direction      VARCHAR(10) NOT NULL,
    quantity       INTEGER NOT NULL,
    entry_price    NUMERIC(12,4) NOT NULL,
    exit_price     NUMERIC(12,4),
    entry_time     TIMESTAMPTZ NOT NULL,
    exit_time      TIMESTAMPTZ,
    entry_regime   VARCHAR(50),
    exit_regime    VARCHAR(50),
    pnl            NUMERIC(12,2),
    commission     NUMERIC(8,2),
    slippage       NUMERIC(8,2),
    exit_reason    VARCHAR(50),
    strategy_name  VARCHAR(50),
    holding_bars   INTEGER,
    metadata       JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades (entry_time DESC);

-- System state snapshots (for crash recovery)
CREATE TABLE IF NOT EXISTS system_state (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    state_type  VARCHAR(50) NOT NULL,
    state_data  JSONB NOT NULL
);
