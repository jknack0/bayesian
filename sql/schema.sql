-- ============================================
-- BayesBot Supabase Schema
-- ============================================
-- Covers: market data (OHLCV + L1), dollar bars,
-- HMM regime detection, strategies, signals,
-- trades, round trips, and performance tracking.
-- ============================================


-- ============================================
-- 1. MARKET DATA (partitioned by timeframe)
-- ============================================
-- Uses double precision for performance on 150M+ rows.
-- Precision loss (~15 significant digits) is irrelevant for price data.

create table market_data (
    symbol         text             not null,
    timeframe      text             not null,  -- '1s', '1m', '5m', '15m', '1h', '1d'
    open_time      timestamptz      not null,
    open           double precision,
    high           double precision,
    low            double precision,
    close          double precision not null,
    volume         double precision not null default 0,
    vwap           double precision,
    trade_count    integer,
    source         text             not null default 'databento',

    primary key (symbol, timeframe, open_time)
) partition by list (timeframe);

create table market_data_1s     partition of market_data for values in ('1s');
create table market_data_1m     partition of market_data for values in ('1m');
create table market_data_5m     partition of market_data for values in ('5m');
create table market_data_15m    partition of market_data for values in ('15m');
create table market_data_1h     partition of market_data for values in ('1h');
create table market_data_1d     partition of market_data for values in ('1d');

create index on market_data (symbol, open_time desc);


-- ============================================
-- 2. L1 TICK DATA (separate table — different shape, 100x more rows)
-- ============================================

create table market_data_l1 (
    symbol         text             not null,
    ts             timestamptz      not null,
    bid            double precision,
    ask            double precision,
    bid_size       double precision,
    ask_size       double precision,
    last_price     double precision,
    last_size      double precision,
    source         text             not null default 'databento',

    primary key (symbol, ts)
);

create index on market_data_l1 (symbol, ts desc);


-- ============================================
-- 3. DOLLAR BARS (variable-length bars, not a standard timeframe)
-- ============================================

create table dollar_bars (
    id             bigint generated always as identity primary key,
    symbol         text             not null,
    bar_index      integer          not null,
    open_time      timestamptz      not null,
    close_time     timestamptz      not null,
    open           double precision not null,
    high           double precision not null,
    low            double precision not null,
    close          double precision not null,
    volume         double precision not null,
    vwap           double precision,
    buy_volume     double precision,
    sell_volume    double precision,
    tick_count     integer,
    dollar_value   double precision,          -- total dollar volume in this bar
    threshold      double precision,          -- dollar threshold used to build this bar
    source         text             not null default 'databento',

    unique (symbol, bar_index)
);

create index on dollar_bars (symbol, open_time desc);


-- ============================================
-- 4. STRATEGIES
-- ============================================

create table strategies (
    id             text        primary key,  -- 'momentum_v1', 'mean_reversion_v1', 'defensive_v1'
    name           text        not null,
    description    text,
    version        text        not null,
    config         jsonb       not null default '{}',  -- thresholds, multipliers, etc.
    asset_class    text,
    status         text        not null default 'active',  -- 'active', 'paper', 'retired'
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now()
);


-- ============================================
-- 5. MODEL ARTIFACTS (trained HMM params, BOCPD state)
-- ============================================

create table model_artifacts (
    id             bigint generated always as identity primary key,
    strategy_id    text        not null references strategies(id),
    model_type     text        not null,  -- 'hmm', 'bocpd', 'pca'
    version        text        not null,
    artifact       jsonb       not null,  -- transition matrix, emission params, PCA transform, etc.
    train_bars     integer,
    train_start    timestamptz,
    train_end      timestamptz,
    metrics        jsonb,                 -- { bic, rcm, log_likelihood, n_states, covariance_type }
    is_active      boolean     not null default false,  -- currently deployed model
    created_at     timestamptz not null default now()
);

create index on model_artifacts (strategy_id, model_type, created_at desc);
create index on model_artifacts (is_active) where is_active = true;


-- ============================================
-- 6. REGIME PREDICTIONS (HMM + BOCPD output history)
-- ============================================

create table regime_predictions (
    id             bigint generated always as identity primary key,
    symbol         text             not null,
    bar_index      integer          not null,
    predicted_at   timestamptz      not null,
    regime_name    text             not null,  -- 'trending', 'mean_reverting', 'volatile'
    prob_mean_rev  double precision not null,
    prob_trending  double precision not null,
    prob_volatile  double precision not null,
    confidence     double precision not null,
    bocpd_cp_prob  double precision,           -- change-point probability at this bar
    bocpd_is_cp    boolean,                    -- whether BOCPD flagged a change point
    model_id       bigint           references model_artifacts(id),

    unique (symbol, bar_index)
);

create index on regime_predictions (symbol, predicted_at desc);
create index on regime_predictions (regime_name, predicted_at desc);


-- ============================================
-- 7. SIGNALS
-- ============================================

create table signals (
    id             bigint generated always as identity primary key,
    strategy_id    text        not null references strategies(id),
    symbol         text        not null,
    signal_time    timestamptz not null,
    bar_index      integer,
    direction      text        not null,  -- 'long', 'short', 'close', 'flatten'
    strength       double precision,
    entry_price    double precision,
    stop_loss      double precision,
    profit_target  double precision,
    time_barrier   integer,               -- max bars to hold
    regime         text,                  -- regime at signal time
    regime_confidence double precision,
    features       jsonb,                 -- snapshot of normalized features that produced signal
    metadata       jsonb,
    created_at     timestamptz not null default now()
);

create index on signals (strategy_id, symbol, signal_time desc);
create index on signals (signal_time desc);


-- ============================================
-- 8. TRADES (individual executions)
-- ============================================
-- Uses numeric for financial-critical fields (commission, slippage).

create table trades (
    id              bigint generated always as identity primary key,
    strategy_id     text        not null references strategies(id),
    signal_id       bigint      references signals(id),
    symbol          text        not null,
    side            text        not null,  -- 'buy', 'sell'
    quantity        numeric     not null,
    price           numeric     not null,
    commission      numeric     not null default 0,
    slippage        numeric,
    order_type      text,                  -- 'market', 'limit', 'stop'
    executed_at     timestamptz not null,
    broker          text        not null,  -- 'ib', 'databento'
    broker_order_id text,
    metadata        jsonb,
    created_at      timestamptz not null default now()
);

create index on trades (strategy_id, executed_at desc);
create index on trades (symbol, executed_at desc);


-- ============================================
-- 9. ROUND TRIPS (entry + exit pairs)
-- ============================================

create table round_trips (
    id             bigint generated always as identity primary key,
    strategy_id    text        not null references strategies(id),
    symbol         text        not null,
    direction      text        not null,  -- 'long', 'short'
    entry_trade_id bigint      not null references trades(id),
    exit_trade_id  bigint      references trades(id),
    entry_price    numeric     not null,
    exit_price     numeric,
    quantity       numeric     not null,
    gross_pnl      numeric,
    net_pnl        numeric,
    hold_duration  interval,
    hold_bars      integer,               -- bars held (for dollar-bar context)
    max_adverse    numeric,               -- max adverse excursion (MAE)
    max_favorable  numeric,               -- max favorable excursion (MFE)
    exit_reason    text,                  -- 'STOP_LOSS', 'PROFIT_TARGET', 'TIME_BARRIER', 'REGIME_CHANGE'
    entry_regime   text,                  -- regime at entry
    exit_regime    text,                  -- regime at exit
    opened_at      timestamptz not null,
    closed_at      timestamptz,
    metadata       jsonb
);

create index on round_trips (strategy_id, opened_at desc);
create index on round_trips (exit_reason);


-- ============================================
-- 10. STRATEGY PERFORMANCE SNAPSHOTS
-- ============================================

create table strategy_snapshots (
    id             bigint generated always as identity primary key,
    strategy_id    text        not null references strategies(id),
    snapshot_date  date        not null,
    symbol         text,                  -- null for aggregate across all symbols
    total_trades   integer     not null,
    win_rate       double precision,
    avg_pnl        numeric,
    total_pnl      numeric,
    sharpe         double precision,
    sortino        double precision,
    max_drawdown   double precision,
    profit_factor  double precision,
    time_in_market double precision,      -- fraction of time with open positions
    metadata       jsonb,                 -- walk-forward results, per-regime stats, etc.

    unique (strategy_id, snapshot_date, symbol)
);


-- ============================================
-- 11. WALK-FORWARD RESULTS
-- ============================================

create table walk_forward_results (
    id              bigint generated always as identity primary key,
    strategy_id     text        not null references strategies(id),
    run_date        timestamptz not null default now(),
    n_windows       integer     not null,
    train_bars      integer     not null,
    test_bars       integer     not null,
    sharpe_mean     double precision,
    sharpe_std      double precision,
    max_drawdown    double precision,
    win_rate        double precision,
    go_no_go        jsonb,                -- { sharpe_positive: true, max_dd_ok: false, ... }
    is_passing      boolean     not null,
    window_details  jsonb,                -- per-window metrics array
    metadata        jsonb
);

create index on walk_forward_results (strategy_id, run_date desc);
