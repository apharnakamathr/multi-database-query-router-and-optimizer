-- Phase 1: backend schema
-- One Postgres instance plays two roles:
--   1. Plain relational tables  -> "structured DB" backend
--   2. pgvector-extended table  -> "vector DB" backend
-- Redis (seeded separately in data/seed_data.py) plays the "cache" backend.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- Structured / transactional data (customers, orders)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id     SERIAL PRIMARY KEY,
    full_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    country         TEXT NOT NULL,
    signup_date     DATE NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS products (
    product_id      SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    price_cents     INTEGER NOT NULL,
    in_stock        BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL DEFAULT 1,
    order_date      TIMESTAMP NOT NULL DEFAULT NOW(),
    status          TEXT NOT NULL DEFAULT 'placed'
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id);
CREATE INDEX IF NOT EXISTS idx_customers_country ON customers(country);

-- ---------------------------------------------------------------------
-- Vector data: product description embeddings for semantic search
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_embeddings (
    product_id      INTEGER PRIMARY KEY REFERENCES products(product_id),
    description     TEXT NOT NULL,
    embedding       vector(32)   -- keep dims small; see .env EMBEDDING_DIM
);

-- IVFFlat index for approximate nearest-neighbor search once the table has
-- enough rows (needs ANALYZE afterwards; harmless to create early).
CREATE INDEX IF NOT EXISTS idx_product_embeddings_ann
    ON product_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- ---------------------------------------------------------------------
-- Query performance log (used later by the feedback loop in Phase 4;
-- created now so Phase 3's router can start writing to it immediately)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_log (
    log_id          SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    predicted_backend TEXT NOT NULL,
    actual_backend  TEXT NOT NULL,       -- differs from predicted on Redis-down fallback
    fallback        BOOLEAN NOT NULL DEFAULT FALSE,
    confidence      NUMERIC,
    latency_ms      NUMERIC NOT NULL,
    cost_usd        NUMERIC NOT NULL DEFAULT 0,
    success         BOOLEAN NOT NULL,
    source          TEXT NOT NULL DEFAULT 'route',  -- 'route' | 'nl-query' | 'workload'
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_query_log_backend ON query_log(actual_backend);

-- ---------------------------------------------------------------------
-- Phase 2/4: naive-baseline log ("send everything to Postgres") kept in
-- its own table so /metrics can compute router-vs-baseline improvement
-- without the baseline polluting the router's own performance numbers.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS baseline_log (
    log_id          SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    backend         TEXT NOT NULL DEFAULT 'postgres',
    latency_ms      NUMERIC NOT NULL,
    cost_usd        NUMERIC NOT NULL DEFAULT 0,
    success         BOOLEAN NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_baseline_log_created ON baseline_log(created_at);

-- ---------------------------------------------------------------------
-- Phase 4: one row per retrain attempt, so classifier accuracy over time
-- is visible on the dashboard and retrains that regress can be audited.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS classifier_metrics (
    metric_id           SERIAL PRIMARY KEY,
    trained_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    model_name          TEXT NOT NULL,
    n_bootstrap_samples  INTEGER NOT NULL,
    n_live_samples       INTEGER NOT NULL,
    accuracy_frozen_test NUMERIC NOT NULL,  -- eval'd on the ORIGINAL held-out bootstrap split only
    promoted             BOOLEAN NOT NULL,   -- did this run replace model.pkl?
    notes                TEXT
);
