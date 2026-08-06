"""
Thin connectors for the three backends.

Phase 6 hardening added here:
  - connection goes through PgBouncer by default (POSTGRES_HOST=pgbouncer)
  - Redis-down -> RedisUnavailable, caught by main.py to fall back to Postgres
    instead of crashing the request
  - a rough per-query $ cost model, so /metrics can show cost, not just
    latency.
"""
import os
import re
import time

import psycopg2
import psycopg2.extras
import redis

# Route Postgres traffic through PgBouncer by default (Phase 6). Falls back
# to POSTGRES_HOST/PORT directly if PGBOUNCER_HOST isn't set, so this still
# works with the plain docker-compose from Phases 1-3.
PG_CONF = dict(
    host=os.getenv("PGBOUNCER_HOST", os.getenv("POSTGRES_HOST", "localhost")),
    port=os.getenv("PGBOUNCER_PORT", os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "router"),
    password=os.getenv("POSTGRES_PASSWORD", "router_pw"),
    dbname=os.getenv("POSTGRES_DB", "router_db"),
)
REDIS_CONF = dict(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
    socket_connect_timeout=float(os.getenv("REDIS_CONNECT_TIMEOUT", "0.5")),
    socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "0.5")),
)

# Rough $/query cost model (illustrative, not real billing data): reflects
# that a semantic ANN scan costs more compute than a cached key lookup.
# Swap for real cost-per-query telemetry once deployed on managed services
# (e.g. RDS/ElastiCache CloudWatch billing metrics).
COST_PER_QUERY_USD = {
    "postgres": 0.00008,
    "vector": 0.00035,
    "redis": 0.00001,
}

# Only SELECTs are allowed through the NL-to-query path -- a Claude-generated
# statement is still untrusted input as far as the DB is concerned.
_WRITE_STATEMENT_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create)\b", re.IGNORECASE
)


class RedisUnavailable(Exception):
    """Raised when Redis can't be reached; main.py catches this to fall back."""


def _assert_read_only(sql: str):
    if _WRITE_STATEMENT_RE.search(sql or ""):
        raise ValueError("Only read-only SELECT statements are permitted here.")


def get_pg_connection():
    return psycopg2.connect(**PG_CONF, connect_timeout=3)


def get_redis_client():
    return redis.Redis(**REDIS_CONF)


def execute_structured(query_text: str, translated_sql: str = None):
    """
    Backend: postgres (structured/transactional).
    If `translated_sql` is supplied, run it
    (read-only enforced); otherwise fall back to the Phase 1/3 demo query.
    """
    start = time.perf_counter()
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if translated_sql:
            _assert_read_only(translated_sql)
            cur.execute(translated_sql)
        else:
            cur.execute(
                """
                SELECT p.product_id, p.name, p.category, p.price_cents, p.in_stock
                FROM products p
                ORDER BY p.product_id
                LIMIT 10
                """
            )
        rows = cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000
        return {"backend": "postgres", "rows": rows, "sql": translated_sql}, latency_ms
    finally:
        conn.close()


def execute_vector(query_text: str, embedding_dim: int = 32, translated_description: str = None):
    """
    Backend: pgvector (semantic/similarity search).
    Swap the toy hash-embedding for a real encoder
    (sentence-transformers, or an embeddings API) when moving past demo.
    """
    import numpy as np

    start = time.perf_counter()
    seed_text = (translated_description or query_text).lower()
    rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
    vec = rng.normal(size=embedding_dim)
    vec = (vec / np.linalg.norm(vec)).tolist()

    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT product_id, description, embedding <=> %s::vector AS distance
            FROM product_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """,
            (vec, vec),
        )
        rows = cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000
        return {"backend": "vector", "rows": rows, "embedded_text": seed_text}, latency_ms
    finally:
        conn.close()


def execute_cache(query_text: str, cache_key: str = None):
    """
    Backend: redis (hot-path key lookups).
    Raises RedisUnavailable on connection failure so main.py can fall back
    to Postgres instead of the request 502-ing (Phase 6 hardening).
    """
    start = time.perf_counter()
    key = cache_key
    if key is None:
        m = re.search(r"\d{1,6}", query_text)
        key = f"product:{m.group(0)}" if m else None

    try:
        r = get_redis_client()
        value = r.get(key) if key else None
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
        raise RedisUnavailable(str(e))

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "backend": "redis",
        "cache_key": key,
        "hit": value is not None,
        "value": value,
    }, latency_ms


def estimate_cost_usd(backend: str) -> float:
    return COST_PER_QUERY_USD.get(backend, 0.0)
