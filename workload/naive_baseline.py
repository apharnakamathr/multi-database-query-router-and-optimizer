"""
Phase 2 step 4 / Phase 4 step 10: the naive baseline the doc calls for --
"send everything to Postgres" -- run against the *same* generated workload
so the comparison in /metrics is apples-to-apples. Logged to baseline_log,
kept separate from the router's own query_log.

Run with:
    python workload/naive_baseline.py --n 300
"""
import argparse
import os
import random
import time

import psycopg2
from dotenv import load_dotenv

from query_templates import random_query

load_dotenv()

PG_CONF = dict(
    host=os.getenv("PGBOUNCER_HOST", os.getenv("POSTGRES_HOST", "localhost")),
    port=os.getenv("PGBOUNCER_PORT", os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "router"),
    password=os.getenv("POSTGRES_PASSWORD", "router_pw"),
    dbname=os.getenv("POSTGRES_DB", "router_db"),
)


def run_naive_query(conn, query_text: str):
    """The naive baseline doesn't classify anything -- every query, whatever
    its shape (structured, semantic, or a cache-style id lookup), runs as a
    generic products scan against Postgres. This is deliberately the wrong
    tool for 2 of the 3 archetypes, which is the point of the comparison."""
    start = time.perf_counter()
    cur = conn.cursor()
    cur.execute("SELECT product_id, name, category, price_cents FROM products ORDER BY product_id LIMIT 10")
    cur.fetchall()
    return (time.perf_counter() - start) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    conn = psycopg2.connect(**PG_CONF)
    log_cur = conn.cursor()
    ok, failed = 0, 0
    for i in range(args.n):
        query_text, _ = random_query(rng)
        try:
            latency_ms = run_naive_query(conn, query_text)
            # Naive baseline cost model: it's always Postgres, so it never
            # gets Redis's near-zero cost even for what should've been a
            # cheap cache hit -- that gap is exactly what the router avoids.
            log_cur.execute(
                """INSERT INTO baseline_log (query_text, backend, latency_ms, cost_usd, success)
                   VALUES (%s, 'postgres', %s, 0.00008, TRUE)""",
                (query_text, latency_ms),
            )
            conn.commit()
            ok += 1
        except Exception as e:
            failed += 1
            conn.rollback()
            print(f"  baseline query failed: {e}")

    conn.close()
    print(f"Done. {ok} succeeded, {failed} failed out of {args.n}.")


if __name__ == "__main__":
    main()
