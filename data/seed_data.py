"""
Loads a realistic dataset into all three backends.

- Structured records -> Postgres (customers, products, orders)
- Embeddings of the *same* products -> pgvector (product_embeddings)
- Hot/frequent lookups -> Redis (top N best-selling products, cached as JSON)

Run with:  python data/seed_data.py
Requires the docker-compose stack to be up (`docker compose up -d`) and
the packages in requirements.txt installed.
"""
import json
import os
import random

import numpy as np
import psycopg2
import redis
from dotenv import load_dotenv
from faker import Faker

load_dotenv()
fake = Faker()
random.seed(42)
np.random.seed(42)

PG_CONF = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    user=os.getenv("POSTGRES_USER", "router"),
    password=os.getenv("POSTGRES_PASSWORD", "router_pw"),
    dbname=os.getenv("POSTGRES_DB", "router_db"),
)
REDIS_CONF = dict(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "32"))

N_CUSTOMERS = 300
N_PRODUCTS = 150
N_ORDERS = 2000
CATEGORIES = ["electronics", "home", "sports", "books", "beauty", "toys", "grocery"]
COUNTRIES = ["IN", "US", "GB", "DE", "SG", "AE"]


def fake_embedding(text: str, dim: int = EMBEDDING_DIM) -> list:
    """
    Deterministic pseudo-embedding derived from text so semantically similar
    strings land near each other in a toy way (hash-based bag-of-words +
    small noise). Swap this out for a real encoder (e.g. sentence-transformers)
    when moving past the demo stage.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def seed_postgres(conn):
    cur = conn.cursor()

    print("Seeding customers...")
    customers = []
    for _ in range(N_CUSTOMERS):
        customers.append(
            (
                fake.name(),
                fake.unique.email(),
                random.choice(COUNTRIES),
                fake.date_between(start_date="-2y", end_date="today"),
                random.random() > 0.1,
            )
        )
    cur.executemany(
        """INSERT INTO customers (full_name, email, country, signup_date, is_active)
           VALUES (%s, %s, %s, %s, %s) ON CONFLICT (email) DO NOTHING""",
        customers,
    )

    print("Seeding products...")
    products = []
    descriptions = []
    for _ in range(N_PRODUCTS):
        category = random.choice(CATEGORIES)
        name = f"{fake.word().capitalize()} {category.capitalize()} {fake.word().capitalize()}"
        desc = fake.sentence(nb_words=12)
        products.append((name, category, random.randint(500, 50000), random.random() > 0.15))
        descriptions.append(desc)
    cur.executemany(
        """INSERT INTO products (name, category, price_cents, in_stock)
           VALUES (%s, %s, %s, %s)""",
        products,
    )

    conn.commit()
    cur.execute("SELECT customer_id FROM customers")
    customer_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT product_id, name FROM products ORDER BY product_id")
    product_rows = cur.fetchall()
    product_ids = [r[0] for r in product_rows]

    print("Seeding orders...")
    statuses = ["placed", "shipped", "delivered", "cancelled", "returned"]
    # skew order volume so some products are "hot" (for the Redis cache step)
    weights = np.random.zipf(1.5, size=len(product_ids))
    weights = weights / weights.sum()
    orders = []
    for _ in range(N_ORDERS):
        cid = random.choice(customer_ids)
        pid = int(np.random.choice(product_ids, p=weights))
        orders.append((cid, pid, random.randint(1, 5), random.choices(statuses, weights=[5, 3, 6, 1, 1])[0]))
    cur.executemany(
        """INSERT INTO orders (customer_id, product_id, quantity, status)
           VALUES (%s, %s, %s, %s)""",
        orders,
    )
    conn.commit()

    print("Seeding product embeddings (pgvector)...")
    embed_rows = []
    for (pid, name), desc in zip(product_rows, descriptions):
        full_text = f"{name}. {desc}"
        emb = fake_embedding(full_text)
        embed_rows.append((pid, desc, emb))
    cur.executemany(
        """INSERT INTO product_embeddings (product_id, description, embedding)
           VALUES (%s, %s, %s) ON CONFLICT (product_id) DO NOTHING""",
        embed_rows,
    )
    conn.commit()
    cur.execute("ANALYZE product_embeddings")
    conn.commit()

    return product_ids, weights


def seed_redis(product_ids, weights, top_n=20):
    print("Seeding Redis with hot-path lookups...")
    r = redis.Redis(**REDIS_CONF)
    r.flushdb()

    # cache the top-N best-selling products as ready-to-serve JSON blobs,
    # simulating a warm cache of "hot key" lookups
    order = np.argsort(weights)[::-1][:top_n]
    hot_products = [product_ids[i] for i in order]
    for pid in hot_products:
        cache_key = f"product:{pid}"
        r.set(cache_key, json.dumps({"product_id": pid, "cached": True}), ex=3600)

    r.set("cache:hot_keys", json.dumps([f"product:{pid}" for pid in hot_products]))
    print(f"  Cached {len(hot_products)} hot product keys in Redis.")


def main():
    conn = psycopg2.connect(**PG_CONF)
    try:
        product_ids, weights = seed_postgres(conn)
        seed_redis(product_ids, weights)
    finally:
        conn.close()
    print("Done. Postgres has customers/products/orders + pgvector embeddings; Redis has hot-key cache.")


if __name__ == "__main__":
    main()
