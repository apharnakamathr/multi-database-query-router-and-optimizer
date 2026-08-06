"""
Phase 3 needs labeled (query -> best backend) examples to train on. In the
full project those come from Phase 2's workload generator run against all
three backends. Since this build covers Phases 1 + 3 only, this script
produces a synthetic-but-realistic bootstrap dataset of the three query
archetypes the doc describes:

  - structured/transactional (SQL-shaped filters/joins)  -> postgres
  - semantic/similarity (NL -> embedding search)          -> vector
  - hot-path repeated key lookups                         -> redis

Replace this with real logged traffic (query_log table) once Phase 2/4
are built -- train_classifier.py doesn't care where the CSV comes from.
"""
import csv
import os
import random

random.seed(7)

HERE = os.path.dirname(os.path.abspath(__file__))

STRUCTURED_TEMPLATES = [
    "select orders where customer_id = {n} and status = 'shipped'",
    "count orders group by status for country IN",
    "sum revenue where category = 'electronics' and price greater than {n}",
    "join customers and orders where signup_date between last month",
    "filter products where in_stock = true and category = 'sports'",
    "average order quantity group by product_id",
    "select customers where country = 'US' and is_active = true",
]

SEMANTIC_TEMPLATES = [
    "find products similar to a cozy noise-cancelling headphone",
    "recommend items like this cotton yoga mat",
    "what products resemble a minimalist desk lamp",
    "find me something with a similar vibe to a leather backpack",
    "products semantically related to outdoor camping gear",
    "closest matches to a lightweight running shoe description",
    "recommend related items to a ceramic coffee mug",
]

CACHE_TEMPLATES = [
    "get product by id {n}",
    "lookup product id: {n}",
    "fetch id {n} details",
    "product:{n}",
    "quick lookup for id {n}",
]


def make_row(text, label, repeat_count=0):
    return {"query_text": text, "label": label, "repeat_count": repeat_count}


def generate(n_per_class: int = 120):
    rows = []
    for _ in range(n_per_class):
        t = random.choice(STRUCTURED_TEMPLATES).format(n=random.randint(1, 500))
        rows.append(make_row(t, "postgres", repeat_count=random.choice([0, 0, 1])))

    for _ in range(n_per_class):
        t = random.choice(SEMANTIC_TEMPLATES)
        rows.append(make_row(t, "vector", repeat_count=random.choice([0, 0, 1])))

    # hot-path lookups repeat heavily by construction -- a handful of ids
    # get looked up over and over, which is exactly the "is this a repeat
    # lookup" signal the router should learn to associate with Redis.
    hot_ids = [random.randint(1, 150) for _ in range(8)]
    for _ in range(n_per_class):
        pid = random.choice(hot_ids)
        t = random.choice(CACHE_TEMPLATES).format(n=pid)
        rows.append(make_row(t, "redis", repeat_count=random.randint(3, 15)))

    random.shuffle(rows)
    return rows


def main(out_path=None, n_per_class=120):
    if out_path is None:
        out_path = os.path.join(HERE, "training_queries.csv")
    rows = generate(n_per_class)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_text", "label", "repeat_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled training queries to {out_path}")


if __name__ == "__main__":
    main()
