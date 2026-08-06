"""
Phase 2 (bridge): the mixed query workload the doc's Phase 2 step 3 asks
for. Reuses the three archetypes from classifier/generate_training_queries.py
so live traffic looks like the same distribution the classifier was
bootstrapped on, but adds more lexical variety and a skewed "hot id" set so
the repeat-lookup -> redis signal actually shows up in real traffic, not
just in the synthetic training CSV.
"""
import random

STRUCTURED_TEMPLATES = [
    "select orders where customer_id = {n} and status = 'shipped'",
    "count orders group by status for country IN",
    "sum revenue where category = 'electronics' and price greater than {n}",
    "join customers and orders where signup_date between last month",
    "filter products where in_stock = true and category = 'sports'",
    "average order quantity group by product_id",
    "select customers where country = 'US' and is_active = true",
    "how many orders were cancelled last week",
    "list top 10 products by price in the beauty category",
]

SEMANTIC_TEMPLATES = [
    "find products similar to a cozy noise-cancelling headphone",
    "recommend items like this cotton yoga mat",
    "what products resemble a minimalist desk lamp",
    "find me something with a similar vibe to a leather backpack",
    "products semantically related to outdoor camping gear",
    "closest matches to a lightweight running shoe description",
    "recommend related items to a ceramic coffee mug",
    "suggest something that matches the vibe of a rainy-day reading nook",
]

CACHE_TEMPLATES = [
    "get product by id {n}",
    "lookup product id: {n}",
    "fetch id {n} details",
    "product:{n}",
    "quick lookup for id {n}",
]

# A handful of ids get looked up disproportionately often, simulating a
# real hot-key access pattern (bestsellers, homepage widgets, etc.).
_HOT_IDS = [random.randint(1, 150) for _ in range(8)]


def random_query(rng: random.Random):
    """Returns (query_text, archetype) with a realistic mix: ~40% structured,
    ~25% semantic, ~35% cache-style hot lookups (skewed toward _HOT_IDS)."""
    roll = rng.random()
    if roll < 0.40:
        t = rng.choice(STRUCTURED_TEMPLATES)
        return t.format(n=rng.randint(1, 500)), "postgres"
    elif roll < 0.65:
        return rng.choice(SEMANTIC_TEMPLATES), "vector"
    else:
        pid = rng.choice(_HOT_IDS) if rng.random() < 0.7 else rng.randint(1, 500)
        t = rng.choice(CACHE_TEMPLATES)
        return t.format(n=pid), "redis"
