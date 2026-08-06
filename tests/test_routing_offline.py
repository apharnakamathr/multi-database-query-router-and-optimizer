"""
Offline sanity check for the classifier + feature extraction (no live
Postgres/Redis required -- runs with execute=False against the FastAPI
app's routing logic directly).

Run with:  python -m tests.test_routing_offline   (from project root)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "router"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classifier"))

from fastapi.testclient import TestClient
from main import app  # noqa: E402

client = TestClient(app)


def route(query_text, **kwargs):
    payload = {"query_text": query_text, "execute": False, **kwargs}
    resp = client.post("/route", json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    print("--- Archetype routing ---")
    samples = [
        "select orders where customer_id = 42 and status = shipped",
        "count orders group by status for country",
        "find products similar to a cozy noise-cancelling headphone",
        "recommend items like this cotton yoga mat",
        "get product by id 8842",
    ]
    for q in samples:
        r = route(q)
        print(f"{q!r:70s} -> {r['predicted_backend']:8s} (conf={r['confidence']:.2f}) "
              f"actual={r['actual_backend']} fallback={r['fallback']}")

    print("\n--- Repeat-lookup effect (same ambiguous-ish query asked repeatedly) ---")
    q = "product id 991"
    for i in range(5):
        r = route(q)
        print(f"  call {i+1}: backend={r['predicted_backend']:8s} "
              f"conf={r['confidence']:.2f} freq_feature={r['features_used']['historical_frequency']}")

    print("\n--- Phase 5: /nl-query without ANTHROPIC_API_KEY (should fall back cleanly) ---")
    resp = client.post("/nl-query", json={"query_text": "find products similar to cozy headphones", "execute": False})
    resp.raise_for_status()
    body = resp.json()
    print(f"  predicted_backend={body['predicted_backend']} translation_used={body['translation_used']}")


if __name__ == "__main__":
    main()
