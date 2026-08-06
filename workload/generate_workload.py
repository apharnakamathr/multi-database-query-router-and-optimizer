"""
Phase 2 step 3-4 (bridge) / Phase 4 fuel: send a mixed, realistic query
workload against the live router API. Every call goes through /route,
which now logs to query_log (Phase 4) -- so running this a few times is
what gives retrain_from_logs.py and the dashboard's /metrics something
real to work with.

Run with:
    python workload/generate_workload.py --n 500 --host http://localhost:8000
"""
import argparse
import random
import time

import requests

from query_templates import random_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of queries to send")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--sleep-ms", type=float, default=5.0, help="delay between requests")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ok, failed = 0, 0
    for i in range(args.n):
        query_text, _ = random_query(rng)
        try:
            resp = requests.post(
                f"{args.host}/route",
                json={"query_text": query_text, "execute": True},
                timeout=5,
            )
            resp.raise_for_status()
            body = resp.json()
            ok += 1
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{args.n}] {query_text!r} -> "
                      f"{body['actual_backend']} ({body['latency_ms']:.1f}ms"
                      f"{' fallback' if body['fallback'] else ''})")
        except Exception as e:
            failed += 1
            print(f"  request failed: {e}")
        time.sleep(args.sleep_ms / 1000.0)

    print(f"\nDone. {ok} succeeded, {failed} failed out of {args.n}.")


if __name__ == "__main__":
    main()
