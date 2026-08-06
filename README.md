# Multi-Database Query Router & Optimizer

A query router that classifies incoming queries and sends each one to the
right backend, structured lookups to Postgres, hot-path key lookups to
Redis, and semantic/similarity search to a vector store (pgvector)
instead of hardcoding routing rules by hand. The classifier retrains
itself from live traffic on a schedule, with guardrails against
reinforcing its own mistakes.

## Architecture

```
                     ┌─────────────┐
  query_text ──────▶ │  classifier │──▶ predicted_backend
                     └─────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌────────┐   ┌─────────┐   ┌──────────┐
         │Postgres│   │  Redis  │   │ pgvector │
         │(struct)│   │ (cache) │   │(semantic)│
         └────────┘   └─────────┘   └──────────┘
```

- **Classification**: a scikit-learn model (logistic regression /
  random forest) trained on labeled query patterns like filter/aggregate
  keywords → Postgres, numeric-ID lookups → Redis, semantic-similarity
  phrasing → pgvector.
- **Connection pooling**: PgBouncer sits in front of Postgres so pool
  limits are respected under load.
- **Graceful degradation**: if Redis is unreachable, cache-style
  lookups fall back to Postgres instead of failing the request
  (`fallback: true` in the response).
- **Continuous retraining**: live routing decisions are logged; a
  retrain job periodically folds *confident, successful, non-fallback*
  predictions back into the training set evaluated against a frozen
  held-out split so a bad retrain never silently gets promoted.

## What's here

```
db-query-router/
├── docker-compose.yml       # postgres+pgvector, pgbouncer, redis, router API, dashboard
├── Dockerfile                # router API image
├── requirements.txt
├── .env.example
├── sql/init.sql              # schema: structured + vector tables, query_log, baseline_log, classifier_metrics
├── data/
│   └── seed_data.py          # populates all three backends with sample data
├── classifier/
│   ├── features.py           # query -> feature vector
│   ├── generate_training_queries.py  # bootstrap labeled dataset
│   ├── train_classifier.py   # initial training, writes model.pkl
│   ├── retrain_from_logs.py  # retrain from live query_log, with guardrails
│   └── model.pkl             # pre-trained model (regenerate anytime)
├── router/
│   ├── backends.py           # connectors, fallback logic, cost model
│   └── main.py                # FastAPI app: /route, /metrics, /query-log, /classifier-accuracy
├── workload/
│   ├── query_templates.py    # mixed-workload query generator
│   ├── generate_workload.py  # drives live traffic through the router
│   └── naive_baseline.py     # "send everything to Postgres" comparison baseline
├── pgbouncer/                 # connection pooling config
├── dashboard/                  # React/Vite dashboard
├── .github/workflows/
│   ├── ci-cd.yml               # test -> build -> push
│   └── scheduled-retrain.yml   # daily classifier retrain
└── tests/test_routing_offline.py  # sanity check, no live DBs required
```

## Running it locally

### 1. Start the backends

```bash
cp .env.example .env
# edit .env and set POSTGRES_PASSWORD

docker compose up -d postgres pgbouncer redis
docker compose ps   # wait until postgres and pgbouncer report healthy
```

### 2. Seed data and train the initial classifier

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/seed_data.py            # populate Postgres + pgvector + Redis

cd classifier
python generate_training_queries.py # bootstrap labeled dataset
python train_classifier.py          # writes model.pkl
cd ..
```

### 3. Start the router API + dashboard

```bash
docker compose up -d --build router_api dashboard
# or run the API locally without Docker:
#   uvicorn router.main:app --reload
```

- API: http://localhost:8000 (docs at `/docs`)
- Dashboard: http://localhost:5173

### 4. Sanity check

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query_text": "find products similar to a cozy noise-cancelling headphone"}'
```

```json
{
  "predicted_backend": "vector",
  "actual_backend": "vector",
  "fallback": false,
  "confidence": 0.99,
  "executed": true,
  "result": {"backend": "vector", "rows": [...]},
  "latency_ms": 12.4
}
```

Offline sanity check without any live DBs (`execute: false`):

```bash
python -m tests.test_routing_offline
```

### 5. Generate traffic + naive baseline, then compare

```bash
# real router traffic (fills query_log for retraining + the dashboard)
python workload/generate_workload.py --n 500 --host http://localhost:8000

# naive comparison: same query mix, always sent to Postgres
python workload/naive_baseline.py --n 500

curl "http://localhost:8000/metrics?hours=1" | python -m json.tool
```

`router_vs_baseline.latency_improvement_pct` and `.cost_improvement_pct`
are the headline numbers. They should come out solidly positive since
the naive baseline runs semantic and cache-style lookups against
Postgres too, which is the wrong tool for both.

### 6. Retrain the classifier from live traffic

```bash
python classifier/retrain_from_logs.py --since-hours 168
```

What this does, and why it's not just "retrain on whatever came in":

- Only uses live rows where the router's own prediction was
  **confident, successful, and not a fallback** as reinforcement
  signal, everything else (low-confidence, failed, or Redis-down-
  fallback rows) is left out of training entirely, since those are
  signals something needs *investigating*, not a label to learn from.
- Live examples are **added to** the original bootstrap set, never
  used alone, so the model can't fully drift onto whatever traffic
  shape happened to dominate recently.
- Every candidate is scored against a **frozen** held-out split of the
  *original* bootstrap CSV, the same split every run; so accuracy
  numbers are comparable across retrains regardless of what live
  traffic looked like that week. If the new model regresses beyond a
  small tolerance on that frozen set, it's logged (`classifier_metrics`
  table) but **not promoted**  `model.pkl` stays untouched.
- The router hot-reloads `model.pkl` on its next request (checks the
  file's mtime), so a promoted retrain takes effect without a restart.

Check the accuracy history:

```bash
curl "http://localhost:8000/classifier-accuracy" | python -m json.tool
```

To run this on a schedule instead of by hand, see
`.github/workflows/scheduled-retrain.yml` (daily cron) — point its
secrets at your deployed Postgres and it'll retrain and commit a
promoted `model.pkl` automatically.

### 7. Fallback behavior

```bash
docker compose stop redis
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query_text": "get product by id 42"}'
# predicted_backend: "redis", actual_backend: "postgres", fallback: true
docker compose start redis
```

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /route` | classify + execute a query; logs to `query_log`; falls back to Postgres if Redis is down |
| `GET /frequency-table` | debug view of what the router considers "hot" |
| `GET /metrics?hours=24` | router vs naive-baseline latency & cost comparison |
| `GET /query-log?limit=100` | recent routing decisions (for the dashboard) |
| `GET /classifier-accuracy?limit=50` | retrain history, evaluated on the frozen held-out set |

## Known simplifications

- **Cost model** (`router/backends.py: COST_PER_QUERY_USD`) is an
  illustrative flat-rate estimate, not real cloud billing data.
- **Embeddings** are a toy hash-based encoder (`data/seed_data.py:
  fake_embedding`), not a real sentence encoder fine for
  demonstrating routing/ANN mechanics, not for real semantic quality.
  Swapping in a real sentence-embedding model is a drop-in replacement
  at the `fake_embedding` call sites.
- **PgBouncer auth** in `docker-compose.yml` uses `AUTH_TYPE=plain` for
  local-dev simplicity. `pgbouncer/pgbouncer.ini` +
  `userlist.txt.template` are there as the `auth_type=md5` reference
  for a real deployment.
- **Classifier accuracy** (100% on the bootstrap set) reflects
  synthetic, cleanly-separable training data and not a claim about
  real-world routing difficulty.
