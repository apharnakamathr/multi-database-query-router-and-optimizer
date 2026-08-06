"""
combining everything
"""
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "classifier"))
sys.path.append(os.path.dirname(__file__))
import joblib  # noqa: E402
import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402
from features import extract_features  # noqa: E402
from backends import (  # noqa: E402
    execute_structured,
    execute_vector,
    execute_cache,
    estimate_cost_usd,
    get_pg_connection,
    RedisUnavailable,
)

try:
    import nl_to_query  #optional
except Exception:  # pragma: no cover - import-time issues shouldn't kill the API
    nl_to_query = None

app = FastAPI(title="Multi-Database Query Router", version="0.4.0 (Phase 1-6)")

# Dashboard (Phase 6) runs on a different origin during local dev (vite on
# :5173); lock this down to a real domain list once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "classifier", "model.pkl")
_bundle = None
_bundle_mtime = None

# In-memory frequency table (Phase 3 step 5's "historical frequency" signal).
_frequency_table: Dict[str, int] = defaultdict(int)

BACKEND_EXECUTORS = {
    "postgres": execute_structured,
    "vector": execute_vector,
    "redis": execute_cache,
}


def _load_model():
    """Hot-reloads model.pkl if it changed on disk, so a Phase 4 retrain
    that promotes a new model gets picked up without restarting the API."""
    global _bundle, _bundle_mtime
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. "
            f"Run `python classifier/train_classifier.py` first."
        )
    mtime = os.path.getmtime(MODEL_PATH)
    if _bundle is None or mtime != _bundle_mtime:
        _bundle = joblib.load(MODEL_PATH)
        _bundle_mtime = mtime
    return _bundle


def _classify(query_text: str, filters=None, semantic_hint=None):
    bundle = _load_model()
    model = bundle["model"]
    le = bundle["label_encoder"]

    text_key = query_text.strip().lower()
    feats = extract_features(
        query_text,
        filters=filters,
        semantic_hint=semantic_hint,
        frequency_table=_frequency_table,
    )
    _frequency_table[text_key] += 1

    X = [feats.to_vector()]
    pred_idx = model.predict(X)[0]
    predicted_backend = le.inverse_transform([pred_idx])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        scores = {cls: float(p) for cls, p in zip(le.classes_, proba)}
    else:
        scores = {cls: (1.0 if cls == predicted_backend else 0.0) for cls in le.classes_}
    return predicted_backend, scores, feats


def _execute_with_fallback(predicted_backend, query_text, **kwargs):
    """
     if the predicted backend is Redis and Redis is unreachable,
    fall back to Postgres rather than failing the request. Returns
    (result, latency_ms, actual_backend, fell_back).
    """
    executor = BACKEND_EXECUTORS.get(predicted_backend)
    if executor is None:
        raise HTTPException(500, f"No executor registered for backend '{predicted_backend}'")
    try:
        result, latency_ms = executor(query_text, **kwargs) if kwargs else executor(query_text)
        return result, latency_ms, predicted_backend, False
    except RedisUnavailable:
        result, latency_ms = execute_structured(query_text)
        return result, latency_ms, "postgres", True


def _log_execution(query_text, predicted_backend, actual_backend, fallback,
                    confidence, latency_ms, success, source="route"):
    """Phase 4 step 8: persist every execution so retrain_from_logs.py and
    the dashboard's /metrics endpoint have real performance data to use."""
    cost = estimate_cost_usd(actual_backend)
    try:
        conn = get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO query_log
                   (query_text, predicted_backend, actual_backend, fallback,
                    confidence, latency_ms, cost_usd, success, source)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (query_text, predicted_backend, actual_backend, fallback,
                 confidence, latency_ms, cost, success, source),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        # Logging failures should never take down a successful query response.
        print(f"[query_log] failed to write log row: {e}", file=sys.stderr)


class QueryRequest(BaseModel):
    query_text: str
    filters: Optional[Dict[str, Any]] = None
    semantic_hint: Optional[bool] = None
    execute: bool = True  # if False, only return the routing decision


class RouteResponse(BaseModel):
    query_text: str
    predicted_backend: str
    actual_backend: str
    fallback: bool
    confidence: float
    all_backend_scores: Dict[str, float]
    features_used: Dict[str, Any]
    executed: bool
    result: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None


class NLQueryResponse(RouteResponse):
    translated_query: Optional[str] = None
    translation_explanation: Optional[str] = None
    translation_used: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route_query(req: QueryRequest):
    predicted_backend, scores, feats = _classify(req.query_text, req.filters, req.semantic_hint)
    confidence = scores[predicted_backend]

    result, latency_ms, actual_backend, fell_back, executed, success = (
        None, None, predicted_backend, False, False, True
    )
    if req.execute:
        try:
            result, latency_ms, actual_backend, fell_back = _execute_with_fallback(
                predicted_backend, req.query_text
            )
            executed = True
        except HTTPException:
            raise
        except Exception as e:
            success = False
            _log_execution(req.query_text, predicted_backend, predicted_backend,
                            False, confidence, 0.0, False, source="route")
            raise HTTPException(502, f"Backend '{predicted_backend}' execution failed: {e}")
        _log_execution(req.query_text, predicted_backend, actual_backend,
                        fell_back, confidence, latency_ms, True, source="route")

    return RouteResponse(
        query_text=req.query_text,
        predicted_backend=predicted_backend,
        actual_backend=actual_backend,
        fallback=fell_back,
        confidence=confidence,
        all_backend_scores=scores,
        features_used=feats.__dict__,
        executed=executed,
        result=result,
        latency_ms=latency_ms,
    )


@app.post("/nl-query", response_model=NLQueryResponse)
def nl_query(req: QueryRequest):
    predicted_backend, scores, feats = _classify(req.query_text, req.filters, req.semantic_hint)
    confidence = scores[predicted_backend]

    translated_query, explanation, translation_used = None, None, False
    exec_kwargs = {}
    if nl_to_query is not None and os.getenv("ANTHROPIC_API_KEY"):
        try:
            translation = nl_to_query.translate(req.query_text, predicted_backend, req.filters)
            translated_query = translation["translated_query"]
            explanation = translation.get("explanation")
            translation_used = True
            if predicted_backend == "postgres":
                exec_kwargs = {"translated_sql": translated_query}
            elif predicted_backend == "vector":
                exec_kwargs = {"translated_description": translated_query}
            elif predicted_backend == "redis":
                exec_kwargs = {"cache_key": translated_query}
        except Exception as e:
            print(f"[nl_to_query] translation failed, using canned query: {e}", file=sys.stderr)

    result, latency_ms, actual_backend, fell_back, executed = None, None, predicted_backend, False, False
    if req.execute:
        try:
            result, latency_ms, actual_backend, fell_back = _execute_with_fallback(
                predicted_backend, req.query_text, **exec_kwargs
            )
            executed = True
        except Exception as e:
            _log_execution(req.query_text, predicted_backend, predicted_backend,
                            False, confidence, 0.0, False, source="nl-query")
            raise HTTPException(502, f"Backend '{predicted_backend}' execution failed: {e}")
        _log_execution(req.query_text, predicted_backend, actual_backend,
                        fell_back, confidence, latency_ms, True, source="nl-query")

    return NLQueryResponse(
        query_text=req.query_text,
        predicted_backend=predicted_backend,
        actual_backend=actual_backend,
        fallback=fell_back,
        confidence=confidence,
        all_backend_scores=scores,
        features_used=feats.__dict__,
        executed=executed,
        result=result,
        latency_ms=latency_ms,
        translated_query=translated_query,
        translation_explanation=explanation,
        translation_used=translation_used,
    )


@app.get("/frequency-table")
def frequency_table():
    """Debug endpoint: see what the router currently considers 'hot'."""
    return dict(_frequency_table)

@app.get("/metrics")
def metrics(hours: int = 24):
    """
    Aggregate router performance (query_log) vs the naive baseline
    (baseline_log) over the last `hours` hours -- this is the number Phase 4
    step 10 / Phase 6 step 15 want quoted: router vs naive, latency & cost.
    """
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT actual_backend AS backend,
                   COUNT(*) AS n,
                   AVG(latency_ms) AS avg_latency_ms,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                   SUM(cost_usd) AS total_cost_usd,
                   AVG(CASE WHEN fallback THEN 1 ELSE 0 END) AS fallback_rate,
                   AVG(CASE WHEN success THEN 1 ELSE 0 END) AS success_rate
            FROM query_log
            WHERE created_at > NOW() - INTERVAL '%s hours'
            GROUP BY actual_backend
            """,
            (hours,),
        )
        router_by_backend = cur.fetchall()

        cur.execute(
            """
            SELECT COUNT(*) AS n,
                   AVG(latency_ms) AS avg_latency_ms,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                   SUM(cost_usd) AS total_cost_usd
            FROM query_log
            WHERE created_at > NOW() - INTERVAL '%s hours'
            """,
            (hours,),
        )
        router_overall = cur.fetchone()

        cur.execute(
            """
            SELECT COUNT(*) AS n,
                   AVG(latency_ms) AS avg_latency_ms,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                   SUM(cost_usd) AS total_cost_usd
            FROM baseline_log
            WHERE created_at > NOW() - INTERVAL '%s hours'
            """,
            (hours,),
        )
        baseline_overall = cur.fetchone()

        improvement = None
        if router_overall and baseline_overall and baseline_overall["avg_latency_ms"]:
            r, b = router_overall, baseline_overall
            improvement = {
                "latency_improvement_pct": (
                    100.0 * (float(b["avg_latency_ms"]) - float(r["avg_latency_ms"]))
                    / float(b["avg_latency_ms"])
                ) if r["avg_latency_ms"] is not None else None,
                "cost_improvement_pct": (
                    100.0 * (float(b["total_cost_usd"] or 0) - float(r["total_cost_usd"] or 0))
                    / float(b["total_cost_usd"])
                ) if b["total_cost_usd"] else None,
            }

        return {
            "window_hours": hours,
            "router_by_backend": router_by_backend,
            "router_overall": router_overall,
            "naive_baseline_overall": baseline_overall,
            "router_vs_baseline": improvement,
        }
    finally:
        conn.close()


@app.get("/query-log")
def query_log(limit: int = 100):
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT log_id, query_text, predicted_backend, actual_backend, fallback,
                      confidence, latency_ms, cost_usd, success, source, created_at
               FROM query_log ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        return cur.fetchall()
    finally:
        conn.close()


@app.get("/classifier-accuracy")
def classifier_accuracy(limit: int = 50):
    """ accuracy-over-time series (evaluated on the frozen held-out
    bootstrap split each retrain -- see classifier/retrain_from_logs.py)."""
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT metric_id, trained_at, model_name, n_bootstrap_samples,
                      n_live_samples, accuracy_frozen_test, promoted, notes
               FROM classifier_metrics ORDER BY trained_at DESC LIMIT %s""",
            (limit,),
        )
        return cur.fetchall()
    finally:
        conn.close()
