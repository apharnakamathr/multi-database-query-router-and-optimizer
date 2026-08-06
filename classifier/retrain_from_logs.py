"""
Phase 4 steps 8-9: periodically recalibrate the classifier using real
execution history from query_log, instead of only the synthetic bootstrap
set from Phase 3.

The hard problem the project doc calls out: avoiding a feedback loop that
reinforces bad early routing decisions, and actually *validating* that a
retrain improves routing rather than just memorizing recent traffic. This
script handles that with three guardrails:

  1. Only successful, non-fallback, high-confidence live rows are treated
     as positive examples ("predicted_backend was probably right"). Low-
     confidence or failed/fallback rows are never used as ground truth --
     they're signal that something needs *investigation*, not a label to
     train on.
  2. Live examples are ADDED to the original frozen bootstrap set, never
     used alone -- so the model can't drift entirely onto whatever shape
     of traffic happened to dominate recently.
  3. Every retrain is scored against a FROZEN held-out split of the
     ORIGINAL bootstrap CSV (not live data). If the new model's accuracy
     on that frozen set regresses beyond a small tolerance, the retrain is
     logged but NOT promoted (model.pkl is left untouched). This is what
     lets you tell "the retrain actually generalizes better" apart from
     "the retrain just overfit to this week's traffic."

Run with:
    python classifier/retrain_from_logs.py
Intended to run on a schedule (cron / GitHub Actions schedule -- see
README's "Phase 4: scheduling retrains" section).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features import extract_features, FEATURE_ORDER

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))

BOOTSTRAP_CSV = os.path.join(HERE, "training_queries.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")

MIN_CONFIDENCE_FOR_LABEL = 0.75  # only trust confident, successful predictions as labels
ACCURACY_REGRESSION_TOLERANCE = 0.02  # allow up to 2pp drop before refusing to promote

PG_CONF = dict(
    host=os.getenv("PGBOUNCER_HOST", os.getenv("POSTGRES_HOST", "localhost")),
    port=os.getenv("PGBOUNCER_PORT", os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "router"),
    password=os.getenv("POSTGRES_PASSWORD", "router_pw"),
    dbname=os.getenv("POSTGRES_DB", "router_db"),
)


def load_bootstrap_rows():
    with open(BOOTSTRAP_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_live_rows(conn, since_hours: int, limit: int):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT query_text, predicted_backend, confidence
        FROM query_log
        WHERE success = TRUE
          AND fallback = FALSE
          AND confidence >= %s
          AND predicted_backend = actual_backend
          AND created_at > NOW() - INTERVAL '%s hours'
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (MIN_CONFIDENCE_FOR_LABEL, since_hours, limit),
    )
    return cur.fetchall()


def build_feature_matrix(rows, freq_table):
    X, y = [], []
    for row in rows:
        text = row["query_text"].strip().lower()
        label = row.get("label") or row.get("predicted_backend")
        repeat_count = int(row.get("repeat_count", 0))
        freq_table[text] += repeat_count
        feats = extract_features(text, frequency_table=freq_table)
        X.append(feats.to_vector())
        y.append(label)
        freq_table[text] += 1
    return X, y


def train_candidate(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X = np.array(X, dtype=float)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    }
    best_name, best_model, best_acc = None, None, -1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc
    return best_name, best_model, le


def evaluate_on_frozen_holdout(model, le, bootstrap_rows):
    """The one number that actually matters: accuracy on a split of the
    ORIGINAL bootstrap set that's held out the same way every single
    retrain, so it's comparable run over run and can't be gamed by live
    traffic composition."""
    freq_table = defaultdict(int)
    X, y = build_feature_matrix(bootstrap_rows, freq_table)
    X = np.array(X, dtype=float)
    y_enc = le.transform(y)
    _, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def log_retrain(conn, model_name, n_bootstrap, n_live, accuracy, promoted, notes):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO classifier_metrics
           (model_name, n_bootstrap_samples, n_live_samples, accuracy_frozen_test, promoted, notes)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (model_name, n_bootstrap, n_live, float(accuracy), promoted, notes),
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=int, default=24 * 7,
                     help="how far back to pull live query_log rows from")
    ap.add_argument("--max-live-rows", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true",
                     help="evaluate and log, but never overwrite model.pkl")
    args = ap.parse_args()

    old_bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    bootstrap_rows = load_bootstrap_rows()

    conn = psycopg2.connect(**PG_CONF)
    try:
        live_rows = load_live_rows(conn, args.since_hours, args.max_live_rows)
        print(f"Loaded {len(bootstrap_rows)} bootstrap rows + {len(live_rows)} live rows "
              f"(confidence>={MIN_CONFIDENCE_FOR_LABEL}, non-fallback, successful).")

        freq_table = defaultdict(int)
        X_boot, y_boot = build_feature_matrix(bootstrap_rows, freq_table)
        X_live, y_live = build_feature_matrix(live_rows, freq_table)
        X_all, y_all = X_boot + X_live, y_boot + y_live

        model_name, candidate_model, le = train_candidate(X_all, y_all)
        new_acc = evaluate_on_frozen_holdout(candidate_model, le, bootstrap_rows)

        old_acc = None
        if old_bundle is not None:
            old_acc = evaluate_on_frozen_holdout(
                old_bundle["model"], old_bundle["label_encoder"], bootstrap_rows
            )

        promote = args.dry_run is False and (old_acc is None or new_acc >= old_acc - ACCURACY_REGRESSION_TOLERANCE)
        notes = (
            f"old_acc={old_acc:.3f} new_acc={new_acc:.3f}" if old_acc is not None
            else f"new_acc={new_acc:.3f} (no previous model to compare)"
        )
        print(notes, f"-> {'PROMOTED' if promote else 'not promoted'}")

        if promote:
            bundle = {
                "model": candidate_model,
                "model_name": model_name,
                "label_encoder": le,
                "feature_order": FEATURE_ORDER,
                "trained_at": datetime.utcnow().isoformat(),
                "n_bootstrap_samples": len(bootstrap_rows),
                "n_live_samples": len(live_rows),
            }
            joblib.dump(bundle, MODEL_PATH)
            print(f"Promoted new model to {MODEL_PATH}")

        log_retrain(conn, model_name, len(bootstrap_rows), len(live_rows), new_acc, promote, notes)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
