"""
Phase 3, steps 5-6:
  - extract features per query
  - train a simple, explainable classifier to predict the best backend

Run with:  python classifier/train_classifier.py
Produces:  classifier/model.pkl  (joblib bundle: model + label encoder + feature order)
"""
import csv
import os
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features import extract_features, FEATURE_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "training_queries.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")


def load_dataset(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_feature_matrix(rows):
    # simulate a running frequency table: a query's "historical_frequency"
    # is how many times we've seen it *before* this occurrence, so repeat
    # cache-style lookups naturally accumulate a higher count -- mirrors
    # what the live router's frequency_table will do in production.
    freq_table = defaultdict(int)
    X, y = [], []
    for row in rows:
        text = row["query_text"].strip().lower()
        repeat_count = int(row["repeat_count"])
        freq_table[text] += repeat_count  # seed frequency for repeat-heavy rows
        feats = extract_features(text, frequency_table=freq_table)
        X.append(feats.to_vector())
        y.append(row["label"])
        freq_table[text] += 1  # this occurrence itself
    return np.array(X, dtype=float), np.array(y)


def train():
    rows = load_dataset(DATA_PATH)
    X, y = build_feature_matrix(rows)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    }

    best_name, best_model, best_acc = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nSelected model: {best_name} (accuracy={best_acc:.3f})")

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": le,
        "feature_order": FEATURE_ORDER,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")

    if hasattr(best_model, "feature_importances_"):
        print("\nFeature importances:")
        for name, imp in sorted(
            zip(FEATURE_ORDER, best_model.feature_importances_), key=lambda t: -t[1]
        ):
            print(f"  {name:30s} {imp:.3f}")


if __name__ == "__main__":
    train()
