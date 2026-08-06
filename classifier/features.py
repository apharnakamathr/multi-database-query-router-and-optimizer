"""
Phase 3, step 5: turn an incoming query into a feature vector the
classifier can use to pick a backend.

Design note: at this stage (Phase 1 + 3 only, no real workload yet from
Phase 2 and no feedback loop yet from Phase 4) features are derived from
query *shape* via simple, explainable heuristics rather than a learned NLU
model. That keeps the classifier's decisions inspectable ("why did this
query get routed to Postgres?") which the doc calls out as a requirement.
"""
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Words that signal semantic/similarity intent vs. exact structured lookups
SEMANTIC_KEYWORDS = {
    "similar", "like", "recommend", "related", "resembling", "find me",
    "sounds like", "matches the vibe", "closest to", "semantically",
}
FILTER_KEYWORDS = {
    "where", "equals", "greater than", "less than", "between", "filter",
    "sort by", "group by", "count", "sum", "average", "join",
}
CACHE_HINT_KEYWORDS = {"lookup", "get product", "fetch id", "by id"}

# In a running system this would be a real frequency counter backed by
# query_log; here it's injected so features.py stays a pure function that's
# easy to unit test, and the router (Phase 3 step 7) supplies live counts.
DEFAULT_FREQUENCY_TABLE: Dict[str, int] = {}


@dataclass
class QueryFeatures:
    query_length: int
    n_tokens: int
    has_filter_keywords: int      # 0/1 -- structured/SQL-shaped intent
    has_semantic_keywords: int    # 0/1 -- similarity/semantic intent
    has_numeric_id: int           # 0/1 -- looks like a direct key lookup
    is_repeat_lookup: int         # 0/1 -- seen this exact query recently?
    historical_frequency: int     # how many times we've seen this query
    has_explicit_filter_dict: int # 0/1 -- caller passed structured filters
    has_explicit_semantic_flag: int  # 0/1 -- caller flagged semantic intent

    def to_vector(self):
        d = asdict(self)
        return [d[k] for k in FEATURE_ORDER]


FEATURE_ORDER = [
    "query_length",
    "n_tokens",
    "has_filter_keywords",
    "has_semantic_keywords",
    "has_numeric_id",
    "is_repeat_lookup",
    "historical_frequency",
    "has_explicit_filter_dict",
    "has_explicit_semantic_flag",
]


def extract_features(
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    semantic_hint: Optional[bool] = None,
    frequency_table: Optional[Dict[str, int]] = None,
) -> QueryFeatures:
    text = (query_text or "").strip().lower()
    freq_table = frequency_table if frequency_table is not None else DEFAULT_FREQUENCY_TABLE
    historical_frequency = freq_table.get(text, 0)

    return QueryFeatures(
        query_length=len(text),
        n_tokens=len(text.split()),
        has_filter_keywords=int(any(k in text for k in FILTER_KEYWORDS)),
        has_semantic_keywords=int(any(k in text for k in SEMANTIC_KEYWORDS)),
        has_numeric_id=int(bool(re.search(r"\bid\s*[:=]?\s*\d+\b|\b\d{2,}\b", text))
                            or any(k in text for k in CACHE_HINT_KEYWORDS)),
        is_repeat_lookup=int(historical_frequency >= 3),
        historical_frequency=historical_frequency,
        has_explicit_filter_dict=int(bool(filters)),
        has_explicit_semantic_flag=int(bool(semantic_hint)),
    )
