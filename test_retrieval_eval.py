"""Retrieval regression tests: guard the headline eval numbers. No network calls, no API keys."""
import json
from pathlib import Path

import pytest

from bm25_retriever import BM25Retriever, load_chunks
from evaluate_retrieval import evaluate

EVAL_PATH = Path("data/processed/eval_set.json")

pytestmark = pytest.mark.skipif(
    not EVAL_PATH.exists(), reason="data/processed/eval_set.json is not in the repo"
)


@pytest.fixture(scope="module")
def results():
    from hybrid_retriever import HybridRetriever  # imported here so the skip above works without the model

    chunks = load_chunks()
    eval_set = json.load(open(EVAL_PATH))
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(chunks, bm25=bm25)
    return evaluate(bm25, eval_set), evaluate(hybrid, eval_set)


def test_hybrid_hit_rate_stays_high(results):
    _, hybrid = results
    # Currently 98/102 (96.1%). 97/102 (95.1%) still passes; 96/102 (94.1%) would fail.
    assert hybrid["hit_rate_at_k"] >= 0.95


def test_hybrid_is_not_worse_than_bm25(results):
    bm25, hybrid = results
    assert hybrid["hit_rate_at_k"] >= bm25["hit_rate_at_k"]
