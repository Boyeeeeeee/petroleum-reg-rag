import os

import pytest
from fastapi.testclient import TestClient

from app import app, RETRIEVER_MODE  # importing app loads .env, so APP_API_KEY is available below

client = TestClient(app)
HEADERS = {"X-API-Key": os.environ["APP_API_KEY"]}

needs_hybrid = pytest.mark.skipif(
    RETRIEVER_MODE != "hybrid", reason="hybrid retrieval is not enabled (RETRIEVER_MODE=bm25)"
)


# ---- /health ----

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_loaded"] == 478
    assert body["retriever"] in {"bm25", "hybrid"}


# ---- /query (local retrieval only, no external calls) ----

def test_query_returns_gas_flaring_penalty_regulation():
    resp = client.post("/query", json={
        "question": "What is the penalty for gas flaring without authorisation?",
        "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["passages"]) == 5
    assert body["passages"][0]["rank"] == 1
    # The Regulation 16 penalty must be in the top 5. Hybrid ranking puts PIA Section 105
    # (penalty "prescribed pursuant to" the regulations) first; BM25 alone had Regulation 16 first.
    citations = [p["citation"] for p in body["passages"]]
    assert any("Gas Flaring" in c and c.endswith("Regulation 16") for c in citations)


@needs_hybrid
def test_query_hybrid_reports_component_ranks():
    resp = client.post("/query", json={
        "question": "What is the penalty for gas flaring without authorisation?",
        "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["retriever"] == "hybrid"
    top = body["passages"][0]
    assert top["bm25_rank"] is not None
    assert top["dense_rank"] is not None


@needs_hybrid
def test_query_hybrid_finds_paraphrased_decommissioning_notice():
    # BM25 alone misses this one (expected Regulation 15); hybrid retrieval finds it in the top 5.
    resp = client.post("/query", json={
        "question": "How soon after completing decommissioning and abandonment must a licensee notify the Commission?",
        "top_k": 5,
    })
    assert resp.status_code == 200
    citations = [p["citation"] for p in resp.json()["passages"]]
    assert any("Decommissioning" in c and c.endswith("Regulation 15") for c in citations)


@needs_hybrid
def test_query_hybrid_finds_regulation_named_in_question():
    # BM25 alone misses this one (expected Royalty Regulations, Regulation 47).
    resp = client.post("/query", json={
        "question": "How does Regulation 47 define 'arm's length'?",
        "top_k": 5,
    })
    assert resp.status_code == 200
    citations = [p["citation"] for p in resp.json()["passages"]]
    assert any("Royalty" in c and c.endswith("Regulation 47") for c in citations)


def test_query_respects_top_k():
    resp = client.post("/query", json={
        "question": "decommissioning fund",
        "top_k": 3,
    })
    assert resp.status_code == 200
    assert len(resp.json()["passages"]) == 3


def test_query_rejects_too_short_question():
    resp = client.post("/query", json={"question": "ab", "top_k": 5})
    assert resp.status_code == 422  # min_length=3 validation


def test_query_rejects_missing_question():
    resp = client.post("/query", json={"top_k": 5})
    assert resp.status_code == 422


# ---- /ask (real Groq calls, no mocking; requires X-API-Key) ----

def test_ask_grounds_answerable_question():
    resp = client.post("/ask", json={
        "question": "What is the penalty for gas flaring without authorisation?",
        "top_k": 5,
    }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    # The specific figure from Regulation 16 should appear in the synthesized answer
    assert "3.50" in body["answer"]
    assert any("Regulation 16" in p["citation"] for p in body["passages_used"])


def test_ask_declines_unanswerable_question():
    resp = client.post("/ask", json={
        "question": "What is the current price of Brent crude oil?",
        "top_k": 5,
    }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is False


def test_ask_returns_passages_used_matching_top_k():
    resp = client.post("/ask", json={
        "question": "What must a licensee do before decommissioning a well?",
        "top_k": 3,
    }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["passages_used"]) == 3


# ---- /ask auth ----

def test_ask_rejects_missing_api_key():
    resp = client.post("/ask", json={"question": "What is gas flaring?", "top_k": 3})
    assert resp.status_code == 401


def test_ask_rejects_wrong_api_key():
    resp = client.post(
        "/ask",
        json={"question": "What is gas flaring?", "top_k": 3},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401