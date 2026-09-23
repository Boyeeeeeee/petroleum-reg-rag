import os

import pytest
from fastapi.testclient import TestClient

from app import app  # importing app loads .env, so APP_API_KEY is available below

client = TestClient(app)
HEADERS = {"X-API-Key": os.environ["APP_API_KEY"]}


# ---- /health ----

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_loaded"] == 478


# ---- /query (BM25 only, no external calls, fast) ----

def test_query_returns_correct_top_citation():
    resp = client.post("/query", json={
        "question": "What is the penalty for gas flaring without authorisation?",
        "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["passages"]) == 5
    top = body["passages"][0]
    assert top["rank"] == 1
    assert "Gas Flaring" in top["citation"]
    assert "Regulation 16" in top["citation"]


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