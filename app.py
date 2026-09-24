import os
import secrets

from dotenv import load_dotenv
from groq import Groq
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from bm25_retriever import BM25Retriever, load_chunks

# Load .env (local dev) before anything reads environment variables.
# On Render/Docker there is no .env file; real env vars are used instead.
load_dotenv()

app = FastAPI(
    title="Petroleum Regulation RAG API",
    description="Retrieval over the Petroleum Industry Act 2021 and NUPRC regulations (Royalty, Gas Flaring, Decommissioning).",
    version="0.2.0",
)

# ---------- Retriever (loaded once at startup, not per-request) ----------
# RETRIEVER_MODE=hybrid (default): BM25 + dense embeddings fused with reciprocal rank fusion.
# RETRIEVER_MODE=bm25: sparse-only fallback, e.g. if the host runs out of memory.
RETRIEVER_MODE = os.environ.get("RETRIEVER_MODE", "hybrid").strip().lower()
if RETRIEVER_MODE not in {"bm25", "hybrid"}:
    raise RuntimeError(f"RETRIEVER_MODE must be 'bm25' or 'hybrid', got {RETRIEVER_MODE!r}")

_chunks = load_chunks()
if RETRIEVER_MODE == "hybrid":
    # Imported lazily so bm25 mode works without fastembed installed.
    from hybrid_retriever import HybridRetriever
    _retriever = HybridRetriever(_chunks)
else:
    _retriever = BM25Retriever(_chunks)

# Warm up now so the embedding model is loaded before the first real request.
_retriever.search("warm-up query", top_k=1)

# NOTE ON CONFIDENCE: an earlier version flagged low-relevance queries with a fixed
# BM25 score threshold. On the eval set the score distributions of answerable and
# unanswerable questions overlapped, so no fixed cutoff worked. Hybrid scores are
# reciprocal-rank-fusion values (max ~0.033) that only reflect rank agreement, so
# they are even less usable as a confidence signal. Relevance is judged by /ask,
# where the LLM reads the retrieved text and states ANSWERABLE: yes/no.
SCORE_NOTES = {
    "hybrid": (
        "Scores are reciprocal-rank-fusion values that reflect rank agreement between BM25 "
        "and dense retrieval, not answer confidence. Use /ask for an answer grounded in the retrieved text."
    ),
    "bm25": (
        "Scores are raw BM25 term-overlap values, not a reliable confidence signal. "
        "Use /ask for an answer grounded in the retrieved text."
    ),
}

# Passages are sent to the LLM in full (not just the 220-char preview), capped so a very
# long section (e.g. the Act's interpretation section) cannot blow up the prompt.
MAX_PASSAGE_CHARS = 2000


# ---------- Auth (protects /ask, which spends Groq quota) ----------

def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("APP_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Server is missing APP_API_KEY configuration.")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# ---------- Groq client (created lazily so /health and /query work without a key) ----------

GROQ_MODEL = "openai/gpt-oss-120b"
_groq_client = None


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


# ---------- Models ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="A question about Nigerian upstream petroleum regulation.")
    top_k: int = Field(5, ge=1, le=20)


class RetrievedPassage(BaseModel):
    rank: int
    score: float
    citation: str
    text_preview: str
    # Present in hybrid mode: where each retriever ranked this passage on its own.
    bm25_rank: int | None = None
    dense_rank: int | None = None


class QueryResponse(BaseModel):
    question: str
    retriever: str
    passages: list[RetrievedPassage]
    note: str | None = None


class AskResponse(BaseModel):
    question: str
    grounded: bool
    answer: str
    passages_used: list[RetrievedPassage]


def _to_passage(r: dict) -> RetrievedPassage:
    return RetrievedPassage(
        rank=r["rank"],
        score=r["score"],
        citation=r["citation"],
        text_preview=r["text_preview"],
        bm25_rank=r.get("bm25_rank"),
        dense_rank=r.get("dense_rank"),
    )


def _passage_for_prompt(r: dict) -> str:
    text = r["text"]
    if len(text) > MAX_PASSAGE_CHARS:
        text = text[:MAX_PASSAGE_CHARS] + " [truncated]"
    return f"[{r['citation']}]\n{text}"


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(_chunks), "retriever": RETRIEVER_MODE}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    results = _retriever.search(req.question, top_k=req.top_k)
    return QueryResponse(
        question=req.question,
        retriever=RETRIEVER_MODE,
        passages=[_to_passage(r) for r in results],
        note=SCORE_NOTES[RETRIEVER_MODE],
    )


SYSTEM_PROMPT = """You are a legal research assistant answering questions about \
Nigerian upstream petroleum regulation, using ONLY the numbered source passages \
provided below. Do not use any outside knowledge, even if you know the answer.

Rules:
1. Answer only from the passages given. Cite the exact citation label for every claim.
2. If the passages do not clearly answer the question, say so explicitly. Do not guess or fill gaps with general knowledge.
3. Your response MUST start with exactly one of these two lines:
   ANSWERABLE: yes
   ANSWERABLE: no
   Then a blank line, then your answer (or a one-sentence explanation of why the passages don't answer it, if "no").
"""


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: QueryRequest):
    results = _retriever.search(req.question, top_k=req.top_k)

    context = "\n\n".join(_passage_for_prompt(r) for r in results)
    user_prompt = f"Question: {req.question}\n\nSource passages:\n\n{context}"

    try:
        completion = get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {type(e).__name__}")

    raw = completion.choices[0].message.content.strip()

    grounded = raw.upper().startswith("ANSWERABLE: YES")
    answer = raw.split("\n\n", 1)[1].strip() if "\n\n" in raw else raw

    return AskResponse(
        question=req.question,
        grounded=grounded,
        answer=answer,
        passages_used=[_to_passage(r) for r in results],
    )