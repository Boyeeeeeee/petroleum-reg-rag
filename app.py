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
    version="0.1.0",
)

# Loaded once at startup, not per-request.
_chunks = load_chunks()
_retriever = BM25Retriever(_chunks)

# NOTE ON CONFIDENCE: an earlier version of this API used a fixed BM25
# score threshold (25.0) to flag low-relevance queries. Checking the score
# distributions on the eval set showed real overlap between answerable and
# unanswerable questions (11/52 answerable questions scored below the
# highest unanswerable score), so a fixed cutoff produces both false
# negatives (good answers marked unconfident) and false positives if set
# too low. Raw BM25 score is a term-overlap statistic, not a semantic
# relevance judgment, so it's not a reliable confidence signal on its own.
#
# v1 (this file): report the score plainly and let the caller judge
# relevance from the retrieved text itself.
# v2: replace this with either (a) dense-embedding cosine similarity,
# which typically separates better, or (b) an LLM reading the top passage
# and judging whether it actually answers the question - a semantic check
# rather than a numeric threshold.

# Below the lowest score seen for ANY question in eval, answerable or not:
# a true floor, not a decision boundary.
LOW_RELEVANCE_HINT = 15.0
# Upper bound of unanswerable scores seen in eval.
UNANSWERABLE_CEILING = 27.4


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


class QueryResponse(BaseModel):
    question: str
    is_confident: bool
    passages: list[RetrievedPassage]
    note: str | None = None


class AskResponse(BaseModel):
    question: str
    grounded: bool
    answer: str
    passages_used: list[RetrievedPassage]


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(_chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    results = _retriever.search(req.question, top_k=req.top_k)
    top_score = results[0]["score"] if results else 0.0

    note = None
    if top_score < LOW_RELEVANCE_HINT:
        note = (
            "Very low retrieval score - this question is likely outside "
            "the loaded corpus (Petroleum Industry Act 2021, Royalty, Gas "
            "Flaring, and Decommissioning regulations)."
        )
    elif top_score < UNANSWERABLE_CEILING:
        note = (
            "Retrieval score is in a range where both relevant and "
            "out-of-scope questions have scored in testing - check the "
            "retrieved passage actually addresses the question before "
            "trusting the citation."
        )

    return QueryResponse(
        question=req.question,
        is_confident=top_score >= UNANSWERABLE_CEILING,
        passages=[
            RetrievedPassage(rank=r["rank"], score=r["score"], citation=r["citation"], text_preview=r["text_preview"])
            for r in results
        ],
        note=note,
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

    context = "\n\n".join(
        f"[{r['citation']}]\n{r['text_preview']}" for r in results
    )
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
        passages_used=[
            RetrievedPassage(rank=r["rank"], score=r["score"], citation=r["citation"], text_preview=r["text_preview"])
            for r in results
        ],
    )