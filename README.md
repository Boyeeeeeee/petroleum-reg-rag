# Petroleum Regulation RAG

A retrieval-augmented question-answering system over Nigerian upstream
petroleum regulation: the Petroleum Industry Act 2021 plus three NUPRC
regulations (Royalty, Gas Flaring/Venting/Methane, Decommissioning &
Abandonment).

## Live demo: https://petroleum-reg-rag.onrender.com/docs
Free-tier hosting, so the first request after a period of inactivity can take about a minute to wake up. `/health` and `/query` are open; `/ask` (LLM answer synthesis) requires an `X-API-Key` header to protect the free Groq quota.

## Status
- [x] PDF ingestion (`extract_text.py`)
- [x] Section/regulation-aware chunking (`chunk_by_section.py`) — 478 chunks
- [x] Eval set (`build_eval_set.py`) — 52 answerable + 10 deliberately unanswerable questions
- [x] BM25 sparse retrieval (`bm25_retriever.py`)
- [x] Dense retrieval (`dense_retriever.py`) — `BAAI/bge-small-en-v1.5` embeddings, generated in Colab
- [x] Hybrid retrieval (`hybrid_retriever.py`) — BM25 + dense, fused with reciprocal rank fusion
- [x] Retrieval evaluation (`evaluate_retrieval.py`) — **hybrid Hit Rate@5: 98.1%, MRR: 0.859** (BM25 baseline: 90.4%, 0.821)
- [x] FastAPI service (`app.py`) — `/query` (raw retrieval) and `/ask` (LLM answer synthesis + grounding check)
- [x] Tests (`test_app.py`, pytest) — 13 tests; the `/ask` tests make real Groq calls, not mocks
- [x] Docker + deployment (Render, free tier)
- [ ] CI

## Corpus
| Document | Units |
|---|---|
| Petroleum Industry Act, 2021 | 319 Sections + 57 Schedule paragraphs |
| Petroleum Royalty Regulations, 2022 | 48 Regulations |
| Gas Flaring, Venting & Methane Emissions Regs, 2023 | 28 Regulations |
| Decommissioning & Abandonment Regs, 2023 | 26 Regulations |

Source PDFs are official Federal Republic of Nigeria gazette copies —
not included in this repo (see `data/raw/README.md` for how to obtain them).

## Pipeline
```
data/raw/*.pdf
  -> extract_text.py        -> data/processed/*_pages.json (per-page text)
  -> chunk_by_section.py    -> data/processed/all_chunks.json (478 chunks)
  -> build_eval_set.py      -> data/processed/eval_set.json (52 + 10 items)
  -> notebooks/dense_embeddings.ipynb (Colab)
                            -> data/processed/embeddings.npy + embeddings_meta.json
  -> bm25_retriever.py      -> sparse retrieval
  -> dense_retriever.py     -> dense retrieval (cosine similarity over embeddings.npy)
  -> hybrid_retriever.py    -> reciprocal rank fusion of the two
  -> evaluate_retrieval.py  -> Hit Rate@5 / MRR against eval_set.json
  -> app.py                 -> FastAPI service (/health, /query, /ask)
```

## Retrieval eval results
Run `python3 evaluate_retrieval.py bm25 dense hybrid` to reproduce
(52 answerable questions, top-5).

| Retriever | Hit Rate@5 | MRR | Misses |
|---|---|---|---|
| BM25 | 90.4% (47/52) | 0.821 | 5 |
| Dense (`bge-small-en-v1.5`) | 94.2% (49/52) | 0.843 | 3 |
| **Hybrid (BM25 + dense, RRF)** | **98.1% (51/52)** | **0.859** | 1 |

**BM25 and dense fail on different questions.** BM25 alone missed three
questions that dense got right, and dense alone missed one that BM25 got
right. That includes the genuine paraphrase-gap failure found in the BM25
baseline: a query using "notify the Commission" failed to retrieve the
decommissioning regulation whose heading is "Post-completion of
decommissioning and abandonment programme" — different wording, same
meaning. Dense retrieval bridges that gap. One question (Seventh Schedule,
paragraph 1) was missed by BM25 and by dense on their own but found by the
fused ranking.

**Hybrid mostly improves recall, not the top slot.** MRR moves less than
Hit Rate (0.843 to 0.859), so the right passage is more often *somewhere in
the top 5* but not always first. For example, on "What is the penalty for
gas flaring without authorisation?" hybrid ranks PIA Section 105 first and
the Regulation 16 penalty schedule inside the top 5, where BM25 alone had
Regulation 16 first. Since `/ask` passes all retrieved passages to the LLM,
top-5 recall is the metric that matters most here.

**Remaining miss:** PIA Section 309 ("If another law conflicts with the
Petroleum Industry Act, which prevails, and what is the notable exception?")
is missed by all three retrievers. Not yet diagnosed.

**Read these numbers with care.** The eval set has 52 answerable questions,
so the gaps between retrievers are one to four questions each: encouraging,
not statistically established. The questions were generated per chunk, and
some name the section or schedule directly ("Regulation 47", "paragraph 1 of
the Seventh Schedule"), which favours methods that can match those labels.
The hybrid constants (RRF k=60, top-100 pool, equal weights) were left at
their defaults on purpose, so the numbers are not tuned to this small set.

## Why there's no fixed confidence threshold
An earlier version used a fixed BM25 score cutoff to flag likely-unanswerable
questions. Checking the actual score distributions showed real overlap
between answerable and unanswerable questions — 11 of 52 legitimately
answerable questions scored *below* the highest-scoring unanswerable
question. Two nearly-identical BM25 scores (17.557 vs. 17.421) turned out
to belong to one perfectly valid question and one completely out-of-scope
one — a fixed number cannot tell them apart.

Hybrid scores make this even less usable: reciprocal rank fusion scores
depend only on ranks (they top out around 0.033), so they say nothing about
whether a passage answers the question. `/query` no longer returns a
confidence flag.

Instead, `/ask` uses an LLM to read the retrieved passages and judge
directly whether they answer the question (`ANSWERABLE: yes/no`), which
correctly separated the same two cases where the raw score could not.

## API
- **`GET /health`** — `{"status", "chunks_loaded", "retriever"}`; `retriever`
  is `hybrid` or `bm25`.
- **`POST /query`** — `{"question": "...", "top_k": 5}`. Raw retrieval only
  (hybrid by default). Returns ranked passages with citations, a fused
  score, and `bm25_rank` / `dense_rank` showing where each retriever ranked
  that passage on its own. No LLM call, no cost, useful for debugging
  retrieval in isolation.
- **`POST /ask`** — same body plus an `X-API-Key` header. Full RAG:
  retrieves passages, sends them to an LLM (full passage text, capped at
  2,000 characters each) to synthesize a cited answer, and judges whether the
  passages actually ground it (`grounded: true/false`). Returns 401 on a
  missing or wrong key and 502 if the LLM call fails.

Answer synthesis runs on **Groq's free tier** (`openai/gpt-oss-120b`),
not a paid API — a deliberate cost-conscious choice for a portfolio
project, not a compromise. Groq's SDK is OpenAI-compatible.

**Configuration (environment variables)**

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq key for `/ask` |
| `APP_API_KEY` | Key clients must send in `X-API-Key` for `/ask` |
| `RETRIEVER_MODE` | Optional. `hybrid` (default) or `bm25`. BM25-only fallback if a host is short on memory |

## Setup
```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Ingestion (run once; source PDFs must be in data/raw/ first)
python3 extract_text.py
python3 chunk_by_section.py
python3 build_eval_set.py

# Dense embeddings (run once, in Colab)
# Open notebooks/dense_embeddings.ipynb in Colab, put all_chunks.json in
# MyDrive/petroleum-reg-rag/ on Google Drive, run all cells, then download
# embeddings.npy and embeddings_meta.json into data/processed/

python3 evaluate_retrieval.py bm25 dense hybrid   # sanity check: hybrid should print Hit Rate@5 98.1%

# API
# Create a .env file with:
#   GROQ_API_KEY=your_key_here      (free, no credit card — https://console.groq.com)
#   APP_API_KEY=any_long_random_string
# Generate a key with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
uvicorn app:app --reload
# then open http://127.0.0.1:8000/docs

# Tests (the /ask tests call Groq for real, so they need the keys above)
python3 -m pytest -q
```

**Docker**
```bash
docker build -t petroleum-reg-rag .
docker run --rm -p 8000:8000 --env-file .env petroleum-reg-rag
```
The embedding model is downloaded at image build time, so the first request
doesn't wait on a download. In a local Docker test the running container
used about 250 MiB of memory (Render's free tier limit is 512 MB).

## Known limitations and next steps
- **Small eval set.** 52 answerable questions, auto-generated per chunk and
  only spot-checked. A larger, human-reviewed set would make the retriever
  comparison meaningful at the level of a percentage point.
- **Long chunks are only partly embedded.** 88 of the 478 chunks are over
  ~350 words, and `bge-small` reads roughly the first 512 tokens. The longest
  is PIA Section 318 (6,235 words, the Act's interpretation section). BM25
  sees the full text, which is one reason hybrid beats dense alone. Next
  step: embed long chunks as overlapping windows that map back to their
  parent chunk, keeping citations unchanged.
- **Citation labels are not part of the indexed text.** Queries that name a
  section or schedule ("Regulation 47", "Seventh Schedule") depend on that
  number appearing in the chunk text. Prepending the citation label to the
  indexed text is a cheap experiment, best judged on a larger eval set.
- **No reranker yet.** A cross-encoder over the fused top results is untested.
- **Section 309 miss** (above) still needs a diagnosis.
- **Free-tier cold starts.** The service sleeps when idle, so the first
  request after a pause can be slow.
- **No CI yet.**