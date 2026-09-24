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
- [x] Retrieval evaluation (`evaluate_retrieval.py`) — **Hit Rate@5: 90.4%, MRR: 0.821**
- [x] FastAPI service (`app.py`) — `/query` (raw retrieval) and `/ask` (LLM answer synthesis + grounding check)
- [ ] Dense embeddings + vector store (Colab — needs Hugging Face access)
- [ ] Hybrid retrieval (BM25 + dense + rerank)
- [ ] Tests + CI
- [x] Docker + deployment

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
  -> bm25_retriever.py      -> sparse retrieval
  -> evaluate_retrieval.py  -> Hit Rate@5 / MRR against eval_set.json
  -> app.py                 -> FastAPI service (/query, /ask)
```

## Retrieval eval results (BM25 baseline)
Run `python3 evaluate_retrieval.py` to reproduce.

- **Hit Rate@5: 90.4%** (47/52)
- **MRR: 0.821**
- 5 misses, including one genuine paraphrase-gap failure: a query using
  "notify the Commission" failed to retrieve the regulation whose actual
  heading is "Post-completion of decommissioning and abandonment
  programme" — different wording, same meaning. BM25's exact-term
  matching can't bridge that; motivates adding dense embeddings.

## Why there's no fixed confidence threshold
An earlier version used a fixed BM25 score cutoff to flag likely-unanswerable
questions. Checking the actual score distributions showed real overlap
between answerable and unanswerable questions — 11 of 52 legitimately
answerable questions scored *below* the highest-scoring unanswerable
question. Two nearly-identical BM25 scores (17.557 vs. 17.421) turned out
to belong to one perfectly valid question and one completely out-of-scope
one — a fixed number cannot tell them apart.

Instead, `/ask` uses an LLM to read the retrieved passage and judge
directly whether it answers the question (`ANSWERABLE: yes/no`), which
correctly separated the same two cases where the raw score could not.

## API
Two endpoints, both POST with `{"question": "...", "top_k": 5}`:

- **`/query`** — raw BM25 retrieval only. Returns ranked passages with
  citations and scores. No LLM call, no cost, useful for debugging retrieval
  in isolation.
- **`/ask`** — full RAG: retrieves passages, then calls an LLM to
  synthesize a cited answer and judge whether the passages actually
  ground it (`grounded: true/false`).

Answer synthesis runs on **Groq's free tier** (`openai/gpt-oss-120b`),
not a paid API — a deliberate cost-conscious choice for a portfolio
project, not a compromise. Groq's SDK is OpenAI-compatible.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Ingestion (run once; source PDFs must be in data/raw/ first)
python3 extract_text.py
python3 chunk_by_section.py
python3 build_eval_set.py
python3 evaluate_retrieval.py   # sanity check: should print Hit Rate@5 90.4%

# API
# Create a .env file with: GROQ_API_KEY=your_key_here
# (free, no credit card — https://console.groq.com)
uvicorn app:app --reload
# then open http://127.0.0.1:8000/docs
```