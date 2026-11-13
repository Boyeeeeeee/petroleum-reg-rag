# Petroleum Regulation RAG
[![CI](https://github.com/Boyeeeeeee/petroleum-reg-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Boyeeeeeee/petroleum-reg-rag/actions/workflows/ci.yml)

A retrieval-augmented question-answering system over Nigerian upstream
petroleum regulation: the Petroleum Industry Act 2021 plus three NUPRC
regulations (Royalty, Gas Flaring/Venting/Methane, Decommissioning &
Abandonment).

## Live demo: https://petroleum-reg-rag.onrender.com/docs
Free-tier hosting, so the first request after a period of inactivity can take about a minute to wake up. `/health` and `/query` are open; `/ask` (LLM answer synthesis) requires an `X-API-Key` header to protect the free Groq quota.

## Status
- [x] PDF ingestion (`extract_text.py`)
- [x] Section/regulation-aware chunking (`chunk_by_section.py`) — 478 chunks
- [x] Eval set — 102 answerable + 10 deliberately unanswerable questions. The original 52 come from `build_eval_set.py`, the 50 added in gold set v2 from `gold_additions.py`; `merge_gold.py` combines them into `eval_set.json`
- [x] BM25 sparse retrieval (`bm25_retriever.py`)
- [x] Dense retrieval (`dense_retriever.py`) — `BAAI/bge-small-en-v1.5` embeddings, generated in Colab
- [x] Hybrid retrieval (`hybrid_retriever.py`) — BM25 + dense, fused with reciprocal rank fusion
- [x] Retrieval evaluation (`evaluate_retrieval.py`) — **hybrid Hit Rate@5: 96.1%, MRR: 0.882 on 102 questions** (BM25 baseline: 93.1%, 0.854). On the original 52 questions alone: hybrid 98.1%, 0.859
- [x] Gold set label checks (`verify_gold.py`) — confirms every new label's citation exists, is unique, and that an anchor phrase from the answer appears in the cited chunk
- [x] FastAPI service (`app.py`) — `/query` (raw retrieval) and `/ask` (LLM answer synthesis + grounding check)
- [x] Per-question regression gate (`build_baseline.py`, `test_regression_gate.py`) — CI fails if a question the baseline found is now missed, or if MRR drops by more than 0.01
- [x] Tests (pytest) — 20 tests: API tests in `test_app.py` (three `/ask` tests call Groq for real, not mocks), aggregate retrieval tests in `test_retrieval_eval.py` and the per-question gate in `test_regression_gate.py`
- [x] Docker + deployment (Render, free tier)
- [x] CI — GitHub Actions runs the offline tests (no Groq calls) on every push

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
  -> build_eval_set.py      (original 52 + 10 questions, imported by merge_gold.py)
  -> gold_additions.py      (50 new questions, imported by merge_gold.py)
  -> merge_gold.py          -> data/processed/eval_set.json (102 + 10 items)
  -> verify_gold.py         -> label checks on the 50 new questions
  -> build_baseline.py      -> data/processed/baseline.json (per-question ranks + data hashes)
  -> notebooks/dense_embeddings.ipynb (Colab)
                            -> data/processed/embeddings.npy + embeddings_meta.json
  -> bm25_retriever.py      -> sparse retrieval
  -> dense_retriever.py     -> dense retrieval (cosine similarity over embeddings.npy)
  -> hybrid_retriever.py    -> reciprocal rank fusion of the two
  -> evaluate_retrieval.py  -> Hit Rate@5 / MRR against eval_set.json
  -> app.py                 -> FastAPI service (/health, /query, /ask)
```

## Gold set v2 (102 questions)
The eval set grew from 52 to 102 answerable questions (plus 10 unanswerable).
The original 52 were kept word for word so earlier numbers stay comparable.
The 50 new questions (ids `g001`–`g050`) were written without naming a
section, regulation or paragraph number, so retrieval has to work from the
meaning of the question. Each item now has an id, a type (`number`, `who`,
`process`, `yes_no`), a list of acceptable citations, and a `known_failure`
flag (Section 309 only, for now). The 24 original
questions that name a unit (none of the new ones do) carry `names_unit: true`,
so they can be reported separately later.

Every new label was checked by script (`verify_gold.py`): the cited chunk
exists, no citation is duplicated, and a short anchor phrase from the answer
appears in the cited chunk. Anchors that also appear in other chunks are
flagged, and one label was replaced after that check because another section
gave an identical answer. I also checked each label against the chunk text by
hand. `merge_gold.py` generates `eval_set.json`; run it instead of
`build_eval_set.py`, which on its own writes only the original 52 + 10.

## Regression gate
The aggregate gate in `test_retrieval_eval.py` (hybrid Hit Rate@5 of at least
95%) can stay green while one question quietly breaks and another happens to
start working. `test_regression_gate.py` closes that gap. `build_baseline.py`
records, for every answerable question, the rank (1-5) at which each retriever
found the expected chunk, or `null` for a miss, in `data/processed/baseline.json`.
It also stores a hash of the chunk text and of the eval set content (parsed
JSON, so Windows and Linux line endings give the same value). On every CI run
the tests rebuild the rankings for BM25 and hybrid and fail if:

- a question the baseline found is now a miss (a new miss), or
- MRR drops by more than 0.01 (about two questions moving from rank 1 to rank 2), or
- the chunks or the eval set no longer match the baseline.

Questions the baseline already misses, including the Section 309 known failure,
cannot trigger a new-miss failure, and fixing them never fails a test. To accept
a deliberate change (new chunking, a retriever change, more questions), re-run
`python3 build_baseline.py bm25 dense hybrid` and commit `baseline.json` in the
same commit, with a message saying why the numbers moved. Dense is recorded in
the baseline for comparison but is not gated on its own; hybrid contains it.
BM25 is gated because `RETRIEVER_MODE=bm25` is a real deployment mode.

## Retrieval eval results
Run `python3 merge_gold.py` and then `python3 evaluate_retrieval.py bm25 dense hybrid`
to reproduce (top-5).

**Full gold set: 102 answerable questions**

| Retriever | Hit Rate@5 | MRR | Misses |
|---|---|---|---|
| BM25 | 93.1% (95/102) | 0.854 | 7 |
| Dense (`bge-small-en-v1.5`) | 94.1% (96/102) | 0.868 | 6 |
| **Hybrid (BM25 + dense, RRF)** | **96.1% (98/102)** | **0.882** | 4 |

**Original 52 questions only** (unchanged from before gold set v2)

| Retriever | Hit Rate@5 | MRR | Misses |
|---|---|---|---|
| BM25 | 90.4% (47/52) | 0.821 | 5 |
| Dense | 94.2% (49/52) | 0.843 | 3 |
| Hybrid | 98.1% (51/52) | 0.859 | 1 |

**The 50 new questions only:** BM25 48/50, dense 47/50, hybrid 47/50. The
three retrievers are within one question of each other here, so this set does
not separate them.

**BM25 and dense fail on different questions.** BM25 alone missed three
questions that dense got right, and dense alone missed one that BM25 got
right (the same split on the original 52 and on all 102, because the new
questions added no questions missed by only one of those two). That includes
the genuine paraphrase-gap failure found in the BM25 baseline: a query using
"notify the Commission" failed to retrieve the decommissioning regulation
whose heading is "Post-completion of decommissioning and abandonment
programme" — different wording, same meaning. Dense retrieval bridges that
gap. One question (Seventh Schedule, paragraph 1) was missed by BM25 and by
dense on their own but found by the fused ranking.

**Hybrid mostly improves recall, not the top slot.** MRR moves less than
Hit Rate (dense to hybrid on 102 questions: Hit Rate 94.1% to 96.1%, MRR
0.868 to 0.882), so the right passage is more often *somewhere in the top 5*
but not always first. For example, on "What is the penalty for gas flaring
without authorisation?" hybrid ranks PIA Section 105 first and the
Regulation 16 penalty schedule inside the top 5, where BM25 alone had
Regulation 16 first. Since `/ask` passes all retrieved passages to the LLM,
top-5 recall is the metric that matters most here.

**Remaining misses:** hybrid misses four questions. Three are missed by all
three retrievers (PIA Section 309, Section 23 and Section 281); the fourth,
Section 119 (notice before ceasing licensed activity), is missed by dense and
hybrid but found by BM25. See the error analysis below.

**Read these numbers with care.** The eval set has 102 answerable questions,
so the gaps between retrievers are one to three questions each: encouraging,
not statistically established. Twenty-four questions name the section or
schedule directly ("Regulation 47", "paragraph 1 of the Seventh Schedule"),
which favours methods that can match those labels. The hybrid constants
(RRF k=60, top-100 pool, equal weights) were left at their defaults on
purpose, so the numbers are not tuned to this set.

## Error analysis: questions all three retrievers miss
Three questions are missed by BM25, dense and hybrid. Section 309 is analysed
in detail; the two found with gold set v2 are described after it.

### Section 309 (known failure)
**Question:** "If another law conflicts with the Petroleum Industry Act, which
prevails, and what is the notable exception?" **Expected:** PIA Section 309
(the PIA prevails over inconsistent laws, except the Nigeria Oil and Gas
Industry Content Development Act).

`diagnose_miss.py` shows where the target chunk ranks (out of 478) for the
original question and for reworded versions:

| Query | BM25 | Dense | Hybrid |
|---|---|---|---|
| Original question | 64 | 22 | 21 |
| "Which law prevails if another law is **inconsistent** with the Petroleum Industry Act?" | 1 | 4 | 1 |
| "Petroleum Industry Act shall prevail over any other law except" | 1 | 45 | 11 |
| "conflict with other laws Petroleum Industry Act prevails" | 146 | 83 | 131 |
| The chunk's own opening text (sanity check) | 1 | 1 | 1 |

- **The chunk is fine.** Querying with the chunk's own text returns it at
  rank 1 in all three retrievers, so this is not a chunking failure.
- **It is a vocabulary gap.** The question says "conflicts", "prevails" and
  "notable exception"; Section 309 says "inconsistent", "prevail" and
  "except". None of "conflicts", "prevails", "exception" or "notable" appear
  in the chunk, and BM25 has no stemming, so "prevails" does not match
  "prevail". Changing one word ("conflicts" to "inconsistent") puts the
  chunk at rank 1 in BM25 and hybrid.
- **Dense retrieval only partly bridges the gap.** It improves the original
  question (rank 64 to 22) but not enough to reach the top 5, and it ranks the
  chunk 83rd for a short keyword-style paraphrase.
- **BM25's top hits look off-topic** (long sections on board membership and
  suspension). One possible cause is matches on common words in long
  sections. This is a hypothesis; stopword removal has not been tested.
- **User-facing effect:** in hybrid mode the target is at rank 21, beyond even
  the maximum `top_k` of 20, so `/ask` never sees Section 309 for this
  phrasing. Rewording the question with the statute's own vocabulary fixes it.

### Two more misses from gold set v2 (not yet diagnosed)
- **Section 23: Secretary of the Commission.** "What qualifications must the
  Secretary of the Nigerian Upstream Petroleum Regulatory Commission have?"
  All three retrievers return Section 4 (the Commission's legal status) at
  rank 1. Section 46 has near-identical wording for the Authority's
  Secretary, so these two sections are a pair of parallel provisions. Not yet
  checked: where Section 23 itself ranks, and whether Section 46 is among the
  top results.
- **Section 281: extension of time for returns.** "Can a company that cannot
  file its hydrocarbon tax returns on time get more time, and who grants it?"
  Top-1 is Section 282 (assessment of tax payable) for BM25 and hybrid, and
  Section 296 for dense. A possible cause is that Section 281 speaks of
  "accounts and particulars" while the question says "returns"; this is a
  hypothesis and has not been tested with `diagnose_miss.py`.

Not changed yet, deliberately: tuning the retriever to fix individual
questions would overfit a 102-question eval set. Candidate improvements are
listed under "Known limitations" and should be judged on the full eval,
ideally a larger one.

## Why there's no fixed confidence threshold
An earlier version used a fixed BM25 score cutoff to flag likely-unanswerable
questions. Checking the actual score distributions (on the original 52
questions) showed real overlap between answerable and unanswerable
questions — 11 of 52 legitimately answerable questions scored *below* the
highest-scoring unanswerable question. Two nearly-identical BM25 scores
(17.557 vs. 17.421) turned out to belong to one perfectly valid question and
one completely out-of-scope one — a fixed number cannot tell them apart.

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
python3 merge_gold.py      # builds data/processed/eval_set.json (102 + 10 questions)
python3 verify_gold.py     # optional: label checks on the 50 new questions

# Dense embeddings (run once, in Colab)
# Open notebooks/dense_embeddings.ipynb in Colab, put all_chunks.json in
# MyDrive/petroleum-reg-rag/ on Google Drive, run all cells, then download
# embeddings.npy and embeddings_meta.json into data/processed/

python3 evaluate_retrieval.py bm25 dense hybrid   # sanity check: hybrid should print Hit Rate@5 96.1%
python3 build_baseline.py bm25 dense hybrid       # records data/processed/baseline.json for the regression gate

# API
# Create a .env file with:
#   GROQ_API_KEY=your_key_here      (free, no credit card — https://console.groq.com)
#   APP_API_KEY=any_long_random_string
# Generate a key with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
uvicorn app:app --reload
# then open http://127.0.0.1:8000/docs

# Tests. Three /ask tests call Groq for real and need the keys above;
# -m "not groq" skips them (this is what CI runs)
python3 -m pytest -q
python3 -m pytest -q -m "not groq"
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
- **Eval set is still small.** 102 answerable questions, up from 52. A larger
  set would make the retriever comparison meaningful at the level of a
  percentage point, and the 24 questions that name their unit partly test
  label matching.
- **The aggregate CI gate has little slack.** `test_retrieval_eval.py` requires
  hybrid Hit Rate@5 of at least 95%, which is 97 of 102; hybrid currently gets
  98. The per-question gate is stricter: any new miss fails it. The first CI
  run on gold set v2 passed on the Ubuntu runner, but whether Windows and Linux
  always rank nearly tied scores identically is not proven. If a CI-only
  failure ever appears, check for a tie first. The gate covers retrieval only;
  answer faithfulness and abstention on unanswerable questions are not yet
  measured.
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
- **Vocabulary gap on paraphrased questions.** The Section 309 analysis shows
  that BM25 (exact tokens, no stemming or stopword handling) and `bge-small`
  both struggle when a question says "conflicts" and the statute says
  "inconsistent". Candidates: stemming and stopword removal in the BM25
  tokenizer, LLM query rewriting into statutory phrasing, a reranker. Each
  should be judged on the full eval, not tuned to one question.
- **Parallel provisions confuse retrieval.** The Commission and the Authority
  have near-identical sections (for example, the Secretary of each). The
  Section 23 miss is probably an instance of this; it has not been diagnosed.
- **Marginal notes are interleaved with body text** in some chunks (visible in
  Section 309: "Consequential amendments.", "Cap. C23, LFN, 2004."), which adds
  noise to both indexes.
- **Free-tier cold starts.** The service sleeps when idle, so the first
  request after a pause can be slow.