# Petroleum Regulation RAG

A retrieval-augmented question-answering system over Nigerian upstream
petroleum regulation: the Petroleum Industry Act 2021 plus three NUPRC
regulations (Royalty, Gas Flaring/Venting/Methane, Decommissioning &
Abandonment).

## Status
- [x] PDF ingestion (`extract_text.py`)
- [x] Section/regulation-aware chunking (`chunk_by_section.py`) — 478 chunks
- [x] BM25 sparse retrieval (`bm25_retriever.py`)
- [ ] Dense embeddings + vector store (Colab — see `notebooks/`)
- [ ] Hybrid retrieval (BM25 + dense + rerank)
- [ ] Eval set (LLM-generated Q&A pairs per chunk)
- [ ] FastAPI service with citations
- [ ] Docker + tests + CI + deployment

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
  -> bm25_retriever.py      -> sparse retrieval, testable now
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 extract_text.py
python3 chunk_by_section.py
python3 bm25_retriever.py   # runs sample queries
```
