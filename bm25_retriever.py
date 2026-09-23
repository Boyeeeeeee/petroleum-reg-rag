
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

PROCESSED_DIR = Path("data/processed")

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def load_chunks():
    with open(PROCESSED_DIR / "all_chunks.json") as f:
        return json.load(f)


def citation_label(chunk: dict) -> str:
    doc_title = chunk["doc_title"]
    unit = f"{chunk['unit_type']} {chunk['unit_number']}"
    if chunk.get("schedule"):
        # Schedule paragraphs cite the schedule name instead of a section number
        return f"{doc_title}, {chunk['schedule']}, paragraph {chunk['unit_number']}"
    return f"{doc_title}, {unit}"


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": round(float(scores[idx]), 3),
                "citation": citation_label(chunk),
                "doc_id": chunk["doc_id"],
                "unit_number": chunk["unit_number"],
                "text_preview": chunk["text"][:220].replace("\n", " "),
            })
        return results


if __name__ == "__main__":
    chunks = load_chunks()
    retriever = BM25Retriever(chunks)
    print(f"Loaded {len(chunks)} chunks into BM25 index.\n")

    test_queries = [
        "What is the penalty for gas flaring without authorisation?",
        "How is royalty calculated for deep offshore production?",
        "What must a licensee do before decommissioning a well?",
        "Who administers the decommissioning and abandonment fund?",
    ]
    for q in test_queries:
        print(f"Query: {q}")
        for r in retriever.search(q, top_k=3):
            print(f"  [{r['rank']}] {r['citation']} (score={r['score']})")
            print(f"      {r['text_preview']}...")
        print()
