"""Diagnose why the retrievers miss one eval question.

Usage:  python diagnose_miss.py
Edit TARGET (and PROBES) below to investigate a different miss.
"""
import json
from pathlib import Path

from bm25_retriever import BM25Retriever, load_chunks, citation_label, tokenize
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever

TARGET = "Petroleum Industry Act, 2021, Section 309"
EVAL_PATH = Path("data/processed/eval_set.json")

# Reworded versions of the question. Edit freely once you have read the chunk text.
PROBES = [
    "Which law prevails if another law is inconsistent with the Petroleum Industry Act?",
    "conflict with other laws Petroleum Industry Act prevails",
    "Petroleum Industry Act shall prevail over any other law except",
]

chunks = load_chunks()
eval_set = json.load(open(EVAL_PATH))
item = next(i for i in eval_set["answerable"] if i["citation"] == TARGET)
question = item["q"]
target_idx = next(i for i, c in enumerate(chunks) if citation_label(c) == TARGET)
target = chunks[target_idx]

bm25 = BM25Retriever(chunks)
dense = DenseRetriever(chunks)
hybrid = HybridRetriever(chunks, bm25=bm25, dense=dense)
RETRIEVERS = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
N = len(chunks)


def rank_of(retriever, q):
    """Rank of the target chunk in the full ranking, or None if it is not ranked at all."""
    for r in retriever.search(q, top_k=N):
        if r["citation"] == TARGET:
            return r["rank"]
    return None


def ranks_line(q):
    return "  ".join(f"{name}: {rank_of(r, q)}" for name, r in RETRIEVERS.items())


print("=" * 78)
print("1. THE QUESTION AND THE CHUNK IT SHOULD FIND")
print("=" * 78)
print("Question:", question)
print("Expected answer:", item.get("a"))
print(f"\nTarget: {TARGET}  (chunk index {target_idx}, pages {target.get('pages')})")
print(f"Length: {len(target['text'].split())} words")
print("\nChunk text (first 900 chars):")
print(target["text"][:900])

print("\n" + "=" * 78)
print("2. VOCABULARY OVERLAP (question words vs chunk words)")
print("=" * 78)
q_tokens = sorted(set(tokenize(question)))
chunk_tokens = set(tokenize(target["text"]))
print("In the chunk:    ", [t for t in q_tokens if t in chunk_tokens])
print("NOT in the chunk:", [t for t in q_tokens if t not in chunk_tokens])

print("\n" + "=" * 78)
print("3. WHERE THE TARGET RANKS FOR THE ORIGINAL QUESTION (None = not ranked)")
print("=" * 78)
print(ranks_line(question))
for name, r in RETRIEVERS.items():
    print(f"\nTop 5 for {name}:")
    for hit in r.search(question, top_k=5):
        print(f"  [{hit['rank']}] {hit['citation']}")
        print(f"       {hit['text_preview'][:130]}")

print("\n" + "=" * 78)
print("4. SELF-RETRIEVAL: the chunk's own opening text as the query")
print("=" * 78)
print("If the target is not rank 1 here, suspect the chunk itself (text cut, merged or garbled).")
print(ranks_line(target["text"][:300]))

print("\n" + "=" * 78)
print("5. REWORDED PROBES (target rank per retriever)")
print("=" * 78)
for p in PROBES:
    print(f"\nQ: {p}")
    print("  " + ranks_line(p))