
import json
from pathlib import Path

from bm25_retriever import BM25Retriever, load_chunks, citation_label

PROCESSED_DIR = Path("data/processed")
TOP_K = 5


def evaluate(retriever: BM25Retriever, eval_set: dict, top_k: int = TOP_K):
    answerable = eval_set["answerable"]
    hits = 0
    reciprocal_ranks = []
    misses = []

    for item in answerable:
        results = retriever.search(item["q"], top_k=top_k)
        found_rank = None
        for r in results:
            if r["citation"] == item["citation"]:
                found_rank = r["rank"]
                break
        if found_rank:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append({
                "question": item["q"],
                "expected_citation": item["citation"],
                "top_result": results[0]["citation"] if results else None,
            })

    hit_rate = hits / len(answerable)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    unanswerable = eval_set["unanswerable"]
    top1_scores_unanswerable = []
    for item in unanswerable:
        results = retriever.search(item["q"], top_k=1)
        if results:
            top1_scores_unanswerable.append(results[0]["score"])

    top1_scores_answerable = []
    for item in answerable:
        results = retriever.search(item["q"], top_k=1)
        if results:
            top1_scores_answerable.append(results[0]["score"])

    return {
        "hit_rate_at_k": hit_rate,
        "mrr": mrr,
        "n_answerable": len(answerable),
        "misses": misses,
        "top1_score_avg_answerable": sum(top1_scores_answerable) / len(top1_scores_answerable),
        "top1_score_avg_unanswerable": sum(top1_scores_unanswerable) / len(top1_scores_unanswerable),
        "top1_scores_unanswerable": top1_scores_unanswerable,
    }


if __name__ == "__main__":
    chunks = load_chunks()
    retriever = BM25Retriever(chunks)
    eval_set = json.load(open(PROCESSED_DIR / "eval_set.json"))

    results = evaluate(retriever, eval_set, top_k=TOP_K)

    print(f"=== BM25 retrieval eval (top-{TOP_K}) ===")
    print(f"Hit Rate @ {TOP_K}: {results['hit_rate_at_k']:.1%}  ({results['n_answerable'] - len(results['misses'])}/{results['n_answerable']})")
    print(f"MRR:              {results['mrr']:.3f}")
    print()
    print(f"Avg top-1 BM25 score, answerable questions:   {results['top1_score_avg_answerable']:.2f}")
    print(f"Avg top-1 BM25 score, unanswerable questions: {results['top1_score_avg_unanswerable']:.2f}")
    print(f"(A big gap here means a score-threshold cutoff can catch unanswerable questions.)")
    print()

    if results["misses"]:
        print(f"=== Misses ({len(results['misses'])}) ===")
        for m in results["misses"]:
            print(f"- Q: {m['question']}")
            print(f"  expected: {m['expected_citation']}")
            print(f"  got instead (top-1): {m['top_result']}")
            print()
