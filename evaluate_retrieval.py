import argparse
import json
from pathlib import Path

from bm25_retriever import BM25Retriever, load_chunks

PROCESSED_DIR = Path("data/processed")
TOP_K = 5


def evaluate(retriever, eval_set: dict, top_k: int = TOP_K):
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


def build_retriever(name: str, chunks: list[dict]):
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "dense":
        from dense_retriever import DenseRetriever  # imported lazily so bm25 runs without fastembed
        return DenseRetriever(chunks)
    if name == "hybrid":
        from hybrid_retriever import HybridRetriever
        return HybridRetriever(chunks)
    raise ValueError(f"Unknown retriever: {name}")


def print_report(name: str, results: dict):
    n = results["n_answerable"]
    print(f"=== {name} retrieval eval (top-{TOP_K}) ===")
    print(f"Hit Rate @ {TOP_K}: {results['hit_rate_at_k']:.1%}  ({n - len(results['misses'])}/{n})")
    print(f"MRR:              {results['mrr']:.3f}")
    print()
    print(f"Avg top-1 score, answerable questions:   {results['top1_score_avg_answerable']:.3f}")
    print(f"Avg top-1 score, unanswerable questions: {results['top1_score_avg_unanswerable']:.3f}")
    print()
    if results["misses"]:
        print(f"--- Misses ({len(results['misses'])}) ---")
        for m in results["misses"]:
            print(f"- Q: {m['question']}")
            print(f"  expected: {m['expected_citation']}")
            print(f"  got instead (top-1): {m['top_result']}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "retrievers",
        nargs="*",
        default=["bm25"],
        choices=["bm25", "dense", "hybrid"],
        help="which retriever(s) to evaluate, e.g. `bm25 dense hybrid`",
    )
    args = parser.parse_args()

    chunks = load_chunks()
    eval_set = json.load(open(PROCESSED_DIR / "eval_set.json"))

    all_results = {}
    for name in args.retrievers:
        retriever = build_retriever(name, chunks)
        results = evaluate(retriever, eval_set, top_k=TOP_K)
        all_results[name] = results
        print_report(name, results)

        with open(PROCESSED_DIR / f"eval_results_{name}.json", "w") as f:
            json.dump({
                "retriever": name,
                "top_k": TOP_K,
                "hit_rate_at_k": results["hit_rate_at_k"],
                "mrr": results["mrr"],
                "misses": results["misses"],
            }, f, indent=2)

    if len(all_results) > 1:
        missed = {name: {m["question"] for m in r["misses"]} for name, r in all_results.items()}
        names = list(missed)
        print("=== Comparison ===")
        for name in names:
            others = set().union(*(missed[o] for o in names if o != name))
            only_this = missed[name] - others
            print(f"Missed only by {name}: {len(only_this)}")
            for q in sorted(only_this):
                print(f"  - {q}")
        both = set.intersection(*missed.values())
        print(f"Missed by all: {len(both)}")
        for q in sorted(both):
            print(f"  - {q}")