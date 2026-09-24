import numpy as np

from bm25_retriever import BM25Retriever, load_chunks, citation_label, tokenize
from dense_retriever import DenseRetriever


class HybridRetriever:
    """Reciprocal rank fusion (RRF) of BM25 and dense rankings.

    Each chunk gets sum(weight / (rrf_k + rank)) over the retrievers that rank it
    inside their top `pool`. RRF uses ranks, not raw scores, so BM25 scores
    (~10-40) and cosine similarities (~0.6-0.8) never need to be normalised.
    """

    def __init__(
        self,
        chunks: list[dict],
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        rrf_k: int = 60,
        pool: int = 100,
        w_bm25: float = 1.0,
        w_dense: float = 1.0,
    ):
        self.chunks = chunks
        self.bm25 = bm25 or BM25Retriever(chunks)
        self.dense = dense or DenseRetriever(chunks)
        self.rrf_k = rrf_k
        self.pool = pool
        self.w_bm25 = w_bm25
        self.w_dense = w_dense

    @staticmethod
    def _ranks(scores: np.ndarray, pool: int) -> dict[int, int]:
        order = np.argsort(-scores, kind="stable")[:pool]
        return {int(idx): pos + 1 for pos, idx in enumerate(order)}

    def search(self, query: str, top_k: int = 5):
        bm25_scores = np.asarray(self.bm25.bm25.get_scores(tokenize(query)))
        dense_scores = self.dense.score_all(query)

        bm25_rank = self._ranks(bm25_scores, self.pool)
        dense_rank = self._ranks(dense_scores, self.pool)

        fused: dict[int, float] = {}
        for idx, r in bm25_rank.items():
            fused[idx] = fused.get(idx, 0.0) + self.w_bm25 / (self.rrf_k + r)
        for idx, r in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + self.w_dense / (self.rrf_k + r)

        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": round(float(fused[idx]), 5),   # RRF score: only meaningful for ordering
                "citation": citation_label(chunk),
                "doc_id": chunk["doc_id"],
                "unit_number": chunk["unit_number"],
                "text_preview": chunk["text"][:220].replace("\n", " "),
                "text": chunk["text"],
                "bm25_rank": bm25_rank.get(idx),
                "dense_rank": dense_rank.get(idx),
            })
        return results


if __name__ == "__main__":
    chunks = load_chunks()
    retriever = HybridRetriever(chunks)

    test_queries = [
        "What is the penalty for gas flaring per 1,000 standard cubic feet?",
        "How soon after completing decommissioning and abandonment must a licensee notify the Commission?",
        "How does Regulation 47 define 'arm's length'?",
    ]
    for q in test_queries:
        print(f"Query: {q}")
        for r in retriever.search(q, top_k=3):
            print(f"  [{r['rank']}] {r['citation']}  (bm25 rank {r['bm25_rank']}, dense rank {r['dense_rank']})")
        print()