import hashlib
import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from bm25_retriever import load_chunks, citation_label

PROCESSED_DIR = Path("data/processed")


class DenseRetriever:
    """Cosine-similarity retrieval over precomputed chunk embeddings.

    Same search() output shape as BM25Retriever, so evaluate_retrieval.py
    and app.py can use either one.
    """

    def __init__(
        self,
        chunks: list[dict],
        embeddings_path: Path = PROCESSED_DIR / "embeddings.npy",
        meta_path: Path = PROCESSED_DIR / "embeddings_meta.json",
    ):
        self.chunks = chunks
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if self.embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings.npy has {self.embeddings.shape[0]} rows but there are "
                f"{len(chunks)} chunks. Re-run the Colab notebook."
            )

        # The notebook embedded chunk["text"] in file order; make sure that still holds.
        text_hash = hashlib.sha256(
            "\n".join(c["text"] for c in chunks).encode("utf-8")
        ).hexdigest()
        if meta["text_hash"] != text_hash:
            raise ValueError(
                "Chunk texts no longer match the ones that were embedded. "
                "Re-run the Colab notebook (or check for a file-encoding difference)."
            )

        self.model_name = meta["model"]
        self.query_prefix = meta["query_prefix"]
        self._model = None
        self._query_cache: dict[str, np.ndarray] = {}

    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self._query_cache:
            if self._model is None:
                self._model = TextEmbedding(
                    self.model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH")
                )
            vec = np.array(
                list(self._model.embed([self.query_prefix + query])), dtype=np.float32
            )[0]
            self._query_cache[query] = vec / np.linalg.norm(vec)
        return self._query_cache[query]

    def score_all(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every chunk (used later for hybrid fusion)."""
        return self.embeddings @ self._embed_query(query)

    def search(self, query: str, top_k: int = 5):
        scores = self.score_all(query)
        ranked = np.argsort(-scores)[:top_k]
        results = []
        for rank, idx in enumerate(ranked, start=1):
            chunk = self.chunks[int(idx)]
            results.append({
                "rank": rank,
                "score": round(float(scores[idx]), 3),
                "citation": citation_label(chunk),
                "doc_id": chunk["doc_id"],
                "unit_number": chunk["unit_number"],
                "text_preview": chunk["text"][:220].replace("\n", " "),
                "text": chunk["text"],
            })
        return results


if __name__ == "__main__":
    chunks = load_chunks()
    retriever = DenseRetriever(chunks)
    print(f"Loaded {len(chunks)} chunks, embeddings {retriever.embeddings.shape}, model {retriever.model_name}\n")

    test_queries = [
        "What is the penalty for gas flaring per 1,000 standard cubic feet?",
        "notify the Commission",
        "How is royalty calculated for deep offshore production?",
        "Who administers the decommissioning and abandonment fund?",
    ]
    for q in test_queries:
        print(f"Query: {q}")
        for r in retriever.search(q, top_k=3):
            print(f"  [{r['rank']}] {r['citation']} (score={r['score']})")
        print()