
import argparse
import hashlib
import json
from pathlib import Path

from bm25_retriever import load_chunks
from evaluate_retrieval import build_retriever, TOP_K

PROCESSED = Path("data/processed")
EVAL_PATH = PROCESSED / "eval_set.json"
OUT_PATH = PROCESSED / "baseline.json"


def corpus_hash(chunks):
    # Same recipe dense_retriever.py uses for embeddings_meta.json.
    return hashlib.sha256("\n".join(c["text"] for c in chunks).encode("utf-8")).hexdigest()


def content_hash(obj):
    # Hash the parsed JSON, not the file bytes, so Windows (CRLF) and Linux (LF)
    # produce the same value for the same content.
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rank_of(retriever, item, top_k=TOP_K):
    accepted = set(item.get("accept") or [item["citation"]])
    for r in retriever.search(item["q"], top_k=top_k):
        if r["citation"] in accepted:
            return r["rank"]
    return None


def main(names):
    chunks = load_chunks()
    eval_set = json.load(open(EVAL_PATH))
    answerable = eval_set["answerable"]
    out = {
        "top_k": TOP_K,
        "corpus_hash": corpus_hash(chunks),
        "eval_set_hash": content_hash(eval_set),
        "n_chunks": len(chunks),
        "n_answerable": len(answerable),
        "retrievers": {},
    }
    for name in names:
        retriever = build_retriever(name, chunks)
        ranks = {item["id"]: rank_of(retriever, item) for item in answerable}
        hits = [v for v in ranks.values() if v is not None]
        out["retrievers"][name] = {
            "hit_rate": len(hits) / len(ranks),
            "mrr": sum(1 / v for v in hits) / len(ranks),
            "ranks": ranks,
            "misses": sorted(k for k, v in ranks.items() if v is None),
        }
        r = out["retrievers"][name]
        print(f"{name}: hit {len(hits)}/{len(ranks)} = {r['hit_rate']:.1%}  MRR {r['mrr']:.3f}  misses {r['misses']}")
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}  (corpus_hash {out['corpus_hash'][:12]}..., eval_set_hash {out['eval_set_hash'][:12]}...)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("retrievers", nargs="*", default=["bm25"], choices=["bm25", "dense", "hybrid"])
    main(p.parse_args().retrievers)