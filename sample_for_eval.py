"""
Pick a stratified random sample of chunks to build the eval set from.
Proportional to each document's share of the corpus, so the eval set
reflects the corpus composition rather than over- or under-testing any
one document.
"""
import json
import random
from pathlib import Path

random.seed(42)

PROCESSED_DIR = Path("data/processed")
SAMPLE_TARGETS = {
    "pia_2021": 34,
    "royalty_regs_2022": 8,
    "gas_flaring_regs_2023": 5,
    "decommissioning_regs_2023": 5,
}  # sums to 52


def main():
    chunks = json.load(open(PROCESSED_DIR / "all_chunks.json"))
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    sample = []
    for doc_id, n in SAMPLE_TARGETS.items():
        pool = by_doc[doc_id]
        picked = random.sample(pool, min(n, len(pool)))
        sample.extend(picked)

    out_path = PROCESSED_DIR / "eval_sample_chunks.json"
    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Sampled {len(sample)} chunks -> {out_path}")
    for c in sample:
        label = f"{c['doc_id']} {c['unit_type']} {c['unit_number']}"
        print(" -", label)


if __name__ == "__main__":
    main()
