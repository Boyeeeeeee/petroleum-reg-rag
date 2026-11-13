
import json
from pathlib import Path

import pytest

from bm25_retriever import BM25Retriever, load_chunks
from build_baseline import content_hash, corpus_hash, rank_of

PROCESSED = Path("data/processed")
BASELINE_PATH = PROCESSED / "baseline.json"
EVAL_PATH = PROCESSED / "eval_set.json"

# One question moving from rank 1 to rank 2 costs about 0.005 MRR on 102 questions,
# so 0.01 tolerates two such shifts and catches anything larger.
MRR_TOLERANCE = 0.01

pytestmark = pytest.mark.skipif(
    not (BASELINE_PATH.exists() and EVAL_PATH.exists()),
    reason="baseline.json or eval_set.json is not in the repo",
)


@pytest.fixture(scope="module")
def baseline():
    return json.load(open(BASELINE_PATH))


@pytest.fixture(scope="module")
def eval_set():
    return json.load(open(EVAL_PATH))


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def bm25(chunks):
    return BM25Retriever(chunks)


@pytest.fixture(scope="module", params=["bm25", "hybrid"])
def current(request, chunks, bm25, eval_set):
    """(retriever name, {question id: rank or None}) for the current code and data."""
    name = request.param
    if name == "bm25":
        retriever = bm25
    else:
        from hybrid_retriever import HybridRetriever  # imported here so bm25-only runs work

        retriever = HybridRetriever(chunks, bm25=bm25)
    return name, {item["id"]: rank_of(retriever, item) for item in eval_set["answerable"]}


def test_data_matches_baseline(baseline, chunks, eval_set):
    assert corpus_hash(chunks) == baseline["corpus_hash"], (
        "The chunks changed since the baseline was recorded. "
        "Re-run `python build_baseline.py bm25 dense hybrid` and commit baseline.json."
    )
    assert content_hash(eval_set) == baseline["eval_set_hash"], (
        "The eval set changed since the baseline was recorded. "
        "Re-run `python build_baseline.py bm25 dense hybrid` and commit baseline.json."
    )


def test_no_new_misses(current, baseline, eval_set):
    name, ranks = current
    text = {item["id"]: item["q"] for item in eval_set["answerable"]}
    new_misses = [
        qid for qid, base_rank in baseline["retrievers"][name]["ranks"].items()
        if base_rank is not None and ranks.get(qid) is None
    ]
    assert not new_misses, f"{name}: new misses vs baseline: " + "; ".join(
        f"{qid} (was rank {baseline['retrievers'][name]['ranks'][qid]}): {text.get(qid)}" for qid in new_misses
    )


def test_mrr_has_not_dropped(current, baseline):
    name, ranks = current
    mrr = sum(1 / r for r in ranks.values() if r) / len(ranks)
    floor = baseline["retrievers"][name]["mrr"] - MRR_TOLERANCE
    assert mrr >= floor, f"{name}: MRR {mrr:.3f} is below baseline {baseline['retrievers'][name]['mrr']:.3f} minus tolerance {MRR_TOLERANCE}"