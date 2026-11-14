
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

PROCESSED = Path("data/processed")
OUT_PATH = PROCESSED / "answerable_pilot.json"
N_HITS = 16
SEED = 0
PAUSE_SECONDS = 2


def _clean(text):
    """Lower-case, fold non-breaking spaces and hyphens to plain ones, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2010", "-").replace("\u2011", "-")
    return re.sub(r"\s+", " ", text).strip()


def citation_match(answer, citation):
    """'full' if the whole label is in the answer, 'short' if only its last part is
    (for example 'Regulation 24'), else None. Word boundaries stop 'Section 23' matching 'Section 230'."""
    low = _clean(answer)
    if _clean(citation) in low:
        return "full"
    unit = _clean(citation.rsplit(", ", 1)[1])
    return "short" if re.search(r"\b" + re.escape(unit) + r"\b", low) else None


def classify(retrieved, grounded, cited):
    if retrieved and grounded and cited == "full":
        return "GOOD: answered and cited the expected section"
    if retrieved and grounded and cited == "short":
        return "GOOD (short citation): answered, expected section named without the document title"
    if retrieved and not grounded:
        return "WRONG REFUSAL: expected section was retrieved but the model refused"
    if retrieved and grounded and not cited:
        return "CHECK: retrieved and answered, but the expected label is not in the answer"
    if not retrieved and not grounded:
        return "SAFE MISS: section not retrieved and the model refused"
    return "CHECK: section NOT retrieved but the model answered anyway"


def rescore():
    rows = json.load(open(OUT_PATH))
    for row in rows:
        if "answer" not in row:
            continue
        cited = citation_match(row["answer"], row["expected"])
        row["cited_expected"] = cited
        row["outcome"] = classify(row["retrieved"], row["grounded"], cited)
        print(f"{row['id']}: {row['outcome']}")
        if row["outcome"].startswith("CHECK"):
            odd = sorted({hex(ord(c)) for c in row["answer"] if ord(c) > 127})
            print(f"    Q: {row['q']}\n    expected: {row['expected']}\n    FULL ANSWER: {row['answer']}\n    non-ASCII characters in answer: {odd}\n")
    ok = [r for r in rows if "error" not in r]
    print("\n=== Summary (re-scored) ===")
    for label in sorted({r["outcome"].split(":")[0] for r in ok}):
        print(f"{label}: {sum(r['outcome'].split(':')[0] == label for r in ok)}")
    json.dump(rows, open(OUT_PATH, "w"), indent=2)


def main():
    from fastapi import HTTPException

    import app

    baseline = json.load(open(PROCESSED / "baseline.json"))
    items = {x["id"]: x for x in json.load(open(PROCESSED / "eval_set.json"))["answerable"]}
    ranks = baseline["retrievers"]["hybrid"]["ranks"]
    misses = sorted(k for k, v in ranks.items() if v is None)
    hits = sorted(k for k, v in ranks.items() if v is not None)
    chosen = misses + sorted(random.Random(SEED).sample(hits, N_HITS))

    rows = []
    for i, qid in enumerate(chosen, start=1):
        item = items[qid]
        row = {"id": qid, "q": item["q"], "expected": item["citation"], "baseline_hybrid_rank": ranks[qid]}
        try:
            resp = app.ask(app.QueryRequest(question=item["q"], top_k=5))
            retrieved = any(p.citation == item["citation"] for p in resp.passages_used)
            cited = citation_match(resp.answer, item["citation"])
            row.update(grounded=resp.grounded, retrieved=retrieved, cited_expected=cited,
                       outcome=classify(retrieved, resp.grounded, cited), answer=resp.answer)
        except HTTPException as e:
            row["error"] = f"{e.status_code}: {e.detail}"
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(f"[{i}/{len(chosen)}] {qid} (hybrid rank {ranks[qid]}) -> {row.get('outcome') or row['error']}")
        print(f"    Q: {item['q']}")
        print(f"    expected: {item['citation']}")
        if "answer" in row:
            print(f"    A: {row['answer'][:300].replace(chr(10), ' ')}")
        time.sleep(PAUSE_SECONDS)

    ok = [r for r in rows if "error" not in r]
    print("\n=== Summary ===")
    for label in sorted({r["outcome"].split(":")[0] for r in ok}):
        print(f"{label}: {sum(r['outcome'].split(':')[0] == label for r in ok)}")
    print(f"Errors (not counted): {len(rows) - len(ok)}")
    json.dump(rows, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    rescore() if "--rescore" in sys.argv else main()