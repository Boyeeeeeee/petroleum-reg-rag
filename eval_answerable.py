
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

PROCESSED = Path("data/processed")
ALL = "--all" in sys.argv
OUT_PATH = PROCESSED / ("answerable_full.json" if ALL else "answerable_pilot.json")
N_HITS = 16
SEED = 0
PAUSE_SECONDS = 2
RETRY_PAUSE_SECONDS = 30
MAX_RETRIES = 3


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


def summarise(rows):
    ok = [r for r in rows if "error" not in r and "answer" in r]
    print("\n=== Summary ===")
    labels = sorted({r["outcome"].split(":")[0] for r in ok})
    for label in labels:
        print(f"{label}: {sum(r['outcome'].split(':')[0] == label for r in ok)}")
    ret = [r for r in ok if r["retrieved"]]
    miss = [r for r in ok if not r["retrieved"]]
    good = lambda r: r["outcome"].startswith("GOOD")
    print(f"\nExpected section retrieved: {len(ret)}")
    print(f"  answered {sum(r['grounded'] for r in ret)}, refused {sum(not r['grounded'] for r in ret)}, "
          f"answered and named the expected section {sum(good(r) for r in ret)}")
    print(f"Expected section NOT retrieved: {len(miss)}")
    print(f"  refused {sum(not r['grounded'] for r in miss)}, answered anyway {sum(r['grounded'] for r in miss)}")
    if not ALL:
        print("(pilot sample is hand-picked: these counts are not rates)")
    print(f"Errors (not counted): {len(rows) - len(ok)}")
    to_check = [r["id"] for r in ok if r["outcome"].startswith("CHECK")]
    if to_check:
        print("To read by hand (CHECK):", ", ".join(to_check))


def rescore():
    rows = json.load(open(OUT_PATH))
    for row in rows:
        if "answer" not in row:
            continue
        cited = citation_match(row["answer"], row["expected"])
        row["cited_expected"] = cited
        row["outcome"] = classify(row["retrieved"], row["grounded"], cited)
        if row["outcome"].startswith(("CHECK", "WRONG")):
            odd = sorted({hex(ord(c)) for c in row["answer"] if ord(c) > 127})
            print(f"{row['id']}: {row['outcome']}\n    Q: {row['q']}\n    expected: {row['expected']}\n"
                  f"    FULL ANSWER: {row['answer']}\n    non-ASCII characters in answer: {odd}\n")
        elif not ALL:
            print(f"{row['id']}: {row['outcome']}")
    summarise(rows)
    json.dump(rows, open(OUT_PATH, "w"), indent=2)


def ask_with_retry(app, HTTPException, question):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return app.ask(app.QueryRequest(question=question, top_k=5))
        except HTTPException as e:
            if e.status_code == 502 and "RateLimit" in str(e.detail) and attempt < MAX_RETRIES:
                wait = RETRY_PAUSE_SECONDS * (attempt + 1)
                print(f"    rate limited, waiting {wait}s then retrying...")
                time.sleep(wait)
                continue
            raise


def main():
    from fastapi import HTTPException

    import app

    baseline = json.load(open(PROCESSED / "baseline.json"))
    eval_items = json.load(open(PROCESSED / "eval_set.json"))["answerable"]
    items = {x["id"]: x for x in eval_items}
    ranks = baseline["retrievers"]["hybrid"]["ranks"]
    if ALL:
        chosen = [x["id"] for x in eval_items]
    else:
        misses = sorted(k for k, v in ranks.items() if v is None)
        hits = sorted(k for k, v in ranks.items() if v is not None)
        chosen = misses + sorted(random.Random(SEED).sample(hits, N_HITS))

    done = {}
    if ALL and OUT_PATH.exists():
        done = {r["id"]: r for r in json.load(open(OUT_PATH)) if "answer" in r}
        print(f"Resuming: {len(done)} questions already answered in {OUT_PATH}")

    rows = []
    for i, qid in enumerate(chosen, start=1):
        if qid in done:
            rows.append(done[qid])
            continue
        item = items[qid]
        row = {"id": qid, "q": item["q"], "expected": item["citation"], "baseline_hybrid_rank": ranks[qid]}
        try:
            resp = ask_with_retry(app, HTTPException, item["q"])
            retrieved = any(p.citation == item["citation"] for p in resp.passages_used)
            cited = citation_match(resp.answer, item["citation"])
            row.update(grounded=resp.grounded, retrieved=retrieved, cited_expected=cited,
                       outcome=classify(retrieved, resp.grounded, cited), answer=resp.answer)
        except HTTPException as e:
            row["error"] = f"{e.status_code}: {e.detail}"
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(f"[{i}/{len(chosen)}] {qid} -> {row.get('outcome') or row['error']}")
        if not ALL and "answer" in row:
            print(f"    Q: {item['q']}\n    expected: {item['citation']}\n    A: {row['answer'][:300].replace(chr(10), ' ')}")
        json.dump(rows, open(OUT_PATH, "w"), indent=2)  # save after every question
        time.sleep(PAUSE_SECONDS)

    summarise(rows)
    json.dump(rows, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    rescore() if "--rescore" in sys.argv else main()