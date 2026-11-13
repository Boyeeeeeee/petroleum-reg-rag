
import json
import time
from pathlib import Path

from fastapi import HTTPException

import app  # loads the retriever and .env exactly as the service does

EVAL_PATH = Path("data/processed/eval_set.json")
OUT_PATH = Path("data/processed/abstention_pilot.json")
PAUSE_SECONDS = 2


def main():
    questions = json.load(open(EVAL_PATH))["unanswerable"]
    rows = []
    for i, item in enumerate(questions, start=1):
        row = {"id": item["id"], "q": item["q"]}
        try:
            resp = app.ask(app.QueryRequest(question=item["q"], top_k=5))
            row.update(grounded=resp.grounded, abstained=not resp.grounded,
                       top1=resp.passages_used[0].citation, answer=resp.answer)
        except HTTPException as e:
            row.update(error=f"{e.status_code}: {e.detail}")
        except Exception as e:  # e.g. a missing key
            row.update(error=f"{type(e).__name__}: {e}")
        rows.append(row)
        status = row.get("error") or ("ABSTAINED" if row["abstained"] else "ANSWERED (should have abstained)")
        print(f"[{i}/{len(questions)}] {row['id']} {status}\n    {item['q']}")
        if "answer" in row:
            print("    ->", row["answer"][:160].replace("\n", " "))
        time.sleep(PAUSE_SECONDS)

    ok = [r for r in rows if "error" not in r]
    abstained = sum(r["abstained"] for r in ok)
    print(f"\nAbstained correctly: {abstained}/{len(ok)}   Errors (not counted): {len(rows) - len(ok)}")
    json.dump(rows, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()