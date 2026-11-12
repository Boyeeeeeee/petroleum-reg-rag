
import json
import re
from pathlib import Path

from build_eval_set import ANSWERABLE, UNANSWERABLE, PIA
from gold_additions import GOLD_ADDITIONS

OUT = Path("data/processed/eval_set.json")
UNIT_RE = re.compile(r"\b(sections?|regulations?|paragraphs?)\s*\(?\d+|\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+schedule", re.I)

# Types for the original 52, in the order they appear in build_eval_set.py.
ORIGINAL_TYPES = [
    "yes_no", "who", "who", "yes_no", "process", "who", "process", "number", "process", "number",   # e001-e010
    "number", "yes_no", "process", "process", "process", "yes_no", "process", "process", "who", "who",  # e011-e020
    "process", "who", "number", "number", "process", "process", "process", "process", "process", "who",  # e021-e030
    "process", "process", "process", "process", "process", "process", "process", "process",             # e031-e038
    "process", "process", "yes_no", "yes_no", "yes_no", "number", "process", "process",                 # e039-e046
    "number", "process", "process", "process", "number", "number",                                      # e047-e052
]
KNOWN_FAILURES = {f"{PIA}, Section 309"}  # all three retrievers miss this (vocabulary gap)

assert len(ANSWERABLE) == 52, len(ANSWERABLE)
assert len(ORIGINAL_TYPES) == 52, len(ORIGINAL_TYPES)

answerable = []
for i, (item, typ) in enumerate(zip(ANSWERABLE, ORIGINAL_TYPES), start=1):
    answerable.append({
        "id": f"e{i:03d}", "q": item["q"], "a": item["a"], "citation": item["citation"],
        "accept": [item["citation"]], "type": typ,
        "known_failure": item["citation"] in KNOWN_FAILURES,
        "names_unit": bool(UNIT_RE.search(item["q"])), "source": "original",
    })
for g in GOLD_ADDITIONS:
    answerable.append({
        "id": g["id"], "q": g["q"], "a": g["a"], "citation": g["citation"],
        "accept": [g["citation"]], "type": g["type"], "known_failure": False,
        "names_unit": bool(UNIT_RE.search(g["q"])), "source": "v2",
        "anchors": g["anchors"], "parallel": bool(g.get("parallel", False)),
    })

unanswerable = [{"id": f"u{i:02d}", "q": q, "a": None, "citation": None}
                for i, q in enumerate(UNANSWERABLE, start=1)]

ids = [x["id"] for x in answerable + unanswerable]
cites = [x["citation"] for x in answerable]
assert len(ids) == len(set(ids)), "duplicate ids"
assert len(cites) == len(set(cites)), "duplicate citations"

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"answerable": answerable, "unanswerable": unanswerable}, f, indent=2)
print(f"Wrote {len(answerable)} answerable + {len(unanswerable)} unanswerable -> {OUT}")
print("known failures:", [x["id"] for x in answerable if x["known_failure"]])
print("questions naming a unit:", sum(x["names_unit"] for x in answerable))