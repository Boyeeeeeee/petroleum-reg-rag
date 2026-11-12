import json, re, collections
from pathlib import Path
from bm25_retriever import BM25Retriever, load_chunks, citation_label
from build_eval_set import ANSWERABLE
from gold_additions import GOLD_ADDITIONS

def norm(t): return re.sub(r"\s+", " ", t.lower()).strip()
UNIT_RE = re.compile(r"\b(sections?|regulations?|paragraphs?|reg\.?|s\.)\s*\(?\d+", re.I)

chunks = load_chunks()
by_cite = collections.defaultdict(list)
for c in chunks: by_cite[citation_label(c)].append(c)
texts = [norm(c["text"]) for c in chunks]
existing = {a["citation"] for a in ANSWERABLE}
problems = []

ids = [g["id"] for g in GOLD_ADDITIONS]
print(f"{len(GOLD_ADDITIONS)} items; unique ids: {len(set(ids))}")
seen = collections.Counter(g["citation"] for g in GOLD_ADDITIONS)
for g in GOLD_ADDITIONS:
    gid = g["id"]; cit = g["citation"]
    if cit not in by_cite: problems.append((gid, "CITATION NOT FOUND", cit)); continue
    if len(by_cite[cit]) > 1: problems.append((gid, "CITATION MATCHES >1 CHUNK", cit))
    if cit in existing: problems.append((gid, "DUPLICATES EXISTING 52", cit))
    if seen[cit] > 1: problems.append((gid, "DUPLICATE WITHIN NEW SET", cit))
    if UNIT_RE.search(g["q"]): problems.append((gid, "QUESTION NAMES A UNIT", g["q"]))
    own = norm(by_cite[cit][0]["text"])
    for a in g["anchors"]:
        if a not in own: problems.append((gid, "ANCHOR NOT IN CHUNK", a)); continue
        others = [citation_label(chunks[i]) for i, t in enumerate(texts) if a in t and citation_label(chunks[i]) != cit]
        if others: problems.append((gid, f"ANCHOR ALSO IN {len(others)} OTHER CHUNK(S)", f"{a!r} -> {others[:4]}"))

print("\n--- Structural / anchor checks ---")
for p in problems: print(p)
if not problems: print("none")

print("\n--- BM25-only difficulty check (top-5) ---")
r = BM25Retriever(chunks)
hits = 0; rr = []; rows = []
for g in GOLD_ADDITIONS:
    res = r.search(g["q"], top_k=5)
    rank = next((x["rank"] for x in res if x["citation"] == g["citation"]), None)
    rr.append(1/rank if rank else 0.0); hits += bool(rank)
    rows.append((g["id"], rank, res[0]["citation"].split(", ", 1)[1]))
print(f"Hit@5 {hits}/{len(GOLD_ADDITIONS)} = {hits/len(GOLD_ADDITIONS):.1%}   MRR {sum(rr)/len(rr):.3f}")
print("rank distribution:", dict(sorted(collections.Counter(x[1] for x in rows).items(), key=lambda kv: (kv[0] is None, kv[0]))))
print("misses (id, top-1 returned):")
for gid, rank, top in rows:
    if rank is None: print(" ", gid, "->", top)