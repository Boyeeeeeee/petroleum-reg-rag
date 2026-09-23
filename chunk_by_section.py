
import json
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

DOC_META = {
    "pia_2021": {"title": "Petroleum Industry Act, 2021", "unit_name": "Section"},
    "royalty_regs_2022": {"title": "Petroleum Royalty Regulations, 2022", "unit_name": "Regulation"},
    "gas_flaring_regs_2023": {
        "title": "Gas Flaring, Venting and Methane Emissions (Prevention of Waste and Pollution) Regulations, 2023",
        "unit_name": "Regulation",
    },
    "decommissioning_regs_2023": {
        "title": "Nigeria Upstream Petroleum Decommissioning and Abandonment Regulations, 2023",
        "unit_name": "Regulation",
    },
}

CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([0-9IVXLC]+)\s*[—\-–]?\s*(.*)$", re.IGNORECASE)
PART_RE = re.compile(r"^\s*PART\s+([0-9IVXLC]+)\s*[—\-–]?\s*(.*)$", re.IGNORECASE)
# Matches a unit-start pattern anywhere in a line, not just at line start,
# because the gazette layout sometimes places a marginal note (e.g.
# "Establishment") BEFORE the number on the merged line. We only accept a
# match as a real boundary if its number equals the expected next number
# in sequence (see find_boundary below), which filters out cross-references
# like "pursuant to section 5." appearing in body text.
UNIT_START_RE = re.compile(r"(?:^|\s)(\d{1,3})\.(?=[—\-–\s(]|[A-Z])")
# Deliberately case-SENSITIVE and requires all-caps ORDINAL + SCHEDULE,
# because the real headings are typeset in full caps ("SEVENTH SCHEDULE"),
# while body-text cross-references use title case ("the Seventh Schedule
# to the Act") and must NOT be treated as a new schedule boundary.
SCHEDULE_RE = re.compile(
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+SCHEDULE\b"
)


def load_pages(doc_id: str):
    with open(PROCESSED_DIR / f"{doc_id}_pages.json") as f:
        return json.load(f)["pages"]


def build_line_stream(pages):
    for p in pages:
        page_num = p["page"]
        for line in p["text"].split("\n"):
            yield line.strip(), page_num


def find_operative_start(lines):
    """Return index of the 2nd line containing a '1.' unit-start pattern
    (skips the TOC's '1. <title>' entry and lands on the real
    Section/Regulation 1)."""
    first_hits = []
    for i, (line, _) in enumerate(lines):
        m = UNIT_START_RE.search(line)
        if m and m.group(1) == "1":
            first_hits.append(i)
            if len(first_hits) == 2:
                return i
    return first_hits[0] if first_hits else 0


def chunk_document(doc_id: str):
    pages = load_pages(doc_id)
    meta = DOC_META[doc_id]
    lines = list(build_line_stream(pages))
    start_idx = find_operative_start(lines)

    chunks = []
    current_chapter = None
    current_part = None
    current_schedule = None
    in_schedule = False
    current_unit_num = None
    current_unit_type = meta["unit_name"]
    current_unit_lines = []
    current_unit_pages = set()
    expected_next = 1  # the number we're looking for to open a new unit

    def flush():
        if current_unit_num is not None and current_unit_lines:
            text = "\n".join(current_unit_lines).strip()
            if text:
                chunks.append({
                    "doc_id": doc_id,
                    "doc_title": meta["title"],
                    "unit_type": current_unit_type,
                    "unit_number": current_unit_num,
                    "chapter": current_chapter,
                    "part": current_part,
                    "schedule": current_schedule,
                    "pages": sorted(current_unit_pages),
                    "text": text,
                })

    for line, page_num in lines[start_idx:]:
        if not line:
            continue

        sched_m = SCHEDULE_RE.match(line)
        if sched_m:
            flush()
            in_schedule = True
            current_schedule = line
            current_chapter = None
            current_part = None
            current_unit_num = None
            current_unit_lines = []
            current_unit_pages = set()
            expected_next = 1  # schedules restart their own paragraph numbering
            continue

        if not in_schedule:
            chap_m = CHAPTER_RE.match(line)
            if chap_m:
                current_chapter = f"Chapter {chap_m.group(1)} — {chap_m.group(2)}".strip(" —")
                continue
            part_m = PART_RE.match(line)
            if part_m:
                current_part = f"Part {part_m.group(1)} — {part_m.group(2)}".strip(" —")
                continue

        # Look for the expected next unit number anywhere in the line
        # (handles marginal notes merged before the number). We scan all
        # candidate matches and accept the first one equal to expected_next.
        boundary_match = None
        for m in UNIT_START_RE.finditer(line):
            if int(m.group(1)) == expected_next:
                boundary_match = m
                break

        if boundary_match:
            flush()
            current_unit_num = expected_next
            current_unit_type = "Schedule Paragraph" if in_schedule else meta["unit_name"]
            # Keep only from the matched number onward; marginal-note text
            # that preceded it on the same line is dropped as noise.
            remainder = line[boundary_match.start(1):].lstrip()
            current_unit_lines = [remainder]
            current_unit_pages = {page_num}
            expected_next += 1
            continue

        if current_unit_num is not None:
            current_unit_lines.append(line)
            current_unit_pages.add(page_num)

    flush()
    return chunks


if __name__ == "__main__":
    all_chunks = []
    for doc_id in DOC_META:
        doc_chunks = chunk_document(doc_id)
        n_sections = sum(1 for c in doc_chunks if c["unit_type"] != "Schedule Paragraph")
        n_schedule = sum(1 for c in doc_chunks if c["unit_type"] == "Schedule Paragraph")
        print(f"{doc_id}: {n_sections} {DOC_META[doc_id]['unit_name']}s, {n_schedule} schedule paragraphs")
        all_chunks.extend(doc_chunks)

    out_path = PROCESSED_DIR / "all_chunks.json"
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\nTotal chunks: {len(all_chunks)} -> {out_path}")
