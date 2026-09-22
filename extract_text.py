"""
Extract text from each source PDF, page by page, and save as JSON.
Keeping page numbers lets every chunk cite an exact page later.
"""
import json
import pdfplumber
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS = {
    "pia_2021": "Petroleum_Industry_Act_2021.pdf",
    "royalty_regs_2022": "Nigerian_Upstream_Petroleum_Royalty_Regulations_2022_d367e902a641eb1f93711ff6.pdf",
    "gas_flaring_regs_2023": "Gas_Flaring_Venting_and_Methane_Reg_2023_cd4e70a88c3295914a6466e9.pdf",
    "decommissioning_regs_2023": "Nigerian_Upstream_Decommissioning_and_Abandonment_Regulations_2023_bb2ec2f535ebcf69053acf92.pdf",
}


def extract(doc_id: str, filename: str):
    path = RAW_DIR / filename
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
    out_path = OUT_DIR / f"{doc_id}_pages.json"
    with open(out_path, "w") as f:
        json.dump({"doc_id": doc_id, "source_file": filename, "pages": pages}, f, indent=2)
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"{doc_id}: {len(pages)} pages, {total_chars} chars -> {out_path}")


if __name__ == "__main__":
    for doc_id, filename in DOCS.items():
        extract(doc_id, filename)
