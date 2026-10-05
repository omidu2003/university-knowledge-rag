import logging
from collections import defaultdict

from src.config import RAW_DATA_DIR
from src.document_loader import load_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

docs = load_documents(RAW_DATA_DIR)

by_file = defaultdict(list)
for d in docs:
    by_file[d["source"]].append(d)

print(f"\nLoaded {len(docs)} records from {len(by_file)} files\n")
print(f"{'File':<60} {'Type':<5} {'Records':>7} {'Chars':>9}")
print("-" * 84)
for source, records in by_file.items():
    chars = sum(len(r["text"]) for r in records)
    print(f"{source[:58]:<60} {records[0]['file_type']:<5} {len(records):>7} {chars:>9}")

if docs:
    sample = next(d for d in docs if d["file_type"] == "pdf") if any(
        d["file_type"] == "pdf" for d in docs
    ) else docs[0]
    print(f"\nSample record: {sample['source']} (page {sample['page']})")
    print(sample["text"][:400])