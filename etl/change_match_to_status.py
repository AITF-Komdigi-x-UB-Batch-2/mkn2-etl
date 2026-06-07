import json
from pathlib import Path

file_path = Path("metadata_sample/reconciled_sample_metadata.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for record in data:
    # hapus field match jika ada
    record.pop("match", None)

    # tambahkan field status
    record["status"] = {
        "atap": None,
        "dinding": None,
        "lantai": None,
    }

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Updated {len(data)} records")