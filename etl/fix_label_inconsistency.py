#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


VALID_SCHEMA = {
    "atap": [
        "Beton",
        "Genteng",
        "Seng",
        "Asbes",
        "Bambu",
        "Kayu/sirap",
        "Jerami/ijuk/daun-daunan/rumbia",
        "Lainnya",
        "Tidak terdeteksi",
    ],
    "dinding": [
        "Tembok",
        "Plesteran anyaman bambu/kawat",
        "Kayu/papan/gypsum/GRC/calciboard",
        "Anyaman bambu",
        "Batang kayu",
        "Bambu",
        "Lainnya",
        "Tidak terdeteksi",
    ],
    "lantai": [
        "Marmer/granit",
        "Keramik",
        "Parket/vinil/karpet",
        "Ubin/tegel/teraso",
        "Kayu/papan",
        "Semen/bata merah",
        "Bambu",
        "Tanah",
        "Lainnya",
        "Tidak terdeteksi",
    ],
}

# Canonical labels are the exact strings that should be written out.
# Keys are normalized comparison forms (lowercase, stripped, underscore-normalized).
CANONICAL_MAP = {
    "atap": {
        "beton": "Beton",
        "genteng": "Genteng",
        "seng": "Seng",
        "asbes": "Asbes",
        "bambu": "Bambu",
        "kayu/sirap": "Kayu/sirap",
        "kayu sirap": "Kayu/sirap",
        "jerami/ijuk/daun-daunan/rumbia": "Jerami/ijuk/daun-daunan/rumbia",
        "jerami/ijuk/daun_daunan/rumbia": "Jerami/ijuk/daun-daunan/rumbia",
        "lainnya": "Lainnya",
        "tidak terdeteksi": "Tidak terdeteksi",
        "tidak_terdeteksi": "Tidak terdeteksi",
    },
    "dinding": {
        "tembok": "Tembok",
        "plesteran anyaman bambu/kawat": "Plesteran anyaman bambu/kawat",
        "plesteran_anyaman_bambu/kawat": "Plesteran anyaman bambu/kawat",
        "kayu/papan/gypsum/GRC/calciboard": "Kayu/papan/gypsum/GRC/calciboard",
        "kayu/papan/gypsum/grc/calciboard": "Kayu/papan/gypsum/GRC/calciboard",
        "kayu papan gypsum grc calciboard": "Kayu/papan/gypsum/GRC/calciboard",
        "anyaman bambu": "Anyaman bambu",
        "anyaman_bambu": "Anyaman bambu",
        "batang kayu": "Batang kayu",
        "batang_kayu": "Batang kayu",
        "bambu": "Bambu",
        "lainnya": "Lainnya",
        "tidak terdeteksi": "Tidak terdeteksi",
        "tidak_terdeteksi": "Tidak terdeteksi",
    },
    "lantai": {
        "marmer/granit": "Marmer/granit",
        "keramik": "Keramik",
        "parket/vinil/karpet": "Parket/vinil/karpet",
        "ubin/tegel/teraso": "Ubin/tegel/teraso",
        "kayu/papan": "Kayu/papan",
        "semen/bata merah": "Semen/bata merah",
        "semen/bata_merah": "Semen/bata merah",
        "bambu": "Bambu",
        "tanah": "Tanah",
        "lainnya": "Lainnya",
        "tidak terdeteksi": "Tidak terdeteksi",
        "tidak_terdeteksi": "Tidak terdeteksi",
    },
}

FIELD_MAP = {
    "atap": ("atap", "jenis_atap_terluas"),
    "dinding": ("dinding", "jenis_dinding_terluas"),
    "lantai": ("lantai", "jenis_lantai_terluas"),
}


def normalize_key(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def canonicalize(category: str, value: Any) -> Any:
    """
    Normalize a label to its canonical spelling/casing.

    - Returns the original value for None / empty.
    - Keeps 'unclassified' unchanged.
    - Maps unknown values to the original stripped string.
    """
    if value is None:
        return value

    raw = str(value).strip()
    if raw == "":
        return raw

    if raw.lower() == "unclassified":
        return "unclassified"

    norm = normalize_key(raw)
    mapping = CANONICAL_MAP[category]

    # direct map first
    if norm in mapping:
        return mapping[norm]

    # fallback: normalize against original values without underscore-specific spacing
    if raw.lower() in mapping:
        return mapping[raw.lower()]

    return raw


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}") from e
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list of records in {path}, got {type(data).__name__}")


def write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_unique(records: List[Dict[str, Any]], field: str) -> Counter:
    c = Counter()
    for rec in records:
        val = rec.get(field, None)
        if val is not None:
            c[str(val)] += 1
    return c


def fix_labelstudio(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Counter]]:
    out = []
    changes = defaultdict(Counter)
    for rec in records:
        new_rec = deepcopy(rec)
        for category, (_, field) in FIELD_MAP.items():
            old = new_rec.get(field, None)
            new = canonicalize(category, old)
            if new != old:
                changes[field][f"{old} -> {new}"] += 1
                new_rec[field] = new
        out.append(new_rec)
    return out, changes


def fix_metadata(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Counter]]:
    out = []
    changes = defaultdict(Counter)
    for rec in records:
        new_rec = deepcopy(rec)
        actual = new_rec.get("actual_label", {})
        if isinstance(actual, dict):
            for category, (field_name, _) in FIELD_MAP.items():
                old = actual.get(field_name, None)
                new = canonicalize(category, old)
                if new != old:
                    changes[f"actual_label.{field_name}"][f"{old} -> {new}"] += 1
                    actual[field_name] = new
            new_rec["actual_label"] = actual
        out.append(new_rec)
    return out, changes


def print_summary(title: str, counter: Counter) -> None:
    print(f"\n=== {title} ===")
    for k, v in counter.most_common():
        print(f"{k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix label inconsistencies in labelstudio_output.json and sample_metadata_augmented.jsonl"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/ and etl/ directories.",
    )
    parser.add_argument(
        "--labelstudio",
        type=Path,
        default=None,
        help="Path to labelstudio_output.json (default: <root>/data/labelstudio_output.json)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to sample_metadata_augmented.jsonl (default: <root>/metadata_sample/sample_metadata_augmented.jsonl)",
    )
    parser.add_argument(
        "--out-labelstudio",
        type=Path,
        default=None,
        help="Output path for cleaned labelstudio JSON (default: <root>/data/labelstudio_output_fixed.json)",
    )
    parser.add_argument(
        "--out-metadata",
        type=Path,
        default=None,
        help="Output path for cleaned metadata JSONL (default: <root>/metadata_sample/sample_metadata_augmented_fixed.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output path for a JSON report (default: <root>/etl/label_inconsistency_fix_report.json)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input files in place. Use with care.",
    )
    args = parser.parse_args()

    root = args.root
    labelstudio_path = args.labelstudio or (root / "data" / "labelstudio_output.json")
    metadata_path = args.metadata or (root / "metadata_sample" / "sample_metadata_augmented.jsonl")

    if args.inplace:
        out_labelstudio_path = labelstudio_path
        out_metadata_path = metadata_path
    else:
        out_labelstudio_path = args.out_labelstudio or (root / "data" / "labelstudio_output_fixed.json")
        out_metadata_path = args.out_metadata or (root / "metadata_sample" / "sample_metadata_augmented_fixed.jsonl")

    report_path = args.report or (root / "etl" / "label_inconsistency_fix_report.json")

    if not labelstudio_path.exists():
        raise FileNotFoundError(f"LabelStudio file not found: {labelstudio_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    labelstudio_records = load_json(labelstudio_path)
    metadata_records = load_jsonl(metadata_path)

    print("Loaded records:")
    print(f"- labelstudio_output.json: {len(labelstudio_records)}")
    print(f"- sample_metadata_augmented.jsonl: {len(metadata_records)}")

    print("\nUnique labels BEFORE fix:")
    for category, (_, field) in FIELD_MAP.items():
        ls_counter = summarize_unique(labelstudio_records, field)
        md_counter = Counter()
        for rec in metadata_records:
            actual = rec.get("actual_label", {})
            if isinstance(actual, dict):
                val = actual.get(category, None)
                if val is not None:
                    md_counter[str(val)] += 1

        print(f"\n=== {category.upper()} ===")
        print(f"labelstudio unique ({field}): {sorted(ls_counter.keys())}")
        print(f"metadata unique (actual_label.{category}): {sorted(md_counter.keys())}")

    fixed_labelstudio, ls_changes = fix_labelstudio(labelstudio_records)
    fixed_metadata, md_changes = fix_metadata(metadata_records)

    out_labelstudio_path.parent.mkdir(parents=True, exist_ok=True)
    out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(out_labelstudio_path, fixed_labelstudio)
    write_jsonl(out_metadata_path, fixed_metadata)

    report = {
        "inputs": {
            "labelstudio": str(labelstudio_path),
            "metadata": str(metadata_path),
        },
        "outputs": {
            "labelstudio": str(out_labelstudio_path),
            "metadata": str(out_metadata_path),
        },
        "changes": {
            "labelstudio": {k: dict(v) for k, v in ls_changes.items()},
            "metadata": {k: dict(v) for k, v in md_changes.items()},
        },
        "valid_schema": VALID_SCHEMA,
        "notes": [
            "unclassified is preserved as-is",
            "Labels outside the canonical mapping are left unchanged",
        ],
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSaved cleaned files:")
    print(f"- {out_labelstudio_path}")
    print(f"- {out_metadata_path}")
    print(f"- {report_path}")

    print("\nChanges summary:")
    if not ls_changes and not md_changes:
        print("No changes were needed.")
    else:
        for field, counter in ls_changes.items():
            print(f"\nLabelStudio changes for {field}:")
            for k, v in counter.most_common():
                print(f"  {k}: {v}")
        for field, counter in md_changes.items():
            print(f"\nMetadata changes for {field}:")
            for k, v in counter.most_common():
                print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())