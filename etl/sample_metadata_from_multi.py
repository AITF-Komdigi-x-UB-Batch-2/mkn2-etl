from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import math

import pandas as pd


@dataclass
class SampleMetadataConfig:
    # Source metadata dari pipeline download_metadata
    source_metadata_path: Path = Path(
        "metadata/metadata.jsonl"
    ).expanduser()

    # Output metadata sample
    output_dir: Path = Path(
        "metadata_sample"
    ).expanduser()

    # Target sampling
    n_multi_total: int = 24000
    n_keep_multi: int = 18000
    random_state: int = 42


class DinsosHouseMetadataSampler:
    def __init__(self, config: SampleMetadataConfig | None = None):
        self.config = config or SampleMetadataConfig()
        self.output_houses_dir = self.config.output_dir / "houses"
        self.output_houses_dir.mkdir(parents=True, exist_ok=True)

    def load_source_records(self) -> List[Dict[str, Any]]:
        path = self.config.source_metadata_path

        if not path.exists():
            raise FileNotFoundError(f"Source metadata tidak ditemukan: {path}")

        records: List[Dict[str, Any]] = []

        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise ValueError(f"Tidak ada file JSON di folder: {path}")

            for fp in json_files:
                with open(fp, "r", encoding="utf-8") as f:
                    records.append(json.load(f))

        elif path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    records.extend(obj)
                elif isinstance(obj, dict):
                    records.append(obj)
                else:
                    raise ValueError(f"Format JSON tidak dikenali: {path}")

        else:
            raise ValueError(f"Format file tidak didukung: {path}")

        if not records:
            raise ValueError("Tidak ada metadata yang berhasil dibaca.")

        return records

    @staticmethod
    def _normalize_label_value(value: Any) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, float) and math.isnan(value):
            return "unknown"
        text = str(value).strip()
        if text == "":
            return "unknown"
        return text

    @staticmethod
    def _parse_actual_label(record: Dict[str, Any]) -> Dict[str, str]:
        raw = record.get("actual_label", {})

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}

        if not isinstance(raw, dict):
            raw = {}

        return {
            "atap": raw.get("atap"),
            "dinding": raw.get("dinding"),
            "lantai": raw.get("lantai"),
        }

    def _make_stratum(self, record: Dict[str, Any]) -> str:
        label = self._parse_actual_label(record)
        atap = self._normalize_label_value(label.get("atap"))
        dinding = self._normalize_label_value(label.get("dinding"))
        lantai = self._normalize_label_value(label.get("lantai"))
        return f"{atap}||{dinding}||{lantai}"

    def _add_stratum_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["__stratum__"] = df.apply(lambda r: self._make_stratum(r.to_dict()), axis=1)
        return df

    def _seed_for_stratum(self, stratum: str) -> int:
        return self.config.random_state + sum(ord(c) for c in stratum)

    def _stratified_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n > len(df):
            raise ValueError(
                f"Sampling request {n} lebih besar dari data yang tersedia {len(df)}."
            )

        if n == len(df):
            return df.sample(frac=1, random_state=self.config.random_state)

        total = len(df)
        counts = df["__stratum__"].value_counts(dropna=False).to_dict()

        expected = {k: counts[k] * n / total for k in counts}
        alloc = {k: int(math.floor(v)) for k, v in expected.items()}
        remaining = n - sum(alloc.values())

        # Bagikan sisa berdasarkan pecahan terbesar
        frac_order = sorted(
            counts.keys(),
            key=lambda k: (expected[k] - alloc[k], counts[k]),
            reverse=True,
        )

        for key in frac_order:
            if remaining == 0:
                break
            if alloc[key] < counts[key]:
                alloc[key] += 1
                remaining -= 1

        # Jika masih sisa, isi secara greedy ke strata yang masih punya kapasitas
        if remaining > 0:
            for key in counts.keys():
                while remaining > 0 and alloc[key] < counts[key]:
                    alloc[key] += 1
                    remaining -= 1
                if remaining == 0:
                    break

        if remaining > 0:
            raise RuntimeError("Gagal membagi sampling secara stratified.")

        sampled_groups = []
        for stratum, group in df.groupby("__stratum__", sort=False):
            k = alloc.get(stratum, 0)
            if k <= 0:
                continue
            if k == len(group):
                sampled_group = group.copy()
            else:
                sampled_group = group.sample(
                    n=k,
                    random_state=self._seed_for_stratum(stratum),
                )
            sampled_groups.append(sampled_group)

        out = pd.concat(sampled_groups, axis=0)
        out = out.sample(frac=1, random_state=self.config.random_state)
        return out

    @staticmethod
    def _dtsen_null() -> Dict[str, None]:
        return {
            "atap": None,
            "dinding": None,
            "lantai": None,
        }

    @staticmethod
    def _select_image(images: List[Dict[str, Any]], view_type: str) -> Optional[Dict[str, Any]]:
        for img in images:
            if img.get("view_type") == view_type:
                return img
        return None

    def _build_multi_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        images = record.get("images", [])
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except Exception:
                images = []

        if not isinstance(images, list):
            images = []

        return {
            "house_id": record.get("house_id"),
            "no_kk": record.get("no_kk"),
            "house_type": "multi",
            "split": None,
            "match": None,
            "images": images,
            "actual_label": self._parse_actual_label(record),
            "dtsen": self._dtsen_null(),
        }

    def _build_single_record(self, record: Dict[str, Any], view_type: str) -> Optional[Dict[str, Any]]:
        images = record.get("images", [])
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except Exception:
                images = []

        if not isinstance(images, list):
            images = []

        selected_image = self._select_image(images, view_type)
        if selected_image is None:
            return None

        base_label = self._parse_actual_label(record)

        if view_type == "exterior":
            house_type = "single_exterior_only"
            split = "exterior"
            house_id = f"{record.get('house_id')}_EXT"
            actual_label = {
                "atap": base_label.get("atap"),
                "dinding": base_label.get("dinding"),
                "lantai": "Tidak Terdeteksi",
            }
        else:
            house_type = "single_interior_only"
            split = "interior"
            house_id = f"{record.get('house_id')}_INT"
            actual_label = {
                "atap": "Tidak Terdeteksi",
                "dinding": "Tidak Terdeteksi",
                "lantai": base_label.get("lantai"),
            }

        return {
            "house_id": house_id,
            "no_kk": record.get("no_kk"),
            "house_type": house_type,
            "split": split,
            "match": None,
            "images": [selected_image],
            "actual_label": actual_label,
            "dtsen": self._dtsen_null(),
        }

    def _write_jsonl(self, records: List[Dict[str, Any]], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write_per_house_json(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            out_path = self.output_houses_dir / f"{rec['house_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

    def run(self) -> Dict[str, Any]:
        source_records = self.load_source_records()
        df = pd.DataFrame(source_records)

        if "house_type" not in df.columns:
            raise ValueError("Kolom house_type tidak ditemukan di metadata source.")
        if "actual_label" not in df.columns:
            raise ValueError("Kolom actual_label tidak ditemukan di metadata source.")
        if "images" not in df.columns:
            raise ValueError("Kolom images tidak ditemukan di metadata source.")

        # Ambil hanya house multi dari metadata hasil download
        multi_df = df[df["house_type"] == "multi"].copy().reset_index(drop=True)
        if len(multi_df) < self.config.n_multi_total:
            raise ValueError(
                f"Jumlah multi house kurang dari target. "
                f"Tersedia: {len(multi_df):,}, target: {self.config.n_multi_total:,}"
            )

        multi_df = self._add_stratum_column(multi_df)

        # 24k multi house stratified
        sampled_24k = self._stratified_sample(multi_df, self.config.n_multi_total)

        # 18k dipertahankan sebagai multi
        sampled_24k = sampled_24k.copy()
        sampled_24k["__sample_rank__"] = range(len(sampled_24k))

        keep_multi_df = self._stratified_sample(sampled_24k, self.config.n_keep_multi)
        split_source_df = sampled_24k.drop(index=keep_multi_df.index).copy()

        # Bangun output final
        final_records: List[Dict[str, Any]] = []

        # 18k multi tetap multi
        for _, row in keep_multi_df.iterrows():
            final_records.append(self._build_multi_record(row.to_dict()))

        # 6k split source -> 12k records
        for _, row in split_source_df.iterrows():
            rec = row.to_dict()

            exterior_rec = self._build_single_record(rec, "exterior")
            interior_rec = self._build_single_record(rec, "interior")

            if exterior_rec is not None:
                final_records.append(exterior_rec)
            if interior_rec is not None:
                final_records.append(interior_rec)

        # Bersihkan urutan output
        final_records = sorted(final_records, key=lambda x: x["house_id"])

        # Simpan hasil
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.config.output_dir / "sample_metadata.jsonl"

        self._write_jsonl(final_records, jsonl_path)
        self._write_per_house_json(final_records)

        summary = pd.DataFrame(final_records)
        summary_path = self.config.output_dir / "sample_metadata_summary.csv"
        summary.to_csv(summary_path, index=False)

        return {
            "jsonl_path": jsonl_path,
            "summary_path": summary_path,
            "output_dir": self.config.output_dir,
            "total_records": len(final_records),
            "multi_records": int((summary["house_type"] == "multi").sum()),
            "single_exterior_only_records": int((summary["house_type"] == "single_exterior_only").sum()),
            "single_interior_only_records": int((summary["house_type"] == "single_interior_only").sum()),
        }