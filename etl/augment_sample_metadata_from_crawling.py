from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

ALLOWED_ATAP = {"jerami", "kayu/sirap"}
ALLOWED_DINDING = {"batang_kayu", "bambu"}
ALLOWED_LANTAI = {"bambu", "kayu/papan", "parket/vinil/karpet"}
ALLOWED_HOUSE_TYPES = {"multi", "single_exterior_only", "single_interior_only"}


@dataclass
class AugmentConfig:
    sample_metadata_path: Path = Path("metadata_sample/sample_metadata.jsonl")
    crawling_metadata_path: Path = Path("metadata/mkn_house_metadata.json")
    output_metadata_path: Path = Path("metadata_sample/sample_metadata_augmented.jsonl")

    source_image_root: Path = Path("data/mkn_img")

    minio_endpoint: str = None
    minio_access_key: str = None
    minio_secret_key: str = None
    minio_secure: bool = False
    minio_bucket_name: str = None
    minio_public_base_url: str = None

    exterior_prefix: str = "tampak_luar"
    interior_prefix: str = "tampak_dalam"

    random_no_kk_prefix: str = "FAM_"

    target_multi_wanted: Optional[int] = None
    target_ext_wanted: Optional[int] = None
    target_int_wanted: Optional[int] = None

    def __post_init__(self):
        if self.minio_endpoint is None:
            self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
            if not self.minio_endpoint:
                raise ValueError("MINIO_ENDPOINT environment variable is required")

        if self.minio_access_key is None:
            self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
            if not self.minio_access_key:
                raise ValueError("MINIO_ACCESS_KEY environment variable is required")

        if self.minio_secret_key is None:
            self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
            if not self.minio_secret_key:
                raise ValueError("MINIO_SECRET_KEY environment variable is required")

        if self.minio_bucket_name is None:
            self.minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")
            if not self.minio_bucket_name:
                raise ValueError("MINIO_BUCKET_NAME environment variable is required")

        if self.minio_public_base_url is None:
            self.minio_public_base_url = os.getenv("MINIO_PUBLIC_BASE_URL")
            if not self.minio_public_base_url:
                raise ValueError("MINIO_PUBLIC_BASE_URL environment variable is required")

        secure_str = os.getenv("MINIO_SECURE", "False")
        self.minio_secure = secure_str.lower() in ("true", "1", "yes")


class MinioImageUploader:
    def __init__(self, config: AugmentConfig):
        self.config = config
        self.client = Minio(
            endpoint=self.config.minio_endpoint,
            access_key=self.config.minio_access_key,
            secret_key=self.config.minio_secret_key,
            secure=self.config.minio_secure,
        )
        self._ensure_bucket()
        self.next_exterior_idx = self._get_next_object_index("exterior")
        self.next_interior_idx = self._get_next_object_index("interior")

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.config.minio_bucket_name):
            self.client.make_bucket(self.config.minio_bucket_name)

    def _get_next_object_index(self, view_type: str) -> int:
        if view_type == "exterior":
            prefix = self.config.exterior_prefix
            pattern = re.compile(
                rf"{re.escape(prefix)}/mkn2_exterior_img_(\d+)\.jpg$",
                re.IGNORECASE,
            )
        else:
            prefix = self.config.interior_prefix
            pattern = re.compile(
                rf"{re.escape(prefix)}/mkn2_interior_img_(\d+)\.jpg$",
                re.IGNORECASE,
            )

        max_idx = 0
        for obj in self.client.list_objects(
            self.config.minio_bucket_name,
            prefix=f"{prefix}/",
            recursive=True,
        ):
            m = pattern.search(obj.object_name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

    @staticmethod
    def _content_type_from_path(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        return "application/octet-stream"

    def upload(self, local_image_path: Path, view_type: str) -> Dict[str, str]:
        if not local_image_path.exists():
            raise FileNotFoundError(f"Image tidak ditemukan: {local_image_path}")

        if view_type not in {"exterior", "interior"}:
            raise ValueError(f"view_type tidak valid: {view_type}")

        if view_type == "exterior":
            idx = self.next_exterior_idx
            self.next_exterior_idx += 1
            prefix = self.config.exterior_prefix
            filename = f"mkn2_exterior_img_{idx:06d}.jpg"
        else:
            idx = self.next_interior_idx
            self.next_interior_idx += 1
            prefix = self.config.interior_prefix
            filename = f"mkn2_interior_img_{idx:06d}.jpg"

        object_name = f"{prefix}/{filename}"

        self.client.fput_object(
            bucket_name=self.config.minio_bucket_name,
            object_name=object_name,
            file_path=str(local_image_path),
            content_type=self._content_type_from_path(local_image_path),
        )

        image_db_url = (
            f"{self.config.minio_public_base_url.rstrip('/')}/"
            f"{self.config.minio_bucket_name}/{object_name}"
        )
        return {"object_name": object_name, "image_db_url": image_db_url}


class SampleMetadataAugmentor:
    def __init__(self, config: AugmentConfig):
        self.config = config
        self.uploader = MinioImageUploader(config)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        if path.suffix.lower() == ".jsonl":
            return SampleMetadataAugmentor._read_jsonl(path)

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]

        raise ValueError(f"Format metadata tidak dikenali: {path}")

    @staticmethod
    def _extract_numeric(text: str) -> Optional[int]:
        m = re.search(r"(\d+)", str(text))
        return int(m.group(1)) if m else None

    @staticmethod
    def _normalize_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "none", "null"}:
            return None
        if text == "jerami/ijuk/daun-daunan/rumbia":
            return "jerami"
        return text

    @staticmethod
    def _safe_prediksi(record: Dict[str, Any]) -> Dict[str, Optional[str]]:
        prediksi = record.get("prediksi", {})
        if isinstance(prediksi, str):
            try:
                prediksi = json.loads(prediksi)
            except Exception:
                prediksi = {}

        if not isinstance(prediksi, dict):
            prediksi = {}

        return {
            "atap": SampleMetadataAugmentor._normalize_label(prediksi.get("atap")),
            "dinding": SampleMetadataAugmentor._normalize_label(prediksi.get("dinding")),
            "lantai": SampleMetadataAugmentor._normalize_label(prediksi.get("lantai")),
        }

    @staticmethod
    def _dtsen_null_object() -> Dict[str, None]:
        return {"atap": None, "dinding": None, "lantai": None}

    @staticmethod
    def _infer_output_house_type(record: Dict[str, Any]) -> str:
        images = record.get("images", [])
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except Exception:
                images = []

        if not isinstance(images, list):
            images = []

        has_exterior = any(img.get("view_type") == "exterior" for img in images)
        has_interior = any(img.get("view_type") == "interior" for img in images)

        if has_exterior and has_interior:
            return "multi"
        if has_exterior:
            return "single_exterior_only"
        if has_interior:
            return "single_interior_only"

        raw = str(record.get("house_type", "")).strip().lower()
        if raw == "multi":
            return "multi"
        return "single_exterior_only"

    @staticmethod
    def _stratum_key(actual_label: Dict[str, Optional[str]], house_type: str) -> str:
        if house_type == "multi":
            return f"{actual_label.get('atap')}||{actual_label.get('dinding')}||{actual_label.get('lantai')}"
        if house_type == "single_exterior_only":
            return f"{actual_label.get('atap')}||{actual_label.get('dinding')}"
        return f"{actual_label.get('lantai')}"

    @staticmethod
    def _labels_allowed(actual_label: Dict[str, Optional[str]], house_type: str) -> bool:
        atap = actual_label.get("atap")
        dinding = actual_label.get("dinding")
        lantai = actual_label.get("lantai")

        if house_type == "multi":
            return (
                atap in ALLOWED_ATAP
                or dinding in ALLOWED_DINDING
                or lantai in ALLOWED_LANTAI
            )

        if house_type == "single_exterior_only":
            return (
                atap in ALLOWED_ATAP
                or dinding in ALLOWED_DINDING
            )

        if house_type == "single_interior_only":
            return lantai in ALLOWED_LANTAI

        return False

    def _resolve_local_image_path(self, image_path_value: Any) -> Path:
        p = Path(str(image_path_value))
        return p if p.is_absolute() else (self.config.source_image_root / p)

    def _load_existing_state(self) -> Dict[str, Any]:
        sample_records = self._read_jsonl(self.config.sample_metadata_path)
        existing_output_records: List[Dict[str, Any]] = []
        if self.config.output_metadata_path.exists():
            existing_output_records = self._read_jsonl(self.config.output_metadata_path)

        all_existing = sample_records + existing_output_records

        max_house_num = 0
        max_image_num = 0
        used_group_ids = set()

        for rec in all_existing:
            hid = self._extract_numeric(rec.get("house_id", ""))
            if hid is not None:
                max_house_num = max(max_house_num, hid)

            sgid = rec.get("source_group_id")
            if sgid:
                used_group_ids.add(str(sgid))

            images = rec.get("images", [])
            if isinstance(images, str):
                try:
                    images = json.loads(images)
                except Exception:
                    images = []

            if isinstance(images, list):
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    iid = self._extract_numeric(img.get("image_id", ""))
                    if iid is not None:
                        max_image_num = max(max_image_num, iid)

        return {
            "sample_records": sample_records,
            "existing_output_records": existing_output_records,
            "used_group_ids": used_group_ids,
            "next_house_num": max_house_num + 1,
            "next_image_num": max_image_num + 1,
        }

    def _build_candidate_pools(
        self,
        crawling_records: List[Dict[str, Any]],
        used_group_ids: set,
    ) -> Dict[str, List[Dict[str, Any]]]:
        pools: Dict[str, List[Dict[str, Any]]] = {
            "multi": [],
            "single_exterior_only": [],
            "single_interior_only": [],
        }

        for record in crawling_records:
            source_group_id = record.get("source_group_id")
            if not source_group_id:
                continue

            source_group_id = str(source_group_id)
            if source_group_id in used_group_ids:
                continue

            house_type = self._infer_output_house_type(record)
            if house_type not in ALLOWED_HOUSE_TYPES:
                continue

            actual_label = self._safe_prediksi(record)
            if not self._labels_allowed(actual_label, house_type):
                continue

            images = record.get("images", [])
            if isinstance(images, str):
                try:
                    images = json.loads(images)
                except Exception:
                    images = []

            if not isinstance(images, list) or len(images) == 0:
                continue

            pools[house_type].append(
                {
                    "source_group_id": source_group_id,
                    "house_type": house_type,
                    "stratum_key": self._stratum_key(actual_label, house_type),
                    "raw_record": record,
                    "actual_label": actual_label,
                }
            )

        return pools

    def _stratified_unique_sample(
        self,
        pool: List[Dict[str, Any]],
        target_n: int,
        used_group_ids: set,
        random_state: int = 42,
    ) -> List[Dict[str, Any]]:
        if target_n <= 0 or not pool:
            return []

        df = pd.DataFrame(pool).copy()
        df = df[~df["source_group_id"].isin(used_group_ids)].copy()
        if df.empty:
            return []

        df = df.drop_duplicates(subset=["source_group_id"], keep="first").reset_index(drop=True)

        if len(df) <= target_n:
            return df.sample(frac=1, random_state=random_state).to_dict(orient="records")

        strata = df["stratum_key"].value_counts().to_dict()
        total = len(df)

        expected = {k: strata[k] * target_n / total for k in strata}
        alloc = {k: int(v) for k, v in expected.items()}
        remainder = target_n - sum(alloc.values())

        frac_order = sorted(
            strata.keys(),
            key=lambda k: (expected[k] - int(expected[k]), strata[k]),
            reverse=True,
        )
        for key in frac_order:
            if remainder <= 0:
                break
            alloc[key] = alloc.get(key, 0) + 1
            remainder -= 1

        picked_rows: List[Dict[str, Any]] = []

        for stratum, g in df.groupby("stratum_key", sort=False):
            quota = alloc.get(stratum, 0)
            if quota <= 0:
                continue
            g = g.sample(frac=1, random_state=random_state).reset_index(drop=True)
            picked_rows.extend(g.head(quota).to_dict(orient="records"))

        if len(picked_rows) < target_n:
            picked_ids = {x["source_group_id"] for x in picked_rows}
            remaining_df = df[~df["source_group_id"].isin(picked_ids)]
            if not remaining_df.empty:
                extra = remaining_df.sample(
                    n=min(target_n - len(picked_rows), len(remaining_df)),
                    random_state=random_state,
                ).to_dict(orient="records")
                picked_rows.extend(extra)

        return picked_rows[:target_n]

    def _next_house_id(self, next_house_num: int) -> str:
        return f"H{next_house_num:05d}"

    def _next_no_kk(self) -> str:
        return f"{self.config.random_no_kk_prefix}{uuid.uuid4().hex}"

    def _next_image_id(self, image_num: int) -> str:
        return f"IMG{image_num:06d}"

    def _upload_and_build_images(
        self,
        raw_images: List[Dict[str, Any]],
        image_num_start: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        uploaded_images: List[Dict[str, Any]] = []
        image_num = image_num_start

        for img in raw_images:
            if not isinstance(img, dict):
                continue

            view_type = img.get("view_type")
            if view_type not in {"exterior", "interior"}:
                continue

            local_image_path = self._resolve_local_image_path(img.get("image_path"))
            upload_result = self.uploader.upload(local_image_path, view_type=view_type)

            uploaded_images.append(
                {
                    "image_id": self._next_image_id(image_num),
                    "image_path": upload_result["object_name"],
                    "image_ori_url": "",
                    "image_db_url": upload_result["image_db_url"],
                    "view_type": view_type,
                }
            )
            image_num += 1

        return uploaded_images, image_num

    def _normalize_final_record(
        self,
        raw_record: Dict[str, Any],
        actual_label: Dict[str, Any],
        house_type: str,
        house_id: str,
        image_num_start: int,
    ) -> tuple[Dict[str, Any], int]:
        images = raw_record.get("images", [])
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except Exception:
                images = []

        if not isinstance(images, list):
            images = []

        uploaded_images, next_image_num = self._upload_and_build_images(images, image_num_start)

        if isinstance(actual_label, str):
            try:
                actual_label = json.loads(actual_label)
            except Exception:
                actual_label = {}

        if not isinstance(actual_label, dict):
            actual_label = {}

        actual_label = {
            "atap": actual_label.get("atap", "Tidak Terdeteksi"),
            "dinding": actual_label.get("dinding", "Tidak Terdeteksi"),
            "lantai": actual_label.get("lantai", "Tidak Terdeteksi"),
        }

        if house_type == "multi":
            pass
        elif house_type == "single_exterior_only":
            actual_label["lantai"] = "Tidak Terdeteksi"
        elif house_type == "single_interior_only":
            actual_label["atap"] = "Tidak Terdeteksi"
            actual_label["dinding"] = "Tidak Terdeteksi"

        final_record = {
            "house_id": house_id,
            "no_kk": self._next_no_kk(),
            "house_type": house_type,
            "split": None,
            "match": None,
            "images": uploaded_images,
            "actual_label": actual_label,
            "dtsen": self._dtsen_null_object(),
        }

        return final_record, next_image_num

    def run(self) -> Dict[str, Any]:
        state = self._load_existing_state()
        sample_records = state["sample_records"]
        used_group_ids = state["used_group_ids"]
        next_house_num = state["next_house_num"]
        next_image_num = state["next_image_num"]

        crawling_records = self._read_json_or_jsonl(self.config.crawling_metadata_path)
        pools = self._build_candidate_pools(crawling_records, used_group_ids)

        available_multi = len({x["source_group_id"] for x in pools["multi"]})
        available_ext = len({x["source_group_id"] for x in pools["single_exterior_only"]})
        available_int = len({x["source_group_id"] for x in pools["single_interior_only"]})

        target_multi_wanted = self.config.target_multi_wanted or available_multi
        target_ext_wanted = self.config.target_ext_wanted or available_ext
        target_int_wanted = self.config.target_int_wanted or available_int

        target_multi = min(available_multi, target_multi_wanted)
        target_ext = min(available_ext, target_ext_wanted)
        target_int = min(available_int, target_int_wanted)

        selected_candidates: List[Dict[str, Any]] = []
        selected_group_ids = set()

        chosen_multi = self._stratified_unique_sample(
            pool=pools["multi"],
            target_n=target_multi,
            used_group_ids=used_group_ids | selected_group_ids,
            random_state=42,
        )
        for c in chosen_multi:
            selected_group_ids.add(c["source_group_id"])
        selected_candidates.extend(chosen_multi)

        chosen_ext = self._stratified_unique_sample(
            pool=pools["single_exterior_only"],
            target_n=target_ext,
            used_group_ids=used_group_ids | selected_group_ids,
            random_state=43,
        )
        for c in chosen_ext:
            selected_group_ids.add(c["source_group_id"])
        selected_candidates.extend(chosen_ext)

        chosen_int = self._stratified_unique_sample(
            pool=pools["single_interior_only"],
            target_n=target_int,
            used_group_ids=used_group_ids | selected_group_ids,
            random_state=44,
        )
        for c in chosen_int:
            selected_group_ids.add(c["source_group_id"])
        selected_candidates.extend(chosen_int)

        selected_candidates = sorted(
            selected_candidates,
            key=lambda x: (x["house_type"], x["source_group_id"], x["stratum_key"]),
        )

        merged_records = list(sample_records)
        new_records: List[Dict[str, Any]] = []

        for cand in selected_candidates:
            final_rec, next_image_num = self._normalize_final_record(
                raw_record=cand["raw_record"],
                actual_label=cand["actual_label"],
                house_type=cand["house_type"],
                house_id=self._next_house_id(next_house_num),
                image_num_start=next_image_num,
            )
            next_house_num += 1
            new_records.append(final_rec)
            merged_records.append(final_rec)

        self.config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_metadata_path, "w", encoding="utf-8") as f:
            for rec in merged_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        out_summary = Counter([r["house_type"] for r in new_records])

        return {
            "sample_records": len(sample_records),
            "added_records": len(new_records),
            "total_records": len(merged_records),
            "multi_added": out_summary.get("multi", 0),
            "single_exterior_only_added": out_summary.get("single_exterior_only", 0),
            "single_interior_only_added": out_summary.get("single_interior_only", 0),
            "available_multi": available_multi,
            "available_ext": available_ext,
            "available_int": available_int,
            "target_multi": target_multi,
            "target_ext": target_ext,
            "target_int": target_int,
            "output_path": str(self.config.output_metadata_path),
            "next_house_num": next_house_num,
            "next_image_num": next_image_num,
        }