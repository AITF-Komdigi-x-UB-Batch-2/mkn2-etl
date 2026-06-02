from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from minio import Minio
from dotenv import load_dotenv

from etl.common import (
    build_image_db_url,
    map_atap,
    map_dinding,
    map_lantai,
    parse_minio_links,
    infer_extension_from_url,
)

# Load environment variables from .env
load_dotenv()


@dataclass
class DownloadMetadataConfig:
    extracted_excel: Path = Path("data/data_keluarga_dinsos_jatim_extracted.xlsx").expanduser()
    metadata_dir: Path = Path("metadata").expanduser()
    workers: int = 16

    # MinIO configuration from environment variables
    minio_endpoint: str = None
    minio_access_key: str = None
    minio_secret_key: str = None
    minio_secure: bool = False
    bucket_name: str = None
    public_base_url: str = None

    # folder object di bucket
    exterior_prefix: str = "tampak_luar"
    interior_prefix: str = "tampak_dalam"

    def __post_init__(self):
        # Load from environment variables (required - no defaults!)
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
        
        if self.bucket_name is None:
            self.bucket_name = os.getenv("MINIO_BUCKET_NAME")
            if not self.bucket_name:
                raise ValueError("MINIO_BUCKET_NAME environment variable is required")
        
        if self.public_base_url is None:
            self.public_base_url = os.getenv("MINIO_PUBLIC_BASE_URL")
            if not self.public_base_url:
                raise ValueError("MINIO_PUBLIC_BASE_URL environment variable is required")
        
        # Parse secure flag from env
        secure_str = os.getenv("MINIO_SECURE", "False")
        self.minio_secure = secure_str.lower() in ("true", "1", "yes")


@dataclass
class ImageTask:
    house_id: str
    image_id: str
    no_kk: str | None
    view_type: str
    source_url: str
    object_name: str
    local_path: str
    image_db_url: str
    actual_label: Dict[str, str]


class DinsosHouseDownloadMetadataPipeline:
    def __init__(self, config: DownloadMetadataConfig | None = None):
        self.config = config or DownloadMetadataConfig()
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        (self.config.metadata_dir / "houses").mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.config.metadata_dir / "_tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.minio_client = Minio(
            self.config.minio_endpoint,
            access_key=self.config.minio_access_key,
            secret_key=self.config.minio_secret_key,
            secure=self.config.minio_secure,
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        if not self.minio_client.bucket_exists(self.config.bucket_name):
            self.minio_client.make_bucket(self.config.bucket_name)

    def load_data(self) -> pd.DataFrame:
        if not self.config.extracted_excel.exists():
            raise FileNotFoundError(
                f"File extracted tidak ditemukan: {self.config.extracted_excel}"
            )
        return pd.read_excel(self.config.extracted_excel, dtype=str, engine="openpyxl")

    def _build_actual_label(self, row: pd.Series) -> Dict[str, str]:
        return {
            "atap": map_atap(row.get("id_atap_terluas")),
            "dinding": map_dinding(row.get("id_dinding_terluas")),
            "lantai": map_lantai(row.get("id_lantai_terluas")),
        }

    def build_tasks(self, df: pd.DataFrame) -> List[ImageTask]:
        tasks: List[ImageTask] = []

        image_seq = 1
        exterior_seq = 1
        interior_seq = 1

        for idx, row in df.iterrows():
            house_id = f"H{idx + 1:05d}"
            no_kk = row.get("no_kk") or None
            actual_label = self._build_actual_label(row)

            exterior_urls = parse_minio_links(row.get("foto_rumah"))
            interior_urls = parse_minio_links(row.get("foto_rumah_tampak_dalam"))

            for url in exterior_urls:
                ext = infer_extension_from_url(url, default=".jpg")
                object_name = f"{self.config.exterior_prefix}/mkn2_exterior_img_{exterior_seq:06d}{ext}"
                local_path = object_name
                tasks.append(
                    ImageTask(
                        house_id=house_id,
                        image_id=f"IMG{image_seq:06d}",
                        no_kk=no_kk,
                        view_type="exterior",
                        source_url=url,
                        object_name=object_name,
                        local_path=local_path,
                        image_db_url=build_image_db_url(
                            self.config.public_base_url,
                            self.config.bucket_name,
                            object_name,
                        ),
                        actual_label=actual_label,
                    )
                )
                image_seq += 1
                exterior_seq += 1

            for url in interior_urls:
                ext = infer_extension_from_url(url, default=".jpg")
                object_name = f"{self.config.interior_prefix}/mkn2_interior_img_{interior_seq:06d}{ext}"
                local_path = object_name
                tasks.append(
                    ImageTask(
                        house_id=house_id,
                        image_id=f"IMG{image_seq:06d}",
                        no_kk=no_kk,
                        view_type="interior",
                        source_url=url,
                        object_name=object_name,
                        local_path=local_path,
                        image_db_url=build_image_db_url(
                            self.config.public_base_url,
                            self.config.bucket_name,
                            object_name,
                        ),
                        actual_label=actual_label,
                    )
                )
                image_seq += 1
                interior_seq += 1

        return tasks

    def _download_and_upload_one(self, task: ImageTask) -> Dict[str, Any]:
        temp_file = self.tmp_dir / f"{task.image_id}.tmp"
        result = {
            "house_id": task.house_id,
            "image_id": task.image_id,
            "view_type": task.view_type,
            "source_url": task.source_url,
            "object_name": task.object_name,
            "local_path": task.local_path,
            "image_db_url": task.image_db_url,
            "no_kk": task.no_kk,
            "status": "failed",
            "error": "",
        }

        try:
            resp = requests.get(task.source_url, stream=True, timeout=90)
            resp.raise_for_status()

            with open(temp_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            file_size = temp_file.stat().st_size

            with open(temp_file, "rb") as f:
                self.minio_client.put_object(
                    bucket_name=self.config.bucket_name,
                    object_name=task.object_name,
                    data=f,
                    length=file_size,
                    content_type="image/jpeg",
                )

            result["status"] = "downloaded"

        except Exception as e:
            result["error"] = str(e)

        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

        return result

    def _build_house_metadata(self, house_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        success_rows = [r for r in rows if r["status"] == "downloaded"]
        if not success_rows:
            return {}

        first = success_rows[0]
        images = [
            {
                "image_id": r["image_id"],
                "image_path": r["local_path"],
                "image_ori_url": r["source_url"],
                "image_db_url": r["image_db_url"],
                "view_type": r["view_type"],
            }
            for r in success_rows
        ]

        has_exterior = any(img["view_type"] == "exterior" for img in images)
        has_interior = any(img["view_type"] == "interior" for img in images)
        house_type = "multi" if (has_exterior and has_interior) else "single"

        metadata = {
            "house_id": house_id,
            "no_kk": first.get("no_kk"),
            "house_type": house_type,
            "split": None,
            "match": None,
            "images": images,
            "actual_label": first.get("actual_label"),
            "dtsen": {
                "atap": None,
                "dinding": None,
                "lantai": None,
            }
        }
        return metadata

    def run(self) -> Dict[str, Path]:
        df = self.load_data()
        tasks = self.build_tasks(df)

        if not tasks:
            raise ValueError("Tidak ada image task yang bisa diproses.")

        # download + upload dalam satu pipeline
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            results = list(executor.map(self._download_and_upload_one, tasks))

        # simpan manifest
        manifest_path = self.config.metadata_dir / "download_manifest.csv"
        fieldnames = [
            "house_id",
            "image_id",
            "view_type",
            "source_url",
            "object_name",
            "local_path",
            "image_db_url",
            "no_kk",
            "status",
            "error",
        ]

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

        # group per house lalu buat metadata
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        task_by_image_id = {t.image_id: t for t in tasks}

        for r in results:
            task = task_by_image_id[r["image_id"]]
            row_dict = {
                **r,
                "actual_label": task.actual_label,
            }
            grouped.setdefault(task.house_id, []).append(row_dict)

        metadata_jsonl_path = self.config.metadata_dir / "metadata.jsonl"
        houses_dir = self.config.metadata_dir / "houses"
        houses_dir.mkdir(parents=True, exist_ok=True)

        with open(metadata_jsonl_path, "w", encoding="utf-8") as jsonl_f:
            for house_id, rows in grouped.items():
                metadata = self._build_house_metadata(house_id, rows)
                if not metadata:
                    continue

                house_json = houses_dir / f"{house_id}.json"
                with open(house_json, "w", encoding="utf-8") as jf:
                    json.dump(metadata, jf, ensure_ascii=False, indent=2)

                jsonl_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        return {
            "manifest_csv": manifest_path,
            "metadata_jsonl": metadata_jsonl_path,
            "metadata_dir": self.config.metadata_dir,
        }