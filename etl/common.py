from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import re

import pandas as pd


SELECTED_COLUMNS = [
    "no_kk",
    "id_lantai_terluas",
    "id_dinding_terluas",
    "id_atap_terluas",
    "foto_rumah",
    "foto_rumah_tampak_dalam",
]

LANTAI_MAP: Dict[str, str] = {
    "1": "marmer/granit",
    "2": "keramik",
    "3": "parket/vinil/karpet",
    "4": "ubin/tegel/teraso",
    "5": "kayu/papan",
    "6": "semen/bata_merah",
    "7": "bambu",
    "8": "tanah",
    "9": "lainnya",
}

DINDING_MAP: Dict[str, str] = {
    "1": "tembok",
    "2": "plesteran_anyaman_bambu/kawat",
    "3": "kayu/papan/gypsum/GRC/calciboard",
    "4": "anyaman_bambu",
    "5": "batang_kayu",
    "6": "bambu",
    "7": "lainnya",
}

ATAP_MAP: Dict[str, str] = {
    "1": "beton",
    "2": "genteng",
    "3": "seng",
    "4": "asbes",
    "5": "bambu",
    "6": "kayu/sirap",
    "7": "jerami/ijuk/daun-daunan/rumbia",
    "8": "lainnya",
}


def normalize_cell(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "nan", "none", "null", "[]", "[ ]"}:
            return pd.NA
        return cleaned

    return value


def clean_no_kk(value: Any) -> Any:
    """
    Menjaga no_kk tetap aman dibaca sebagai string.
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return pd.NA

    if text.endswith(".0"):
        text = text[:-2]

    return text


def normalize_code(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None

    return str(int(digits))


def map_label(value: Any, mapping: Dict[str, str]) -> str:
    code = normalize_code(value)
    if code is None:
        return "unclassified"
    return mapping.get(code, "unclassified")


def map_lantai(value: Any) -> str:
    return map_label(value, LANTAI_MAP)


def map_dinding(value: Any) -> str:
    return map_label(value, DINDING_MAP)


def map_atap(value: Any) -> str:
    return map_label(value, ATAP_MAP)


def parse_minio_links(cell: Any) -> List[str]:
    """
    Input:
    - '[link]'
    - '[link1, link2]'
    - '[https://...prefix=]' -> dianggap kosong

    Output:
    - list URL valid
    """
    if pd.isna(cell):
        return []

    text = str(cell).strip()
    if not text or text in {"[]", "[ ]"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    if not text:
        return []

    parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
    urls: List[str] = []

    for part in parts:
        if not part:
            continue
        if part.endswith("prefix="):
            continue
        if part.startswith("http://") or part.startswith("https://"):
            urls.append(part)

    return urls


def infer_extension_from_url(url: str, default: str = ".jpg") -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else default


def build_image_db_url(public_base_url: str, bucket_name: str, object_name: str) -> str:
    return f"{public_base_url.rstrip('/')}/{bucket_name}/{object_name}"