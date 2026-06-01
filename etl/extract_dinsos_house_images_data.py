from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from etl.common import SELECTED_COLUMNS, clean_no_kk, normalize_cell


@dataclass
class ExtractConfig:
    source_excel: Path = Path("data/data_keluarga_dinsos_jatim.xlsx")
    output_excel: Path = Path("data/data_keluarga_dinsos_jatim_extracted.xlsx")


class DinsosHouseImagesExtractor:
    def __init__(self, config: ExtractConfig | None = None):
        self.config = config or ExtractConfig()

    def run(self) -> pd.DataFrame:
        if not self.config.source_excel.exists():
            raise FileNotFoundError(f"File sumber tidak ditemukan: {self.config.source_excel}")

        df = pd.read_excel(
            self.config.source_excel,
            usecols=SELECTED_COLUMNS,
            dtype=str,
            engine="openpyxl",
        )

        for col in SELECTED_COLUMNS:
            df[col] = df[col].apply(normalize_cell)

        df["no_kk"] = df["no_kk"].apply(clean_no_kk)

        # hanya baris yang semua kolom terpilih terisi
        df = df.dropna(subset=SELECTED_COLUMNS, how="any").reset_index(drop=True)

        self.config.output_excel.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.config.output_excel, index=False, engine="openpyxl")

        return df