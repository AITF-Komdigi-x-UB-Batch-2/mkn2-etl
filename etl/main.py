from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl.extract_dinsos_house_images_data import DinsosHouseImagesExtractor
from etl.download_images_and_metadata import DinsosHouseDownloadMetadataPipeline


class RutilahuETLPipeline:
    def __init__(self):
        self.extractor = DinsosHouseImagesExtractor()
        self.downloader = DinsosHouseDownloadMetadataPipeline()

    def run_extract(self) -> None:
        df = self.extractor.run()
        print(f"[OK] Extract selesai. Total rows: {len(df):,}")

    def run_download_and_metadata(self) -> None:
        outputs = self.downloader.run()
        print("[OK] Download + metadata selesai.")
        for k, v in outputs.items():
            print(f"{k}: {v}")

    def run_all(self) -> None:
        self.run_extract()
        self.run_download_and_metadata()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rutilahu ETL Pipeline")
    parser.add_argument("--extract_data", action="store_true", help="Jalankan extract data dari excel lokal")
    parser.add_argument(
        "--download_metadata",
        action="store_true",
        help="Jalankan download image ke MinIO + pembuatan metadata",
    )
    parser.add_argument("--all", action="store_true", help="Jalankan semua task")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    pipeline = RutilahuETLPipeline()

    if args.all:
        pipeline.run_all()
        return

    did_run = False

    if args.extract_data:
        pipeline.run_extract()
        did_run = True

    if args.download_metadata:
        pipeline.run_download_and_metadata()
        did_run = True

    if not did_run:
        parser.print_help()


if __name__ == "__main__":
    main()