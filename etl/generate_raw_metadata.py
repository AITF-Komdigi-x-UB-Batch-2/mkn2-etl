import json
import hashlib
from pathlib import Path

    # =========================
    # KONFIGURASI
    # =========================
    ROOT_DIR = Path("data/mkn_img")
    OUTPUT_METADATA = ROOT_DIR / "house_metadata.json"

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    SINGLE_CONFIG = {
        "rlh_ext":  ("layak", "exterior"),
        "rlh_int":  ("layak", "interior"),
        "rtlh_ext": ("tidak_layak", "exterior"),
        "rtlh_int": ("tidak_layak", "interior"),
    }

    MULTI_CONFIG = SINGLE_CONFIG.copy()

    # =========================
    # UTIL
    # =========================
    def file_hash(path, chunk_size=8192):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


    def is_image_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in IMAGE_EXTS


    def sorted_image_files(folder: Path):
        files = [p for p in folder.iterdir() if is_image_file(p)]
        return sorted(files, key=lambda p: p.name.lower())


    def new_house_id(idx: int) -> str:
        return f"H{idx:05d}"


    def extract_numeric_suffix(stem: str) -> int:
        """
        Ambil angka terakhir dari nama file.
        Contoh:
        mkn_multi_rtlh_int_0001 -> 1
        0001 -> 1
        """
        return int(stem.split("_")[-1])


    # =========================
    # BUILD SINGLE INDEX
    # =========================
    def build_single_index(root_dir: Path, single_config: dict):
        """
        Index single images by (folder_name, hash).
        Ini dipakai untuk matching multi -> single.
        """
        index = {}

        for folder_name, (kelayakan, view_type) in single_config.items():
            folder = root_dir / folder_name
            if not folder.exists():
                print(f"[WARN] Folder tidak ada: {folder}")
                continue

            for fpath in sorted_image_files(folder):
                h = file_hash(fpath)
                key = (folder_name, h)

                if key in index:
                    print(f"[WARN] Duplikat hash di folder yang sama: {fpath}")
                    continue

                index[key] = {
                    "hash": h,
                    "rel_path": str(fpath.relative_to(root_dir)),
                    "folder_name": folder_name,
                    "kelayakan_rumah": kelayakan,
                    "view_type": view_type,
                }

        return index


    # =========================
    # BUILD MULTI RECORDS
    # =========================
    def build_multi_houses(root_dir: Path, multi_root: Path, multi_config: dict, single_index: dict):
        """
        Multi-images dikelompokkan per rumah.
        Setiap image di multi hanya menyimpan satu path: image_path
        yang menunjuk ke file single yang cocok berdasarkan hash.
        """
        house_map = {}
        view_sort = {"exterior": 0, "interior": 1}

        for folder_name, (kelayakan, view_type) in multi_config.items():
            folder = multi_root / folder_name
            if not folder.exists():
                print(f"[WARN] Folder multi tidak ada: {folder}")
                continue

            for fpath in sorted_image_files(folder):
                stem = fpath.stem
                pair_key = (kelayakan, stem)

                h = file_hash(fpath)

                # cari referensi ke single dari folder yang sama
                single_key = (folder_name, h)
                single_info = single_index.get(single_key)

                if single_info is None:
                    print(f"[WARN] Tidak ketemu padanan single untuk: {fpath}")
                    continue

                entry = {
                    "image_hash": h,
                    "image_path": single_info["rel_path"],  # satu-satunya path
                    "folder_name": folder_name,
                    "view_type": view_type,
                    "material_atap": None,
                    "kondisi_atap": None,
                    "material_dinding": None,
                    "kondisi_dinding": None,
                    "material_lantai": None,
                    "kondisi_lantai": None,
                }

                if pair_key not in house_map:
                    house_map[pair_key] = {
                        "kelayakan_rumah": kelayakan,
                        "stem": stem,
                        "images": [],
                    }

                house_map[pair_key]["images"].append(entry)

        house_records = []
        for (kelayakan, stem), info in sorted(
            house_map.items(),
            key=lambda x: (x[0][0], extract_numeric_suffix(x[0][1]))
        ):
            images = sorted(info["images"], key=lambda x: view_sort.get(x["view_type"], 99))

            if len(images) < 2:
                print(f"[WARN] Pair tidak lengkap: {kelayakan} - {stem}")

            record = {
                "house_id": None,
                "kelayakan_rumah": info["kelayakan_rumah"],
                "split": "unsplit",
                "dataset_scheme": "multi",
                "pair_key": {
                    "kelayakan_rumah": info["kelayakan_rumah"],
                    "stem": info["stem"],
                },
                "images": images,
            }
            house_records.append(record)

        return house_records


    # =========================
    # BUILD SINGLE RECORDS
    # =========================
    def build_single_records(root_dir: Path, single_config: dict, keep_all=True, exclude_hashes=None):
        exclude_hashes = exclude_hashes or set()
        records = []
        counter = 1

        for folder_name, (kelayakan, view_type) in single_config.items():
            folder = root_dir / folder_name
            if not folder.exists():
                print(f"[WARN] Folder tidak ada: {folder}")
                continue

            for fpath in sorted_image_files(folder):
                h = file_hash(fpath)

                if (not keep_all) and (h in exclude_hashes):
                    continue

                record = {
                    "house_id": new_house_id(counter),
                    "kelayakan_rumah": kelayakan,
                    "split": "unsplit",
                    "dataset_scheme": "single",
                    "source_group_id": h,
                    "images": [
                        {
                            "image_hash": h,
                            "image_path": str(fpath.relative_to(root_dir)),
                            "folder_name": folder_name,
                            "view_type": view_type,
                            "material_atap": None,
                            "kondisi_atap": None,
                            "material_dinding": None,
                            "kondisi_dinding": None,
                            "material_lantai": None,
                            "kondisi_lantai": None,
                        }
                    ],
                }
                records.append(record)
                counter += 1

        return records


    # =========================
    # MAIN
    # =========================
    def main():
        ROOT_DIR.mkdir(parents=True, exist_ok=True)
        multi_root = ROOT_DIR / "multi_images"

        print("Indexing single images...")
        single_index = build_single_index(ROOT_DIR, SINGLE_CONFIG)
        print(f"Total single terindeks: {len(single_index)}")

        print("Building multi...")
        multi_records = build_multi_houses(ROOT_DIR, multi_root, MULTI_CONFIG, single_index)
        print(f"Total record multi: {len(multi_records)}")

        print("Building single...")
        single_records = build_single_records(ROOT_DIR, SINGLE_CONFIG, keep_all=True)
        print(f"Total record single: {len(single_records)}")

        all_records = []
        hid_counter = 1

        for rec in multi_records + single_records:
            rec["house_id"] = new_house_id(hid_counter)
            hid_counter += 1
            all_records.append(rec)

        with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        print(f"Metadata ditulis ke: {OUTPUT_METADATA}")


    if __name__ == "__main__":
        main()