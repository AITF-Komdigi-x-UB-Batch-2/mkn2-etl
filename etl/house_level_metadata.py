import json
import pandas as pd
from pathlib import Path

# =========================
# PATH
# =========================
CNN_JSON = "data/cnn/mkn_image_metadata.json"
EXCEL_PATH = "data/pairing.xlsx"

OUTPUT_JSON = "data/metadata/mkn_house_metadata.json"

# =========================
# LOAD DATA
# =========================
with open(CNN_JSON, "r", encoding="utf-8") as f:
    cnn_data = json.load(f)

df = pd.read_excel(EXCEL_PATH)

# rapikan nama kolom excel
df.columns = [c.strip().lower() for c in df.columns]

# =========================
# INDEX CNN METADATA
# =========================
cnn_by_path = {}
cnn_by_id = {}

for item in cnn_data:
    cnn_by_path[item["image_path"]] = item
    cnn_by_id[item["id"]] = item

# =========================
# OUTPUT CONTAINER
# =========================
houses = []
used_image_ids = set()

house_counter = 1


def new_house_id(idx):
    return f"H{idx:05d}"


# =========================================================
# SOURCE GROUP ID
# =========================================================
# Tujuan:
# image yang muncul di sample multi dan single
# HARUS berada di split yang sama
#
# Maka:
# - multi house dan single house yang berbagi image
#   memakai source_group_id yang sama
#
# Contoh:
# exterior IMG000006 dan interior IMG000577
# muncul di:
#   - multi house
#   - single exterior
#   - single interior
#
# ketiganya share source_group_id yang sama
# =========================================================
image_to_group = {}
group_counter = 1

for _, row in df.iterrows():

    ext_id = str(row["id_exterior"]).strip()
    int_id = str(row["id_interior"]).strip()

    group_id = f"G{group_counter:05d}"
    group_counter += 1

    image_to_group[ext_id] = group_id
    image_to_group[int_id] = group_id


# =========================================================
# AGGREGATE CNN PREDICTION
# =========================================================
def aggregate_prediction(images):

    atap = None
    dinding = None
    lantai = None

    for img in images:

        item = cnn_by_id[img["image_id"]]

        if item["view_type"] == "exterior":
            atap = item["material_atap"]
            dinding = item["material_dinding"]

        elif item["view_type"] == "interior":

            # fallback dinding kalau tidak ada exterior
            if dinding is None:
                dinding = item["material_dinding"]

            lantai = item["material_lantai"]

    return {
        "atap": atap,
        "dinding": dinding,
        "lantai": lantai
    }


# =========================================================
# BUILD MULTI HOUSE
# =========================================================
for _, row in df.iterrows():

    ext_id = str(row["id_exterior"]).strip()
    int_id = str(row["id_interior"]).strip()

    ext_path = str(row["image_path_exterior"]).strip()
    int_path = str(row["image_path_interior"]).strip()

    images = []

    # ======================
    # EXTERIOR
    # ======================
    if ext_id in cnn_by_id:

        ext_item = cnn_by_id[ext_id]

        images.append({
            "image_id": ext_item["id"],
            "image_path": ext_item["image_path"],
            "view_type": ext_item["view_type"]
        })

        used_image_ids.add(ext_item["id"])

    # ======================
    # INTERIOR
    # ======================
    if int_id in cnn_by_id:

        int_item = cnn_by_id[int_id]

        images.append({
            "image_id": int_item["id"],
            "image_path": int_item["image_path"],
            "view_type": int_item["view_type"]
        })

        used_image_ids.add(int_item["id"])

    if len(images) == 0:
        continue

    # gunakan exterior sebagai referensi utama
    base_item = cnn_by_id[ext_id]

    source_group_id = image_to_group.get(ext_id)

    house = {
        "house_id": new_house_id(house_counter),

        "house_type": "multi",

        # split belum diisi
        "split": None,

        # penting untuk anti data leakage
        "source_group_id": source_group_id,

        "kelayakan_rumah": base_item["kelayakan_rumah"],

        "images": images,

        "cnn_prediction": aggregate_prediction(images),

        "dtsen": {
            "atap": None,
            "dinding": None,
            "lantai": None
        }
    }

    houses.append(house)
    house_counter += 1


# =========================================================
# BUILD SINGLE HOUSE
# =========================================================
for item in cnn_data:

    image_id = item["id"]

    image = {
        "image_id": item["id"],
        "image_path": item["image_path"],
        "view_type": item["view_type"]
    }

    # =========================================
    # Jika image termasuk pair multi:
    # gunakan source_group_id yang sama
    # =========================================
    if image_id in image_to_group:
        source_group_id = image_to_group[image_id]

    # =========================================
    # Jika benar-benar standalone:
    # buat group sendiri
    # =========================================
    else:
        source_group_id = f"G{group_counter:05d}"
        group_counter += 1

    # =========================================
    # SINGLE PREDICTION
    # =========================================
    if item["view_type"] == "exterior":

        cnn_prediction = {
            "atap": item["material_atap"],
            "dinding": item["material_dinding"],
            "lantai": None
        }

    else:

        cnn_prediction = {
            "atap": None,
            "dinding": item["material_dinding"],
            "lantai": item["material_lantai"]
        }

    house = {
        "house_id": new_house_id(house_counter),

        "house_type": "single",

        "split": None,

        # =====================================
        # KUNCI ANTI DATA LEAKAGE
        # =====================================
        "source_group_id": source_group_id,

        "kelayakan_rumah": item["kelayakan_rumah"],

        "images": [image],

        "cnn_prediction": cnn_prediction,

        "dtsen": {
            "atap": None,
            "dinding": None,
            "lantai": None
        }
    }

    houses.append(house)
    house_counter += 1


# =========================================================
# SAVE
# =========================================================
Path("data/metadata").mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(houses, f, ensure_ascii=False, indent=2)

print(f"✅ Total houses: {len(houses)}")
print(f"✅ Metadata saved to: {OUTPUT_JSON}")