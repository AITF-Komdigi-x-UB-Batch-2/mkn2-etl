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
cnn_by_id = {}
cnn_by_path = {}

for item in cnn_data:
    cnn_by_id[item["id"]] = item
    cnn_by_path[item["image_path"]] = item

# =========================
# OUTPUT CONTAINER
# =========================
houses = []

house_counter = 1
group_counter = 1

# =========================
# UTIL
# =========================
def new_house_id(idx):
    return f"H{idx:05d}"


def new_group_id(idx):
    return f"GID{idx:05d}"


# =========================================================
# SOURCE GROUP ID
# =========================================================
# Tujuan:
# sample multi dan single yang menggunakan image sama
# HARUS berada di split yang sama
#
# Maka:
# semua image pair share source_group_id yang sama
# =========================================================
image_to_group = {}

for _, row in df.iterrows():

    ext_id = str(row["id_exterior"]).strip()
    int_id = str(row["id_interior"]).strip()

    group_id = new_group_id(group_counter)
    group_counter += 1

    image_to_group[ext_id] = group_id
    image_to_group[int_id] = group_id


# =========================================================
# AGGREGATE PREDICTION
# =========================================================
def aggregate_prediction(images):

    atap = None
    dinding = None
    lantai = None

    for img in images:

        item = cnn_by_id[img["image_id"]]

        # ======================
        # EXTERIOR
        # ======================
        if item["view_type"] == "exterior":

            atap = item["material_atap"]

            # dinding utama dari exterior
            dinding = item["material_dinding"]

        # ======================
        # INTERIOR
        # ======================
        elif item["view_type"] == "interior":

            # fallback kalau tidak ada exterior
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

    # skip kalau kosong
    if len(images) == 0:
        continue

    # referensi utama
    base_item = cnn_by_id[ext_id]

    # source group
    source_group_id = image_to_group.get(ext_id)

    # prediksi agregasi
    prediksi = aggregate_prediction(images)

    house = {
        "house_id": new_house_id(house_counter),

        "house_type": "multi",

        # split nanti diisi setelah splitting
        "split": None,

        # anti leakage
        "source_group_id": source_group_id,

        # validasi DTSEN vs prediksi
        "match": None,

        "kelayakan_rumah": base_item["kelayakan_rumah"],

        "images": images,

        # rename cnn_prediction -> prediksi
        "prediksi": prediksi,

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
    # Kalau image bagian dari pair
    # gunakan source_group_id yang sama
    # =========================================
    if image_id in image_to_group:

        source_group_id = image_to_group[image_id]

    # =========================================
    # Standalone image
    # =========================================
    else:

        source_group_id = new_group_id(group_counter)
        group_counter += 1

    # =========================================
    # SINGLE PREDICTION
    # =========================================
    if item["view_type"] == "exterior":

        prediksi = {
            "atap": item["material_atap"],
            "dinding": item["material_dinding"],
            "lantai": None
        }

    else:

        prediksi = {
            "atap": None,
            "dinding": item["material_dinding"],
            "lantai": item["material_lantai"]
        }

    house = {
        "house_id": new_house_id(house_counter),

        "house_type": "single",

        "split": None,

        # anti leakage
        "source_group_id": source_group_id,

        # hasil validasi nanti
        "match": None,

        "kelayakan_rumah": item["kelayakan_rumah"],

        "images": [image],

        "prediksi": prediksi,

        "dtsen": {
            "atap": None,
            "dinding": None,
            "lantai": None
        }
    }

    houses.append(house)
    house_counter += 1


# =========================================================
# SORT HOUSE
# =========================================================
houses = sorted(houses, key=lambda x: x["house_id"])


# =========================================================
# SAVE
# =========================================================
Path("data/metadata").mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(houses, f, ensure_ascii=False, indent=2)

print(f"✅ Total houses: {len(houses)}")
print(f"✅ Metadata saved to: {OUTPUT_JSON}")