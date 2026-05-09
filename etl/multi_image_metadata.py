import json
import pandas as pd
from pathlib import Path

# =========================
# PATH
# =========================
CNN_JSON = "data/cnn/mkn_image_metadata.json"
EXCEL_PATH = "data/pairing.xlsx"

OUTPUT_JSON = "data/metadata/multi_images_metadata.json"

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

for item in cnn_data:
    cnn_by_id[item["id"]] = item

# =========================
# OUTPUT CONTAINER
# =========================
houses = []

house_counter = 1
group_counter = 1


def new_house_id(idx):
    return f"H{idx:05d}"


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

            # fallback jika tidak ada exterior
            if dinding is None:
                dinding = item["material_dinding"]

            lantai = item["material_lantai"]

    return {
        "atap": atap,
        "dinding": dinding,
        "lantai": lantai
    }


# =========================================================
# BUILD MULTI HOUSE ONLY
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

    # skip jika pair tidak lengkap
    if len(images) < 2:
        print(f"[WARN] Pair tidak lengkap: {ext_id} | {int_id}")
        continue

    # gunakan exterior sebagai referensi utama
    base_item = cnn_by_id[ext_id]

    source_group_id = f"G{group_counter:05d}"
    group_counter += 1

    house = {
        "house_id": new_house_id(house_counter),

        "house_type": "multi",

        "split": None,

        # anti leakage
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
# SAVE
# =========================================================
Path("data/metadata").mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(houses, f, ensure_ascii=False, indent=2)

print(f"✅ Total multi houses: {len(houses)}")
print(f"✅ Metadata saved to: {OUTPUT_JSON}")