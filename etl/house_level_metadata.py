import json
import pandas as pd

# =========================
# PATH
# =========================
CNN_JSON = "data/cnn/mkn_image_metadata.json"
EXCEL_PATH = "data/pairing.xlsx"
OUTPUT_JSON = "data/metadata/house_metadata.json"

# =========================
# LOAD
# =========================
with open(CNN_JSON, "r", encoding="utf-8") as f:
    cnn_data = json.load(f)

df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip().str.lower()

# =========================
# INDEX CNN
# =========================
cnn_index = {item["image_path"]: item for item in cnn_data}

used_images = set()
houses = []
hid_counter = 1

def new_house_id(idx):
    return f"H{idx:05d}"


# =========================
# AGGREGATION FUNCTION
# =========================
def aggregate_cnn(images):
    atap = None
    dinding = None
    lantai = None

    for img in images:
        item = cnn_index[img["image_path"]]

        if img["view_type"] == "exterior":
            atap = item["material_atap"]
            dinding = item["material_dinding"]

        elif img["view_type"] == "interior":
            if dinding is None:  # fallback kalau tidak ada exterior
                dinding = item["material_dinding"]
            lantai = item["material_lantai"]

    return {
        "atap": atap,
        "dinding": dinding,
        "lantai": lantai
    }


# =========================
# BUILD MULTI
# =========================
for _, row in df.iterrows():

    ext_path = row["image_path_exterior"]
    int_path = row["image_path_interior"]

    images = []

    if pd.notna(ext_path) and ext_path in cnn_index:
        item = cnn_index[ext_path]

        images.append({
            "image_id": item["id"],
            "image_path": item["image_path"],
            "view_type": "exterior"
        })

        used_images.add(ext_path)

    if pd.notna(int_path) and int_path in cnn_index:
        item = cnn_index[int_path]

        images.append({
            "image_id": item["id"],
            "image_path": item["image_path"],
            "view_type": "interior"
        })

        used_images.add(int_path)

    if not images:
        continue

    base_item = cnn_index[images[0]["image_path"]]

    house = {
        "house_id": new_house_id(hid_counter),
        "house_type": "multi",
        "split": base_item["split"],
        "kelayakan_rumah": base_item["kelayakan_rumah"],

        "images": images,

        "cnn_prediction": aggregate_cnn(images),

        "dtsen": {
            "atap": None,
            "dinding": None,
            "lantai": None
        }
    }

    houses.append(house)
    hid_counter += 1


# =========================
# BUILD SINGLE
# =========================
for item in cnn_data:

    path = item["image_path"]

    if path in used_images:
        continue

    image = {
        "image_id": item["id"],
        "image_path": item["image_path"],
        "view_type": item["view_type"]
    }

    # aggregation langsung dari 1 image
    if item["view_type"] == "exterior":
        cnn_pred = {
            "atap": item["material_atap"],
            "dinding": item["material_dinding"],
            "lantai": None
        }
    else:
        cnn_pred = {
            "atap": None,
            "dinding": item["material_dinding"],
            "lantai": item["material_lantai"]
        }

    house = {
        "house_id": new_house_id(hid_counter),
        "house_type": "single",
        "split": item["split"],
        "kelayakan_rumah": item["kelayakan_rumah"],

        "images": [image],

        "cnn_prediction": cnn_pred,

        "dtsen": {
            "atap": None,
            "dinding": None,
            "lantai": None
        }
    }

    houses.append(house)
    hid_counter += 1


# =========================
# SAVE
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(houses, f, indent=2, ensure_ascii=False)

print(f"✅ Done. Total houses: {len(houses)}")