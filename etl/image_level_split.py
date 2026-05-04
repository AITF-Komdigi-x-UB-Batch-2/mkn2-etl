import json
import random
from collections import defaultdict

INPUT_JSON = "data/cnn/mkn_image_metadata.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
random.seed(SEED)


# =========================
# LOAD
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)


# =========================
# BUILD STRATIFICATION KEY
# =========================
def build_key(item):
    view = item["view_type"]

    if view == "exterior":
        atap = item["material_atap"] or "none"
        dinding = item["material_dinding"] or "none"
        return f"{view}|{atap}|{dinding}"

    else:  # interior
        dinding = item["material_dinding"] or "none"
        lantai = item["material_lantai"] or "none"
        return f"{view}|{dinding}|{lantai}"


# =========================
# GROUP BY KEY
# =========================
groups = defaultdict(list)

for item in data:
    key = build_key(item)
    groups[key].append(item)


# =========================
# STRATIFIED SPLIT (WITH RARE HANDLING)
# =========================
for key, samples in groups.items():

    random.shuffle(samples)
    n = len(samples)

    # ===== HANDLE RARE CASE =====
    if n == 1:
        samples[0]["split"] = "train"
        print(f"{key}: train=1, val=0, test=0 (rare=1)")
        continue

    elif n == 2:
        samples[0]["split"] = "train"
        samples[1]["split"] = "test"
        print(f"{key}: train=1, val=0, test=1 (rare=2)")
        continue

    # ===== NORMAL SPLIT =====
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    # safety guard (biar ga kosong)
    if n_val == 0:
        n_val = 1
        n_train -= 1

    if n_test == 0:
        n_test = 1
        n_train -= 1

    for x in samples[:n_train]:
        x["split"] = "train"

    for x in samples[n_train:n_train+n_val]:
        x["split"] = "val"

    for x in samples[n_train+n_val:]:
        x["split"] = "test"

    print(
        f"{key}: "
        f"train={n_train}, "
        f"val={n_val}, "
        f"test={n_test}"
    )


# =========================
# SAVE
# =========================
with open(INPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\nSplit stratified multi-attribute (with rare handling) selesai.")