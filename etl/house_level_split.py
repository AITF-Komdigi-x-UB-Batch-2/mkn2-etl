import json
import random
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_JSON = "data/metadata/mkn_house_metadata.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

# =========================
# LOAD
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    houses = json.load(f)

# =========================
# BUILD GROUP
# =========================
# IMPORTANT:
# Semua house dengan source_group_id sama
# HARUS masuk split yang sama
#
# Karena:
# - single dan multi bisa share image
# - supaya tidak data leakage
# =========================
grouped = defaultdict(list)

for house in houses:
    gid = house["source_group_id"]
    grouped[gid].append(house)

groups = list(grouped.values())

# =========================
# BUILD STRATIFICATION KEY
# =========================
def build_key(group):

    # pakai sample pertama sebagai representasi
    sample = group[0]

    view_types = sorted(
        list(set(
            img["view_type"]
            for img in sample["images"]
        ))
    )

    view_key = "+".join(view_types)

    pred = sample["prediksi"]

    # =====================
    # MULTI
    # =====================
    if view_key == "exterior+interior":

        atap = pred["atap"] or "none"
        dinding = pred["dinding"] or "none"
        lantai = pred["lantai"] or "none"

        return f"multi|{atap}|{dinding}|{lantai}"

    # =====================
    # SINGLE EXTERIOR
    # =====================
    elif view_key == "exterior":

        atap = pred["atap"] or "none"
        dinding = pred["dinding"] or "none"

        return f"single_ext|{atap}|{dinding}"

    # =====================
    # SINGLE INTERIOR
    # =====================
    else:

        dinding = pred["dinding"] or "none"
        lantai = pred["lantai"] or "none"

        return f"single_int|{dinding}|{lantai}"


# =========================
# GROUP BY STRATIFICATION KEY
# =========================
strata = defaultdict(list)

for group in groups:
    key = build_key(group)
    strata[key].append(group)

# =========================
# SPLIT CONTAINER
# =========================
train_groups = []
val_groups = []
test_groups = []

# =========================
# STRATIFIED SPLIT
# =========================
for key, items in strata.items():

    random.shuffle(items)

    n = len(items)

    # =====================
    # RARE CASE
    # =====================
    if n == 1:

        train_groups.extend(items)

        print(f"{key}: train=1 val=0 test=0 (rare=1)")
        continue

    elif n == 2:

        train_groups.append(items[0])
        test_groups.append(items[1])

        print(f"{key}: train=1 val=0 test=1 (rare=2)")
        continue

    # =====================
    # NORMAL SPLIT
    # =====================
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    # safety guard
    if n_val == 0:
        n_val = 1
        n_train -= 1

    if n_test == 0:
        n_test = 1
        n_train -= 1

    train_groups.extend(items[:n_train])

    val_groups.extend(
        items[n_train:n_train+n_val]
    )

    test_groups.extend(
        items[n_train+n_val:]
    )

    print(
        f"{key}: "
        f"train={n_train}, "
        f"val={n_val}, "
        f"test={n_test}"
    )

# =========================
# ASSIGN SPLIT
# =========================
def assign_split(groups, split_name):

    for group in groups:

        for house in group:
            house["split"] = split_name


assign_split(train_groups, "train")
assign_split(val_groups, "val")
assign_split(test_groups, "test")

# =========================
# VALIDATION
# =========================
# memastikan source_group_id
# tidak bocor antar split
# =========================
gid_check = defaultdict(set)

for house in houses:
    gid_check[
        house["source_group_id"]
    ].add(house["split"])

leak_found = False

for gid, splits in gid_check.items():

    if len(splits) > 1:
        leak_found = True
        print(f"[LEAK] {gid} -> {splits}")

if not leak_found:
    print("\nNo data leakage detected.")

# =========================
# SUMMARY
# =========================
summary = defaultdict(int)

for house in houses:
    summary[house["split"]] += 1

print("\n===== SUMMARY =====")
print(f"Train : {summary['train']}")
print(f"Val   : {summary['val']}")
print(f"Test  : {summary['test']}")

# =========================
# SAVE
# =========================
with open(INPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        houses,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSplit saved to metadata.")