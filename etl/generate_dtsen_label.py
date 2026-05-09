import json
import random

# =========================
# PATH
# =========================
INPUT_JSON = "data/metadata/mkn_house_metadata.json"
OUTPUT_JSON = "data/metadata/mkn_house_metadata.json"

SEED = 42
random.seed(SEED)

# =========================
# LABEL LIST
# =========================
ATAP_LABELS = [
    "beton",
    "genteng",
    "seng",
    "asbes",
    "bambu",
    "kayu/sirap",
    "jerami/ijuk/daun_daunan/rumbia",
    "lainnya"
]

DINDING_LABELS = [
    "tembok",
    "plesteran_anyaman_bambu/plesteran_anyaman_kawat",
    "kayu/papan/gypsum/GRC/calciboar",
    "anyaman_bambu",
    "batang_kayu",
    "bambu",
    "lainnya"
]

LANTAI_LABELS = [
    "marmer/granit",
    "keramik",
    "parket/vinil/karpet",
    "ubin/tegel/teraso",
    "kayu/papan",
    "semen/bata_merah",
    "bambu",
    "tanah",
    "lainnya"
]

# =========================
# LOAD
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    houses = json.load(f)

# =========================
# HELPER
# =========================
def random_different(labels, current_value):
    """
    ambil random label yang berbeda
    dari value sekarang
    """
    candidates = [x for x in labels if x != current_value]

    if not candidates:
        return current_value

    return random.choice(candidates)


# =========================
# SHUFFLE HOUSE INDEX
# =========================
indices = list(range(len(houses)))
random.shuffle(indices)

# 50% match
match_count = len(houses) // 2

match_indices = set(indices[:match_count])

# =========================
# FILL DTSEN
# =========================
for idx, house in enumerate(houses):

    pred = house["prediksi"]

    # =====================
    # MATCH
    # =====================
    if idx in match_indices:

        house["match"] = True

        house["dtsen"] = {
            "atap": pred["atap"],
            "dinding": pred["dinding"],
            "lantai": pred["lantai"]
        }

    # =====================
    # NOT MATCH
    # =====================
    else:

        house["match"] = False

        images = house["images"]

        # cek tipe image
        has_exterior = any(
            img["view_type"] == "exterior"
            for img in images
        )

        has_interior = any(
            img["view_type"] == "interior"
            for img in images
        )

        dtsen = {
            "atap": None,
            "dinding": None,
            "lantai": None
        }

        # =====================================
        # EXTERIOR COMPONENT
        # =====================================
        if has_exterior:

            dtsen["atap"] = random_different(
                ATAP_LABELS,
                pred["atap"]
            )

            dtsen["dinding"] = random_different(
                DINDING_LABELS,
                pred["dinding"]
            )

        # =====================================
        # INTERIOR COMPONENT
        # =====================================
        if has_interior:

            # kalau single interior only
            # dinding bisa overwrite exterior
            dtsen["dinding"] = random_different(
                DINDING_LABELS,
                pred["dinding"]
            )

            dtsen["lantai"] = random_different(
                LANTAI_LABELS,
                pred["lantai"]
            )

        house["dtsen"] = dtsen

# =========================
# SAVE
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        houses,
        f,
        ensure_ascii=False,
        indent=2
    )

# =========================
# SUMMARY
# =========================
total_match = sum(
    1 for x in houses
    if x["match"] is True
)

total_not_match = sum(
    1 for x in houses
    if x["match"] is False
)

print(f"✅ Total houses : {len(houses)}")
print(f"✅ Match        : {total_match}")
print(f"✅ Not Match    : {total_not_match}")
print("✅ DTSEN generation selesai.")