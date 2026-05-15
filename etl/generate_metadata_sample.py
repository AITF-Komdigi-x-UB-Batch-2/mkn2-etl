import json
import random

# =========================
# CONFIG
# =========================
INPUT_JSON = "data/metadata/mkn_house_metadata.json"
OUTPUT_JSON = "data/metadata/mkn_house_metadata_sample.json"

N_SAMPLE = 5
SEED = 42

random.seed(SEED)

# =========================
# LOAD
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    houses = json.load(f)

# =========================
# GROUP BY SPLIT
# =========================
train_data = [
    x for x in houses
    if x["split"] == "train"
]

val_data = [
    x for x in houses
    if x["split"] == "val"
]

test_data = [
    x for x in houses
    if x["split"] == "test"
]

# =========================
# RANDOM SAMPLE
# =========================
train_sample = random.sample(
    train_data,
    min(N_SAMPLE, len(train_data))
)

val_sample = random.sample(
    val_data,
    min(N_SAMPLE, len(val_data))
)

test_sample = random.sample(
    test_data,
    min(N_SAMPLE, len(test_data))
)

# =========================
# COMBINE
# =========================
final_samples = (
    train_sample +
    val_sample +
    test_sample
)

# =========================
# SAVE JSON
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        final_samples,
        f,
        ensure_ascii=False,
        indent=2
    )

# =========================
# SUMMARY
# =========================
print("✅ Sample metadata saved.")
print(f"📄 Output : {OUTPUT_JSON}")
print(f"📦 Total  : {len(final_samples)} samples")
print(f"   - Train : {len(train_sample)}")
print(f"   - Val   : {len(val_sample)}")
print(f"   - Test  : {len(test_sample)}")