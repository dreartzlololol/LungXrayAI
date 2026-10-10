import os
import shutil
import random

SOURCE_DIR = "Dataset"
OUTPUT_DIR = "Dataset_split"

classes = ["NORMAL", "PNEUMONIA"]

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

random.seed(42)

for split in SPLIT_RATIO:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in classes:
    src_dir = os.path.join(SOURCE_DIR, cls)

    if not os.path.exists(src_dir):
        print(f"ไม่เจอ {cls}")
        continue

    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT_RATIO["train"])
    val_end = train_end + int(total * SPLIT_RATIO["val"])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        for f in files:
            shutil.copy2(
                os.path.join(src_dir, f),
                os.path.join(OUTPUT_DIR, split, cls, f)
            )

    print(f"✅ {cls}: {total} images")

print("พี่จ๋า เสร็จแล้ว💦💦💦")
