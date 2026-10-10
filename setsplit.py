import os
import shutil
import random
from math import floor

SRC_DIR = "Dataset_v3_all"
DST_DIR = "Dataset_v3"

SPLIT_RATIO = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

IMG_EXT = (".png", ".jpg", ".jpeg")
random.seed(42)

def ensure_dirs(classes):
    for split in SPLIT_RATIO:
        for c in classes:
            os.makedirs(os.path.join(DST_DIR, split, c), exist_ok=True)

def split_class(class_name):
    src_class_dir = os.path.join(SRC_DIR, class_name)
    images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(IMG_EXT)]
    random.shuffle(images)

    total = len(images)
    n_train = floor(total * SPLIT_RATIO["train"])
    n_val = floor(total * SPLIT_RATIO["val"])
    n_test = total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        for f in files:
            src = os.path.join(src_class_dir, f)
            dst = os.path.join(DST_DIR, split, class_name, f)
            shutil.copy2(src, dst)

    return n_train, n_val, n_test

if __name__ == "__main__":
    classes = os.listdir(SRC_DIR)
    ensure_dirs(classes)

    print("📦 Splitting Dataset_v3_all → Dataset_v3\n")

    for c in classes:
        n_train, n_val, n_test = split_class(c)
        print(f"{c}: train={n_train}, val={n_val}, test={n_test}")

    print("\n✅ Phase 1.5 Complete (train / val / test ready)")
