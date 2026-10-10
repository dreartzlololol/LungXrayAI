import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# =====================
# CONFIG
# =====================
DATASET_DIR = "Dataset_v3"
OUT_DIR = "statistics_v3/incremental_learning"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
PERCENTS = [20, 40, 60, 80, 100]

random.seed(42)
tf.random.set_seed(42)

# =====================
# DATA GENERATOR
# =====================
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=0.05,
    brightness_range=[0.8, 1.2]
)

val_aug = ImageDataGenerator(rescale=1./255)

# =====================
# BUILD MODEL
# =====================
def build_model(num_classes):
    base = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =====================
# MAIN LOOP
# =====================
results = []

for pct in PERCENTS:
    print(f"\n🔥 Training with {pct}% of training data")

    train_gen_full = train_aug.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_aug.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    total_samples = train_gen_full.samples
    use_samples = int(total_samples * pct / 100)

    indices = np.random.permutation(total_samples)[:use_samples]
    train_gen_full.index_array = indices
    train_gen_full.samples = use_samples

    labels = train_gen_full.classes[indices]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(train_gen_full.num_classes)

    history = model.fit(
        train_gen_full,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1
    )

    results.append({
        "train_size_percent": pct,
        "train_accuracy": history.history["accuracy"][-1],
        "val_accuracy": history.history["val_accuracy"][-1]
    })

# =====================
# SAVE CSV
# =====================
df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "incremental_results.csv")
df.to_csv(csv_path, index=False)

# =====================
# PLOT CURVE
# =====================
plt.figure(figsize=(8, 6))
plt.plot(df["train_size_percent"], df["train_accuracy"], marker="o", label="Train Accuracy")
plt.plot(df["train_size_percent"], df["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Accuracy")
plt.title("Incremental Learning Curve (MobileNetV2, Epoch=8)")
plt.legend()
plt.grid(True)

png_path = os.path.join(OUT_DIR, "incremental_learning_curve.png")
plt.savefig(png_path, dpi=300)
plt.close()

print("\n✅ Incremental learning analysis complete")
print(f"Saved: {csv_path}")
print(f"Saved: {png_path}")
