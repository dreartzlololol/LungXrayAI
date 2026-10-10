import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

DATASET_DIR = "Dataset_v4"
TRAIN_DIR = f"{DATASET_DIR}/train"
VAL_DIR = f"{DATASET_DIR}/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3

MODEL_NAME = "densenet121_xray_3class"
MODEL_OUT = f"models/{MODEL_NAME}.h5"
STAT_DIR = "statistics_v4/learning_curve"

os.makedirs("models", exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

# 1. ImageDataGenerator with improved data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.85, 1.15],
    horizontal_flip=True
).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# 2. Compute Class Weights for Imbalanced Dataset
class_labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights_dict = dict(enumerate(class_weights))
print(f"[INFO] Dynamic class weights calculated: {class_weights_dict}")

# 3. Load Pretrained DenseNet121
base_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Add Robust Classification Head with Regularization
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# 4. Stage 1: Train Classification Head Only (Warm-up)
print("\n[START] Stage 1: Training classifier head (5 epochs)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Optimized steps_per_epoch and validation_steps to speed up CPU training
history_stage1 = model.fit(
    train_gen,
    steps_per_epoch=180,
    validation_data=val_gen,
    validation_steps=60,
    epochs=5,
    class_weight=class_weights_dict,
    verbose=1
)

# 5. Stage 2: Fine-Tuning the Whole Model
print("\n[START] Stage 2: Fine-tuning top layers of base model (7 epochs)...")
base_model.trainable = True

# Freeze all layers of the base model except the last 30 layers to speed up CPU training
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with small learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

history_stage2 = model.fit(
    train_gen,
    steps_per_epoch=180,
    validation_data=val_gen,
    validation_steps=60,
    epochs=7,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# 6. Save Model and Combine Training Histories
model.save(MODEL_OUT)
print("\n[SUCCESS] Model saved to:", MODEL_OUT)

combined_history = {}
for key in history_stage1.history.keys():
    combined_history[key] = history_stage1.history[key] + history_stage2.history.get(key, [])

df = pd.DataFrame(combined_history)
df["epoch"] = range(1, len(df) + 1)
df["model"] = MODEL_NAME
df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

history_csv = f"{STAT_DIR}/training_history_log.csv"
df.to_csv(history_csv, index=False)
print("[SUCCESS] Unified training history log saved to:", history_csv)
