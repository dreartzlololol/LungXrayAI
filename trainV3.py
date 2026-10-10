import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# =====================
# CONFIG
# =====================
DATASET_DIR = "Dataset_v3"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16        # CPU friendly
EPOCHS = 25
NUM_CLASSES = 5

MODEL_NAME = "xray_5class_mobilenetv2"
MODEL_OUT = "models"
os.makedirs(MODEL_OUT, exist_ok=True)

# =====================
# DATA GENERATORS
# =====================
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=0.05,
    brightness_range=[0.8, 1.2],
    preprocessing_function=lambda x: x + tf.random.normal(
        shape=tf.shape(x), mean=0.0, stddev=0.02
    )
)

val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_aug.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

class_names = list(train_gen.class_indices.keys())
print("Classes:", class_names)

# =====================
# CLASS WEIGHT
# =====================
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# =====================
# MODEL
# =====================
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# CALLBACKS
# =====================
ckpt = ModelCheckpoint(
    filepath=os.path.join(MODEL_OUT, MODEL_NAME + ".h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =====================
# TRAIN
# =====================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[ckpt]
)

# =====================
# SAVE FINAL MODEL
# =====================
model.save(os.path.join(MODEL_OUT, MODEL_NAME + "_final.h5"))
print("✅ Training complete & model saved")
