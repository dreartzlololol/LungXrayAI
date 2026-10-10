import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/densenet121_xray_3class.h5"
DATASET_DIR = "Dataset_v4/test"
CSV_PATH = "statistics_v4/evaluation/detailed_predictions.csv"
OUT_DIR = "statistics_v4/gradcam"

IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "relu"

for d in ["correct", "false_positive", "false_negative"]:
    os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

def gradcam(img_array, model, layer_name, class_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

df = pd.read_csv(CSV_PATH)

samples = []
samples += (
    df[df["is_correct"] == True]
    .sort_values("prob_score", ascending=False)
    .groupby("class_name")
    .head(1)
    .to_dict("records")
)
samples += df[df["error_type"] == "type_I"].head(2).to_dict("records")
samples += df[df["error_type"] == "type_II"].head(2).to_dict("records")

for row in samples:
    img_path = os.path.join(DATASET_DIR, row["class_name"], row["image_id"])
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    heatmap = gradcam(img_array, model, LAST_CONV_LAYER)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    if row["is_correct"]:
        folder = "correct"
    elif row["error_type"] == "type_I":
        folder = "false_positive"
    else:
        folder = "false_negative"

    out_path = os.path.join(
        OUT_DIR,
        folder,
        f"{row['class_name']}_{row['image_id'].split('.')[0]}_gradcam.png"
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
