import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = "Dataset_v4"
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_PATH = "models/densenet121_xray_3class.h5"

STAT_DIR = "statistics_v4"
EVAL_DIR = os.path.join(STAT_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)

test_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_indices = test_gen.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

pred_probs = model.predict(test_gen, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_gen.classes

report = classification_report(
    true_labels,
    pred_labels,
    target_names=[inv_class_indices[i] for i in range(len(inv_class_indices))],
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(EVAL_DIR, "evaluation_summary_report.csv")
)

rows = []
filenames = test_gen.filenames

for i, fname in enumerate(filenames):
    true_id = true_labels[i]
    pred_id = pred_labels[i]

    true_class = inv_class_indices[true_id]
    pred_class = inv_class_indices[pred_id]

    prob_score = float(pred_probs[i][pred_id])
    is_correct = true_id == pred_id

    if is_correct:
        error_type = "none"
    else:
        if true_class == "normal" and pred_class != "normal":
            error_type = "type_I"
        elif true_class != "normal" and pred_class == "normal":
            error_type = "type_II"
        else:
            error_type = "other"

    rows.append({
        "image_id": os.path.basename(fname),
        "true_class": true_class,
        "predicted_class": pred_class,
        "class_name": true_class,
        "prob_score": prob_score,
        "is_correct": is_correct,
        "error_type": error_type
    })

df = pd.DataFrame(rows)
df.to_csv(
    os.path.join(EVAL_DIR, "detailed_predictions.csv"),
    index=False
)

print("[SUCCESS] Evaluation complete")
print("[SUCCESS] Saved to:", EVAL_DIR)
