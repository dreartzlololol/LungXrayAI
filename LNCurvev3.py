import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

CSV_PATH = "statistics_v3/learning_curve/training_history_log.csv"
OUT_DIR = "statistics_v3/learning_curve"
OUT_IMG = "training_learning_curves.png"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

epochs = df["epoch"]
train_acc = df["accuracy"]
val_acc = df["val_accuracy"]
train_loss = df["loss"]
val_loss = df["val_loss"]

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = "MobileNetV2_Xray_5Class"

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

plt.suptitle(f"{model_name}\nTraining Run: {timestamp}", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig(os.path.join(OUT_DIR, OUT_IMG), dpi=300, bbox_inches="tight")
plt.close()

print("✅ Learning curve generated")
print("✅ Saved:", os.path.join(OUT_DIR, OUT_IMG))
