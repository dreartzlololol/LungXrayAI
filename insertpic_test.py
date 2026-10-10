import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

IMG_SIZE = 224

# โหลดโมเดล
model = load_model("xray_pneumonia_baseline.h5")

# เปิด dialog เลือกรูป
root = tk.Tk()
root.withdraw()  # ซ่อนหน้าต่างหลัก

file_path = filedialog.askopenfilename(
    title="Select Chest X-ray Image",
    filetypes=[("Image files", "*.jpg *.png *.jpeg")]
)

if not file_path:
    print("❌ No image selected")
    exit()

# โหลดและ preprocess รูป
img = cv2.imread(file_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# ทำนาย
pred = model.predict(img)[0][0]

print("Selected file:", file_path)
print("Confidence score:", round(float(pred), 3))

if pred > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")
