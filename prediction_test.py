import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("xray_pneumonia_baseline.h5")

img = cv2.imread("sample_images/lung.jpg")
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

print("Score:", pred)

if pred > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")