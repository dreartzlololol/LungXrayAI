import cv2
import os
import time

os.makedirs("captured", exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found")
    exit()

time.sleep(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Capture failed")
    exit()

cv2.imwrite("captured/capture.jpg", frame)
print("✅ Saved captured/capture.jpg")
