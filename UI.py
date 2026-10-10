import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

MODEL_PATH = "models/densenet121_xray_3class.h5"
CLASSES = ["Normal", "Tuberculosis", "Pneumonia"]

model = tf.keras.models.load_model(MODEL_PATH)

root = tk.Tk()
root.title("AI Lung X-ray")
root.geometry("720x760")

video_label = tk.Label(root)
video_label.pack()

result_label = tk.Label(root, text="Result: -", font=("Arial", 18))
result_label.pack(pady=10)

confidence_label = tk.Label(root, text="", font=("Arial", 14))
confidence_label.pack()

control_frame = tk.Frame(root)
control_frame.pack(pady=10)

tk.Label(control_frame, text="Camera").grid(row=0, column=0, padx=5)

camera_var = tk.StringVar()
camera_combo = ttk.Combobox(control_frame, textvariable=camera_var, width=10)
camera_combo.grid(row=0, column=1, padx=5)

cap = None
frame_count = 0

def scan_cameras():
    devices = []
    for i in range(6):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            devices.append(str(i))
            c.release()
    camera_combo["values"] = devices
    if devices:
        camera_combo.current(0)

def open_camera():
    global cap, frame_count
    if cap:
        cap.release()
    cap = cv2.VideoCapture(int(camera_var.get()))
    frame_count = 0

def update_frame():
    global frame_count
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display = Image.fromarray(frame_rgb).resize((560, 420))
            imgtk = ImageTk.PhotoImage(display)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            if frame_count % 24 == 0:
                img = cv2.resize(frame_rgb, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                pred = model.predict(img, verbose=0)
                idx = np.argmax(pred)
                conf = float(pred[0][idx]) * 100
                result_label.config(text=f"Result: {CLASSES[idx]}")
                confidence_label.config(text=f"Confidence: {conf:.2f}%")
    root.after(10, update_frame)

open_btn = tk.Button(control_frame, text="OPEN CAMERA", width=15, command=open_camera)
open_btn.grid(row=0, column=2, padx=10)

scan_cameras()
update_frame()
root.mainloop()

if cap:
    cap.release()
