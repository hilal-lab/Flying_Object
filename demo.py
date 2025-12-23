from ultralytics import YOLO
import cv2

# 1. Load Model Hasil Fine-Tuning
# Nanti, copy file best.pt hasil training teman ke folder ini
model = YOLO('runs/train/finetune_result/weights/best.pt') 

# 2. Lakukan Prediksi pada Video/Webcam
# source=0 untuk Webcam, atau ganti nama file video 'test_video.mp4'
results = model.predict(source='test_video.mp4', show=True, conf=0.5, save=True)

print("Demo Selesai.")