import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Flying Object Detection Demo",
    page_icon="✈️",
    layout="wide"
)

# --- JUDUL & SIDEBAR ---
st.title("✈️ Demo Deteksi Objek Terbang (YOLOv11)")
st.markdown("""
Aplikasi ini mendeteksi **Drone, Burung, Pesawat, dan Balon Udara** menggunakan model hasil fine-tuning.
""")

st.sidebar.header("Pengaturan Model")

# 1. Load Model (Otomatis mencari path Anda)
# Ganti path ini jika lokasi file berubah
default_model_path = 'runs/train/finetune_result4/weights/best.pt'

# Cek apakah model ada
if not os.path.exists(default_model_path):
    st.sidebar.error(f"File model tidak ditemukan di: {default_model_path}")
    st.sidebar.info("Pastikan Anda sudah 'git pull' data dari GitHub.")
    model = None
else:
    try:
        model = YOLO(default_model_path)
        st.sidebar.success("Model Berhasil Dimuat! ✅")
    except Exception as e:
        st.sidebar.error(f"Error memuat model: {e}")
        model = None

# Slider Confidence
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)

# --- PILIHAN MODE ---
mode = st.radio("Pilih Mode Demo:", ["Upload Gambar", "Upload Video", "Webcam Real-time"], horizontal=True)
st.divider()

if model is None:
    st.warning("Silakan perbaiki path model terlebih dahulu.")
    st.stop()

# ==========================================
# MODE 1: GAMBAR
# ==========================================
if mode == "Upload Gambar":
    uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Tampilkan Gambar Asli
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='Gambar Asli', use_column_width=True)
        
        # Tombol Prediksi
        if st.button("Deteksi Objek"):
            # Lakukan prediksi
            results = model.predict(image, conf=conf_threshold)
            
            # Ambil hasil plot (array numpy BGR)
            res_plotted = results[0].plot()
            
            # Convert BGR ke RGB agar warna benar di Streamlit
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(res_rgb, caption='Hasil Deteksi', use_column_width=True)
                
                # Tampilkan detail teks
                st.success(f"Terdeteksi {len(results[0].boxes)} objek.")

# ==========================================
# MODE 2: VIDEO
# ==========================================
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Pilih video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Simpan video ke file sementara (karena OpenCV butuh path file)
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder untuk video player
        
        stop_button = st.button("Stop Video")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prediksi
            results = model.predict(frame, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            # Convert ke RGB
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Tampilkan di Streamlit
            st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
            
        cap.release()

# ==========================================
# MODE 3: WEBCAM
# ==========================================
elif mode == "Webcam Real-time":
    st.write("Tekan tombol **Start** untuk memulai webcam laptop Anda.")
    run = st.checkbox('Nyalakan Webcam')
    st_frame = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0) # 0 adalah ID webcam default laptop
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca webcam.")
                break
            
            # Prediksi
            results = model.predict(frame, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            # Convert ke RGB
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
        
        cap.release()