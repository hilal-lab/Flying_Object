from ultralytics import YOLO

def main():
    # ---------------------------------------------------------
    # 1. LOAD MODEL (Transfer Learning)
    # ---------------------------------------------------------
    # Kita memuat 'best.pt' dari paper sebagai otak dasar.
    # Pastikan file best.pt ada di folder 'weights'
    print("Memuat Model Pre-trained dari Paper...")
    model = YOLO('weights/best.pt') 

    # ---------------------------------------------------------
    # 2. START TRAINING (Fine-Tuning)
    # ---------------------------------------------------------
    # Di sini kita melakukan Fine-Tuning dengan dataset baru.
    # Parameter augmentasi di bawah ini adalah bagian "Preprocessing"
    # untuk mensimulasikan kondisi sulit (rotasi, gelap, dll).
    
    results = model.train(
        data='data.yaml',   # Config dataset
        epochs=50,          # 50 Epoch cukup untuk fine-tuning
        imgsz=640,          # Resolusi gambar
        batch=16,           # Batch size (turunkan ke 8 jika GPU teman memori kecil)
        project='runs/train',
        name='finetune_result',
        
        # --- Preprocessing & Augmentation Pipeline ---
        degrees=15.0,      # Rotasi gambar +/- 15 derajat (simulasi terbang miring)
        mosaic=1.0,        # Mosaic augmentation (gabung 4 gambar jadi 1)
        hsv_v=0.4,         # Ubah brightness (simulasi kondisi gelap/terang)
        fliplr=0.5,        # Flip horizontal (kiri-kanan)
        scale=0.5,         # Scaling (zoom in/out)
    )
    
    print("Training Selesai! Model baru tersimpan di runs/train/finetune_result/weights/best.pt")

if __name__ == '__main__':
    main()