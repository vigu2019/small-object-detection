from ultralytics import YOLO

MODEL_PATH = "runs/detect/train_cloatt_elam_50ep/weights/best.pt"
SOURCE = "data_final/test/images"   # ← IMPORTANT FIX

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.25,
        save=True
    )
