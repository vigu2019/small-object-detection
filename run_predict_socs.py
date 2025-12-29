from ultralytics import YOLO

MODEL = "runs/detect/train24/weights/best.pt"
SOURCE = "data_final/test/images"

model = YOLO(MODEL)
model.predict(
    source=SOURCE,
    imgsz=640,
    conf=0.25,
    device=0,   # GPU
    save=True
)
