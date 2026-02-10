from ultralytics import YOLO

MODEL_PATH = "runs/detect/train24/weights/best.pt"
VIDEO_PATH = "traffic.mp4"

model = YOLO(MODEL_PATH)

model.predict(
    source=VIDEO_PATH,
    conf=0.5,
    iou=0.6,
    save=True,
    stream=True,
    show=False,
    visualize=False   # <<< ADD THIS
)
