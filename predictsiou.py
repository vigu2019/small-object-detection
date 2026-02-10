from ultralytics import YOLO

# load trained model
model = YOLO("runs/detect/train24/weights/best.pt")
# or use last.pt if you want latest
# model = YOLO("runs/detect/train_siou/weights/last.pt")

# run prediction
results = model.predict(
    source="data_final/test/images",
    conf=0.25,
    save=True
)
