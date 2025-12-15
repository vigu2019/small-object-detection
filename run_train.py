from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolov8_c2f_cloatt.yaml")

    model.train(
        data="data.yaml",
        epochs=50,            # same as train-9
        imgsz=640,
        batch=4,              # use the same value as train-9
        task="detect",
        name="train_cloatt_elam_50ep"
    )
