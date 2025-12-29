from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("models/yolov8_c2f_cloatt_agbifpn_socs.yaml")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,          # reduce if VRAM issues
        device=0,         # GPU
        workers=2,        # IMPORTANT on Windows
        pretrained=False
    )

if __name__ == "__main__":
    freeze_support()
    main()
