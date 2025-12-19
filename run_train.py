import sys
import os
from multiprocessing import freeze_support

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "ultralytics_backup"))

    from ultralytics import YOLO

    model = YOLO("models/yolov8_c2f_cloatt_agbifpn.yaml")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,        # keeps temps safe
        device=0,       # GPU
        workers=0,      # Windows-safe
        amp=False,      # stability > speed
        cache=False
    )

if __name__ == "__main__":
    freeze_support()
    main()
