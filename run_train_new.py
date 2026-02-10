import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ultralytics"))

from ultralytics import YOLO

def main():
    model = YOLO("models/yolov8_c2f_cloatt_agbifpn.yaml")

    model.train(
        data="data.yaml",
        epochs=35,                 # half run
        imgsz=640,
        batch=4,
        device=0,
        workers=4,
        name="train24_siou_recovered"
    )

if __name__ == "__main__":
    main()
