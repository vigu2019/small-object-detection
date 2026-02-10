import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(ROOT, "ultralytics_siou"))
sys.path.insert(0, ROOT)

print("USING ULTRALYTICS FROM:", sys.path[0])

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# custom layers
from custom_layers.c2f_cloatt import C2fCloAtt
from custom_layers.elam import ELAM

tasks.C2fCloAtt = C2fCloAtt
tasks.ELAM = ELAM


def main():
    print("Training with SIoU loss (ultralytics_siou)")

    model = YOLO("models/yolov8_c2f_cloatt.yaml")
    
    model.train(
        data="data.yaml",
        epochs=75,
        imgsz=512,
        batch=2,
        device=0,
        workers=1,
        amp=True,
        name="train_siou"
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
