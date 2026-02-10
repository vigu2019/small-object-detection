import sys
import os
from multiprocessing import freeze_support

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "ultralytics_backup"))

    from ultralytics import YOLO

    # Load weights from epoch-25 run
    model = YOLO("runs/detect/nwpu_c2fcloatt_agbifpn2/weights/last.pt")

    # Start a NEW run, extending training
    model.train(
        data="data_nwpu.yaml",
        epochs=50,          # NEW total epochs
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        amp=False,
        cache=False,
        name="nwpu_c2fcloatt_agbifpn2_continued"
    )

if __name__ == "__main__":
    freeze_support()
    main()
