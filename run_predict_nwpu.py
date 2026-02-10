import sys
import os

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "ultralytics_backup"))

    from ultralytics import YOLO

    # load trained NWPU weights
    model = YOLO("runs/detect/nwpu_c2fcloatt_agbifpn2_continued/weights/best.pt")

    # run prediction
    model.predict(
        source="Data/images/test",   # folder OR single image
        imgsz=640,
        conf=0.60,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        name="nwpu_predictions"
    )

if __name__ == "__main__":
    main()
