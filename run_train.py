from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

from custom_layers.c2f_cloatt import C2fCloAtt
from custom_layers.elam import ELAM

tasks.C2fCloAtt = C2fCloAtt
tasks.ELAM = ELAM

def main():
    print("RUN_TRAIN.PY EXECUTED")
    print("C2fCloAtt registered:", "C2fCloAtt" in tasks.__dict__)
    print("ELAM registered:", "ELAM" in tasks.__dict__)

    model = YOLO("runs/detect/train_siou2/weights/last.pt")

    model.train(
        data="data.yaml",
        epochs=100,        
        imgsz=512,
        batch=2,
        device=0,
        workers=1,
        close_mosaic=5,
        amp=True,
        name="train_siou2_ext"  
    )

if __name__ == "__main__":
    main()
