from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# Register custom layers
from custom_layers.c2f_cloatt import C2fCloAtt
from custom_layers.elam import ELAM

tasks.C2fCloAtt = C2fCloAtt
tasks.ELAM = ELAM


def main():
    model = YOLO("runs/detect/train_siou2_ext/weights/best.pt")

    model.predict(
        source="data_final/test/images",
        imgsz=512,
        conf=0.25,
        device=0,
        save=True,
        show=False
    )

    print("Prediction done")


if __name__ == "__main__":
    main()
