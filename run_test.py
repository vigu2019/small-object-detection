from ultralytics_backup import YOLO

model = YOLO("models/yolov8_c2f_cloatt.yaml")  # load custom model

model.info()  # print model summary
