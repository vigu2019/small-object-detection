from ultralytics_backup.utils import YAML

yaml = YAML()
cfg = yaml.load("models/yolov8_c2f_cloatt.yaml")

print(cfg["backbone"])
