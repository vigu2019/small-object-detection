from ultralytics import YOLO
import torchvision.ops as ops
import torch

# -------- Small-object post-processing --------
def small_object_head_refinement(
    boxes,
    scores,
    classes,
    small_area_thresh=32 * 32,
    conf_boost=1.1,
    small_nms_iou=0.3,
    normal_nms_iou=0.5
):
    if boxes.numel() == 0:
        return boxes, scores, classes

    boxes = boxes.clone()
    scores = scores.clone()
    classes = classes.clone()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    small_mask = areas < small_area_thresh
    scores[small_mask] = scores[small_mask] * conf_boost

    avg_area = areas.mean()
    iou_thresh = small_nms_iou if avg_area < small_area_thresh else normal_nms_iou

    keep = ops.nms(boxes, scores, iou_thresh)
    return boxes[keep], scores[keep], classes[keep]

# -------- Prediction --------
MODEL_PATH = "runs/detect/gpu_clean_30ep/weights/last.pt"
SOURCE = "data_final/test/images"

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.05,
        save=True,
        verbose=True
    )

    for r in results:
        boxes = r.boxes.xyxy
        scores = r.boxes.conf
        classes = r.boxes.cls

        boxes, scores, classes = small_object_head_refinement(
            boxes, scores, classes
        )

        print("Final refined detections:", len(boxes))
