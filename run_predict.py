from ultralytics import YOLO
import torch
import torchvision.ops as ops


def small_object_head_refinement(
    boxes,
    scores,
    classes,
    small_area_thresh=32 * 32,
    conf_boost=1.1,
    small_nms_iou=0.3,
    normal_nms_iou=0.5
):
    """
    Small-object–aware post-processing head
    """

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    # 1️⃣ Small-object confidence reweighting
    small_mask = areas < small_area_thresh
    scores[small_mask] *= conf_boost

    # 2️⃣ Small-object–aware NMS
    avg_area = areas.mean()
    iou_thresh = small_nms_iou if avg_area < small_area_thresh else normal_nms_iou

    keep = ops.nms(boxes, scores, iou_thresh)

    return boxes[keep], scores[keep], classes[keep]


MODEL_PATH = "runs/detect/train18/weights/best.pt"
SOURCE = "data_final/test/images"

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.25,
        save=False   # IMPORTANT: disable default saving
    )

    for r in results:
        boxes = r.boxes.xyxy
        scores = r.boxes.conf
        classes = r.boxes.cls

        # 🔥 Apply proposed head-level enhancement
        boxes, scores, classes = small_object_head_refinement(
            boxes=boxes,
            scores=scores,
            classes=classes
        )

        # 🔹 (Optional) Print to verify
        print(f"Final detections: {len(boxes)}")

        # 🔹 You can now save / visualize refined results if needed
