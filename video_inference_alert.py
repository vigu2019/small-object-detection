import cv2
from ultralytics import YOLO
import winsound  # Windows beep

VIDEO_PATH = "video_data/raw/yt-video-1.mp4"
OUTPUT_PATH = "video_data/outputs/detected_video.mp4"
CONF_THRES = 0.25

model = YOLO("yolov8s.pt") 

cap = cv2.VideoCapture(VIDEO_PATH)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES)
    detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Object {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    if detected:
        winsound.Beep(1000, 200)  # 🔔 alert

    out.write(frame)
    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video inference + alert completed")
