import os
import cv2
import uuid
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_video(file, model_type):
    print("Video inference started")

    if model_type == "enhanced":
        model_path = "models/enhanced.pt"
    else:
        model_path = "models/baseline.pt"

    print("Loading model:", model_path)
    model = YOLO(model_path)

    video_id = str(uuid.uuid4())
    input_path = os.path.join(BASE_DIR, "uploads", "videos", f"{video_id}.mp4")
    output_path = os.path.join(BASE_DIR, "outputs", "videos", f"{video_id}_out.mp4")

    with open(input_path, "wb") as f:
        f.write(file.file.read())

    print("Video saved:", input_path)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("OpenCV could not open the video")

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    if fps == 0:
        fps = 25

    print("Video properties:", width, height, fps)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise Exception("VideoWriter failed to open")

    print("Writer initialized")

    results = model(source=input_path, stream=True)

    frame_count = 0
    for r in results:
        frame = r.plot()
        writer.write(frame)
        frame_count += 1

    print("Frames processed:", frame_count)

    cap.release()
    writer.release()

    print("Video saved:", output_path)

    filename = os.path.basename(output_path)

    return {
    "video_url": f"http://127.0.0.1:8000/videos/{filename}"
    }

