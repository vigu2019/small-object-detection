import cv2
import os

video_path = "video_data/raw/yt-video-1.mp4"  
output_dir = "video_data/frames/video1"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
save_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % 10 == 0:
        frame_name = f"frame_{save_id:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        save_id += 1

    frame_id += 1

cap.release()
print(f"Frames saved: {save_id}")