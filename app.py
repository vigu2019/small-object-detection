from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uuid
import os
import shutil
import video

# -------------------------------------------------
# Base paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "videos")

# Create folders if not exist (important for Render)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve outputs folder
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    return video.run_video(file, model_type)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    uid = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_DIR, f"{uid}.jpg")

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model_path = "models/baseline.pt" if model_type == "baseline" else "models/enhanced.pt"
    model = YOLO(model_path)

    results = model(img_path, conf=0.25)
    result = results[0]

    out_path = os.path.join(OUTPUT_DIR, f"{uid}_pred.jpg")
    result.save(out_path)

    counts = {}
    for c in result.boxes.cls.tolist():
        name = result.names[int(c)]
        counts[name] = counts.get(name, 0) + 1

    return {
        "image_url": f"/outputs/{uid}_pred.jpg",
        "total": sum(counts.values()),
        "counts": counts
    }
