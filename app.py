from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uuid, os, shutil
import video

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/videos",
    StaticFiles(directory=os.path.join(BASE_DIR, "outputs", "videos")),
    name="videos",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=False,    
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

baseline_model = YOLO("models/baseline.pt")
enhanced_model = YOLO("models/enhanced.pt")

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

    model = baseline_model if model_type == "baseline" else enhanced_model

    results = model(img_path, conf=0.25)
    results = model(img_path, conf=0.25, save=False)
    result = results[0]

    out_path = os.path.join(OUTPUT_DIR, f"{uid}_pred.jpg")
    result.save(out_path)

    counts = {}
    for c in result.boxes.cls.tolist():
        name = result.names[int(c)]
        counts[name] = counts.get(name, 0) + 1

    return {
        "image_url": f"http://localhost:8000/outputs/{uid}_pred.jpg",
        "total": sum(counts.values()),
        "counts": counts
    }

