## 🚀 Project Objective

Small objects in aerial and drone imagery are difficult to detect due to:

* Low resolution
* Dense clustering
* Scale variation
* Background noise

This project improves detection performance by enhancing YOLOv8 with:

* Custom attention modules
* Improved feature fusion
* Decoupled detection head
* Loss function refinements

---

## 🧠 Models Used

### 1️⃣ Baseline Model

* Standard YOLOv8
* Default loss functions
* Standard backbone and neck

### 2️⃣ Enhanced Model

* C2fCloAtt blocks (Channel + Local + Spatial Attention)
* ELAM attention mechanism
* AGBiFPN feature fusion
* Optional SIoU loss integration
* Decoupled detection head for better localization

---

## 📊 Dataset

Primary dataset used:

* **VisDrone Dataset**

  * 10 object classes
  * Focused on aerial and drone-based imagery
  * Contains small and densely packed objects

Classes:

```
pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor
```

---

## 🏗️ Project Structure

```
small-object-detection/
│
├── models/                  # Model YAMLs and weights
├── train/                   # Training scripts
├── custom_layers/           # Attention modules (C2fCloAtt, ELAM)
├── data.yaml                # Dataset configuration
├── runs/                    # Training outputs
├── app.py                   # FastAPI backend
├── video.py                 # Video inference module
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/vigu2019/small-object-detection.git
cd small-object-detection

python -m venv yolovenv
yolovenv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 🏋️ Training

### Baseline Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

### Enhanced Training

```python
from ultralytics import YOLO

model = YOLO("models/yolov8_c2f_cloatt_agbifpn.yaml")
model.train(data="data.yaml", epochs=100, imgsz=640)
```

---

## 📈 Evaluation

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml
```

Metrics analyzed:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* Confusion Matrix
* PR Curves

---

## 🎥 Inference

### Image Inference

```bash
yolo detect predict model=best.pt source=test.jpg
```

### Video Inference (FastAPI Backend)

Start backend:

```bash
uvicorn app:app --reload
```

Endpoints:

* `/predict` → Image detection
* `/predict/video` → Video detection

---

## 🌍 Web Application Integration

The detection system is integrated with:

* FastAPI backend
* Flutter Web frontend
* Model selection (Baseline / Enhanced)
* Real-time detection visualization
* Object count statistics

---

## 📌 Key Contributions

✔ Improved detection of small and dense objects
✔ Enhanced feature extraction using attention modules
✔ Better bounding box localization with SIoU
✔ Real-time inference for image and video
✔ Comparative evaluation against baseline

---

## 📊 Applications

* Smart Traffic Monitoring
* Drone Surveillance
* Urban Density Analysis
* Disaster Response Monitoring
* Small Target Detection in Aerial Imaging

---
## Authors

Vignesh Murali
Shivani Krishnan
Neha A R
Poorvaja M Sooraj


## 📜 Future Work

* Crack detection using specialized datasets (e.g., SDNET)
* Model quantization for edge deployment
* Explainability integration (Grad-CAM)
* Deployment on cloud platforms
