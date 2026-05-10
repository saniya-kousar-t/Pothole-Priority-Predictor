# 🚧 Pothole Priority Predictor
### AI-Driven Road Maintenance Decision Support System

An end-to-end AI-powered system that detects potholes from road images,
assesses severity, predicts deterioration, estimates repair costs, and
prioritizes maintenance using YOLOv11 and Machine Learning — built for
Indian municipal road authorities (BBMP, Bengaluru).

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![mAP50](https://img.shields.io/badge/mAP50-88.52%25-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌟 Features

- 📸 Image & video input (Street View / CCTV)
- 🔍 YOLOv11-seg pothole detection with bounding boxes & masks
- 📊 Severity scoring (1–5) using MiDaS DPT Hybrid depth estimation
- 🏆 Priority score (0–100) using RICE model
- 💰 Repair cost estimation (INR) via IRC:SP:72 & Karnataka PWD SOR 2024
- 🌦️ Formation analysis using Weather API (rainfall) & Traffic API (load)
- 📈 Deterioration prediction at 30, 60, 90 days using Random Forest
- 🗺️ Geospatial dashboard with real map (Leaflet.js + Folium)
- 🤖 Kannada chatbot for road safety queries
- 📄 PDF report download for PWD authorities
- 🌐 Bilingual support (English + Kannada)

---

## 📊 Model Performance

### Detection Model (YOLOv11-seg)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| mAP50 | 88.52% | Detects 88.5% of potholes correctly |
| mAP50-95 | 69.86% | Strong across all IoU thresholds |
| Precision | 86.79% | 86.8% of detections are real potholes |
| Recall | 81.38% | Finds 81.4% of all actual potholes |
| F1 Score | 84.00% | Balanced precision-recall performance |
| Status | ✅ Production Ready | mAP50 ≥ 85% threshold met |

### Segmentation Model Results

| Metric | Value |
|--------|-------|
| mAP50 (Box) | 88.58% |
| mAP50-95 (Box) | 66.97% |
| Precision (Box) | 85.87% |
| Recall (Box) | 81.18% |
| mAP50 (Mask) | 88.34% |
| mAP50-95 (Mask) | 62.23% |
| Precision (Mask) | 86.34% |
| Recall (Mask) | 81.39% |

---

## 🔄 End-to-End Workflow
Image Input (Street View / CCTV)
↓
Preprocessing (CLAHE + Gaussian)
↓
Detection (YOLOv11-seg)
↓
Severity Assessment (MiDaS DPT Hybrid) → Score 1-5
↓
Formation Analysis (Weather API + Traffic API)
↓
Deterioration Prediction (Random Forest) → 30/60/90 days
↓
Priority Ranking (Severity + Traffic + Formation + Deterioration)
↓
Repair & Cost Calculation (IRC Guidelines + Karnataka PWD SOR 2024)
↓
Geospatial Dashboard (Flask + Leaflet.js + Folium)

---

## 🧮 Key Formulas

### Severity Score
severity_score = 0.4 × area_norm
+ 0.35 × depth_term
+ 0.15 × (road_std / 255)
+ 0.10 × (depth_p95 / 100)

### Priority Score
priority_score = (Severity/5 × 60)
+ (Traffic/10 × 25)
+ (Risk/10 × 15)
+ urgency_bonus

### Formation Risk
formation_risk = 0.35 × rainfall
+ 0.30 × traffic
+ 0.20 × temperature
+ 0.15 × road_age

---

## 🗂️ Dataset

- **Source:** Roboflow Universe
- **Link:** https://universe.roboflow.com/major-vl1h9/pothole-bwzav/dataset/2
- **Format:** YOLOv11 (PyTorch TXT)
- **Task:** Object Detection + Segmentation
- **Classes:** 1 (pothole)

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 6,819 | 70% |
| Validation | 1,945 | 20% |
| Test | 974 | 10% |
| **Total** | **9,738** | **100%** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv11-seg (Ultralytics) |
| Depth Estimation | MiDaS DPT Hybrid |
| Deterioration | Random Forest Regressor |
| Backend | Flask, Python, OpenCV |
| Frontend | HTML, CSS, JavaScript |
| Maps | Leaflet.js, Folium |
| Charts | Chart.js |
| External APIs | Weather API, Traffic API |
| Training | Google Colab (GPU) |
| Dataset | Roboflow Universe |
| Version Control | GitHub |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/saniya-kousar-t/Pothole-Priority-Predictor.git
cd Pothole-Priority-Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask App
```bash
python app.py
```

### 4. Open in Browser
http://127.0.0.1:5000

### 5. Run Model Directly
```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(
    source="your_image.jpg",
    conf=0.25,
    save=True
)
```

---

## 📡 API Reference

### POST /predict
Upload a road image → Returns complete analysis

**Response:**
```json
{
  "potholes_detected": 3,
  "severity": 4,
  "area_sqm": 0.545,
  "depth_score": 0.876,
  "formation": {
    "risk_score": 3.05,
    "risk_level": "LOW",
    "dominant_factor": "traffic_loading"
  },
  "deterioration": {
    "sev_30d": 4.02,
    "sev_60d": 4.04,
    "sev_90d": 4.07,
    "urgency": "HIGH",
    "days_to_critical": 0
  },
  "priority": {
    "score": 39.8,
    "label": "LOW",
    "rice_component": 19.8,
    "formation_component": 4.6,
    "deterioration_bonus": 15.35
  },
  "repair_cost_inr": 1768.2
}
```

---

## 📁 Project Structure
Pothole-Priority-Predictor/
│
├── app.py                  # Flask backend
├── best.pt                 # Trained YOLOv11 model
├── requirements.txt        # Dependencies
│
├── models/
│   ├── yolo_model.py       # YOLOv11 detection
│   ├── severity.py         # MiDaS depth estimation
│   ├── deterioration.py    # Random Forest prediction
│   └── cost_estimator.py   # IRC cost calculation
│
├── static/
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript files
│   └── images/             # Static assets
│
├── templates/
│   └── index.html          # Main dashboard
│
├── notebooks/
│   ├── testandvalidation1.ipynb  # CPU validation
│   ├── testandvalidation2.ipynb  # GPU + DAV2
│   └── testandvalidation3.ipynb  # Roboflow training
│
└── README.md

---

## 🎯 Target Users

- BBMP (Bruhat Bengaluru Mahanagara Palike)
- Karnataka PWD Authorities
- Municipal Road Maintenance Teams
- Smart City Infrastructure Planners

---

## 📜 Standards Used

- IRC:SP:72 — Road Repair Guidelines
- Karnataka PWD SOR 2024 — Cost Rates
- COCO Evaluation — Model Metrics

---

## 👥 Team

- Developed as part of DSML Internship Cohort 12
- Organization: IIMSTC, Bengaluru
- Academic Year: 2025-26

---

## 📄 License

MIT License — Free to use with attribution
