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

### Ground Truth Validation (5 Test Images)

| Image | Detections | GT | TP | FP | FN | Precision | Recall | F1 | Avg IoU |
|-------|-----------|----|----|----|----|-----------|--------|-----|---------|
| t1.webp | 8 | 8 | 8 | 0 | 0 | 100% | 100% | 100% | 0.9614 |
| t2.webp | 6 | 6 | 6 | 0 | 0 | 100% | 100% | 100% | 0.9348 |
| t3.webp | 2 | 2 | 2 | 0 | 0 | 100% | 100% | 100% | 0.9704 |
| t4.webp | 1 | 1 | 1 | 0 | 0 | 100% | 100% | 100% | 0.9763 |
| t5.webp | 1 | 1 | 1 | 0 | 0 | 100% | 100% | 100% | 0.9408 |
| **TOTAL** | **18** | **18** | **18** | **0** | **0** | **100%** | **100%** | **100%** | **~0.955** |

---

## 🔄 End-to-End Workflow
