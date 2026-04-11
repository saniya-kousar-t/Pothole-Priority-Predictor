# Pothole-Priority-Predictor
# 🚧 Pothole Detection using YOLOv11

A deep learning model to detect potholes in images using YOLOv11.

## 📊 Model Performance
| Metric | Score |
|---|---|
| mAP50 | 0.883 |
| mAP50-95 | 0.658 |
| Precision | 0.861 |
| Recall | 0.810 |

## 📁 Dataset
Dataset from Roboflow Universe:
https://universe.roboflow.com/major-vl1h9/pothole-bwzav/dataset/2

## 🛠️ Requirements
pip install ultralytics roboflow

## 🚀 How to Run
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="your_image.jpg", conf=0.25, save=True)

## 🏋️ Training
- Model: YOLOv11n
- Epochs: 50
- Image Size: 640
- Batch Size: 16
