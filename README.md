# Pothole-Priority-Predictor

# Pothole Detection using YOLOv11

An AI-powered system that detects potholes from images/videos and provides severity analysis, repair suggestions, and priority scoring using a YOLOv11 model.

## Features
- Image & video input  
- YOLOv11 pothole detection  
- Severity scoring (0–5)  
- Priority score (0–100)  
- Repair cost estimation  
- RICE model for prioritization  
- Interactive dashboard


## Model Performance
| Metric        | Score |
|--------------|------|
| mAP50        | 0.883 |
| mAP50-95     | 0.658 |
| Precision    | 0.861 |
| Recall       | 0.810 |


## Training
- Model: YOLOv11n
- Epochs: 50
- Image Size: 640
- Batch Size: 16

## Dataset
Dataset from Roboflow Universe:
https://universe.roboflow.com/major-vl1h9/pothole-bwzav/dataset/2

## Tech Stack
- **Backend:** Flask, YOLOv11, OpenCV  
- **Frontend:** HTML, CSS, JavaScript 

## API
POST /predict
Upload an image/video → Returns:
- detections
- severity
- priority score
- repair estimate

## How to Run Model
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="your_image.jpg", conf=0.25, save=True)
