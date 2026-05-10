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
- 🌦️ Formation analysis using Weather API (rainfall) & Traffic API
