# YOLOv11 Detection API — Integration Guide

> **Audience:** Team 2 (Frontend / Integration)  
> **Base URL (local dev):** `http://localhost:5000`  
> **Base URL (production):** *(set by DevOps — replace throughout)*

---

## Table of Contents
1. [Quick Start](#1-quick-start)
2. [Authentication](#2-authentication)
3. [Endpoints](#3-endpoints)
   - [GET /](#get-)
   - [GET /health](#get-health)
   - [POST /predict](#post-predict)
4. [Request Reference](#4-request-reference)
5. [Response Reference](#5-response-reference)
6. [Error Codes](#6-error-codes)
7. [curl Examples](#7-curl-examples)
8. [Postman Setup](#8-postman-setup)
9. [JavaScript / Python Integration Snippets](#9-integration-snippets)
10. [Limits & Notes](#10-limits--notes)

---

## 1. Quick Start

```bash
# 1. Clone / unzip the project
cd flask_yolo_api

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model weights
cp /path/to/best.pt .

# 5. Run the development server
python app.py
# → Listening on http://0.0.0.0:5000

# 6. Run in production (gunicorn)
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

> **GPU inference:** Install the CUDA-enabled PyTorch wheel **before**
> `ultralytics`. See https://pytorch.org/get-started/locally/

---

## 2. Authentication

The current version has **no authentication**. If deployed publicly, place it
behind a reverse proxy (nginx) with an API key header or OAuth token. Team 2
should pass an `X-API-Key` header once auth is wired up.

---

## 3. Endpoints

### GET /

Basic liveness check.

**Response `200`**
```json
{
  "status": "ok",
  "model": "best.pt",
  "endpoints": [
    "GET  /",
    "GET  /health",
    "POST /predict"
  ]
}
```

---

### GET /health

Verifies the model weights are loaded and inference is ready.

| Status | Meaning |
|--------|---------|
| `200`  | Model loaded — API is ready |
| `503`  | Model failed to load — not ready |

**Response `200`**
```json
{
  "status": "ok",
  "model_path": "best.pt",
  "model": "loaded"
}
```

**Response `503`**
```json
{
  "status": "degraded",
  "model_path": "best.pt",
  "model": "Model file not found: best.pt"
}
```

---

### POST /predict

Run object detection on an uploaded image.

| Property | Value |
|----------|-------|
| Method | `POST` |
| Content-Type | `multipart/form-data` |
| Max file size | **16 MB** |

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | file | ✅ | — | Image file (JPEG, PNG, BMP, WEBP, TIFF) |
| `conf` | float | ❌ | `0.25` | Confidence threshold `(0, 1)` |
| `iou` | float | ❌ | `0.45` | NMS IoU threshold `(0, 1)` |

#### Response `200 OK`

```json
{
  "success": true,
  "inference_time_ms": 42.31,
  "image": {
    "filename": "photo.jpg",
    "width": 1280,
    "height": 720
  },
  "count": 2,
  "predictions": [
    {
      "label": "car",
      "class_id": 2,
      "confidence": 0.9512,
      "bbox": {
        "x1": 120.5,
        "y1": 85.0,
        "x2": 480.3,
        "y2": 310.7,
        "width": 359.8,
        "height": 225.7
      }
    },
    {
      "label": "person",
      "class_id": 0,
      "confidence": 0.8834,
      "bbox": {
        "x1": 530.0,
        "y1": 100.2,
        "x2": 620.1,
        "y2": 400.0,
        "width": 90.1,
        "height": 299.8
      }
    }
  ]
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Always `true` on 200 |
| `inference_time_ms` | float | End-to-end inference time in milliseconds |
| `image.filename` | string | Sanitised original filename |
| `image.width` | int | Image width in pixels |
| `image.height` | int | Image height in pixels |
| `count` | int | Number of detections returned |
| `predictions[].label` | string | Human-readable class name |
| `predictions[].class_id` | int | Numeric class index |
| `predictions[].confidence` | float | Score ∈ (0,1), 4 decimal places |
| `predictions[].bbox.x1` | float | Left edge (pixels from left) |
| `predictions[].bbox.y1` | float | Top edge (pixels from top) |
| `predictions[].bbox.x2` | float | Right edge |
| `predictions[].bbox.y2` | float | Bottom edge |
| `predictions[].bbox.width` | float | `x2 - x1` |
| `predictions[].bbox.height` | float | `y2 - y1` |

> **Coordinate system:** origin `(0, 0)` is the **top-left** corner of the
> image. `x` increases right, `y` increases downward — matches HTML5 Canvas,
> CSS, and most drawing libraries.

> **Sort order:** predictions are returned sorted by `confidence` descending
> (highest confidence first).

---

## 4. Request Reference

### Setting thresholds

| Use-case | Recommended `conf` | Recommended `iou` |
|----------|--------------------|-------------------|
| High recall (find everything) | `0.1` | `0.5` |
| Balanced (default) | `0.25` | `0.45` |
| High precision (few false positives) | `0.5` | `0.4` |

---

## 5. Response Reference

### Empty detections

When the model finds nothing above the threshold, the response is still
`200 OK` with an empty array:

```json
{
  "success": true,
  "inference_time_ms": 38.1,
  "image": { "filename": "empty.jpg", "width": 640, "height": 480 },
  "count": 0,
  "predictions": []
}
```

---

## 6. Error Codes

| HTTP Status | `error` message | Cause |
|-------------|-----------------|-------|
| `400` | `No 'image' field in request.` | Missing file field |
| `400` | `No file selected.` | Empty filename |
| `400` | `Unsupported file type…` | Not JPEG/PNG/BMP/WEBP/TIFF |
| `400` | `'conf' and 'iou' must be floats.` | Non-numeric parameter |
| `400` | `'conf' must be between 0 and 1…` | Out-of-range threshold |
| `413` | `File too large. Maximum size is 16 MB.` | Upload exceeds 16 MB |
| `404` | `Endpoint not found.` | Wrong URL |
| `405` | `Method not allowed.` | e.g. GET on /predict |
| `500` | `Inference error: <detail>` | Runtime inference failure |
| `503` | `Model unavailable: <detail>` | Weights file missing / corrupt |

**All error responses share this shape:**
```json
{
  "success": false,
  "error": "Human-readable reason."
}
```

---

## 7. curl Examples

```bash
# Basic prediction
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/photo.jpg"

# Custom thresholds
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/photo.jpg" \
  -F "conf=0.5" \
  -F "iou=0.4"

# Save response to file
curl -X POST http://localhost:5000/predict \
  -F "image=@photo.jpg" \
  -o result.json

# Health check
curl http://localhost:5000/health
```

---

## 8. Postman Setup

1. **New Request** → Method: `POST`, URL: `http://localhost:5000/predict`
2. **Body tab** → select **form-data**
3. Add key `image`, change type dropdown to **File**, upload your image
4. *(Optional)* Add keys `conf` and `iou` with type **Text**
5. Click **Send**

---

## 9. Integration Snippets

### JavaScript (Fetch API)

```javascript
async function detectObjects(imageFile, conf = 0.25, iou = 0.45) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("conf", conf);
  form.append("iou", iou);

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error);
  }

  return res.json(); // { success, count, predictions, ... }
}

// Draw bounding boxes on a canvas
function drawBoxes(canvas, predictions) {
  const ctx = canvas.getContext("2d");
  predictions.forEach(({ label, confidence, bbox }) => {
    const { x1, y1, width, height } = bbox;
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, width, height);
    ctx.fillStyle = "#00ff00";
    ctx.font = "14px sans-serif";
    ctx.fillText(`${label} ${(confidence * 100).toFixed(1)}%`, x1, y1 - 4);
  });
}
```

### Python (requests)

```python
import requests

def predict(image_path: str, conf: float = 0.25, iou: float = 0.45):
    url = "http://localhost:5000/predict"
    with open(image_path, "rb") as f:
        files = {"image": f}
        data  = {"conf": conf, "iou": iou}
        resp  = requests.post(url, files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()

result = predict("photo.jpg", conf=0.3)
for pred in result["predictions"]:
    print(f"{pred['label']} ({pred['confidence']:.2%}): {pred['bbox']}")
```

---

## 10. Limits & Notes

| Limit | Value |
|-------|-------|
| Max upload size | 16 MB |
| Accepted formats | JPEG, PNG, BMP, WEBP, TIFF |
| Model | YOLOv11 (`best.pt`) |
| Default confidence | 0.25 |
| Default IoU | 0.45 |

- Uploaded images are **deleted immediately** after inference — nothing is stored on disk.
- The model is loaded **once** at first request and kept in memory for all subsequent requests.
- `inference_time_ms` covers only the model forward pass + NMS, not network/file-I/O time.
- For production use, run behind **gunicorn** (`gunicorn -w 2 app:app`) — the Flask dev server is single-threaded.
