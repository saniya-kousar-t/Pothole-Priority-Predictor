import os
import io
import time
import random
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow all origins; restrict in production

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "best.pt")
MAX_CONTENT  = int(os.getenv("MAX_CONTENT_MB", "16")) * 1024 * 1024   # 16 MB
CONF_THRESH  = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESH   = float(os.getenv("IOU_THRESHOLD",  "0.45"))
ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT

# ── Base coordinates (Bengaluru) for demo random offsets ──────────────────────
BASE_LAT = float(os.getenv("BASE_LAT", "12.9716"))
BASE_LNG = float(os.getenv("BASE_LNG", "77.5946"))

# ── lazy-load model ───────────────────────────────────────────────────────────
_model = None

def get_model():
    """Load YOLOv11 model once and cache it."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading model from %s …", MODEL_PATH)
            _model = YOLO(MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Model load error: {exc}") from exc
    return _model


# ── helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def format_predictions(results) -> list[dict]:
    """Convert ultralytics Results → clean JSON-serialisable list."""
    predictions = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append({
                "label":      names[int(box.cls[0].item())],
                "class_id":   int(box.cls[0].item()),
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width":  round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                },
            })
    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    return predictions


def compute_severity(predictions: list[dict], count: int) -> int:
    """
    Derive a severity score (1–5) from model outputs.

    Logic:
      - Base severity from pothole count (more potholes = worse)
      - Boosted by average confidence (high confidence = clearer damage)
    """
    if count == 0:
        return 0

    avg_conf = sum(p["confidence"] for p in predictions) / count

    # Map count → raw severity
    if count == 1:
        base = 1
    elif count == 2:
        base = 2
    elif count <= 4:
        base = 3
    elif count <= 6:
        base = 4
    else:
        base = 5

    # Confidence boost: if average confidence > 0.75, bump up by 1
    if avg_conf > 0.75 and base < 5:
        base += 1

    return base


def compute_priority_score(severity: int, count: int) -> int:
    """
    Return a 0–100 priority score.
    Formula: weighted mix of severity and pothole count.
    """
    # Severity contributes 70%, count contributes 30% (capped at 10)
    sev_score   = (severity / 5) * 70
    count_score = min(count / 10, 1.0) * 30
    return round(sev_score + count_score)


def compute_repair(severity: int, count: int) -> dict:
    """
    Return repair_type and estimated_cost (INR) based on severity.

    Severity 1–2 → Minor patch        (₹1,500 – ₹4,000 per pothole)
    Severity 3–4 → Partial repair     (₹5,000 – ₹12,000 per pothole)
    Severity 5   → Full-depth repair  (₹15,000 – ₹25,000 per pothole)
    """
    if severity <= 2:
        repair_type   = "Minor patch"
        cost_per_hole = random.randint(1500, 4000)
    elif severity <= 4:
        repair_type   = "Partial repair"
        cost_per_hole = random.randint(5000, 12000)
    else:
        repair_type   = "Full-depth repair"
        cost_per_hole = random.randint(15000, 25000)

    # Total cost = per-hole cost × number of potholes (minimum 1)
    total_cost = cost_per_hole * max(count, 1)

    return {
        "repair_type":    repair_type,
        "estimated_cost": total_cost,
    }


def random_coordinates() -> dict:
    """
    Return GPS coordinates near the base location with a small random offset.
    Used for demo purposes since the model doesn't provide real GPS data.
    Offset range: ±0.015 degrees (~1.5 km radius)
    """
    lat = round(BASE_LAT + random.uniform(-0.015, 0.015), 6)
    lng = round(BASE_LNG + random.uniform(-0.015, 0.015), 6)
    return {"lat": lat, "lng": lng}


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_path":   MODEL_PATH,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    ─────────────
    Form-data key : image  (file)
    Query params  : conf   (float, default 0.25)
                    iou    (float, default 0.45)

    Returns JSON:
    {
        "success": true,
        "inference_time_ms": 42.3,
        "image":        { "filename": "...", "width": 640, "height": 480 },
        "predictions":  [ { "label", "confidence", "bbox": {...} }, … ],
        "count":        2,

        // ── dashboard fields ──
        "severity":        4,          // 1–5
        "priority_score":  82,         // 0–100
        "repair_type":    "Partial repair",
        "estimated_cost":  8500,       // INR
        "lat":             12.9731,
        "lng":             77.5961
    }
    """
    # ── 1. Validate file ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in form-data."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTS)}",
        }), 415

    # ── 2. Parse optional thresholds ─────────────────────────────────────────
    try:
        conf = float(request.args.get("conf", CONF_THRESH))
        iou  = float(request.args.get("iou",  IOU_THRESH))
        if not (0 < conf < 1) or not (0 < iou < 1):
            raise ValueError
    except ValueError:
        return jsonify({"success": False,
                        "error": "conf and iou must be floats in (0, 1)."}), 400

    # ── 3. Run inference ──────────────────────────────────────────────────────
    try:
        model     = get_model()
        img_bytes = file.read()
        img_buf   = io.BytesIO(img_bytes)

        from PIL import Image as PILImage
        pil_img = PILImage.open(img_buf).convert("RGB")
        w, h    = pil_img.size

        t0      = time.perf_counter()
        results = model.predict(source=pil_img, conf=conf, iou=iou, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"success": False, "error": f"Inference failed: {exc}"}), 500

    # ── 4. Build response ─────────────────────────────────────────────────────
    predictions = format_predictions(results)
    count       = len(predictions)

    severity       = compute_severity(predictions, count)
    priority_score = compute_priority_score(severity, count)
    repair_info    = compute_repair(severity, count)
    coords         = random_coordinates()

    return jsonify({
        "success":            True,
        "inference_time_ms":  round(elapsed, 2),
        "image": {
            "filename": secure_filename(file.filename),
            "width":    w,
            "height":   h,
        },
        "predictions":    predictions,
        "count":          count,

        # ── Dashboard fields ──
        "severity":        severity,
        "priority_score":  priority_score,
        "repair_type":     repair_info["repair_type"],
        "estimated_cost":  repair_info["estimated_cost"],
        "lat":             coords["lat"],
        "lng":             coords["lng"],
    }), 200


# ── error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size is {MAX_CONTENT // (1024*1024)} MB.",
    }), 413

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"success": False, "error": "Method not allowed."}), 405

@app.errorhandler(404)
def not_found(_):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
