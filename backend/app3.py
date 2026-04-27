"""
app3.py  —  Pothole Priority Predictor | Flask Backend
Cohort 12 | IIMSTC | Stage 3 → Stage 9 API

Endpoints
─────────
GET  /health          liveness probe
POST /predict         YOLO detection + severity + RICE (Stage 3–4)
POST /formation       formation factor analysis       (Stage 5)
POST /deterioration   deterioration prediction        (Stage 6)
GET  /                landing page
GET  /dashboard       dashboard page
"""

import os
import io
import time
import logging
import tempfile
import math
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "best.pt")
MAX_CONTENT = int(os.getenv("MAX_CONTENT_MB", "64")) * 1024 * 1024
CONF_THRESH = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESH  = float(os.getenv("IOU_THRESHOLD",  "0.45"))

ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_EXTS       = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT

# Bengaluru centre — fallback when no GPS supplied
BASE_LAT = float(os.getenv("BASE_LAT", "12.9716"))
BASE_LNG = float(os.getenv("BASE_LNG", "77.5946"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── lazy-load YOLO model ───────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO model from %s …", MODEL_PATH)
            _model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded.")
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise RuntimeError(f"Model load error: {exc}") from exc
    return _model


# ── lazy-load formation module ────────────────────────────────────────────────
# NOTE: imported lazily inside the /formation route so a missing module
# does NOT crash the entire Flask app at startup.
def _get_formation_analyser():
    try:
        from formation_analysis import analyse_formation_from_image
        return analyse_formation_from_image
    except ImportError as exc:
        raise RuntimeError(
            "formation_analysis.py not found. "
            "Ensure the file is in the same directory as app3.py."
        ) from exc


# ── lazy-load deterioration module ────────────────────────────────────────────
def _get_deterioration_predictor():
    try:
        from deterioration_predictor import predict_deterioration
        return predict_deterioration
    except ImportError as exc:
        raise RuntimeError(
            "deterioration_predictor.py not found. "
            "Ensure the file is in the same directory as app3.py."
        ) from exc


# ── helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def is_video(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTS

def get_coords(form) -> dict:
    """
    Use real GPS from form-data if provided, else fall back to Bengaluru centre.
    Always prefer real GPS — random scatter only for pure demo mode.
    """
    lat = form.get("lat", type=float)
    lng = form.get("lng", type=float)
    if lat is not None and lng is not None:
        return {"lat": lat, "lng": lng, "source": "provided"}
    return {"lat": BASE_LAT, "lng": BASE_LNG, "source": "default_bengaluru"}


def format_predictions(results) -> list[dict]:
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
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "width":  round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                },
            })
    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    return predictions


def extract_video_frame(file_bytes: bytes, ext: str):
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(1, int(total_frames * 0.10))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame   = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not read frame from video.")
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return pil_img, target_frame
    finally:
        os.unlink(tmp_path)


# ── Severity (placeholder until Stage 4 depth model integrated) ───────────────

def compute_severity(predictions: list[dict], count: int, image_w: int = 640, image_h: int = 640) -> int:
    """
    Placeholder severity from detection output.
    REPLACE with Depth Anything V2 output once Stage 4 is integrated.

    NOTE: this is a detection-based approximation only.
    The canonical severity will come from segmentation + depth estimation.
    """
    if count == 0:
        return 0
    avg_conf = sum(p["confidence"] for p in predictions) / count
    if count == 1:   base = 1
    elif count == 2: base = 2
    elif count <= 4: base = 3
    elif count <= 6: base = 4
    else:            base = 5
    if avg_conf > 0.75 and base < 5:
        base += 1
    return base


def compute_rice_severity_score(predictions: list[dict], count: int,
                                image_w: int = 640, image_h: int = 640) -> float:
    """Continuous 0–10 severity score. Uses actual image dims for area normalization."""
    if count == 0:
        return 0.0
    count_component = min(math.log(count + 1) / math.log(11), 1.0) * 4.0
    avg_conf        = sum(p["confidence"] for p in predictions) / count
    conf_component  = avg_conf * 3.0
    max_area        = image_w * image_h * 0.25        # 25% of frame = max expected pothole
    avg_area        = sum(p["bbox"]["width"] * p["bbox"]["height"] for p in predictions) / count
    area_component  = min(avg_area / max_area, 1.0) * 3.0
    return round(count_component + conf_component + area_component, 2)


# ── RICE model ────────────────────────────────────────────────────────────────

def compute_rice_components(predictions: list[dict], count: int, severity: int) -> dict:
    if count == 0:
        return {"reach": 0, "impact": 0.0, "confidence": 0.0,
                "effort": 1, "rice_score": 0.0}

    avg_conf   = sum(p["confidence"] for p in predictions) / count
    reach_map  = {1: 1000, 2: 2000, 3: 3500, 4: 7000, 5: 10000}
    impact_map = {1: 0.25, 2: 0.5,  3: 1.0,  4: 2.0,  5: 3.0}
    weeks_map  = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4}

    reach      = reach_map.get(severity, 1000)
    impact     = impact_map.get(severity, 0.25)
    confidence = round(min(max(avg_conf, 0.0), 1.0), 4)
    effort     = max(count * weeks_map.get(severity, 2), 1)
    rice_score = round((reach * impact * confidence) / effort, 2)

    # Normalise to 0–100 for dashboard compatibility
    # Empirical max ≈ (10000 × 3.0 × 1.0) / 1 = 30000
    rice_normalised = round(min(rice_score / 300, 100.0), 2)

    return {
        "reach":          reach,
        "impact":         impact,
        "confidence":     confidence,
        "effort":         effort,
        "rice_score_raw": rice_score,
        "rice_score":     rice_normalised,   # 0–100, this is what the dashboard uses
    }


# ── Repair suggestion (deterministic, IRC:SP:50 aligned) ─────────────────────

# Cost formula: (Volume_m3 × material_rate + workers × hours × 250 + equipment) × 1.20
# Karnataka PWD SSR 2024
_REPAIR_TABLE = {
    1: {"type": "Crack Sealing",            "mat_rate": 8000,  "workers": 1, "hours": 1.0, "equip": 200,  "irc": "IRC:SP:50-1999 §7.1"},
    2: {"type": "Cold Mix Patching",         "mat_rate": 12000, "workers": 2, "hours": 2.0, "equip": 400,  "irc": "IRC:SP:50-1999 §8.1"},
    3: {"type": "Hot Mix Asphalt Patching",  "mat_rate": 18000, "workers": 3, "hours": 2.5, "equip": 800,  "irc": "IRC:SP:50-1999 §8.3"},
    4: {"type": "Partial Depth Repair",      "mat_rate": 25000, "workers": 4, "hours": 4.0, "equip": 2000, "irc": "IRC:SP:50-1999 §9.1"},
    5: {"type": "Full Depth Reclamation",    "mat_rate": 40000, "workers": 6, "hours": 8.0, "equip": 5000, "irc": "IRC:SP:50-1999 §10"},
}

def compute_repair(severity: int, count: int,
                   area_sqm: float = 0.25, depth_score: float = 0.3) -> dict:
    """
    Deterministic repair cost from severity + estimated area.
    area_sqm: pothole area in m² (from Stage 4; default 0.25 m² if unavailable)
    depth_score: 0–1 normalised depth (from Stage 4; default 0.3)
    """
    sev   = max(1, min(severity, 5))
    entry = _REPAIR_TABLE[sev]

    volume_m3    = area_sqm * (depth_score * 0.20 + 0.05)   # depth_score → 5–25 cm
    material_cost = volume_m3 * entry["mat_rate"]
    labour_cost   = entry["workers"] * entry["hours"] * 250  # ₹250/hr per worker
    equip_cost    = entry["equip"]
    total_base    = material_cost + labour_cost + equip_cost
    total_inr     = round(total_base * 1.20 * count)         # 20% overhead × count

    # Cost band: ±25%
    return {
        "repair_type":     entry["type"],
        "irc_reference":   entry["irc"],
        "workers_required": entry["workers"],
        "estimated_hours": entry["hours"],
        "material_cost":   round(material_cost * count),
        "labour_cost":     round(labour_cost * count),
        "equipment_cost":  round(equip_cost * count),
        "estimated_cost":  total_inr,
        "cost_range":      f"₹{int(total_inr*0.75):,} – ₹{int(total_inr*1.25):,}",
        "basis":           "Karnataka PWD SSR 2024 × 1.20 overhead",
    }


# ── HTML routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def landing():
    return send_from_directory(BASE_DIR, "landing.html")

@app.route("/dashboard", methods=["GET"])
@app.route("/final_dashboard.html", methods=["GET"])
def dashboard():
    return send_from_directory(BASE_DIR, "final_dashboard.html")


# ── /health ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_path":   MODEL_PATH,
        "modules": {
            "formation":    (Path(BASE_DIR) / "formation_analysis.py").exists(),
            "deterioration": (Path(BASE_DIR) / "deterioration_predictor.py").exists(),
            "det_model_pkl": (Path(BASE_DIR) / "deterioration_model.pkl").exists(),
        },
    }), 200


# ── /predict ─────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Stage 3–4 detection endpoint.
    Accepts: image (file), lat (float, optional), lng (float, optional)
    Returns: detection results + severity + RICE + repair estimate
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in form-data."}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Empty or unsupported file."}), 400

    try:
        conf = float(request.args.get("conf", CONF_THRESH))
        iou  = float(request.args.get("iou",  IOU_THRESH))
        if not (0 < conf < 1) or not (0 < iou < 1):
            raise ValueError
    except ValueError:
        return jsonify({"success": False, "error": "conf and iou must be floats in (0,1)."}), 400

    frame_number = None
    source_type  = "image"
    file_bytes   = file.read()
    ext          = file.filename.rsplit(".", 1)[1].lower()

    try:
        from PIL import Image as PILImage
        if is_video(file.filename):
            source_type = "video"
            pil_img, frame_number = extract_video_frame(file_bytes, ext)
        else:
            pil_img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = pil_img.size
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not read file: {exc}"}), 400

    try:
        model   = get_model()
        t0      = time.perf_counter()
        results = model.predict(source=pil_img, conf=conf, iou=iou, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"success": False, "error": f"Inference failed: {exc}"}), 500

    predictions    = format_predictions(results)
    count          = len(predictions)
    coords         = get_coords(request.form)
    severity       = compute_severity(predictions, count, w, h)
    severity_score = compute_rice_severity_score(predictions, count, w, h)
    rice           = compute_rice_components(predictions, count, severity)
    repair         = compute_repair(severity, count)

    count_comp = round(min(math.log(count + 1) / math.log(11), 1.0) * 4.0, 2) if count else 0.0
    conf_comp  = round((sum(p["confidence"] for p in predictions) / count * 3.0), 2) if count else 0.0
    area_comp  = round(severity_score - count_comp - conf_comp, 2) if count else 0.0

    response = {
        "success":           True,
        "inference_time_ms": round(elapsed, 2),
        "source_type":       source_type,
        "image": {
            "filename": secure_filename(file.filename),
            "width": w, "height": h,
        },
        "predictions": predictions,
        "count":       count,
        "lat":         coords["lat"],
        "lng":         coords["lng"],
        "gps_source":  coords["source"],

        # Severity (placeholder — will be replaced by Stage 4 depth output)
        "severity":        severity,
        "severity_note":   "detection-based placeholder; Stage 4 depth model pending",
        "severity_score":  severity_score,

        # RICE
        "rice_score":      rice["rice_score"],   # normalised 0–100
        "analysis_details": {
            "detected_items":        [p["label"] for p in predictions],
            "estimated_reach":       rice["reach"],
            "estimated_impact":      rice["impact"],
            "estimated_confidence":  rice["confidence"],
            "estimated_effort":      rice["effort"],
            "severity_factors": {
                "count_component":      count_comp,
                "confidence_component": conf_comp,
                "area_component":       area_comp,
            },
        },

        # Repair
        "repair_type":    repair["repair_type"],
        "estimated_cost": repair["estimated_cost"],
        "repair_details": repair,
    }
    if frame_number is not None:
        response["frame_number"] = frame_number

    return jsonify(response), 200


# ── /formation ────────────────────────────────────────────────────────────────

@app.route("/formation", methods=["POST"])
def formation():
    """
    Stage 5 — Formation Factor Analysis.
    Accepts: image (file, optional), lat (float), lng (float)
    Returns: risk_score, risk_level, dominant_factor, shap_explanation, ...
    """
    lat = request.form.get("lat", type=float)
    lng = request.form.get("lng", type=float)
    image_path = None

    if "image" in request.files:
        file = request.files["image"]
        tmp  = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(file.read())
        tmp.close()
        image_path = tmp.name

    try:
        analyser = _get_formation_analyser()
        result   = analyser(
            image_path=image_path,
            lat=lat,
            lng=lng,
            api_key=os.getenv("WEATHER_API_KEY"),
        )
        return jsonify({"success": True, **result}), 200

    except Exception as e:
        logger.exception("Formation analysis error")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except Exception:
                pass


# ── /deterioration ────────────────────────────────────────────────────────────

@app.route("/deterioration", methods=["POST"])
def deterioration():
    """
    Stage 6 — Deterioration Prediction.

    Accepts JSON body OR form-data with the following fields:
        current_severity      float  1–5  (required)
        monthly_rainfall_mm   float       (required — from OWM or /formation output)
        vehicles_per_hour     float       (required — from TomTom or OSM estimate)
        road_age_years        float       (required — from OSM start_date)
        drainage_condition    int  0/1/2  (optional, default 1)
        construction_quality  int  0/1/2  (optional, default 1)
        temperature_range     float       (optional, default 12.0)
        crack_intensity       float  0–15 (optional, default 2.0)
        road_type             str         (optional — auto-inferred if absent)

    Returns: full deterioration prediction including 30/60/90d forecasts,
             urgency label, days_to_critical, and priority bonus score.

    Integration tip: if calling after /formation, pass monthly_rainfall_mm
    and vehicles_per_hour from the /formation response's features_used block.
    """
    # Accept both JSON and form-data
    if request.is_json:
        data = request.get_json(force=True) or {}
    else:
        data = request.form.to_dict()

    def _float(key, default=None):
        v = data.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _int(key, default=None):
        v = _float(key)
        return int(v) if v is not None else default

    # ── Required fields ───────────────────────────────────────────────────────
    current_severity = _float("current_severity")
    if current_severity is None:
        return jsonify({
            "success": False,
            "error": "current_severity is required (float 1–5)."
        }), 400

    monthly_rainfall_mm = _float("monthly_rainfall_mm")
    if monthly_rainfall_mm is None:
        return jsonify({
            "success": False,
            "error": "monthly_rainfall_mm is required."
        }), 400

    vehicles_per_hour = _float("vehicles_per_hour")
    if vehicles_per_hour is None:
        return jsonify({
            "success": False,
            "error": "vehicles_per_hour is required."
        }), 400

    road_age_years = _float("road_age_years")
    if road_age_years is None:
        return jsonify({
            "success": False,
            "error": "road_age_years is required."
        }), 400

    # ── Optional fields ───────────────────────────────────────────────────────
    drainage_condition   = _int("drainage_condition",   default=1)
    construction_quality = _int("construction_quality", default=1)
    temperature_range    = _float("temperature_range",  default=12.0)
    crack_intensity      = _float("crack_intensity",    default=2.0)
    road_type            = data.get("road_type")        # None = auto-inferred

    try:
        predictor = _get_deterioration_predictor()
        result    = predictor(
            current_severity=current_severity,
            monthly_rainfall_mm=monthly_rainfall_mm,
            vehicles_per_hour=vehicles_per_hour,
            road_age_years=road_age_years,
            drainage_condition=drainage_condition,
            construction_quality=construction_quality,
            temperature_range=temperature_range,
            crack_intensity=crack_intensity,
            road_type=road_type,
        )

        # Attach priority bonus for Stage 7 consumption
        from deterioration_predictor import deterioration_priority_bonus
        result["priority_bonus"] = deterioration_priority_bonus(result)

        return jsonify({"success": True, **result}), 200

    except Exception as e:
        logger.exception("Deterioration prediction error")
        return jsonify({"success": False, "error": str(e)}), 500


# ── error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False,
                    "error": f"File too large. Max {MAX_CONTENT//(1024*1024)} MB."}), 413

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
