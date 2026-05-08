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

from dotenv import load_dotenv
load_dotenv()

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


# ── lazy-load YOLO & Depth models ──────────────────────────────────────────────
_model = None
_depth_model = None

def get_model():
    global _model, _depth_model
    if _model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO model from %s …", MODEL_PATH)
            _model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded.")
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise RuntimeError(f"Model load error: {exc}") from exc
            
    if _depth_model is None:
        try:
            import sys
            import torch
            depth_path = os.path.join(BASE_DIR, "Depth-Anything-V2", "metric_depth")
            if depth_path not in sys.path:
                sys.path.insert(0, depth_path)
            
            from depth_anything_v2.dpt import DepthAnythingV2
            
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            }
            logger.info("Loading Depth-Anything-V2 metric depth model...")
            _depth_model = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': 20})
            _depth_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "latest_pothole_depth_v2.pth"), map_location='cpu'))
            _depth_model = _depth_model.to(DEVICE).eval()
            logger.info("Depth model loaded.")
        except Exception as exc:
            logger.warning("Failed to load Depth model (will fallback to heuristic): %s", exc)
            _depth_model = "FAILED"
            
    return _model, _depth_model


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
    for result in results:
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                if i < len(predictions):
                    px = float(mask.sum().item())
                    predictions[i]["area_sqm"] = round(px * 0.000039, 4)
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

def compute_severity(predictions: list[dict], results, depth_map, count: int, image_w: int = 640, image_h: int = 640) -> int:
    """
    Physical severity calculation using a weighted combination:
    80% from Aggregate YOLO Mask Area + 20% from Max Depth-Anything Metric Depth.
    This resolves the monocular depth micro-geometry limitation while remaining physically accurate.
    """
    if count == 0:
        return 0
        
    import cv2
    import numpy as np

    total_mask_pixels = 0.0
    max_depth_score = 0.0
    
    for result in results:
        if result.masks is None:
            continue
            
        masks = result.masks.data.cpu().numpy()
        
        for i, mask in enumerate(masks):
            # 1. Aggregate Area
            total_mask_pixels += float(mask.sum())
            
            # 2. Find Max Depth Score
            if depth_map is not None:
                mask_resized = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5
                
                if np.any(mask_bool):
                    kernel = np.ones((20, 20), np.uint8)
                    dilated = cv2.dilate(mask_resized, kernel, iterations=1)
                    ring_mask = (dilated > 0.5) & (~mask_bool)
                    
                    if np.any(ring_mask):
                        road_baseline = np.median(depth_map[ring_mask])
                        hole_max = np.max(depth_map[mask_bool])
                        physical_depth_m = abs(hole_max - road_baseline)
                        
                        # Monocular depth heavily smooths micro-depressions.
                        depth_score = min(physical_depth_m / 0.02, 1.0)
                        max_depth_score = max(max_depth_score, depth_score)
            else:
                conf = predictions[i].get("confidence", 0.0) if i < len(predictions) else 0.0
                max_depth_score = max(max_depth_score, min(conf, 1.0))

    # Calculate Total Area Ratio
    total_pixels = float(image_w * image_h)
    area_ratio = total_mask_pixels / total_pixels
    
    # A pothole taking up 30% of the total camera frame is considered massive (Score = 1.0)
    area_score = min(area_ratio / 0.30, 1.0)

    # Count Score: 5+ potholes in a single frame = heavily damaged road (Score = 1.0)
    count_score = min(count / 5.0, 1.0)

    # Weighted Combination (Balanced 30-40-30 Model):
    #   30% Total Area   — Account for large structural damage
    #   40% Count         — Density of potholes
    #   30% Worst Depth   — Critical safety factor
    combined_score = (0.30 * area_score) + (0.40 * count_score) + (0.30 * max_depth_score)
    
    # Force high severity if 5+ potholes detected (Road Failure)
    if count >= 5:
        combined_score = max(combined_score, 0.85) 
    
    # Map 0.0-1.0 to 1-5 Severity Scale
    if combined_score < 0.15:
        sev = 1
    elif combined_score < 0.35:
        sev = 2
    elif combined_score < 0.55:
        sev = 3
    elif combined_score < 0.75:
        sev = 4
    else:
        sev = 5
            
    return sev


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
    # Capped effort: Don't let high effort (more holes) mask the high danger
    effort     = max(min(count * weeks_map.get(severity, 2), 15), 1) 
    rice_score = round((reach * impact * confidence) / effort, 2)

    # Normalise to 0–100 for dashboard visibility
    # Adjusted divisor (20) ensures bad roads hit the 70+ "High Priority" threshold
    rice_normalised = round(min(rice_score / 20, 100.0), 2)

    return {
        "reach":          reach,
        "impact":         impact,
        "confidence":     confidence,
        "effort":         effort,
        "rice_score_raw": rice_score,
        "rice_score":     rice_normalised,   # 0–100, this is what the dashboard uses
    }


# ── Cost Estimation (IRC-standard, integrated from teammate's notebook) ───────

# Pixel-to-meter conversion: dashcam ~6m wide at 640px → 1px ≈ 0.009375m
PIXEL_TO_METER = 0.009375
DEPTH_SCALE    = 3000

# IRC material rates (₹ per m³)
_IRC_RATES = {
    "Small":  3950,    # depth < 2.5 cm
    "Medium": 10000,   # depth < 5 cm
    "Large":  20240,   # depth ≥ 5 cm
}

def compute_cost_irc(results, depth_map, predictions: list[dict],
                     image_w: int = 640, image_h: int = 640) -> dict:
    """
    IRC-standard per-pothole cost estimation.
    Uses real YOLO bounding box areas + Depth-Anything depth values.
    Formula from teammate's Cost estimation.ipynb.
    """
    import cv2
    import numpy as np

    total_cost       = 0.0
    total_area_m2    = 0.0
    total_volume_m3  = 0.0
    pothole_details  = []
    worst_severity   = "Small"
    severity_order   = {"Small": 0, "Medium": 1, "Large": 2}

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Get masks if available (for depth extraction)
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        for i, box in enumerate(result.boxes):
            # ── Bounding box area ──
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_px   = x2 - x1
            height_px  = y2 - y1
            area_pixels = width_px * height_px
            confidence  = float(box.conf[0])

            # Area in m² (capped 0.01 – 0.5 m², same as notebook)
            area_m2 = area_pixels * (PIXEL_TO_METER ** 2)
            area_m2 = max(0.01, min(area_m2, 0.5))

            # ── Depth ──
            if depth_map is not None and masks is not None and i < len(masks):
                mask_resized = cv2.resize(masks[i],
                                          (depth_map.shape[1], depth_map.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5
                if np.any(mask_bool):
                    depth_mean = float(np.mean(depth_map[mask_bool]))
                    depth_m = depth_mean / DEPTH_SCALE
                else:
                    depth_m = 0.02 + (area_pixels / (image_w * image_h)) * 0.06
            elif depth_map is not None:
                # Use bounding box region from depth map
                patch = depth_map[y1:y2, x1:x2]
                if patch.size > 0:
                    depth_m = float(np.mean(patch)) / DEPTH_SCALE
                else:
                    depth_m = 0.02 + (area_pixels / (image_w * image_h)) * 0.06
            else:
                # Fallback: estimate from bbox size (same as notebook)
                depth_m = 0.02 + (area_pixels / (image_w * image_h)) * 0.06

            depth_m = max(0.01, min(depth_m, 0.08))

            # ── Volume ──
            volume = area_m2 * depth_m

            # ── Severity & Rate (IRC standards) ──
            if depth_m < 0.025:
                sev_label = "Small"
                rate      = 3950
            elif depth_m < 0.05:
                sev_label = "Medium"
                rate      = 10000
            else:
                sev_label = "Large"
                rate      = 20240

            if severity_order[sev_label] > severity_order[worst_severity]:
                worst_severity = sev_label

            # ── Cost (exact formula from notebook) ──
            C_mat    = volume * rate
            C_tack   = area_m2 * 34       # tack coat @ ₹34/m²
            C_labour = 400                # flat per pothole
            C_equip  = 300                # flat per pothole
            C_sub    = C_mat + C_tack + C_labour + C_equip
            C_total  = C_sub * 1.20       # 20% overhead

            total_cost      += C_total
            total_area_m2   += area_m2
            total_volume_m3 += volume

            pothole_details.append({
                "pothole_id":  i + 1,
                "confidence":  round(confidence, 2),
                "area_m2":     round(area_m2, 4),
                "depth_m":     round(depth_m, 4),
                "volume_m3":   round(volume, 6),
                "severity":    sev_label,
                "cost_inr":    round(C_total, 2),
            })

    # ── Repair type from worst severity ──
    repair_type_map = {
        "Small":  "Crack Sealing / Cold Patch",
        "Medium": "Hot Mix Asphalt Patching",
        "Large":  "Full Depth Repair",
    }

    count = len(pothole_details)
    return {
        "repair_type":      repair_type_map.get(worst_severity, "Crack Sealing / Cold Patch"),
        "estimated_cost":   round(total_cost, 2),
        "total_area_m2":    round(total_area_m2, 4),
        "total_volume_m3":  round(total_volume_m3, 6),
        "pothole_count":    count,
        "worst_severity":   worst_severity,
        "cost_range":       f"₹{int(total_cost * 0.75):,} – ₹{int(total_cost * 1.25):,}",
        "basis":            "IRC:SP:50-1999 + Karnataka PWD SSR 2024 × 1.20 overhead",
        "pothole_details":  pothole_details,
    }


# ── HTML routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def landing():
    return send_from_directory(BASE_DIR, "landing.html")

@app.route("/dashboard", methods=["GET"])
@app.route("/final_dashboard.html", methods=["GET"])
def dashboard():
    return send_from_directory(BASE_DIR, "final_dashboard.html")

@app.route("/chatbot/<path:filename>")
def chatbot_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "chatbot"), filename)

@app.route("/pdf_report.js")
def pdf_report_js():
    return send_from_directory(BASE_DIR, "pdf_report.js")


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
        model, depth_model = get_model()
        t0      = time.perf_counter()
        results = model.predict(source=pil_img, conf=conf, iou=iou, verbose=False, retina_masks=True)
        
        depth_map = None
        if depth_model != "FAILED" and depth_model is not None:
            import cv2
            import numpy as np
            raw_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            depth_map = depth_model.infer_image(raw_image, 518)
            
        elapsed = (time.perf_counter() - t0) * 1000
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"success": False, "error": f"Inference failed: {exc}"}), 500

    predictions    = format_predictions(results)
    count          = len(predictions)
    coords         = get_coords(request.form)
    severity       = compute_severity(predictions, results, depth_map, count, w, h)
    severity_score = compute_rice_severity_score(predictions, count, w, h)
    rice           = compute_rice_components(predictions, count, severity)
    cost_result    = compute_cost_irc(results, depth_map, predictions, w, h)

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

        # Severity (weighted 80% Area + 20% Depth)
        "severity":        severity,
        "severity_note":   "Weighted: 80% YOLO mask area + 20% Depth-Anything-V2",
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

        # Cost Estimation (IRC-standard from teammate's notebook)
        "repair_type":    cost_result["repair_type"],
        "estimated_cost": cost_result["estimated_cost"],
        "repair_details": cost_result,
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


