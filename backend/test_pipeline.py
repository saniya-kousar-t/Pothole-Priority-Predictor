"""
test_pipeline.py
══════════════════════════════════════════════════════════════════
End-to-End Pipeline Test — Pothole Priority Predictor
Cohort 12 | IIMSTC

Tests the full chain:
  Image → YOLO Detection → Segmentation → Depth Estimation
        → Severity → Formation Analysis → Deterioration → Priority

Run from your project folder:
    python test_pipeline.py --image path/to/pothole.jpg

Or test without a real image (uses a synthetic test image):
    python test_pipeline.py --synthetic

Requires (all in same folder):
    best.pt                      ← YOLO detection weights
    best_seg.pt                  ← YOLO segmentation weights  (rename yours to this)
    latest_pothole_depth_v2.pth    ← Depth Anything V2 fine-tuned weights
    deterioration_model.pkl      ← trained RF model
    formation_analysis.py
    deterioration_predictor.py
    .env                         ← WEATHER_API_KEY=...
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env loaded")
except ImportError:
    print("⚠️  python-dotenv not installed — run: pip install python-dotenv")
    print("   Continuing without .env (weather API may not work)")

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")
def step(n, msg): print(f"\n{BOLD}{'─'*55}\nSTEP {n}: {msg}\n{'─'*55}{RESET}")


# ══════════════════════════════════════════════════════════════════
# STEP 0 — File existence check
# ══════════════════════════════════════════════════════════════════
def check_files():
    step(0, "File Existence Check")

    required = {
        "best.pt":                   "YOLO detection weights",
        "formation_analysis.py":     "Formation analysis module",
        "deterioration_predictor.py":"Deterioration predictor module",
        "deterioration_model.pkl":   "Trained RF model (run train_deterioration.py if missing)",
    }
    optional = {
        "best_seg.pt":                 "YOLO segmentation weights (rename your seg best.pt)",
        "latest_pothole_depth_v2.pth":   "Depth Anything V2 fine-tuned weights",
        ".env":                        "Environment file with WEATHER_API_KEY",
    }

    all_ok = True
    for fname, desc in required.items():
        if Path(fname).exists():
            ok(f"{fname} — {desc}")
        else:
            fail(f"{fname} MISSING — {desc}")
            all_ok = False

    for fname, desc in optional.items():
        if Path(fname).exists():
            ok(f"{fname} — {desc}")
        else:
            warn(f"{fname} not found — {desc}")

    weather_key = os.getenv("WEATHER_API_KEY", "")
    if weather_key and len(weather_key) > 10:
        ok(f"WEATHER_API_KEY set ({weather_key[:6]}...)")
    else:
        warn("WEATHER_API_KEY not set — formation will use Bengaluru defaults")

    return all_ok


# ══════════════════════════════════════════════════════════════════
# STEP 1 — YOLO Detection
# ══════════════════════════════════════════════════════════════════
def test_detection(image_path: str) -> dict:
    step(1, "YOLO Detection (YOLOv11n)")
    t0 = time.perf_counter()
    try:
        from ultralytics import YOLO
        from PIL import Image as PILImage

        model = YOLO("best.pt")
        img   = PILImage.open(image_path).convert("RGB")
        w, h  = img.size
        results = model.predict(source=img, conf=0.25, iou=0.45, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "w": x2-x1, "h": y2-y1}
                })

        elapsed = (time.perf_counter() - t0) * 1000
        count = len(detections)

        if count > 0:
            ok(f"Detected {count} pothole(s) in {elapsed:.1f}ms")
            for i, d in enumerate(detections):
                info(f"  Pothole {i+1}: conf={d['confidence']:.3f}  "
                     f"bbox=[{d['bbox']['x1']:.0f},{d['bbox']['y1']:.0f},"
                     f"{d['bbox']['x2']:.0f},{d['bbox']['y2']:.0f}]")
        else:
            warn("No potholes detected (try a clearer image or lower conf threshold)")

        return {"success": True, "count": count, "detections": detections,
                "image_w": w, "image_h": h, "elapsed_ms": elapsed}

    except Exception as e:
        fail(f"Detection failed: {e}")
        traceback.print_exc()
        return {"success": False, "count": 0, "detections": [], "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# STEP 2 — YOLO Segmentation
# ══════════════════════════════════════════════════════════════════
def test_segmentation(image_path: str, detections: list) -> dict:
    step(2, "YOLO Segmentation (YOLOv11n-seg)")

    seg_model_path = "best_seg.pt"
    if not Path(seg_model_path).exists():
        warn("best_seg.pt not found — skipping segmentation")
        warn("Action needed: rename your segmentation best.pt to best_seg.pt")
        # Return estimated values from detection bboxes
        area_sqm = 0.0
        if detections:
            avg_bbox_area_px = sum(
                d["bbox"]["w"] * d["bbox"]["h"] for d in detections
            ) / len(detections)
            # Assume ~1m of road = ~100px at typical dashcam distance
            area_sqm = round(avg_bbox_area_px / 10000, 3)
        return {"success": False, "skipped": True,
                "area_sqm": max(area_sqm, 0.1), "mask_pixels": 0,
                "note": "Estimated from detection bbox — replace with real seg model"}

    try:
        from ultralytics import YOLO
        from PIL import Image as PILImage
        import numpy as np

        t0    = time.perf_counter()
        model = YOLO(seg_model_path)
        img   = PILImage.open(image_path).convert("RGB")
        results = model.predict(source=img, conf=0.25, verbose=False)

        total_mask_px = 0
        seg_count = 0
        for result in results:
            if result.masks is not None:
                
                for mask in result.masks.data:
                    total_mask_px += int(mask.sum().item())
                    seg_count += 1

        elapsed = (time.perf_counter() - t0) * 1000

        # Convert mask pixels to approximate m² (rough calibration)
        # Assume dashcam image covers ~4m wide road, image is 640px wide
        # → 1px ≈ 0.00625m → 1px² ≈ 0.000039m²
        PX_TO_M2 = 0.000039
        area_sqm = round(total_mask_px * PX_TO_M2, 3)

        ok(f"Segmented {seg_count} mask(s) in {elapsed:.1f}ms")
        info(f"  Total mask area: {total_mask_px:,} px → ~{area_sqm:.3f} m²")

        return {"success": True, "seg_count": seg_count,
                "mask_pixels": total_mask_px, "area_sqm": area_sqm,
                "elapsed_ms": elapsed}

    except Exception as e:
        fail(f"Segmentation failed: {e}")
        traceback.print_exc()
        return {"success": False, "area_sqm": 0.25, "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Depth Estimation (Depth Anything V2)
# ══════════════════════════════════════════════════════════════════
def test_depth(image_path: str, detections: list) -> dict:
    step(3, "Depth Estimation (Depth Anything V2)")

    depth_weights = "latest_pothole_depth_v2.pth"
    da_dir        = Path("Depth-Anything-V2")

    if not da_dir.exists():
        warn("Depth-Anything-V2/ folder not found")
        warn("Action needed: git clone https://github.com/DepthAnything/Depth-Anything-V2.git")
        return {"success": False, "skipped": True, "depth_score": 0.4,
                "note": "Depth-Anything-V2 repo not cloned locally"}

    if not Path(depth_weights).exists():
        warn(f"{depth_weights} not found — trying pretrained vits checkpoint")
        depth_weights = "Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
        if not Path(depth_weights).exists():
            warn("No depth weights found at all")
            warn("Action needed: download from HuggingFace or copy from Google Drive")
            return {"success": False, "skipped": True, "depth_score": 0.4,
                    "note": "No depth weights available"}

    try:
        import sys
        sys.path.insert(0, str(da_dir))

        import cv2
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        model_configs = {
            "vits": {"encoder": "vits", "features": 64,
                     "out_channels": [48, 96, 192, 384]}
        }
        depth_model = DepthAnythingV2(**model_configs["vits"])
        state = torch.load(depth_weights, map_location="cpu")
        depth_model.load_state_dict(state, strict=False)
        depth_model = depth_model.to(DEVICE).eval()

        t0      = time.perf_counter()
        raw_img = cv2.imread(image_path)
        depth   = depth_model.infer_image(raw_img)   # HxW numpy array
        elapsed = (time.perf_counter() - t0) * 1000

        ok(f"Depth map generated in {elapsed:.1f}ms — shape: {depth.shape}")
        info(f"  Depth range: min={depth.min():.3f}  max={depth.max():.3f}  "
             f"mean={depth.mean():.3f}")

        # Extract depth score within each detected pothole bbox
        depth_scores = []
        if detections:
            h_img, w_img = depth.shape
            for det in detections:
                x1 = int(det["bbox"]["x1"] / 640 * w_img)
                y1 = int(det["bbox"]["y1"] / 640 * h_img)
                x2 = int(det["bbox"]["x2"] / 640 * w_img)
                y2 = int(det["bbox"]["y2"] / 640 * h_img)
                roi = depth[y1:y2, x1:x2]
                if roi.size > 0:
                    # Relative depth: how deep is this ROI vs the full frame
                    rel_depth = float(roi.max() / (depth.max() + 1e-9))
                    depth_scores.append(rel_depth)
                    info(f"  Pothole bbox depth score: {rel_depth:.3f} (0=shallow, 1=deep)")

        avg_depth_score = round(float(np.mean(depth_scores)) if depth_scores else 0.4, 3)
        ok(f"Average depth score: {avg_depth_score:.3f}")

        return {"success": True, "depth_score": avg_depth_score,
                "depth_shape": list(depth.shape), "elapsed_ms": elapsed}

    except Exception as e:
        fail(f"Depth estimation failed: {e}")
        traceback.print_exc()
        return {"success": False, "depth_score": 0.4, "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# STEP 4 — Severity Computation
# ══════════════════════════════════════════════════════════════════
def test_severity(detection_result: dict, seg_result: dict, depth_result: dict) -> dict:
    step(4, "Severity Assessment")

    count       = detection_result.get("count", 0)
    area_sqm    = seg_result.get("area_sqm", 0.25)
    depth_score = depth_result.get("depth_score", 0.4)

    if count == 0:
        warn("No potholes detected — severity = 0")
        return {"severity": 0, "continuous_score": 0.0, "source": "detection"}

    # Physics-informed severity from three signals
    # Depth contributes 50%, area 30%, count 20%
    depth_component = min(depth_score * 5.0, 5.0)   # 0–5
    area_component  = min(area_sqm / 0.5 * 5.0, 5.0)  # 0–5 (0.5m² = max)
    count_component = min(count * 1.0, 5.0)             # 0–5

    continuous = (0.5 * depth_component +
                  0.3 * area_component  +
                  0.2 * count_component)

    severity = max(1, min(5, round(continuous)))

    # Source label
    if depth_result.get("success") and seg_result.get("success"):
        source = "depth+segmentation (Stage 4 complete)"
    elif depth_result.get("skipped") and seg_result.get("skipped"):
        source = "detection-only (depth+seg pending)"
    else:
        source = "partial (some Stage 4 components available)"

    ok(f"Severity: {severity}/5  (continuous: {continuous:.2f})")
    info(f"  Depth component:  {depth_component:.2f} × 0.5 = {0.5*depth_component:.2f}")
    info(f"  Area component:   {area_component:.2f} × 0.3 = {0.3*area_component:.2f}")
    info(f"  Count component:  {count_component:.2f} × 0.2 = {0.2*count_component:.2f}")
    info(f"  Source: {source}")

    return {"severity": severity, "continuous_score": round(continuous, 2),
            "area_sqm": area_sqm, "depth_score": depth_score, "source": source}


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Formation Analysis
# ══════════════════════════════════════════════════════════════════
def test_formation(image_path: str, lat: float, lng: float) -> dict:
    step(5, "Formation Factor Analysis")
    try:
        from formation_analysis import analyse_formation_from_image

        t0     = time.perf_counter()
        result = analyse_formation_from_image(
            image_path=image_path,
            lat=lat,
            lng=lng,
            api_key=os.getenv("WEATHER_API_KEY"),
        )
        elapsed = (time.perf_counter() - t0) * 1000

        ok(f"Formation analysis complete in {elapsed:.1f}ms")
        info(f"  Risk Score    : {result['risk_score']}/10  →  {result['risk_level']}")
        info(f"  Dominant cause: {result['dominant_factor']}")
        info(f"  Confidence    : {result['confidence']}")
        info(f"  Data sources  : {result['data_sources']}")

        print(f"\n  {CYAN}Factor breakdown:{RESET}")
        for item in result["shap_explanation"]:
            bar = "█" * int(item["score"])
            print(f"    {item['label'][:40]:<40} {bar} {item['score']:.1f}  "
                  f"[contrib: {item['contribution']:.2f}]")

        irc = result.get("irc_thresholds_crossed", [])
        if irc:
            warn(f"IRC thresholds crossed: {irc}")

        return {"success": True, **result}

    except Exception as e:
        fail(f"Formation analysis failed: {e}")
        traceback.print_exc()
        return {"success": False, "risk_score": 5.0, "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# STEP 6 — Deterioration Prediction
# ══════════════════════════════════════════════════════════════════
def test_deterioration(severity_result: dict, formation_result: dict,
                       road_age: float = 8.0) -> dict:
    step(6, "Deterioration Prediction (30/60/90 days)")
    try:
        from deterioration_predictor import predict_deterioration, deterioration_priority_bonus

        severity   = severity_result["severity"]
        # Pull weather from formation result if available
        weather    = formation_result.get("weather_snapshot", {})
        road_data  = formation_result.get("road_data", {})

        monthly_rain = weather.get("monthly_rainfall_mm", 80.0)
        temp_range   = weather.get("diurnal_range_degC", 12.0)
        road_age_yr  = road_data.get("road_age_years", road_age)
        road_type    = road_data.get("road_type", None)
        drain        = 1  # moderate default

        t0     = time.perf_counter()
        result = predict_deterioration(
            current_severity=severity,
            monthly_rainfall_mm=monthly_rain,
            vehicles_per_hour=500,     # mid-range Bengaluru default
            road_age_years=road_age_yr,
            drainage_condition=drain,
            construction_quality=1,
            temperature_range=temp_range,
            crack_intensity=2.0,
            road_type=road_type,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        bonus = deterioration_priority_bonus(result)

        ok(f"Deterioration prediction complete in {elapsed:.1f}ms")
        print(f"\n  {CYAN}Severity trajectory:{RESET}")
        print(f"    Now   : {'█' * int(result['current_severity'])} "
              f"{result['current_severity']:.1f}/5")
        print(f"    30d   : {'█' * int(result['predicted_sev_30d'])} "
              f"{result['predicted_sev_30d']:.2f}/5")
        print(f"    60d   : {'█' * int(result['predicted_sev_60d'])} "
              f"{result['predicted_sev_60d']:.2f}/5")
        print(f"    90d   : {'█' * int(result['predicted_sev_90d'])} "
              f"{result['predicted_sev_90d']:.2f}/5")
        print(f"    Physics baseline (IRC:37): {result['physics_sev_90d']:.2f}/5")

        urgency_str = f"{result['urgency_emoji']} {result['urgency_label']}"
        dtc = result['days_to_critical']
        dtc_str = f"{dtc} days" if dtc is not None else "stable (>90 days)"

        info(f"  Urgency          : {urgency_str}")
        info(f"  Trajectory       : {result['trajectory']}")
        info(f"  Will worsen?     : {result['will_worsen']}  "
             f"(probability: {result['worsen_probability']:.1%})")
        info(f"  Days to critical : {dtc_str}")
        info(f"  Model used       : {result['model_used']}")
        info(f"  Priority bonus   : +{bonus} (for Stage 7 RICE score)")

        return {"success": True, **result, "priority_bonus": bonus}

    except Exception as e:
        fail(f"Deterioration prediction failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# STEP 7 — Priority Score
# ══════════════════════════════════════════════════════════════════
def compute_priority(severity_result: dict, formation_result: dict,
                     deterioration_result: dict, count: int) -> dict:
    step(7, "Priority Score (RICE + Trajectory Adjustment)")

    severity     = severity_result.get("severity", 1)
    risk_score   = formation_result.get("risk_score", 5.0)
    det_bonus    = deterioration_result.get("priority_bonus", 0.0)

    # Base RICE-derived score (simplified — your team will have the full version)
    reach_map  = {1: 1000, 2: 2000, 3: 3500, 4: 7000, 5: 10000}
    impact_map = {1: 0.25, 2: 0.5,  3: 1.0,  4: 2.0,  5: 3.0}
    effort_map = {1: 1,    2: 1,    3: 2,    4: 2,    5: 4}

    reach  = reach_map.get(severity, 1000)
    impact = impact_map.get(severity, 1.0)
    effort = max(count * effort_map.get(severity, 2), 1)
    rice_raw = (reach * impact * 0.85) / effort
    rice_norm = min(rice_raw / 300, 65.0)  # cap base at 65 so bonus can push to 100

    # Formation risk contribution (0–15)
    formation_component = risk_score / 10.0 * 15.0

    # Final score
    final_score = min(rice_norm + formation_component + det_bonus, 100.0)

    if final_score >= 80:   label = "CRITICAL 🚨"
    elif final_score >= 60: label = "HIGH 🔴"
    elif final_score >= 40: label = "MEDIUM 🟡"
    else:                   label = "LOW 🟢"

    ok(f"Priority Score: {final_score:.1f}/100  →  {label}")
    info(f"  Base RICE component   : {rice_norm:.1f}")
    info(f"  Formation component   : {formation_component:.1f}")
    info(f"  Deterioration bonus   : {det_bonus:.1f}")

    return {"score": round(final_score, 1), "label": label,
            "rice_component": round(rice_norm, 1),
            "formation_component": round(formation_component, 1),
            "deterioration_bonus": det_bonus}


# ══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════
def print_final_report(image_path, lat, lng, det, seg, dep, sev, form, detr, pri):
    print(f"\n{'═'*55}")
    print(f"{BOLD}  FINAL REPORT — Pothole Priority Predictor{RESET}")
    print(f"{'═'*55}")
    print(f"  Image        : {image_path}")
    print(f"  Location     : {lat}, {lng}")
    print(f"  Potholes     : {det['count']}")
    print(f"  Severity     : {sev['severity']}/5  ({sev['continuous_score']:.2f} continuous)")
    print(f"  Area         : ~{seg.get('area_sqm', 0):.3f} m²")
    print(f"  Depth score  : {dep.get('depth_score', 0):.3f}")
    print(f"\n  Formation    : {form.get('risk_score', 'N/A')}/10 — {form.get('risk_level', 'N/A')}")
    print(f"  Cause        : {form.get('dominant_factor', 'N/A')}")
    print(f"\n  Now  →90d    : {detr.get('current_severity','?')} → {detr.get('predicted_sev_90d','?')}")
    print(f"  Urgency      : {detr.get('urgency_emoji','')} {detr.get('urgency_label','N/A')}")
    dtc = detr.get('days_to_critical')
    print(f"  Days to crit : {dtc if dtc else '90+'}")
    print(f"\n  {BOLD}PRIORITY SCORE: {pri['score']}/100 — {pri['label']}{RESET}")
    print(f"{'═'*55}\n")

    # JSON summary
    summary = {
        "image": image_path, "location": {"lat": lat, "lng": lng},
        "potholes_detected": det["count"],
        "severity": sev["severity"],
        "area_sqm": seg.get("area_sqm"),
        "depth_score": dep.get("depth_score"),
        "formation": {
            "risk_score": form.get("risk_score"),
            "risk_level": form.get("risk_level"),
            "dominant_factor": form.get("dominant_factor"),
        },
        "deterioration": {
            "sev_30d": detr.get("predicted_sev_30d"),
            "sev_60d": detr.get("predicted_sev_60d"),
            "sev_90d": detr.get("predicted_sev_90d"),
            "urgency": detr.get("urgency_label"),
            "days_to_critical": detr.get("days_to_critical"),
        },
        "priority": pri,
    }
    out_path = "pipeline_test_result.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    ok(f"Full result saved to {out_path}")


# ══════════════════════════════════════════════════════════════════
# SYNTHETIC IMAGE CREATOR (for testing without a real image)
# ══════════════════════════════════════════════════════════════════
def create_synthetic_image(path: str = "test_pothole_synthetic.jpg"):
    """Creates a simple grey road image with a dark circular pothole."""
    try:
        import cv2
        import numpy as np
        img = np.ones((480, 640, 3), dtype=np.uint8) * 120  # grey road
        cv2.ellipse(img, (320, 280), (80, 50), 0, 0, 360, (40, 40, 40), -1)  # dark pothole
        cv2.ellipse(img, (320, 280), (80, 50), 0, 0, 360, (20, 20, 20), 3)   # edge
        # Add some texture
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(path, img)
        ok(f"Synthetic test image created: {path}")
        return path
    except Exception as e:
        warn(f"Could not create synthetic image ({e}) — trying PIL")
        try:
            from PIL import Image, ImageDraw
            img  = Image.new("RGB", (640, 480), (120, 120, 120))
            draw = ImageDraw.Draw(img)
            draw.ellipse([240, 230, 400, 330], fill=(40, 40, 40), outline=(20, 20, 20))
            img.save(path)
            ok(f"Synthetic test image created (PIL): {path}")
            return path
        except Exception as e2:
            fail(f"Could not create synthetic image: {e2}")
            return None


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pothole Pipeline End-to-End Test")
    parser.add_argument("--image",     type=str, default=None,
                        help="Path to pothole image")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic test image (no real image needed)")
    parser.add_argument("--lat",       type=float, default=12.9716,
                        help="Latitude (default: Indiranagar, Bengaluru)")
    parser.add_argument("--lng",       type=float, default=77.5946,
                        help="Longitude (default: Indiranagar, Bengaluru)")
    parser.add_argument("--road-age",  type=float, default=8.0,
                        help="Road age in years (default: 8)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*55}")
    print("  POTHOLE PRIORITY PREDICTOR — Pipeline Test")
    print(f"{'═'*55}{RESET}")
    print(f"  Location: {args.lat}, {args.lng}")

    # ── File check ────────────────────────────────────────────────
    check_files()

    # ── Image ─────────────────────────────────────────────────────
    if args.synthetic or args.image is None:
        if args.image is None:
            warn("No --image provided. Use --image path/to/pothole.jpg")
            warn("Using synthetic test image instead...")
        image_path = create_synthetic_image()
        if image_path is None:
            fail("Could not create synthetic image. Provide --image argument.")
            sys.exit(1)
    else:
        image_path = args.image
        if not Path(image_path).exists():
            fail(f"Image not found: {image_path}")
            sys.exit(1)
        ok(f"Using image: {image_path}")

    # ── Run pipeline ──────────────────────────────────────────────
    det  = test_detection(image_path)
    seg  = test_segmentation(image_path, det.get("detections", []))
    dep  = test_depth(image_path, det.get("detections", []))
    sev  = test_severity(det, seg, dep)
    form = test_formation(image_path, args.lat, args.lng)
    detr = test_deterioration(sev, form, args.road_age)
    pri  = compute_priority(sev, form, detr, det.get("count", 1))

    # ── Final report ──────────────────────────────────────────────
    print_final_report(image_path, args.lat, args.lng,
                       det, seg, dep, sev, form, detr, pri)

    # ── Summary ───────────────────────────────────────────────────
    steps = {
        "Detection":     det.get("success", False),
        "Segmentation":  seg.get("success", False) or seg.get("skipped", False),
        "Depth":         dep.get("success", False) or dep.get("skipped", False),
        "Severity":      sev.get("severity", 0) > 0,
        "Formation":     form.get("success", False),
        "Deterioration": detr.get("success", False),
        "Priority":      pri.get("score", 0) > 0,
    }
    print(f"\n{BOLD}  Pipeline Status:{RESET}")
    all_pass = True
    for name, passed in steps.items():
        if passed:
            ok(name)
        else:
            fail(name)
            all_pass = False

    if all_pass:
        print(f"\n  {GREEN}{BOLD}All pipeline steps passed! ✅{RESET}")
    else:
        print(f"\n  {YELLOW}{BOLD}Some steps skipped/failed — see warnings above ⚠️{RESET}")
        print(f"  {CYAN}Steps marked skipped are not failures — "
              f"they need model files.{RESET}")


if __name__ == "__main__":
    main()
