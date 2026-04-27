"""
deterioration_predictor.py
══════════════════════════════════════════════════════════════════
Stage 6 — Pothole Deterioration Predictor
Pothole Priority Predictor | Cohort 12 | IIMSTC

Predicts how fast a pothole will worsen at 30 / 60 / 90 days
using a trained Random Forest + IRC:37-2018 physics baseline.

Usage (standalone):
    from deterioration_predictor import predict_deterioration
    result = predict_deterioration(
        current_severity=3,
        monthly_rainfall_mm=320,
        vehicles_per_hour=900,
        road_age_years=12,
        drainage_condition=2,        # 0=good, 1=moderate, 2=poor
        construction_quality=1,      # 0=good, 1=average, 2=poor
        temperature_range=10.0,      # °C diurnal swing
        crack_intensity=4,           # cracks per metre² (0–15)
    )

Flask integration (in app3.py):
    from deterioration_predictor import predict_deterioration
    result = predict_deterioration(**feature_dict)
    # Returns dict ready to embed in JSON response

Output schema:
    {
      "current_severity": 3,
      "predicted_sev_30d": 3.8,
      "predicted_sev_60d": 4.4,
      "predicted_sev_90d": 4.98,
      "will_worsen": true,
      "worsen_probability": 0.969,
      "urgency_label": "CRITICAL",
      "urgency_emoji": "🚨",
      "days_to_critical": 28,          # days until severity ≥ 4
      "trajectory": "rapid",           # "stable" | "slow" | "moderate" | "rapid"
      "physics_sev_90d": 4.71,         # IRC:37 baseline (for transparency)
      "model_used": "RandomForest+IRC37",
      "irc_reference": "IRC:37-2018 Table 3",
      "features_used": { ... }
    }

Data basis: Synthetic IRC:37-2018 + IRC:82-2015 + MoRTH 2023
            5,000 Indian road sections, Bengaluru-calibrated
"""

from __future__ import annotations

import os
import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR         = Path(__file__).parent
MODEL_PATH   = _DIR / "deterioration_model.pkl"

# ── Feature list (must match training order exactly) ──────────────────────────
FEATURES = [
    "current_severity",
    "monthly_rainfall_mm",
    "vehicles_per_hour",
    "road_age_years",
    "drainage_condition",
    "construction_quality",
    "temperature_range",
    "crack_intensity",
]

# ── IRC:37-2018 Table 3 — k-values by road type ───────────────────────────────
# Decay constant per million standard axles (msa)
# PSI(t) = PSI_0 × exp(-k × ESAL_msa)
IRC37_K_VALUES = {
    "highway":   0.10,   # NH — Hosur Rd, Tumkur Rd
    "arterial":  0.20,   # Major district roads, ORR
    "collector": 0.28,   # Ward arterials, layout roads
    "local":     0.40,   # Inner ward roads, lanes
}

# Typical daily ESALs by road type (vehicles_per_hour × 10h × ESAL_per_vehicle)
# ESAL_per_vehicle: car≈0.0002, auto≈0.005, bus≈1.5, truck≈4.0
# Weighted average for Bengaluru mix ≈ 0.25 per vehicle
ESAL_PER_VEHICLE = 0.25   # conservative Bengaluru urban mix

# ── Lazy model cache ──────────────────────────────────────────────────────────
_model_cache: Optional[dict] = None


def _load_model() -> dict:
    """Load deterioration_model.pkl once and cache. Falls back to physics-only."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if MODEL_PATH.exists():
        try:
            _model_cache = joblib.load(MODEL_PATH)
            logger.info("Deterioration model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Could not load deterioration model: %s — using physics fallback", exc)
            _model_cache = {}
    else:
        logger.warning(
            "deterioration_model.pkl not found at %s — using IRC:37 physics-only mode. "
            "Run the training notebook to generate the model file.",
            MODEL_PATH,
        )
        _model_cache = {}

    return _model_cache


# ── IRC:37-2018 Physics Baseline ──────────────────────────────────────────────

def _irc37_physics_severity(
    current_severity: float,
    road_type: str,
    vehicles_per_hour: float,
    monthly_rainfall_mm: float,
    drainage_condition: int,
    road_age_years: float,
    days: int,
) -> float:
    """
    IRC:37-2018 exponential decay model.

    PSI(t) = PSI_0 × exp(-k_eff × ESAL_cumulative_msa)

    Severity maps to PSI: severity = 6 - PSI  (PSI 5→Sev 1, PSI 1→Sev 5)
    Returns clamped severity at `days`.
    """
    k_base = IRC37_K_VALUES.get(road_type, IRC37_K_VALUES["collector"])

    # Water acceleration factor (IRC:SP:50 §4.2)
    drain_deficit = {0: 0.0, 1: 0.5, 2: 1.0}.get(int(drainage_condition), 0.5)
    rain_factor   = 1.0 + 0.015 * (monthly_rainfall_mm / 100.0) * drain_deficit
    k_eff         = k_base * rain_factor

    # Daily ESALs → cumulative msa over `days`
    daily_esal       = vehicles_per_hour * 10 * ESAL_PER_VEHICLE          # 10 operating hours
    esal_cumulative  = daily_esal * days / 1_000_000                       # convert to msa

    # Age penalty: older roads have lower remaining structural capacity
    age_factor = 1.0 + 0.03 * max(road_age_years - 5, 0)                  # +3% per year after yr 5

    # PSI calculation
    psi_0 = max(6.0 - current_severity, 0.1)
    psi_t = psi_0 * math.exp(-k_eff * esal_cumulative * age_factor)
    psi_t = max(psi_t, 0.1)

    sev_t = 6.0 - psi_t
    return round(min(max(sev_t, current_severity), 5.0), 3)   # monotonic: can't improve


# ── ML Inference ──────────────────────────────────────────────────────────────

def _ml_predict_severity_90d(features: dict) -> Optional[float]:
    """Run Random Forest regression. Returns None if model not loaded."""
    model_data = _load_model()
    if not model_data or "regression_model" not in model_data:
        return None

    reg  = model_data["regression_model"]
    feat = model_data.get("features", FEATURES)
    row  = pd.DataFrame([{f: features.get(f, 0) for f in feat}])
    return float(reg.predict(row)[0])


def _ml_predict_worsen_prob(features: dict) -> Optional[float]:
    """Run Random Forest classifier. Returns worsen probability or None."""
    model_data = _load_model()
    if not model_data or "classification_model" not in model_data:
        return None

    cls  = model_data["classification_model"]
    feat = model_data.get("features", FEATURES)
    row  = pd.DataFrame([{f: features.get(f, 0) for f in feat}])
    return float(cls.predict_proba(row)[0][1])


# ── Helper: infer road_type from vehicles_per_hour if not provided ─────────────

def _infer_road_type(vehicles_per_hour: float) -> str:
    """Rough road type inference for physics model when road_type not supplied."""
    if vehicles_per_hour >= 800:
        return "highway"
    elif vehicles_per_hour >= 500:
        return "arterial"
    elif vehicles_per_hour >= 200:
        return "collector"
    return "local"


# ── Days-to-critical estimator ─────────────────────────────────────────────────

def _days_to_critical(
    current_severity: float,
    sev_30: float,
    sev_60: float,
    sev_90: float,
    critical_threshold: float = 4.0,
) -> Optional[int]:
    """
    Linear interpolation to estimate when severity will cross 4.0 (critical).
    Returns None if already critical or never reaches threshold in 90 days.
    """
    if current_severity >= critical_threshold:
        return 0   # already critical

    points = [(0, current_severity), (30, sev_30), (60, sev_60), (90, sev_90)]

    for i in range(len(points) - 1):
        d0, s0 = points[i]
        d1, s1 = points[i + 1]
        if s0 < critical_threshold <= s1:
            # Linear interpolate
            frac = (critical_threshold - s0) / (s1 - s0 + 1e-9)
            return int(d0 + frac * (d1 - d0))

    return None   # won't reach critical in 90 days


# ── Urgency label ──────────────────────────────────────────────────────────────

def _urgency(worsen_prob: float, sev_90: float, days_to_critical: Optional[int]) -> tuple[str, str]:
    """
    Returns (urgency_label, urgency_emoji).
    Combines probability and trajectory for a two-signal decision.
    """
    # if days_to_critical is not None and days_to_critical <= 30:
    #     return "CRITICAL", "🚨"
    if days_to_critical is not None and days_to_critical <= 30 and worsen_prob >= 0.5:
        return "CRITICAL", "🚨"
    if worsen_prob >= 0.80 or sev_90 >= 4.5:
        return "CRITICAL", "🚨"
    if worsen_prob >= 0.60 or sev_90 >= 3.5:
        return "HIGH", "🔴"
    if worsen_prob >= 0.40 or sev_90 >= 2.5:
        return "MEDIUM", "🟡"
    return "LOW", "🟢"


def _trajectory_label(current_severity: float, sev_90: float) -> str:
    delta = sev_90 - current_severity
    if delta < 0.3:
        return "stable"
    elif delta < 1.0:
        return "slow"
    elif delta < 2.0:
        return "moderate"
    return "rapid"


# ── Public API ─────────────────────────────────────────────────────────────────

def predict_deterioration(
    current_severity: float,
    monthly_rainfall_mm: float,
    vehicles_per_hour: float,
    road_age_years: float,
    drainage_condition: int       = 1,   # 0=good, 1=moderate, 2=poor
    construction_quality: int     = 1,   # 0=good, 1=average, 2=poor
    temperature_range: float      = 12.0,
    crack_intensity: float        = 2.0,
    road_type: Optional[str]      = None,
) -> dict:
    """
    Main entry point. Returns a complete deterioration prediction dict.

    Parameters
    ----------
    current_severity       : int/float 1–5 (from Stage 4 segmentation)
    monthly_rainfall_mm    : float from OpenWeatherMap (use monthly average
                             or current month accumulated)
    vehicles_per_hour      : float from TomTom Traffic API or OSM estimate
    road_age_years         : float from OSM start_date or BBMP data
    drainage_condition     : 0=good, 1=moderate, 2=poor  (OSM + field estimate)
    construction_quality   : 0=good, 1=average, 2=poor  (derived from road_age + type)
    temperature_range      : daily max–min °C from OpenWeatherMap
    crack_intensity        : number of visible cracks per m² (0–15); use 2 as default
    road_type              : "highway"|"arterial"|"collector"|"local"|None (auto-inferred)
    """
    # Clamp inputs to valid ranges
    current_severity   = float(min(max(current_severity, 1.0), 5.0))
    monthly_rainfall_mm = float(min(max(monthly_rainfall_mm, 0.0), 600.0))
    vehicles_per_hour  = float(min(max(vehicles_per_hour, 50.0), 2000.0))
    road_age_years     = float(min(max(road_age_years, 0.5), 30.0))
    drainage_condition = int(min(max(drainage_condition, 0), 2))
    construction_quality = int(min(max(construction_quality, 0), 2))
    temperature_range  = float(min(max(temperature_range, 0.0), 40.0))
    crack_intensity    = float(min(max(crack_intensity, 0.0), 15.0))

    if road_type not in IRC37_K_VALUES:
        road_type = _infer_road_type(vehicles_per_hour)

    features = {
        "current_severity":     current_severity,
        "monthly_rainfall_mm":  monthly_rainfall_mm,
        "vehicles_per_hour":    vehicles_per_hour,
        "road_age_years":       road_age_years,
        "drainage_condition":   drainage_condition,
        "construction_quality": construction_quality,
        "temperature_range":    temperature_range,
        "crack_intensity":      crack_intensity,
    }

    # ── 1. IRC:37 physics predictions at all three horizons ──────────────────
    phys_30 = _irc37_physics_severity(current_severity, road_type, vehicles_per_hour,
                                       monthly_rainfall_mm, drainage_condition,
                                       road_age_years, 30)
    phys_60 = _irc37_physics_severity(current_severity, road_type, vehicles_per_hour,
                                       monthly_rainfall_mm, drainage_condition,
                                       road_age_years, 60)
    phys_90 = _irc37_physics_severity(current_severity, road_type, vehicles_per_hour,
                                       monthly_rainfall_mm, drainage_condition,
                                       road_age_years, 90)

    # ── 2. ML prediction at 90d (if model available) ─────────────────────────
    ml_sev_90  = _ml_predict_severity_90d(features)
    ml_prob    = _ml_predict_worsen_prob(features)

    # ── 3. Final severity values: ML@90d + physics-proportioned 30/60 ────────
    #
    # Strategy: if ML model loaded, blend ML@90d with physics.
    # For 30d and 60d, we interpolate proportionally from the physics trajectory.
    # This ensures monotonicity: sev_30 ≤ sev_60 ≤ sev_90.
    # Physics residual = ml_90 - phys_90 captures site-specific adjustment.

    if ml_sev_90 is not None:
        ml_sev_90   = float(min(max(ml_sev_90, current_severity), 5.0))
        residual_90 = ml_sev_90 - phys_90        # can be + or -

        # Distribute residual proportionally to elapsed time
        sev_30 = min(max(phys_30 + residual_90 * (30 / 90), current_severity), 5.0)
        sev_60 = min(max(phys_60 + residual_90 * (60 / 90), current_severity), 5.0)
        sev_90 = ml_sev_90
        model_used = "RandomForest+IRC37"
    else:
        # Physics-only fallback
        sev_30 = phys_30
        sev_60 = phys_60
        sev_90 = phys_90
        model_used = "IRC37-PhysicsOnly"

    # Ensure strict monotonicity (safety clamp)
    sev_30 = min(sev_30, 5.0)
    sev_60 = max(sev_60, sev_30)
    sev_90 = max(sev_90, sev_60)

    # ── 4. Worsen probability ─────────────────────────────────────────────────
    if ml_prob is not None:
        worsen_prob = ml_prob
    else:
        # Physics-derived probability: use delta as proxy
        delta = sev_90 - current_severity
        worsen_prob = float(min(delta / 2.0, 1.0))   # rough sigmoid

    will_worsen = worsen_prob >= 0.5

    # ── 5. Days to critical ───────────────────────────────────────────────────
    dtc = _days_to_critical(current_severity, sev_30, sev_60, sev_90)

    # ── 6. Labels ─────────────────────────────────────────────────────────────
    urgency_label, urgency_emoji = _urgency(worsen_prob, sev_90, dtc)
    trajectory = _trajectory_label(current_severity, sev_90)

    return {
        # ── Core predictions ──
        "current_severity":   round(current_severity, 1),
        "predicted_sev_30d":  round(sev_30, 2),
        "predicted_sev_60d":  round(sev_60, 2),
        "predicted_sev_90d":  round(sev_90, 2),

        # ── Classification ──
        "will_worsen":        will_worsen,
        "worsen_probability": round(worsen_prob, 4),

        # ── Urgency ──
        "urgency_label":      urgency_label,
        "urgency_emoji":      urgency_emoji,
        "days_to_critical":   dtc,         # None if stable / already critical
        "trajectory":         trajectory,  # "stable"|"slow"|"moderate"|"rapid"

        # ── Transparency ──
        "physics_sev_90d":    round(phys_90, 2),
        "model_used":         model_used,
        "road_type_used":     road_type,
        "irc_reference":      "IRC:37-2018 Table 3 + IRC:82-2015",

        # ── Echo features (for dashboard display) ──
        "features_used": {
            "current_severity":     round(current_severity, 1),
            "monthly_rainfall_mm":  round(monthly_rainfall_mm, 1),
            "vehicles_per_hour":    int(vehicles_per_hour),
            "road_age_years":       int(road_age_years),
            "drainage_condition":   drainage_condition,
            "construction_quality": construction_quality,
            "temperature_range":    round(temperature_range, 1),
            "crack_intensity":      round(crack_intensity, 1),
        },
    }


# ── Batch inference (optional, for future /batch endpoint) ────────────────────

def predict_deterioration_batch(records: list[dict]) -> list[dict]:
    """
    Run predict_deterioration over a list of feature dicts.
    Returns list of result dicts in the same order.
    """
    return [predict_deterioration(**r) for r in records]


# ── Priority score adjustment (for Stage 7 integration) ───────────────────────

def deterioration_priority_bonus(det_result: dict) -> float:
    """
    Returns a 0–35 bonus to add to the base RICE priority score.

    Two components:
      trajectory_bonus (0–20): how fast is it getting worse?
      urgency_bonus    (0–15): how soon does it cross the critical threshold?

    Call this from compute_priority_score() in app3.py once
    formation + deterioration are integrated.
    """
    current = det_result["current_severity"]
    sev_90  = det_result["predicted_sev_90d"]
    dtc     = det_result.get("days_to_critical")
    prob    = det_result["worsen_probability"]

    # Trajectory bonus: severity delta × 5, capped at 20
    delta_sev       = max(sev_90 - current, 0)
    trajectory_bonus = min(delta_sev * 5.0, 20.0)

    # Urgency bonus: shorter time to critical = higher bonus
    if dtc is not None:
        urgency_bonus = max(0.0, (90 - dtc) / 90.0 * 15.0)
    elif prob >= 0.8:
        urgency_bonus = 12.0
    else:
        urgency_bonus = prob * 10.0

    return round(trajectory_bonus + urgency_bonus, 2)


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Deterioration Predictor — quick test")
    print("=" * 60)

    test_cases = [
        {
            "label": "Highway pothole — Bengaluru monsoon",
            "params": dict(
                current_severity=3, monthly_rainfall_mm=320,
                vehicles_per_hour=900, road_age_years=12,
                drainage_condition=2, construction_quality=1,
                temperature_range=10, crack_intensity=4,
                road_type="highway",
            ),
        },
        {
            "label": "Minor crack — dry Jayanagar residential",
            "params": dict(
                current_severity=1, monthly_rainfall_mm=40,
                vehicles_per_hour=150, road_age_years=3,
                drainage_condition=0, construction_quality=0,
                temperature_range=10, crack_intensity=0,
                road_type="local",
            ),
        },
        {
            "label": "Old arterial — post-monsoon",
            "params": dict(
                current_severity=4, monthly_rainfall_mm=100,
                vehicles_per_hour=500, road_age_years=18,
                drainage_condition=2, construction_quality=2,
                temperature_range=14, crack_intensity=6,
                road_type="arterial",
            ),
        },
    ]

    for tc in test_cases:
        print(f"\n  📍 {tc['label']}")
        result = predict_deterioration(**tc["params"])
        print(f"     Current severity   : {result['current_severity']}/5")
        print(f"     Predicted @30 days : {result['predicted_sev_30d']}/5")
        print(f"     Predicted @60 days : {result['predicted_sev_60d']}/5")
        print(f"     Predicted @90 days : {result['predicted_sev_90d']}/5")
        print(f"     Physics baseline   : {result['physics_sev_90d']}/5")
        print(f"     Will worsen?       : {result['will_worsen']} (prob: {result['worsen_probability']:.1%})")
        print(f"     Urgency            : {result['urgency_emoji']} {result['urgency_label']}")
        print(f"     Days to critical   : {result['days_to_critical'] if result['days_to_critical'] is not None else 'N/A (stable in 90d)'}")
        print(f"     Trajectory         : {result['trajectory']}")
        print(f"     Model used         : {result['model_used']}")
        bonus = deterioration_priority_bonus(result)
        print(f"     Priority bonus     : +{bonus} (add to RICE score)")
    print()
