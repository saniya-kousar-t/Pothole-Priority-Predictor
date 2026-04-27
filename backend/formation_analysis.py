"""
formation_analysis.py
══════════════════════════════════════════════════════════════════
Stage 5 — Pothole Formation Factor Analysis
Pothole Priority Predictor | Cohort 12 | IIMSTC

Analyses WHY a pothole formed at a given GPS location using:
  • OpenWeatherMap API   — rainfall, temperature, humidity
  • OSM Overpass API     — road type, age, surface, drain proximity
  • IRC:37-2018 physics  — ESAL-based traffic loading
  • Rule-based XGBoost-ready feature engineering

No external ML model needed — the risk score is produced by a
physics-weighted formula that is fully explainable to coordinators.
Each factor maps to an IRC standard clause.

Usage:
    from formation_analysis import analyse_formation_from_image
    result = analyse_formation_from_image(
        image_path="/tmp/pothole.jpg",   # or None
        lat=12.9716,
        lng=77.5946,
        api_key="YOUR_OWM_KEY",
    )

Output schema:
    {
      "risk_score": 7.8,            # 0–10 continuous
      "risk_level": "HIGH",         # LOW / MEDIUM / HIGH / CRITICAL
      "dominant_factor": "water_infiltration",
      "factor_scores": {
        "water_infiltration": 8.2,
        "traffic_loading": 6.5,
        "thermal_stress": 4.1,
        "road_age_pavement": 7.0,
        "drainage_failure": 9.0,
      },
      "factor_weights": { ... },
      "shap_explanation": [          # sorted list for dashboard bar chart
        {"factor": "drainage_failure", "label": "Poor drainage near site",
         "score": 9.0, "weight": 0.25, "contribution": 2.25, "irc_ref": "IRC:SP:42-1994"},
        ...
      ],
      "weather_snapshot": { ... },
      "road_data": { ... },
      "irc_thresholds_crossed": ["rainfall_intensity", "road_age"],
      "data_sources": ["OpenWeatherMap", "OSM Overpass", "IRC:37-2018"],
      "confidence": "medium",       # "high" if all APIs succeeded
    }
"""

from __future__ import annotations

import os
import math
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants — IRC Standards ─────────────────────────────────────────────────

# IRC:37-2018 Table 3 k-values; IRC:82-2015 drainage specs
IRC_THRESHOLDS = {
    "annual_rainfall_high_mm":      900,    # Bengaluru 30-yr mean
    "rainfall_intensity_high_mmhr": 25.0,   # IRC:SP:50 §4.2 infiltration threshold
    "consecutive_wet_days_risk":    7,      # IRC:82 base saturation risk
    "pavement_design_life_years":   10,     # IRC:37-2018 §5.1
    "pavement_high_risk_years":     8,      # >8 yr: non-linear fatigue drop
    "cvpd_high_traffic":            450,    # IRC:37 traffic class D
    "temp_diurnal_crack_risk_degC": 15.0,   # thermal cracking threshold
}

# Factor weights — tuned from IRC:37/IRC:SP:50 literature
# Must sum to 1.0
FACTOR_WEIGHTS = {
    "water_infiltration": 0.30,    # IRC:SP:50-1999 §4.2 — dominant in Bengaluru
    "traffic_loading":    0.25,    # IRC:37-2018 ESAL model
    "drainage_failure":   0.25,    # IRC:SP:42-1994 — strong Bengaluru signal
    "road_age_pavement":  0.12,    # IRC:37-2018 §5.1 design life
    "thermal_stress":     0.08,    # Diurnal temperature cycling
}

# OSM highway tag → road type mapping
OSM_HIGHWAY_TO_TYPE = {
    "motorway": "highway", "trunk": "highway",
    "primary": "arterial", "secondary": "arterial",
    "tertiary": "collector", "unclassified": "collector",
    "residential": "local", "service": "local",
    "living_street": "local", "road": "local",
}

# IRC:37 design traffic (CVPD) per road type — for ESAL estimation
ROAD_TYPE_CVPD_EST = {
    "highway": 1200, "arterial": 600,
    "collector": 300, "local": 80,
}


# ── API helpers ────────────────────────────────────────────────────────────────

def _owm_current(lat: float, lon: float, api_key: str) -> Optional[dict]:
    """Fetch current weather from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        r = requests.get(url, params={
            "lat": lat, "lon": lon,
            "appid": api_key, "units": "metric",
        }, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("OWM current weather failed: %s", exc)
        return None


def _owm_forecast(lat: float, lon: float, api_key: str) -> Optional[dict]:
    """Fetch 5-day / 3-hour forecast for consecutive wet day count."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    try:
        r = requests.get(url, params={
            "lat": lat, "lon": lon,
            "appid": api_key, "units": "metric", "cnt": 40,   # ~5 days
        }, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("OWM forecast failed: %s", exc)
        return None


def _osm_road_data(lat: float, lon: float, radius_m: int = 50) -> Optional[dict]:
    """
    Query OSM Overpass for the nearest road and drain proximity.
    Returns dict with highway type, surface, start_date, drain proximity.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:15];
    (
      way(around:{radius_m},{lat},{lon})["highway"];
      way(around:200,{lat},{lon})["waterway"="drain"];
      way(around:200,{lat},{lon})["waterway"="stream"];
    );
    out body;
    """
    try:
        # r = requests.post(overpass_url, data={"data": query}, timeout=15)
        r = requests.post(overpass_url, data={"data": query}, timeout=15,
                  headers={"User-Agent": "PotholePredictor/1.0"})
        r.raise_for_status()
        data = r.json()
        return data.get("elements", [])
    except Exception as exc:
        logger.warning("OSM Overpass query failed: %s", exc)
        return None


def _parse_osm_elements(elements: list, lat: float, lon: float) -> dict:
    """Extract road features from raw OSM elements list."""
    road_info = {
        "highway_tag":      "residential",
        "road_type":        "local",
        "surface":          "asphalt",
        "start_date":       None,
        "road_age_years":   7,           # Bengaluru median if unknown
        "drain_proximity_m": 999,
        "has_drain_nearby": False,
        "lanes":            2,
    }

    roads   = [e for e in elements if e.get("type") == "way"
               and "highway" in e.get("tags", {})]
    drains  = [e for e in elements if e.get("type") == "way"
               and e.get("tags", {}).get("waterway") in ("drain", "stream")]

    if roads:
        best = roads[0]
        tags = best.get("tags", {})
        hw   = tags.get("highway", "residential")
        road_info["highway_tag"]  = hw
        road_info["road_type"]    = OSM_HIGHWAY_TO_TYPE.get(hw, "local")
        road_info["surface"]      = tags.get("surface", "asphalt")
        road_info["lanes"]        = int(tags.get("lanes", 2))

        # Road age from OSM start_date
        start = tags.get("start_date", tags.get("construction_date", ""))
        if start and len(start) >= 4:
            try:
                build_year = int(start[:4])
                import datetime
                road_info["road_age_years"] = max(
                    datetime.datetime.now().year - build_year, 0)
                road_info["start_date"] = start
            except ValueError:
                pass

    if drains:
        road_info["has_drain_nearby"] = True
        road_info["drain_proximity_m"] = 30    # rough — Overpass doesn't give exact dist

    return road_info


# ── Weather feature extraction ─────────────────────────────────────────────────

def _extract_weather_features(
    owm_current: Optional[dict],
    owm_forecast: Optional[dict],
) -> dict:
    """
    Extract formation-relevant weather features from OWM JSON responses.
    Falls back to Bengaluru annual averages if API unavailable.
    """
    if owm_current:
        rain_1h     = owm_current.get("rain", {}).get("1h", 0.0)
        rain_3h     = owm_current.get("rain", {}).get("3h", 0.0)
        temp_max    = owm_current.get("main", {}).get("temp_max", 28.0)
        temp_min    = owm_current.get("main", {}).get("temp_min", 18.0)
        humidity    = owm_current.get("main", {}).get("humidity", 70)
        description = owm_current.get("weather", [{}])[0].get("description", "clear")
    else:
        # Bengaluru annual defaults
        rain_1h, rain_3h = 0.0, 0.0
        temp_max, temp_min = 28.0, 17.0
        humidity = 65
        description = "unknown (API unavailable)"

    # Estimate monthly rainfall from current conditions + Bengaluru seasonality
    # This is a rough proxy — for production, use OWM historical API
    import datetime
    month = datetime.datetime.now().month
    bengaluru_monthly_avg = {
        1: 5, 2: 8, 3: 15, 4: 45, 5: 110,
        6: 100, 7: 120, 8: 140, 9: 195, 10: 185,
        11: 65, 12: 15,
    }
    monthly_rainfall_est = float(
        rain_3h * 10 * 30  # rough: 10 events/day × 30 days
        or bengaluru_monthly_avg.get(month, 60)
    )
    monthly_rainfall_est = min(monthly_rainfall_est, 600.0)

    # Consecutive wet days from forecast
    consecutive_wet_days = 0
    if owm_forecast:
        for entry in owm_forecast.get("list", []):
            has_rain = entry.get("rain", {}).get("3h", 0) > 0.5
            if has_rain:
                consecutive_wet_days += 1
            else:
                break
        consecutive_wet_days = min(consecutive_wet_days, 30)

    diurnal_range    = max(temp_max - temp_min, 0.0)
    rainfall_intensity = rain_1h   # mm/hr — the directly measured intensity

    return {
        "monthly_rainfall_mm":    round(monthly_rainfall_est, 1),
        "rainfall_intensity_mmhr": round(rainfall_intensity, 2),
        "temp_max_degC":          round(temp_max, 1),
        "temp_min_degC":          round(temp_min, 1),
        "diurnal_range_degC":     round(diurnal_range, 1),
        "humidity_pct":           int(humidity),
        "consecutive_wet_days":   consecutive_wet_days,
        "description":            description,
    }


# ── Factor score calculators ───────────────────────────────────────────────────

def _score_water_infiltration(weather: dict, road: dict) -> tuple[float, str]:
    """
    IRC:SP:50-1999 §4.2 — water infiltration through surface cracks.
    High when: heavy rainfall intensity + saturated base + poor surface.
    Returns (score 0–10, irc_reference)
    """
    score = 0.0

    # Monthly rainfall contribution (0–4 pts)
    rain_mm = weather["monthly_rainfall_mm"]
    if rain_mm > 300:   score += 4.0
    elif rain_mm > 150: score += 2.5
    elif rain_mm > 60:  score += 1.5
    else:               score += 0.5

    # Rainfall intensity — immediate infiltration risk (0–3 pts)
    intensity = weather["rainfall_intensity_mmhr"]
    if intensity > IRC_THRESHOLDS["rainfall_intensity_high_mmhr"]:
        score += 3.0
    elif intensity > 10:
        score += 2.0
    elif intensity > 2:
        score += 1.0

    # Consecutive wet days — base saturation (0–2 pts)
    cwd = weather["consecutive_wet_days"]
    score += min(cwd / IRC_THRESHOLDS["consecutive_wet_days_risk"] * 2.0, 2.0)

    # Surface type penalty (0–1 pt): unpaved = full point
    if road.get("surface") in ("unpaved", "dirt", "gravel", "compacted"):
        score += 1.0
    elif road.get("surface") in ("asphalt", "paved"):
        score += 0.0

    return round(min(score, 10.0), 2), "IRC:SP:50-1999 §4.2"


def _score_traffic_loading(road: dict) -> tuple[float, str]:
    """
    IRC:37-2018 ESAL model — traffic fatigue loading.
    Derived from road type → CVPD estimate → ESALs.
    Returns (score 0–10, irc_reference)
    """
    road_type = road.get("road_type", "local")
    cvpd_est  = ROAD_TYPE_CVPD_EST.get(road_type, 150)

    # IRC:37 traffic classification thresholds
    if cvpd_est >= 1500:   score = 10.0
    elif cvpd_est >= 450:  score = 7.5
    elif cvpd_est >= 150:  score = 5.0
    elif cvpd_est >= 50:   score = 2.5
    else:                  score = 1.0

    # Lane count boost: more lanes → higher traffic density per road section
    lanes = road.get("lanes", 2)
    if lanes >= 6:   score = min(score + 1.5, 10.0)
    elif lanes >= 4: score = min(score + 0.8, 10.0)

    return round(score, 2), "IRC:37-2018 Table 2 (ESAL)"


def _score_thermal_stress(weather: dict) -> tuple[float, str]:
    """
    Thermal cracking from diurnal temperature cycling.
    IRC:37-2018 §6.3 — critical at >15°C diurnal range.
    Bengaluru doesn't freeze, so this is a minor contributor.
    Returns (score 0–10, irc_reference)
    """
    diurnal = weather["diurnal_range_degC"]
    # 0–10 linearly, saturating at 25°C range
    score = min(diurnal / IRC_THRESHOLDS["temp_diurnal_crack_risk_degC"] * 6.0, 10.0)
    return round(score, 2), "IRC:37-2018 §6.3"


def _score_road_age(road: dict) -> tuple[float, str]:
    """
    Pavement design life exhaustion per IRC:37-2018 §5.1.
    Design life = 10–15 years; after year 8, fatigue resistance drops sharply.
    Returns (score 0–10, irc_reference)
    """
    age = road.get("road_age_years", 7)

    if age > 15:      score = 9.5
    elif age > 10:    score = 7.5
    elif age > IRC_THRESHOLDS["pavement_high_risk_years"]: score = 5.5
    elif age > 5:     score = 3.5
    elif age > 2:     score = 1.5
    else:             score = 0.5

    return round(score, 2), "IRC:37-2018 §5.1 (design life)"


def _score_drainage_failure(road: dict, weather: dict) -> tuple[float, str]:
    """
    IRC:SP:42-1994 drainage standard — blocked drains + rainfall → ponding.
    Bengaluru: BBMP blocked drains are the #1 precursor to pothole clusters.
    Returns (score 0–10, irc_reference)
    """
    score = 3.0  # base — most Bengaluru ward roads have drainage issues

    # Drain proximity
    if road.get("has_drain_nearby"):
        # Drain present but may be blocked in heavy rain
        if weather["monthly_rainfall_mm"] > 150:
            score += 3.0   # drain likely overwhelmed
        else:
            score += 1.0   # drain present, manageable
    else:
        # No drain → guaranteed ponding in rain
        score += 4.0

    # High rainfall × no drain = maximum drainage failure
    if weather["monthly_rainfall_mm"] > 200 and not road.get("has_drain_nearby"):
        score = min(score + 2.0, 10.0)

    # Humidity: high humidity → prolonged saturation
    if weather["humidity_pct"] > 85:
        score = min(score + 0.5, 10.0)

    return round(min(score, 10.0), 2), "IRC:SP:42-1994 §3.2"


# ── IRC threshold crossing detector ───────────────────────────────────────────

def _check_irc_thresholds(weather: dict, road: dict) -> list[str]:
    """Return list of IRC threshold violation labels for the report."""
    crossed = []

    if weather["monthly_rainfall_mm"] > IRC_THRESHOLDS["annual_rainfall_high_mm"] / 12:
        crossed.append("monthly_rainfall_high")
    if weather["rainfall_intensity_mmhr"] > IRC_THRESHOLDS["rainfall_intensity_high_mmhr"]:
        crossed.append("rainfall_intensity_critical")
    if weather["consecutive_wet_days"] >= IRC_THRESHOLDS["consecutive_wet_days_risk"]:
        crossed.append("base_saturation_risk")
    if weather["diurnal_range_degC"] >= IRC_THRESHOLDS["temp_diurnal_crack_risk_degC"]:
        crossed.append("thermal_crack_risk")
    if road["road_age_years"] >= IRC_THRESHOLDS["pavement_high_risk_years"]:
        crossed.append("design_life_high_risk")
    if road["road_age_years"] >= IRC_THRESHOLDS["pavement_design_life_years"]:
        crossed.append("design_life_exceeded")

    return crossed


# ── SHAP-style explanation builder ────────────────────────────────────────────

def _build_shap_explanation(factor_scores: dict, plain_labels: dict) -> list[dict]:
    """
    Build a sorted explanation list for the dashboard bar chart.
    Mimics SHAP contribution = score × weight for each factor.
    """
    items = []
    for factor, score in factor_scores.items():
        weight       = FACTOR_WEIGHTS[factor]
        contribution = round(score * weight, 3)
        items.append({
            "factor":       factor,
            "label":        plain_labels[factor]["label"],
            "score":        score,
            "weight":       weight,
            "contribution": contribution,
            "irc_ref":      plain_labels[factor]["irc_ref"],
        })
    items.sort(key=lambda x: x["contribution"], reverse=True)
    return items


# ── Main public function ───────────────────────────────────────────────────────

def analyse_formation_from_image(
    image_path: Optional[str],
    lat: Optional[float],
    lng: Optional[float],
    api_key: Optional[str] = None,
) -> dict:
    """
    Main entry point called from app3.py /formation endpoint.

    Parameters
    ----------
    image_path  : path to temp image file (currently unused in scoring,
                  reserved for future CNN-based surface analysis)
    lat, lng    : GPS coordinates of the pothole
    api_key     : OpenWeatherMap API key (WEATHER_API_KEY env var)

    Returns
    -------
    dict with full formation analysis result (see module docstring)
    """
    # ── 0. Validate coordinates ───────────────────────────────────────────────
    if lat is None or lng is None:
        # Default to centre of Bengaluru for demo
        lat, lng = 12.9716, 77.5946
        logger.warning("No GPS coords provided — using Bengaluru centre")

    data_sources = []
    api_success  = {"weather": False, "osm": False}

    # ── 1. Weather data ───────────────────────────────────────────────────────
    owm_current = owm_forecast = None
    if api_key:
        owm_current  = _owm_current(lat, lng, api_key)
        owm_forecast = _owm_forecast(lat, lng, api_key)
        if owm_current:
            data_sources.append("OpenWeatherMap")
            api_success["weather"] = True
    else:
        logger.warning("No OWM API key — using Bengaluru seasonal defaults")

    weather = _extract_weather_features(owm_current, owm_forecast)

    # ── 2. OSM road data ──────────────────────────────────────────────────────
    osm_elements = _osm_road_data(lat, lng)
    if osm_elements is not None:
        road = _parse_osm_elements(osm_elements, lat, lng)
        data_sources.append("OSM Overpass")
        api_success["osm"] = True
    else:
        # Defaults for Bengaluru if OSM unavailable
        road = {
            "highway_tag": "residential", "road_type": "local",
            "surface": "asphalt", "start_date": None,
            "road_age_years": 7, "drain_proximity_m": 999,
            "has_drain_nearby": False, "lanes": 2,
        }

    data_sources.append("IRC:37-2018")

    # ── 3. Compute factor scores ──────────────────────────────────────────────
    water_score,  water_ref  = _score_water_infiltration(weather, road)
    traffic_score, traf_ref  = _score_traffic_loading(road)
    thermal_score, therm_ref = _score_thermal_stress(weather)
    age_score,    age_ref    = _score_road_age(road)
    drain_score,  drain_ref  = _score_drainage_failure(road, weather)

    factor_scores = {
        "water_infiltration": water_score,
        "traffic_loading":    traffic_score,
        "drainage_failure":   drain_score,
        "road_age_pavement":  age_score,
        "thermal_stress":     thermal_score,
    }

    # ── 4. Weighted composite risk score ─────────────────────────────────────
    risk_score = sum(
        factor_scores[f] * FACTOR_WEIGHTS[f]
        for f in FACTOR_WEIGHTS
    )
    risk_score = round(min(risk_score, 10.0), 2)

    # ── 5. Risk level ─────────────────────────────────────────────────────────
    if risk_score >= 7.5:   risk_level = "CRITICAL"
    elif risk_score >= 5.5: risk_level = "HIGH"
    elif risk_score >= 3.5: risk_level = "MEDIUM"
    else:                   risk_level = "LOW"

    # ── 6. Dominant factor ────────────────────────────────────────────────────
    # Weighted contribution (score × weight) determines dominance
    contributions = {f: factor_scores[f] * FACTOR_WEIGHTS[f] for f in factor_scores}
    dominant_factor = max(contributions, key=contributions.get)

    # ── 7. IRC threshold crossings ────────────────────────────────────────────
    irc_crossed = _check_irc_thresholds(weather, road)

    # ── 8. SHAP-style explanation ─────────────────────────────────────────────
    plain_labels = {
        "water_infiltration": {
            "label":   "Heavy rainfall / water infiltration",
            "irc_ref": "IRC:SP:50-1999 §4.2",
        },
        "traffic_loading": {
            "label":   "High traffic axle loading (ESAL)",
            "irc_ref": "IRC:37-2018 Table 2",
        },
        "drainage_failure": {
            "label":   "Blocked or absent drainage",
            "irc_ref": "IRC:SP:42-1994 §3.2",
        },
        "road_age_pavement": {
            "label":   "Pavement age / design life exceeded",
            "irc_ref": "IRC:37-2018 §5.1",
        },
        "thermal_stress": {
            "label":   "Thermal cycling (day–night temp swing)",
            "irc_ref": "IRC:37-2018 §6.3",
        },
    }
    shap_explanation = _build_shap_explanation(factor_scores, plain_labels)

    # ── 9. Data confidence ────────────────────────────────────────────────────
    if api_success["weather"] and api_success["osm"]:
        confidence = "high"
    elif api_success["weather"] or api_success["osm"]:
        confidence = "medium"
    else:
        confidence = "low (defaults used — provide API key and check connectivity)"

    # ── 10. Assemble result ───────────────────────────────────────────────────
    return {
        # Core output
        "risk_score":     risk_score,
        "risk_level":     risk_level,
        "dominant_factor": dominant_factor,

        # Detailed scores
        "factor_scores":  factor_scores,
        "factor_weights": FACTOR_WEIGHTS,

        # SHAP-style ranked explanation (for dashboard bar chart)
        "shap_explanation": shap_explanation,

        # Raw input snapshots (for transparency / dashboard display)
        "weather_snapshot": weather,
        "road_data": {
            "road_type":       road["road_type"],
            "highway_tag":     road["highway_tag"],
            "surface":         road["surface"],
            "road_age_years":  road["road_age_years"],
            "start_date":      road["start_date"],
            "has_drain_nearby": road["has_drain_nearby"],
            "drain_proximity_m": road["drain_proximity_m"],
            "lanes":           road["lanes"],
        },

        # Academic/presentation layer
        "irc_thresholds_crossed": irc_crossed,
        "data_sources":           data_sources,
        "confidence":             confidence,
        "location":               {"lat": lat, "lng": lng},
    }


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    api_key = os.getenv("WEATHER_API_KEY", "")

    print("=" * 60)
    print("  Formation Analysis — quick test (Indiranagar, Bengaluru)")
    print("=" * 60)

    result = analyse_formation_from_image(
        image_path=None,
        lat=12.9716,
        lng=77.5946,
        api_key=api_key or None,
    )

    print(f"\n  Risk Score  : {result['risk_score']}/10")
    print(f"  Risk Level  : {result['risk_level']}")
    print(f"  Dominant    : {result['dominant_factor']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"\n  Factor breakdown:")
    for item in result["shap_explanation"]:
        bar = "█" * int(item["score"])
        print(f"    {item['label'][:38]:<38} {bar} {item['score']:.1f}  "
              f"(contrib: {item['contribution']:.2f}) [{item['irc_ref']}]")

    print(f"\n  IRC thresholds crossed: {result['irc_thresholds_crossed'] or 'none'}")
    print(f"  Data sources used:      {result['data_sources']}")
    print(f"\n  Weather snapshot: {json.dumps(result['weather_snapshot'], indent=4)}")
    print(f"  Road data:        {json.dumps(result['road_data'], indent=4)}")
