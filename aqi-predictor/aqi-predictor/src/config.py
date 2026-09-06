"""
Central configuration for the AQI Predictor project.
Change CITY_NAME / LAT / LON to point the whole pipeline at a different city.
"""
import os

# ---- Location -------------------------------------------------------------
CITY_NAME = os.getenv("AQI_CITY_NAME", "Rawalpindi")
LATITUDE = float(os.getenv("AQI_LAT", "33.6007"))
LONGITUDE = float(os.getenv("AQI_LON", "73.0679"))
TIMEZONE = os.getenv("AQI_TZ", "auto")

# ---- APIs (Open-Meteo: free, no API key required) --------------------------
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo limits
MAX_PAST_DAYS = 92        # max historical days the air-quality API can return
MAX_FORECAST_DAYS = 7     # max forecast days the API can return

# ---- Paths (act as our local "Feature Store" and "Model Registry") --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_CACHE_PATH = os.path.join(DATA_DIR, "raw_hourly_cache.csv")
FEATURE_STORE_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
SHAP_SUMMARY_PATH = os.path.join(REPORTS_DIR, "shap_summary.png")

# ---- Forecast / modelling settings -----------------------------------------
HORIZONS_HOURS = [24, 48, 72]     # predict AQI 1, 2, and 3 days ahead
LAG_HOURS = [24, 48, 72]          # lag features to build from the AQI series
ROLLING_WINDOW_HOURS = 72

# AQI (US EPA scale) hazard thresholds used for the dashboard alert banner
AQI_CATEGORIES = [
    (0, 50, "Good", "#2ecc71"),
    (51, 100, "Moderate", "#f1c40f"),
    (101, 150, "Unhealthy for Sensitive Groups", "#e67e22"),
    (151, 200, "Unhealthy", "#e74c3c"),
    (201, 300, "Very Unhealthy", "#8e44ad"),
    (301, 500, "Hazardous", "#7f2b41"),
]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
