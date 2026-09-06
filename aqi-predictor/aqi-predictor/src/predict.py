"""
Prediction helper: loads the trained model from the Model Registry and the
latest data from the Feature Store / live API, and produces an AQI forecast
for the next 1, 2, and 3 days.
"""
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from feature_pipeline import fetch_raw_data, build_supervised_table


def load_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train_pipeline.py` first."
        )
    return joblib.load(config.MODEL_PATH)


def get_latest_forecast_rows() -> pd.DataFrame:
    """Pull fresh data and build the feature rows for the most recent
    timestamp we have complete lag/rolling data for, one per horizon."""
    raw = fetch_raw_data(past_days=10, forecast_days=config.MAX_FORECAST_DAYS)
    supervised = build_supervised_table(raw)

    feature_cols = [c for c in supervised.columns if c not in ("time", "target_time", "target_aqi")]
    usable = supervised.dropna(subset=feature_cols)

    latest_time = usable["time"].max()
    latest_rows = usable[usable["time"] == latest_time].copy()
    return latest_rows, raw


def get_forecast() -> pd.DataFrame:
    bundle = load_model()
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    latest_rows, raw_history = get_latest_forecast_rows()
    latest_rows = latest_rows.sort_values("horizon_hours")

    preds = model.predict(latest_rows[feature_cols])
    out = pd.DataFrame({
        "target_time": latest_rows["target_time"].values,
        "horizon_hours": latest_rows["horizon_hours"].values,
        "predicted_aqi": preds,
    })
    return out, raw_history


def categorize_aqi(value: float):
    for lo, hi, label, color in config.AQI_CATEGORIES:
        if lo <= value <= hi:
            return label, color
    return "Hazardous", config.AQI_CATEGORIES[-1][3]


if __name__ == "__main__":
    forecast, _ = get_forecast()
    print(forecast)
