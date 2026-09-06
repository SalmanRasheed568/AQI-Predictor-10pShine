"""
Feature Pipeline
================
1. Fetches raw weather + pollutant data from Open-Meteo (free, no API key).
2. Computes model input features (time-based, lag, rolling, weather) and
   the training targets (AQI N hours ahead).
3. Stores/updates the resulting rows in the local "Feature Store"
   (data/features.csv). Also pushes to Hopsworks if HOPSWORKS_API_KEY is set
   (optional — the pipeline works perfectly without it).

Usage:
    python src/feature_pipeline.py                # normal run (recent data)
    python src/feature_pipeline.py --backfill      # full 92-day historical backfill
"""
import argparse
import sys
import os

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


# -----------------------------------------------------------------------
# Raw data fetchers
# -----------------------------------------------------------------------
def fetch_air_quality(past_days: int, forecast_days: int) -> pd.DataFrame:
    """Fetch hourly pollutant + US AQI data from Open-Meteo air quality API."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": config.TIMEZONE,
    }
    resp = requests.get(config.AIR_QUALITY_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather(past_days: int, forecast_days: int) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo weather API."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": config.TIMEZONE,
    }
    resp = requests.get(config.WEATHER_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_raw_data(past_days: int, forecast_days: int) -> pd.DataFrame:
    """Merge air quality + weather into a single hourly dataframe."""
    aq = fetch_air_quality(past_days, forecast_days)
    wx = fetch_weather(past_days, forecast_days)
    df = pd.merge(aq, wx, on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """All lag/rolling features use only PAST values relative to each row's
    own timestamp -> no leakage."""
    df = df.copy()
    for lag in config.LAG_HOURS:
        df[f"us_aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)

    df["us_aqi_rolling_mean"] = (
        df["us_aqi"].shift(1).rolling(config.ROLLING_WINDOW_HOURS, min_periods=6).mean()
    )
    # AQI change rate: how fast AQI has been changing over the last 24h
    df["aqi_change_rate_24h"] = (df["us_aqi"].shift(1) - df["us_aqi"].shift(25)) / 24.0
    return df


def build_supervised_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand into a long "horizon" table: each timestamp t produces one row
    per forecast horizon (24h/48h/72h ahead), with the target being the
    US AQI observed at t + horizon.
    """
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)

    feature_cols = [
        "hour", "day", "day_of_week", "month",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "surface_pressure", "precipitation",
        "us_aqi_rolling_mean", "aqi_change_rate_24h",
    ] + [f"us_aqi_lag_{lag}h" for lag in config.LAG_HOURS] \
      + [f"pm2_5_lag_{lag}h" for lag in config.LAG_HOURS]

    rows = []
    for horizon in config.HORIZONS_HOURS:
        sub = df.copy()
        sub["horizon_hours"] = horizon
        sub["target_aqi"] = sub["us_aqi"].shift(-horizon)
        sub["target_time"] = sub["time"] + pd.Timedelta(hours=horizon)
        rows.append(sub[["time", "target_time", "horizon_hours"] + feature_cols + ["target_aqi"]])

    long_df = pd.concat(rows, ignore_index=True)
    return long_df


# -----------------------------------------------------------------------
# Feature store (local CSV; optionally also pushed to Hopsworks)
# -----------------------------------------------------------------------
def save_to_feature_store(new_rows: pd.DataFrame) -> None:
    if os.path.exists(config.FEATURE_STORE_PATH):
        existing = pd.read_csv(config.FEATURE_STORE_PATH, parse_dates=["time", "target_time"])
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time", "horizon_hours"], keep="last")
    else:
        combined = new_rows
    combined = combined.sort_values(["time", "horizon_hours"]).reset_index(drop=True)
    combined.to_csv(config.FEATURE_STORE_PATH, index=False)
    print(f"[feature_store] Saved {len(combined)} total rows -> {config.FEATURE_STORE_PATH}")

    _maybe_push_to_hopsworks(combined)


def _maybe_push_to_hopsworks(df: pd.DataFrame) -> None:
    """Optional: if HOPSWORKS_API_KEY env var is set, also write to Hopsworks
    Feature Store. Wrapped in try/except so a missing/invalid key never
    breaks the pipeline (the local CSV store above is always authoritative)."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        return
    try:
        import hopsworks  # noqa: F401  (only imported if key is present)
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["time", "horizon_hours"],
            event_time="time",
            description="AQI forecasting features",
        )
        fg.insert(df)
        print("[feature_store] Also pushed to Hopsworks feature store.")
    except Exception as e:  # pragma: no cover
        print(f"[feature_store] Hopsworks push skipped/failed ({e}). Local CSV store is still up to date.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def run(backfill: bool = False) -> pd.DataFrame:
    past_days = config.MAX_PAST_DAYS if backfill else 5
    forecast_days = config.MAX_FORECAST_DAYS

    print(f"[feature_pipeline] Fetching raw data for {config.CITY_NAME} "
          f"(past_days={past_days}, forecast_days={forecast_days})...")
    raw = fetch_raw_data(past_days=past_days, forecast_days=forecast_days)
    raw.to_csv(config.RAW_CACHE_PATH, index=False)

    print("[feature_pipeline] Computing features + targets...")
    supervised = build_supervised_table(raw)

    # Keep rows that have complete features (targets may legitimately be NaN
    # for the most recent rows where the future hasn't happened yet — those
    # rows are still useful later for *inference*).
    feature_cols = [c for c in supervised.columns if c not in ("time", "target_time", "target_aqi")]
    supervised = supervised.dropna(subset=feature_cols)

    save_to_feature_store(supervised)
    return supervised


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true",
                         help="Fetch full 92-day history to build a training dataset")
    args = parser.parse_args()
    run(backfill=args.backfill)
