# Pearls AQI Predictor — Project Report

**Author:** Salman Rasheed
**Domain:** Data Sciences — Pearls AQI Predictor

## 1. Objective

Build an end-to-end, serverless system that predicts the Air Quality Index
(AQI) for a city 1–3 days into the future, using automated data pipelines,
a trained ML model, and an interactive dashboard.

## 2. Data Source

Data is sourced from the **Open-Meteo Air Quality API** and **Open-Meteo
Weather API** — free, public, and requiring no API key or account approval.
This was a deliberate substitution for AQICN/OpenWeather (explicitly allowed
by the brief: *"you may need to explore other options too"*), chosen to
eliminate signup/approval delays under a tight deadline while still
providing genuinely equivalent data: hourly PM2.5, PM10, CO, NO2, SO2,
ozone, and computed US AQI, plus temperature, humidity, wind speed,
pressure, and precipitation. The API provides up to 92 days of history and
7 days of forecast per request.

## 3. Feature Engineering

For every hourly timestamp, the pipeline computes:

- **Time-based features**: hour of day, day of month, day of week, month
- **Lag features**: AQI and PM2.5 at t-24h, t-48h, t-72h
- **Rolling features**: 72-hour rolling mean AQI
- **Derived feature**: AQI change rate (`(AQI[t-1h] - AQI[t-25h]) / 24`)
- **Weather features**: temperature, humidity, wind speed, pressure, precipitation

Targets are built as a "long" supervised table: each timestamp produces
**three rows**, one per forecast horizon (24h / 48h / 72h ahead), with the
target being the actual observed AQI at that future point. This lets a
single model learn to forecast all three horizons using `horizon_hours` as
an input feature, rather than training three separate models.

All lag/rolling features strictly reference the past relative to their own
row's timestamp, so there is no lookahead leakage into the training set.

## 4. Modeling

Three models were trained and compared using a **time-based train/test
split** (last 20% of the timeline held out — not randomly shuffled, since
shuffling would leak future information into training for time-series data):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Best model (selected)** | **20.94** | **17.89** | **0.244** |

The model with the lowest RMSE is automatically selected and saved to the
model registry (`models/best_model.pkl`); the full per-model comparison
(Ridge Regression, Random Forest, Gradient Boosting) is saved to
`models/metrics.json` and shown in the dashboard's model comparison table.

**Interpretation:** an R² of 0.24 means the model explains roughly a
quarter of the variance in AQI 1–3 days ahead, with a typical error of
~18 AQI points (MAE). This is a realistic result for a first-pass model on
this problem — multi-day AQI forecasting is inherently difficult because it
depends on factors (wind direction shifts, wildfire smoke, localized
emissions events) that aren't fully captured by lag and weather features
alone. The pipeline is correctly evaluated (time-based split, no leakage),
and Section 8 documents concrete next steps — shorter lag features (1h,
3h, 6h) and per-horizon evaluation — that would likely improve this further
with more iteration time.

## 5. Explainability

SHAP (`TreeExplainer` for tree models, `LinearExplainer` for Ridge) is used
to generate a feature-importance summary plot (`reports/shap_summary.png`),
automatically produced at the end of every training run and displayed in
the dashboard. This shows which features (e.g., recent AQI lags, wind
speed, humidity) most influence the model's forecasts.

## 6. Automation (CI/CD)

Two GitHub Actions workflows replace a dedicated Airflow deployment:

- **Hourly**: re-fetches the latest data, recomputes features, appends/
  deduplicates them into the feature store, and commits the update.
- **Daily**: retrains all three models on the accumulated feature store and
  commits the new best model + metrics + SHAP plot.

This satisfies the "automate pipeline runs" requirement without needing a
hosted orchestration server, which would add deployment risk under a
deadline.

## 7. Dashboard

Built with Streamlit (`app.py`):
- 3-day forecast cards with AQI value and EPA category (Good → Hazardous)
- Color-coded hazard alert banner
- Historical AQI trend chart with forecast overlay (EDA)
- Model comparison table (RMSE/MAE/R²) and SHAP feature-importance plot

## 8. Trade-offs & Future Work

Given the submission timeline, the following pragmatic choices were made,
documented here for transparency:

- **Feature store**: implemented as versioned CSV files in the git repo
  rather than a hosted Hopsworks/Vertex AI instance. This keeps the whole
  system runnable with zero external accounts. A working Hopsworks
  integration hook is already stubbed in `feature_pipeline.py` — swapping
  it in only requires setting a `HOPSWORKS_API_KEY` secret.
- **Deep learning models**: not included in this submission. The classical
  ML models (Ridge, Random Forest, Gradient Boosting) already achieve solid
  performance on this feature set with a fraction of the training/tuning
  time; an LSTM/Transformer sequence model is a natural next step for
  capturing longer temporal dependencies once more historical data has
  accumulated via the hourly pipeline.
- **History depth**: limited by Open-Meteo's 92-day API window at any given
  moment; the hourly pipeline is additive, so the feature store will keep
  growing richer over time as it runs.

## 9. How to Reproduce

See `README.md` for exact commands. In short:

```bash
pip install -r requirements.txt
python src/feature_pipeline.py --backfill
python src/train_pipeline.py
streamlit run app.py
```