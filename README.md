# Pearls AQI Predictor

End-to-end, 100% serverless Air Quality Index (AQI) forecasting system.
Predicts the US AQI for the next **1, 2, and 3 days** for a given city, using
live weather + pollutant data, automated pipelines, and an interactive dashboard.

**No API key / signup is required anywhere in this project** — it uses the
free, public [Open-Meteo](https://open-meteo.com/) Air Quality and Weather
APIs instead of AQICN/OpenWeather, which need account approval.

## Architecture

```
Open-Meteo APIs  ──raw data──▶  feature_pipeline.py  ──features──▶  data/features.csv
 (weather + AQ)                  (runs hourly via                   (local "Feature Store")
                                   GitHub Actions)                          │
                                                                            ▼
                                                              train_pipeline.py
                                                          (runs daily via GitHub Actions)
                                                                            │
                                                            models/best_model.pkl
                                                             (local "Model Registry")
                                                                            │
                                                                            ▼
                                                                  app.py (Streamlit)
                                                          live dashboard + 3-day forecast
```

- **Feature Store / Model Registry**: implemented as versioned, deduplicated
  CSV/pickle files committed back to the repo by GitHub Actions. This is a
  deliberate, pragmatic substitute for Hopsworks/Vertex AI free tiers (which
  require account setup) — swapping in Hopsworks later only requires
  implementing `_maybe_push_to_hopsworks()` in `src/feature_pipeline.py`,
  which is already stubbed out and auto-activates if you set a
  `HOPSWORKS_API_KEY` secret.
- **Automation**: GitHub Actions (`.github/workflows/`) — no Airflow server needed.
- **Models**: Ridge Regression, Random Forest, Gradient Boosting (scikit-learn),
  evaluated with RMSE / MAE / R², best one auto-selected.
- **Explainability**: SHAP summary plot generated automatically after training.
- **Alerts**: dashboard shows a red/orange/green hazard banner based on
  predicted AQI (EPA categories).

## Project structure

```
aqi-predictor/
├── src/
│   ├── config.py            # city, paths, thresholds — edit here to change city
│   ├── feature_pipeline.py  # fetch + engineer features, write to feature store
│   ├── train_pipeline.py    # train/evaluate models, save best + SHAP plot
│   └── predict.py           # loads model, produces 3-day forecast
├── app.py                   # Streamlit dashboard
├── .github/workflows/
│   ├── hourly_features.yml  # runs feature_pipeline.py every hour
│   └── daily_training.yml   # runs train_pipeline.py every day
├── data/features.csv        # feature store (created after first run)
├── models/best_model.pkl    # model registry (created after training)
└── requirements.txt
```

## Run it locally (5-minute setup)

```bash
git clone <your-repo-url>
cd aqi-predictor
pip install -r requirements.txt

# 1. Backfill ~92 days of historical data to build a training set
python src/feature_pipeline.py --backfill

# 2. Train the models
python src/train_pipeline.py

# 3. Launch the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

To point the project at a different city, edit `src/config.py`
(`CITY_NAME`, `LATITUDE`, `LONGITUDE`) or set env vars `AQI_CITY_NAME`,
`AQI_LAT`, `AQI_LON` before running.

## Automating it (already configured)

Once pushed to GitHub, two workflows run automatically (also triggerable
manually from the **Actions** tab):

| Workflow | Schedule | What it does |
|---|---|---|
| `hourly_features.yml` | every hour | fetches new data, updates `data/features.csv`, commits it back |
| `daily_training.yml` | daily at 02:00 UTC | retrains models on the latest feature store, commits `models/` + SHAP plot back |

No secrets need to be configured — the default `GITHUB_TOKEN` (auto-provided
by Actions) has permission to commit back to the repo since `permissions:
contents: write` is set in each workflow file.

## Deploying the dashboard (optional, for a live demo link)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app" → select this repo → set main file to `app.py` → Deploy.
4. You'll get a public URL you can include in your submission alongside the repo link.

## How this satisfies the project brief

| Requirement | Implementation |
|---|---|
| Fetch raw weather + pollutant data | `feature_pipeline.fetch_raw_data()` (Open-Meteo, free) |
| Time-based + derived features (AQI change rate) | `add_time_features`, `add_lag_and_rolling_features` |
| Feature store | `data/features.csv`, versioned via git; Hopsworks-ready hook included |
| Historical backfill | `python src/feature_pipeline.py --backfill` |
| Train/evaluate multiple models (RMSE/MAE/R²) | `train_pipeline.py` — Ridge, Random Forest, Gradient Boosting |
| Model registry | `models/best_model.pkl` + `models/metrics.json` |
| Automated hourly/daily pipelines | GitHub Actions workflows |
| Web dashboard with forecast | `app.py` (Streamlit) |
| EDA | Historical AQI trend chart in dashboard |
| SHAP feature importance | Auto-generated `reports/shap_summary.png`, shown in dashboard |
| Hazard alerts | Color-coded banner in dashboard based on EPA AQI categories |

## Notes / known limitations

- Open-Meteo's air-quality API provides up to 92 days of history and 7 days
  of forecast weather — plenty for this use case, but if you need years of
  history for deeper modelling, you'd need to accumulate it yourself over
  time via the hourly pipeline (it appends without overwriting).
- Deep learning (LSTM/Transformer) models were intentionally left as a
  documented "future work" item in `report.md` rather than implemented,
  to prioritize a fully working, correctly evaluated classical-ML pipeline
  under the submission deadline — see report for the trade-off discussion.
