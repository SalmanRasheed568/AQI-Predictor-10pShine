import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config
from predict import get_forecast, categorize_aqi

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

st.title(f"🌫️ Air Quality Index Forecast — {config.CITY_NAME}")
st.caption(
    "Predicts the US AQI for the next 3 days from live weather + pollutant data. "
    "100% serverless data source (Open-Meteo), scikit-learn model, GitHub Actions automation."
)

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
try:
    forecast_df, raw_history = get_forecast()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Run these two commands locally first, then reload this app:\n\n"
        "```\npython src/feature_pipeline.py --backfill\npython src/train_pipeline.py\n```"
    )
    st.stop()

col1, col2, col3 = st.columns(3)
for col, (_, row) in zip([col1, col2, col3], forecast_df.iterrows()):
    label, color = categorize_aqi(row["predicted_aqi"])
    day_num = int(row["horizon_hours"] // 24)
    with col:
        st.metric(f"Day +{day_num}", f"{row['predicted_aqi']:.0f} AQI")
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px 10px;"
            f"border-radius:6px;font-weight:600;'>{label}</span>",
            unsafe_allow_html=True,
        )

# Hazard alert banner
max_pred = forecast_df["predicted_aqi"].max()
if max_pred >= 151:
    st.error(f"⚠️ HAZARD ALERT: Predicted AQI reaches {max_pred:.0f} — Unhealthy or worse. "
              "Consider limiting outdoor exposure.")
elif max_pred >= 101:
    st.warning(f"⚠️ Predicted AQI reaches {max_pred:.0f} — Unhealthy for sensitive groups.")
else:
    st.success(f"Air quality looks good over the next 3 days (max predicted AQI: {max_pred:.0f}).")

st.divider()

# ---------------------------------------------------------------------------
# Historical trend (EDA) + forecast on one chart
# ---------------------------------------------------------------------------
st.subheader("Historical AQI trend + forecast")
hist = raw_history[["time", "us_aqi"]].dropna()

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["time"], y=hist["us_aqi"], mode="lines",
                          name="Observed AQI", line=dict(color="#3498db")))
fig.add_trace(go.Scatter(x=forecast_df["target_time"], y=forecast_df["predicted_aqi"],
                          mode="lines+markers", name="Forecast AQI",
                          line=dict(color="#e74c3c", dash="dash")))
fig.update_layout(xaxis_title="Time", yaxis_title="US AQI", height=420,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Model info + SHAP feature importance
# ---------------------------------------------------------------------------
st.subheader("Model performance & feature importance")
mcol1, mcol2 = st.columns([1, 1])

with mcol1:
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            metrics = json.load(f)
        st.write(f"**Best model:** `{metrics['best_model']}`")
        st.dataframe(pd.DataFrame(metrics["results"]).T.rename(
            columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"}
        ).style.format("{:.2f}"))
    else:
        st.info("No metrics yet — run `python src/train_pipeline.py`.")

with mcol2:
    if os.path.exists(config.SHAP_SUMMARY_PATH):
        st.image(config.SHAP_SUMMARY_PATH, caption="SHAP feature importance (top drivers of predicted AQI)")
    else:
        st.info("SHAP summary not generated yet — it's produced automatically by train_pipeline.py.")

st.divider()
st.caption(
    "Pipelines: feature pipeline runs hourly, training pipeline runs daily — both automated via "
    "GitHub Actions (see `.github/workflows/`). Data source: Open-Meteo Air Quality & Weather APIs (free, no key)."
)
