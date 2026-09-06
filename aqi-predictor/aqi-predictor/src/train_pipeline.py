"""
Training Pipeline
==================
1. Loads historical (features, target) rows from the Feature Store.
2. Trains and evaluates several models (Ridge Regression, Random Forest,
   Gradient Boosting) using a time-based train/test split (no shuffling,
   since this is time-series data).
3. Picks the best model by RMSE, saves it + metrics to the Model Registry
   (models/ directory), and produces a SHAP feature-importance summary plot.

Usage:
    python src/train_pipeline.py
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def load_training_data() -> pd.DataFrame:
    if not os.path.exists(config.FEATURE_STORE_PATH):
        raise FileNotFoundError(
            "Feature store not found. Run `python src/feature_pipeline.py --backfill` first."
        )
    df = pd.read_csv(config.FEATURE_STORE_PATH, parse_dates=["time", "target_time"])
    df = df.dropna(subset=["target_aqi"])  # only rows where we know the real outcome
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"time", "target_time", "target_aqi"}
    return [c for c in df.columns if c not in exclude]


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def run():
    print("[train_pipeline] Loading data from feature store...")
    df = load_training_data()
    feature_cols = get_feature_columns(df)

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} labeled rows available — run the backfill "
            "(`python src/feature_pipeline.py --backfill`) and wait a few hourly "
            "runs so targets 24-72h in the future become known."
        )

    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[feature_cols], train_df["target_aqi"]
    X_test, y_test = test_df[feature_cols], test_df["target_aqi"]

    candidates = {
        "ridge_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    fitted_models = {}
    for name, model in candidates.items():
        print(f"[train_pipeline] Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print(f"    -> RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = fitted_models[best_name]
    print(f"[train_pipeline] Best model: {best_name}")

    joblib.dump({"model": best_model, "model_name": best_name, "feature_cols": feature_cols},
                config.MODEL_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump({"results": results, "best_model": best_name,
                    "n_train": len(train_df), "n_test": len(test_df)}, f, indent=2)
    print(f"[train_pipeline] Saved model -> {config.MODEL_PATH}")
    print(f"[train_pipeline] Saved metrics -> {config.METRICS_PATH}")

    _save_shap_summary(best_name, best_model, X_test, feature_cols)


def _save_shap_summary(model_name: str, model, X_test: pd.DataFrame, feature_cols: list):
    """Best-effort SHAP summary plot. Skipped gracefully if it fails —
    training must never fail because of an explainability step."""
    try:
        import shap
        sample = X_test.sample(min(200, len(X_test)), random_state=42)

        if model_name in ("random_forest", "gradient_boosting"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        else:
            underlying = model.named_steps["model"]
            scaled = model.named_steps["scaler"].transform(sample)
            explainer = shap.LinearExplainer(underlying, scaled)
            shap_values = explainer.shap_values(scaled)

        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(config.SHAP_SUMMARY_PATH, dpi=150)
        plt.close()
        print(f"[train_pipeline] Saved SHAP summary plot -> {config.SHAP_SUMMARY_PATH}")
    except Exception as e:
        print(f"[train_pipeline] SHAP plot skipped ({e}).")


if __name__ == "__main__":
    run()
