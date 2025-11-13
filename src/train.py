from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
	"""
	Load the California Housing dataset as a pandas DataFrame and target Series.
	Returns X (features), y (target), and the list of feature names.
	"""
	data_bundle = fetch_california_housing(as_frame=True)
	feature_names = [name for name in data_bundle.feature_names]
	X = data_bundle.frame[feature_names]
	y = data_bundle.target
	return X, y, feature_names


def build_models() -> Dict[str, Pipeline]:
	"""
	Build candidate regression models as sklearn Pipelines.
	"""
	models: Dict[str, Pipeline] = {
		"LinearRegression": Pipeline(
			steps=[
				("scale", StandardScaler()),
				("regressor", LinearRegression()),
			]
		),
		"RandomForestRegressor": Pipeline(
			steps=[
				("regressor", RandomForestRegressor(
					n_estimators=300,
					random_state=42,
					n_jobs=-1,
				)),
			]
		),
	}
	return models


def evaluate_model(
	model: Pipeline,
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_test: pd.Series,
) -> Dict[str, float]:
	"""
	Fit model, predict, and compute evaluation metrics.
	Returns a dictionary with rmse, mae, r2.
	"""
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
	mae = float(mean_absolute_error(y_test, y_pred))
	r2 = float(r2_score(y_test, y_pred))
	return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_residual_quantiles(y_true: pd.Series, y_pred: np.ndarray, percentiles: List[int]) -> Dict[str, float]:
	"""
	Compute residual quantiles for given integer percentiles [1..99].
	Residuals are defined as (y_true - y_pred).
	Returns a dict like { 'p10': value_at_10th_percentile, ... }.
	"""
	residuals = (np.asarray(y_true) - np.asarray(y_pred)).astype(float)
	quantile_values = np.percentile(residuals, percentiles).astype(float)
	return {f"p{p}": float(v) for p, v in zip(percentiles, quantile_values)}


def save_artifacts(
	best_name: str,
	best_pipeline: Pipeline,
	metrics_by_model: Dict[str, Dict[str, float]],
	feature_names: List[str],
	residual_quantiles: Dict[str, float] | None = None,
) -> None:
	"""
	Save the trained pipeline and metadata (metrics and feature names).
	"""
	project_root = Path(__file__).resolve().parents[1]
	models_dir = project_root / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	model_path = models_dir / "best_model.joblib"
	meta_path = models_dir / "metadata.json"

	joblib.dump(best_pipeline, model_path)

	metadata = {
		"saved_at": datetime.now(timezone.utc).isoformat(),
		"best_model": best_name,
		"metrics_by_model": metrics_by_model,
		"feature_names": feature_names,
		"dataset": "sklearn.datasets.fetch_california_housing",
		"residual_quantiles": residual_quantiles or {},
		"target_units": "hundred_thousands_usd",
	}
	with meta_path.open("w", encoding="utf-8") as fp:
		json.dump(metadata, fp, indent=2)


def main() -> None:
	X, y, feature_names = load_dataset()
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	models = build_models()
	metrics_by_model: Dict[str, Dict[str, float]] = {}
	scores_for_selection: List[Tuple[str, float]] = []

	for name, pipeline in models.items():
		metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
		metrics_by_model[name] = metrics
		scores_for_selection.append((name, metrics["rmse"]))
		print(f"{name}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

	# Select the model with the lowest RMSE
	best_name, _ = min(scores_for_selection, key=lambda pair: pair[1])
	best_pipeline = models[best_name]

	# Compute residual quantiles on the held-out test set for the best model (before refitting on all data)
	# We store p1..p99 so that arbitrary confidence levels (e.g., 78%) can be requested later.
	percentiles = list(range(1, 100))
	y_test_pred_for_best = best_pipeline.predict(X_test)
	residual_quantiles = compute_residual_quantiles(y_test, y_test_pred_for_best, percentiles)

	# Refit on all data for final model
	best_pipeline.fit(X, y)
	save_artifacts(best_name, best_pipeline, metrics_by_model, feature_names, residual_quantiles=residual_quantiles)
	print(f"\nBest model: {best_name}")
	print("Saved pipeline to 'house_price_ml/models/best_model.joblib' and metadata to 'house_price_ml/models/metadata.json'.")


if __name__ == "__main__":
	main()

