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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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


def build_models(preprocessor: ColumnTransformer | Pipeline | None = None) -> Dict[str, Pipeline]:
	"""
	Build candidate regression models as sklearn Pipelines.
	Optionally prepend a preprocessing transformer.
	"""
	prefix = []
	if preprocessor is not None:
		prefix = [("preprocess", preprocessor)]

	models: Dict[str, Pipeline] = {
		"LinearRegression": Pipeline(steps=[*prefix, ("regressor", LinearRegression())]),
		"RandomForestRegressor": Pipeline(steps=[*prefix, ("regressor", RandomForestRegressor(
			n_estimators=300,
			random_state=42,
			n_jobs=-1,
		))]),
		"GradientBoostingRegressor": Pipeline(steps=[*prefix, ("regressor", GradientBoostingRegressor(random_state=42))]),
	}
	return models


def tune_model(
	name: str,
	pipeline: Pipeline,
	X_train: pd.DataFrame,
	y_train: pd.Series,
) -> Tuple[Pipeline, Dict]:
	"""
	Tune the model using GridSearchCV where applicable. Returns best estimator and CV summary.
	"""
	if name == "RandomForestRegressor":
		param_grid = {
			"regressor__n_estimators": [200, 300, 500],
			"regressor__max_depth": [None, 12, 20],
			"regressor__min_samples_leaf": [1, 2, 4],
		}
		cv = KFold(n_splits=3, shuffle=True, random_state=42)
		search = GridSearchCV(
			estimator=pipeline,
			param_grid=param_grid,
			scoring="neg_root_mean_squared_error",
			cv=cv,
			n_jobs=-1,
			refit=True,
			verbose=0,
		)
		search.fit(X_train, y_train)
		best_estimator: Pipeline = search.best_estimator_
		cv_summary = {
			"best_params": search.best_params_,
			"best_score_neg_rmse": float(search.best_score_),
		}
		return best_estimator, cv_summary

	if name == "GradientBoostingRegressor":
		param_grid = {
			"regressor__n_estimators": [100, 200],
			"regressor__learning_rate": [0.05, 0.1],
			"regressor__max_depth": [2, 3],
		}
		cv = KFold(n_splits=3, shuffle=True, random_state=42)
		search = GridSearchCV(
			estimator=pipeline,
			param_grid=param_grid,
			scoring="neg_root_mean_squared_error",
			cv=cv,
			n_jobs=-1,
			refit=True,
			verbose=0,
		)
		search.fit(X_train, y_train)
		best_estimator: Pipeline = search.best_estimator_
		cv_summary = {
			"best_params": search.best_params_,
			"best_score_neg_rmse": float(search.best_score_),
		}
		return best_estimator, cv_summary

	# LinearRegression: no tuning; fit directly
	pipeline.fit(X_train, y_train)
	return pipeline, {"best_params": {}, "best_score_neg_rmse": None}


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


def compute_abs_residual_quantiles(y_true: pd.Series, y_pred: np.ndarray, percentiles: List[int]) -> Dict[str, float]:
	"""
	Compute absolute residual quantiles |y - y_hat| for percentiles [1..99],
	for split-conformal symmetric intervals.
	"""
	abs_residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred)).astype(float)
	quantile_values = np.percentile(abs_residuals, percentiles).astype(float)
	return {f"p{p}": float(v) for p, v in zip(percentiles, quantile_values)}


def load_csv_dataset(csv_path: str, target_column: str, drop_columns: List[str] | None = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
	"""
	Load a CSV dataset with mixed types.
	Returns X, y, raw feature names, numeric feature names, categorical feature names.
	"""
	df = pd.read_csv(csv_path)
	if drop_columns:
		for col in drop_columns:
			if col in df.columns:
				df = df.drop(columns=[col])
	if target_column not in df.columns:
		raise ValueError(f"Target column '{target_column}' not found in CSV.")
	y = df[target_column]
	X = df.drop(columns=[target_column])
	num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
	cat_cols = [c for c in X.columns if c not in num_cols]
	return X, y, list(X.columns), num_cols, cat_cols


def build_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer | Pipeline:
	"""
	Impute/scale numerics; impute/one-hot categoricals.
	"""
	num_pipeline = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scale", StandardScaler()),
	])
	if len(cat_features) == 0:
		return Pipeline(steps=[("num", num_pipeline)])
	cat_pipeline = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
	])
	return ColumnTransformer(transformers=[
		("num", num_pipeline, num_features),
		("cat", cat_pipeline, cat_features),
	])


def save_plots(y_test: pd.Series, y_pred: np.ndarray, output_dir: Path, title_prefix: str = "") -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	# Predicted vs Actual
	plt.figure(figsize=(6, 6))
	sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
	plt.xlabel("Actual")
	plt.ylabel("Predicted")
	plt.title(f"{title_prefix} Predicted vs Actual")
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1)
	plt.tight_layout()
	plt.savefig(output_dir / "pred_vs_actual.png", dpi=150)
	plt.close()

	# Residuals
	residuals = y_test - y_pred
	plt.figure(figsize=(6, 4))
	sns.histplot(residuals, kde=True)
	plt.xlabel("Residual")
	plt.title(f"{title_prefix} Residual Distribution")
	plt.tight_layout()
	plt.savefig(output_dir / "residual_hist.png", dpi=150)
	plt.close()

	# Residuals vs Predicted
	plt.figure(figsize=(6, 4))
	sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor=None)
	plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
	plt.xlabel("Predicted")
	plt.ylabel("Residual")
	plt.title(f"{title_prefix} Residuals vs Predicted")
	plt.tight_layout()
	plt.savefig(output_dir / "residuals_vs_pred.png", dpi=150)
	plt.close()


def save_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str], output_dir: Path, top_n: int = 20) -> None:
	try:
		result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_squared_error")
	except Exception:
		return
	importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False).head(top_n)
	plt.figure(figsize=(8, max(3, int(top_n * 0.4))))
	sns.barplot(x=importances.values, y=importances.index, orient="h")
	plt.xlabel("Permutation importance (mean decrease in score)")
	plt.title("Top feature importances")
	plt.tight_layout()
	output_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_dir / "feature_importance.png", dpi=150)
	plt.close()


def save_artifacts(
	best_name: str,
	best_pipeline: Pipeline,
	metrics_by_model: Dict[str, Dict[str, float]],
	feature_names: List[str],
	residual_quantiles: Dict[str, float] | None = None,
	calibration_abs_residual_quantiles: Dict[str, float] | None = None,
	calibration_size: int | None = None,
	cv_summaries: Dict[str, Dict] | None = None,
	dataset_kind: str = "california",
	target_column: str | None = None,
	num_features: List[str] | None = None,
	cat_features: List[str] | None = None,
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
		"calibration_abs_residual_quantiles": calibration_abs_residual_quantiles or {},
		"calibration_size": calibration_size or 0,
		"conformal_method": "split_conformal_symmetric_abs_residual",
		"cv_summaries": cv_summaries or {},
		"dataset_kind": dataset_kind,
		"target_column": target_column,
		"numeric_features": num_features or [],
		"categorical_features": cat_features or [],
		"target_units": "hundred_thousands_usd",
	}
	with meta_path.open("w", encoding="utf-8") as fp:
		json.dump(metadata, fp, indent=2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train house price models (California or CSV).")
	parser.add_argument("--dataset", choices=["california", "csv"], default="california")
	parser.add_argument("--csv-path", type=str, help="Path to CSV when --dataset=csv")
	parser.add_argument("--target", type=str, default="SalePrice", help="Target column for CSV dataset")
	parser.add_argument("--drop-cols", type=str, nargs="*", default=["Id"], help="Columns to drop for CSV dataset if present")
	parser.add_argument("--plots-dir", type=str, default="models/plots", help="Directory to save evaluation plots")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.dataset == "california":
		X, y, feature_names = load_dataset()
		num_features = feature_names
		cat_features: List[str] = []
		preprocessor = Pipeline(steps=[("scale", StandardScaler())])
		dataset_kind = "california"
		target_col = None
	else:
		if not args.csv_path:
			raise ValueError("Please provide --csv-path for CSV dataset.")
		X, y, feature_names, num_features, cat_features = load_csv_dataset(args.csv_path, args.target, drop_columns=args.drop_cols)
		preprocessor = build_preprocessor(num_features, cat_features)
		dataset_kind = "csv"
		target_col = args.target

	# 60/20/20 split: train/calibration/test
	X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_cal, y_train, y_cal = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

	models = build_models(preprocessor=preprocessor)
	metrics_by_model: Dict[str, Dict[str, float]] = {}
	cv_summaries: Dict[str, Dict] = {}
	scores_for_selection: List[Tuple[str, float]] = []

	for name, pipeline in models.items():
		# Tune on training split
		best_estimator, cv_summary = tune_model(name, pipeline, X_train, y_train)
		cv_summaries[name] = cv_summary
		# Evaluate on test split
		metrics = evaluate_model(best_estimator, X_train, X_test, y_train, y_test)
		metrics_by_model[name] = metrics
		scores_for_selection.append((name, metrics["rmse"]))
		print(f"{name}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

	# Select the model with the lowest RMSE
	best_name, _ = min(scores_for_selection, key=lambda pair: pair[1])
	# Re-tune to obtain the best estimator for the chosen model
	best_pipeline, _ = tune_model(best_name, models[best_name], X_train, y_train)

	# Compute residual quantiles on the held-out test set for the best model (before refitting on all data)
	# We store p1..p99 so that arbitrary confidence levels (e.g., 78%) can be requested later.
	percentiles = list(range(1, 100))
	y_test_pred_for_best = best_pipeline.predict(X_test)
	residual_quantiles = compute_residual_quantiles(y_test, y_test_pred_for_best, percentiles)

	# Split-conformal: absolute residuals on calibration set
	y_cal_pred = best_pipeline.predict(X_cal)
	calibration_abs_residual_quantiles = compute_abs_residual_quantiles(y_cal, y_cal_pred, percentiles)
	calibration_size = int(len(y_cal))

	# Refit on all data for final model
	best_pipeline.fit(X, y)

	# Save plots
	plots_dir = Path(args.plots_dir)
	save_plots(y_test, y_test_pred_for_best, plots_dir, title_prefix=best_name)
	save_permutation_importance(best_pipeline, X_test, y_test, feature_names, plots_dir)

	save_artifacts(
		best_name,
		best_pipeline,
		metrics_by_model,
		feature_names,
		residual_quantiles=residual_quantiles,
		calibration_abs_residual_quantiles=calibration_abs_residual_quantiles,
		calibration_size=calibration_size,
		cv_summaries=cv_summaries,
		dataset_kind=dataset_kind,
		target_column=target_col,
		num_features=num_features,
		cat_features=cat_features,
	)
	print(f"\nBest model: {best_name}")
	print("Saved pipeline to 'house_price_ml/models/best_model.joblib' and metadata to 'house_price_ml/models/metadata.json'.")
	print(f"Saved plots to '{plots_dir}'.")


if __name__ == "__main__":
	main()

