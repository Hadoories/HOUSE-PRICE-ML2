from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from statistics import NormalDist


def load_metadata() -> Dict:
	project_root = Path(__file__).resolve().parents[1]
	meta_path = project_root / "models" / "metadata.json"
	if not meta_path.exists():
		raise FileNotFoundError(
			"Metadata not found. Train the model first: python -m house_price_ml.src.train"
		)
	with meta_path.open("r", encoding="utf-8") as fp:
		return json.load(fp)


def load_model():
	project_root = Path(__file__).resolve().parents[1]
	model_path = project_root / "models" / "best_model.joblib"
	if not model_path.exists():
		raise FileNotFoundError(
			"Model not found. Train the model first: python -m house_price_ml.src.train"
		)
	return joblib.load(model_path)


def prompt_for_features(feature_names: List[str]) -> List[float]:
	values: List[float] = []
	print("Enter feature values (numbers). Press Enter after each value.")
	for name in feature_names:
		while True:
			raw = input(f"{name}: ").strip()
			try:
				values.append(float(raw))
				break
			except ValueError:
				print("Please enter a valid number.")
	return values


def predict_single(values: List[float], feature_names: List[str]) -> float:
	model = load_model()
	input_frame = pd.DataFrame([values], columns=feature_names)
	pred = model.predict(input_frame)
	return float(pred[0])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Predict house price using a saved model (California Housing)."
	)
	parser.add_argument(
		"--values",
		nargs="+",
		type=float,
		help="Eight numeric feature values in order: MedInc HouseAge AveRooms AveBedrms Population AveOccup Latitude Longitude",
	)
	parser.add_argument(
		"--confidence",
		type=float,
		default=80.0,
		help="Confidence level percentage for the prediction interval (e.g., 78 for 78%%). Default: 80.",
	)
	return parser.parse_args()


def _format_usd(amount_in_hundreds_of_thousands: float) -> str:
	"""
	Format a target value (which is in units of $100,000s) as a USD string.
	"""
	dollars = amount_in_hundreds_of_thousands * 100_000.0
	return f"${dollars:,.0f}"


def _compute_interval_linear(
	prediction: float,
	confidence_pct: float,
	metadata: Dict,
) -> List[float]:
	"""
	For linear models, use saved residual quantiles to form an empirical interval:
	lower = pred + Q(residual, tail), upper = pred + Q(residual, 1 - tail).
	If residual quantiles are unavailable, fallback to Normal approximation using RMSE.
	"""
	alpha = max(0.5, min(confidence_pct / 100.0, 0.999))
	tail = (1.0 - alpha) / 2.0
	lower_q_pct = int(round(tail * 100))
	upper_q_pct = int(round((1.0 - tail) * 100))
	residual_quantiles: Dict[str, float] = metadata.get("residual_quantiles", {}) or {}

	# Try to use empirical residual quantiles first
	key_lower = f"p{lower_q_pct}"
	key_upper = f"p{upper_q_pct}"
	if key_lower in residual_quantiles and key_upper in residual_quantiles:
		lower = prediction + float(residual_quantiles[key_lower])
		upper = prediction + float(residual_quantiles[key_upper])
		return [lower, upper]

	# Fallback: Normal approximation using RMSE for the saved best model
	best_model_name = metadata.get("best_model", "")
	metrics = (metadata.get("metrics_by_model", {}) or {}).get(best_model_name, {})
	rmse = float(metrics.get("rmse", 0.0))
	if rmse <= 0.0:
		return [prediction, prediction]
	z = float(NormalDist().inv_cdf(1.0 - tail))
	half_width = z * rmse
	return [prediction - half_width, prediction + half_width]


def _compute_interval_random_forest(
	model_pipeline,
	input_frame: pd.DataFrame,
	prediction: float,
	confidence_pct: float,
) -> List[float]:
	"""
	For RandomForest, compute per-tree predictions and take empirical quantiles.
	"""
	alpha = max(0.5, min(confidence_pct / 100.0, 0.999))
	tail = (1.0 - alpha) / 2.0
	lower_q = tail
	upper_q = 1.0 - tail

	# Access the RF estimator inside the pipeline
	try:
		rf = model_pipeline.named_steps["regressor"]
		estimators = getattr(rf, "estimators_", None)
	except Exception:
		estimators = None

	if not estimators:
		# Fallback: no per-tree distribution available, return degenerate interval
		return [prediction, prediction]

	per_tree_preds = np.array([est.predict(input_frame)[0] for est in estimators], dtype=float)
	lower = float(np.quantile(per_tree_preds, lower_q))
	upper = float(np.quantile(per_tree_preds, upper_q))
	return [lower, upper]


def main() -> None:
	metadata = load_metadata()
	feature_names: List[str] = list(metadata["feature_names"])

	args = parse_args()
	if args.values is None:
		values = prompt_for_features(feature_names)
	else:
		values = list(args.values)
		if len(values) != len(feature_names):
			raise ValueError(
				f"Expected {len(feature_names)} values, got {len(values)}. Feature order: {', '.join(feature_names)}"
			)

	# Predict
	model_pipeline = load_model()
	input_frame = pd.DataFrame([values], columns=feature_names)
	prediction = float(model_pipeline.predict(input_frame)[0])

	# Compute interval
	best_model_name = metadata.get("best_model", "")
	if best_model_name == "RandomForestRegressor":
		lower, upper = _compute_interval_random_forest(model_pipeline, input_frame, prediction, args.confidence)
	else:
		lower, upper = _compute_interval_linear(prediction, args.confidence, metadata)

	# Output in USD
	print(f"Predicted median house value: {_format_usd(prediction)}")
	print(f"Price range: {_format_usd(lower)} – {_format_usd(upper)}")
	print(f"Confidence: {args.confidence:.0f}%")


if __name__ == "__main__":
	main()

