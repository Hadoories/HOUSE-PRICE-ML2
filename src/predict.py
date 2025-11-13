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
	parser.add_argument(
		"--csv-in",
		type=str,
		help="Path to a CSV with columns matching feature names for batch prediction.",
	)
	parser.add_argument(
		"--csv-out",
		type=str,
		help="Optional path to write batch predictions (CSV). Includes prediction, lower, upper, confidence.",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		help="Output prediction(s) as JSON to stdout.",
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


def _compute_intervals_random_forest_batch(
	model_pipeline,
	input_frame: pd.DataFrame,
	confidence_pct: float,
) -> np.ndarray:
	"""
	Vectorized RF intervals for a batch: returns Nx2 array [lower, upper].
	"""
	alpha = max(0.5, min(confidence_pct / 100.0, 0.999))
	tail = (1.0 - alpha) / 2.0
	lower_q = tail
	upper_q = 1.0 - tail

	try:
		rf = model_pipeline.named_steps["regressor"]
		estimators = getattr(rf, "estimators_", None)
	except Exception:
		estimators = None

	if not estimators:
		preds = model_pipeline.predict(input_frame).astype(float)
		return np.column_stack([preds, preds])

	# Stack per-tree predictions as (n_samples, n_trees)
	per_tree = np.column_stack([est.predict(input_frame).astype(float) for est in estimators])
	lower = np.quantile(per_tree, lower_q, axis=1)
	upper = np.quantile(per_tree, upper_q, axis=1)
	return np.column_stack([lower, upper])


def main() -> None:
	metadata = load_metadata()
	feature_names: List[str] = list(metadata["feature_names"])

	args = parse_args()
	model_pipeline = load_model()
	best_model_name = metadata.get("best_model", "")

	# Batch mode
	if args.csv_in:
		df = pd.read_csv(args.csv_in)
		missing = [c for c in feature_names if c not in df.columns]
		if missing:
			raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")

		input_frame = df[feature_names].copy()
		preds = model_pipeline.predict(input_frame).astype(float)

		if best_model_name == "RandomForestRegressor":
			intervals = _compute_intervals_random_forest_batch(model_pipeline, input_frame, args.confidence)
			lower = intervals[:, 0]
			upper = intervals[:, 1]
		else:
			# Linear-like: same residual quantiles for all rows
			alpha = max(0.5, min(args.confidence / 100.0, 0.999))
			tail = (1.0 - alpha) / 2.0
			lower_q_pct = int(round(tail * 100))
			upper_q_pct = int(round((1.0 - tail) * 100))
			residual_quantiles: Dict[str, float] = metadata.get("residual_quantiles", {}) or {}
			key_lower = f"p{lower_q_pct}"
			key_upper = f"p{upper_q_pct}"

			if key_lower in residual_quantiles and key_upper in residual_quantiles:
				lower = preds + float(residual_quantiles[key_lower])
				upper = preds + float(residual_quantiles[key_upper])
			else:
				metrics = (metadata.get("metrics_by_model", {}) or {}).get(best_model_name, {})
				rmse = float(metrics.get("rmse", 0.0))
				if rmse > 0.0:
					from statistics import NormalDist
					z = float(NormalDist().inv_cdf(1.0 - tail))
					half = z * rmse
					lower = preds - half
					upper = preds + half
				else:
					lower = preds
					upper = preds

		out_df = df.copy()
		out_df["prediction"] = preds
		out_df["lower"] = lower
		out_df["upper"] = upper
		out_df["confidence"] = float(args.confidence)

		if args.json:
			records = []
			for _, row in out_df.iterrows():
				records.append({
					"prediction_usd": _format_usd(float(row["prediction"])),
					"lower_usd": _format_usd(float(row["lower"])),
					"upper_usd": _format_usd(float(row["upper"])),
					"confidence": float(args.confidence),
				})
			print(json.dumps(records, indent=2))
		elif args.csv_out:
			out_df.to_csv(args.csv_out, index=False)
			print(f"Wrote predictions to '{args.csv_out}'.")
		else:
			# Print a concise preview
			preview = out_df[["prediction", "lower", "upper", "confidence"]].head(10)
			# Show in USD for readability
			for i, r in preview.iterrows():
				print(f"{i}: Pred={_format_usd(float(r['prediction']))}  Range={_format_usd(float(r['lower']))}–{_format_usd(float(r['upper']))}  Conf={int(r['confidence'])}%")
		return

	# Single prediction path
	if args.values is None:
		values = prompt_for_features(feature_names)
	else:
		values = list(args.values)
		if len(values) != len(feature_names):
			raise ValueError(
				f"Expected {len(feature_names)} values, got {len(values)}. Feature order: {', '.join(feature_names)}"
			)

	input_frame = pd.DataFrame([values], columns=feature_names)
	prediction = float(model_pipeline.predict(input_frame)[0])

	if best_model_name == "RandomForestRegressor":
		lower, upper = _compute_interval_random_forest(model_pipeline, input_frame, prediction, args.confidence)
	else:
		lower, upper = _compute_interval_linear(prediction, args.confidence, metadata)

	if args.json:
		print(json.dumps({
			"prediction_usd": _format_usd(prediction),
			"lower_usd": _format_usd(lower),
			"upper_usd": _format_usd(upper),
			"confidence": float(args.confidence),
		}, indent=2))
	else:
		print(f"Predicted median house value: {_format_usd(prediction)}")
		print(f"Price range: {_format_usd(lower)} – {_format_usd(upper)}")
		print(f"Confidence: {args.confidence:.0f}%")


if __name__ == "__main__":
	main()

