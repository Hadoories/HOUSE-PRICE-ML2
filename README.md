## House Price Prediction (California Housing) - Python ML Project

This project trains a machine learning model to predict median house prices using the California Housing dataset. It includes:
- A training script that evaluates multiple models and saves the best one
- A CLI predictor that loads the saved model and performs predictions

### 1) Setup (Windows PowerShell)
Option A — run from inside this folder (standalone project):
1. Open PowerShell in this folder.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

Option B — run from the parent directory (if this folder is nested in another project):
1. Open PowerShell in the parent folder.
2. Create/activate venv there, then:
   ```powershell
   pip install -r house_price_ml/requirements.txt
   ```

### 2) Train the model
Option A (inside this folder):
```powershell
python -m src.train
```
Option B (from the parent folder):
```powershell
python -m house_price_ml.src.train
```
Outputs:
- Trains Linear Regression, Random Forest (with CV tuning), and Gradient Boosting (with CV tuning)
- Prints RMSE / MAE / R² on the test set
- Saves the best pipeline to `house_price_ml/models/best_model.joblib`
- Saves metadata (features, metrics, CV summary, conformal calibration stats) to `house_price_ml/models/metadata.json`
- Saves evaluation plots to `house_price_ml/models/plots/` (pred vs actual, residuals, feature importances)

Use a Kaggle/CSV dataset:
```powershell
# Example: Kaggle House Prices - copy train.csv locally, then run:
python -m src.train --dataset csv --csv-path C:\path\to\train.csv --target SalePrice --drop-cols Id --plots-dir models/plots
```

Quick start for Kaggle batch predictions:
```powershell
# 1) Train on Kaggle train.csv
python -m src.train --dataset csv --csv-path "C:\Users\Haidar\Downloads\train.csv" --target SalePrice --drop-cols Id

# 2) Use provided sample inputs (same columns as Kaggle features, without SalePrice/Id)
python -m src.predict --csv-in ".\data\sample_inputs_kaggle.csv" --csv-out ".\predictions_kaggle.csv" --confidence 80
```

### 3) Predict with the saved model (CLI)
Interactive mode (you will be prompted for each feature):
```powershell
python -m src.predict
```

One-line mode (pass all 8 features in order):
```powershell
python -m src.predict --values 8.3252 41 6.9841 1.0238 322 2.5556 37.88 -122.23
```

Confidence intervals:
- Add the `--confidence` flag (percentage) to get a price range at that confidence level. Default is 80.
```powershell
python -m src.predict --values 8.3252 41 6.9841 1.0238 322 2.5556 37.88 -122.23 --confidence 78
```
Batch predictions (CSV):
```powershell
# Input CSV must contain columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
python -m src.predict --csv-in data/sample_inputs.csv --csv-out predictions.csv --confidence 80
```

JSON output (single or batch):
```powershell
python -m src.predict --values 8.3252 41 6.9841 1.0238 322 2.5556 37.88 -122.23 --confidence 78 --json
python -m src.predict --csv-in data/sample_inputs.csv --confidence 80 --json > predictions.json
```
Example output:
```
Predicted median house value: $392,790
Price range: $340,000 – $445,000
Confidence: 78%
```
Notes on intervals:
- Uses split-conformal symmetric intervals based on calibration absolute residuals (valid coverage under exchangeability).
- If calibration stats are missing, falls back to: RF per-tree quantiles or empirical residuals/normal approx.

CSV predictions with trained CSV model:
- Ensure your input CSV has the same raw columns used at training time (metadata `feature_names`).
- The CLI will handle preprocessing (imputation/encoding/scaling) automatically.

Feature order (California Housing):
1. MedInc (median income in block group, in tens of thousands)
2. HouseAge (median house age in years)
3. AveRooms (average rooms)
4. AveBedrms (average bedrooms)
5. Population
6. AveOccup (average occupants per household)
7. Latitude
8. Longitude

### Project Structure
```
house_price_ml/
  README.md
  requirements.txt
  .gitignore
  models/
    (saved models and metadata)
  src/
    __init__.py
    train.py
    predict.py
```

### Notes
- The dataset will download on first run; ensure you have internet access.
- If you retrain, the best model and metadata will be overwritten.
- All features are numeric; units are as provided in the California Housing dataset.


