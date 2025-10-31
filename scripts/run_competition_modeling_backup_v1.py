#!/usr/bin/env python3
"""
Competition Modeling Pipeline

Trains and evaluates models for the antibody developability competition.

Author: BITCORE Modeling Team
Date: 2025-10-14
Purpose: Generate competition-ready predictions and model artifacts
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import argparse
import sys
from bioinformatics.feature_integration import FeatureIntegration

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _predict_subset(
feature_lookup: pd.DataFrame,
ids: List[str],
model: Any,
imputer: Optional[Any],
scaler: Optional[Any],
integrator: FeatureIntegration,
) -> List[Tuple[str, float]]:
"""Generate predictions for a subset of samples"""
feature_cols = integrator.get_feature_columns(feature_lookup)
subset = feature_lookup.loc[ids, feature_cols]

# Convert subset to numpy array
values = subset.values

# Ensure values is a numpy array
if not isinstance(values, np.ndarray):
values = np.array(values)

if imputer is not None:
values = imputer.transform(values)
elif np.isnan(values).any():
mask = ~np.isnan(values).any(axis=1)
subset = subset.loc[mask]
values = subset.values
# Ensure values is a numpy array after filtering
if not isinstance(values, np.ndarray):
values = np.array(values)
if subset.empty:
return []

if scaler is not None:
values = scaler.transform(values)

predictions = model.predict(values)
return list(zip(subset.index.tolist(), predictions))


def _generate_predictions(
feature_matrix: pd.DataFrame,
targets_df: pd.DataFrame,
results: Dict,
integrator: FeatureIntegration,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
feature_cols = integrator.get_feature_columns(feature_matrix)
feature_lookup = feature_matrix.set_index('antibody_id')
feature_lookup = feature_lookup[feature_cols]

if targets_df.empty:
target_lookup = pd.DataFrame(columns=['antibody_id']).set_index('antibody_id')
else:
target_lookup = targets_df.set_index('antibody_id')

training_ids = set(target_lookup.index)
all_ids = set(feature_lookup.index)
base_holdout_ids = sorted(all_ids - training_ids)

# Initialize results
training_predictions = []
holdout_predictions = []

# Generate predictions for each model
for target_name, model_info in results.items():
model = model_info['model']

# Get imputer and scaler if they exist
imputer = getattr(model_info, 'imputer', None)
scaler = getattr(model_info, 'scaler', None)

# Training predictions
if training_ids:
train_preds = _predict_subset(
feature_lookup,
list(training_ids),
model,
imputer,
scaler,
integrator
)
training_predictions.extend([(target_name, aid, pred) for aid, pred in train_preds])

# Holdout predictions
if base_holdout_ids:
holdout_preds = _predict_subset(
feature_lookup,
base_holdout_ids,
model,
imputer,
scaler,
integrator
)
holdout_predictions.extend([(target_name, aid, pred) for aid, pred in holdout_preds])

# Convert to DataFrames
training_df = pd.DataFrame(training_predictions, columns=['target', 'antibody_id', 'prediction'])
holdout_df = pd.DataFrame(holdout_predictions, columns=['target', 'antibody_id', 'prediction'])

return training_df, holdout_df


def main():
parser = argparse.ArgumentParser(description="Run competition modeling pipeline")
parser.add_argument("--skip-feature-refresh", action="store_true", 
help="Skip feature refresh and use existing feature matrix")
parser.add_argument("--timestamp", type=str, 
help="Timestamp for output files (YYYYMMDD_HHMMSS format)")
args = parser.parse_args()

integrator = FeatureIntegration()

if args.skip_feature_refresh and (integrator.features_dir / "modeling_feature_matrix.csv").is_file():
combined_df = pd.read_csv(integrator.features_dir / "combined_feature_matrix.csv")
feature_matrix = pd.read_csv(integrator.features_dir / "modeling_feature_matrix.csv")
else:
combined_df = integrator.load_and_combine_features()
feature_matrix = integrator.prepare_features_for_modeling(combined_df)
targets_df = integrator.load_targets()
if targets_df is None:
raise SystemExit("Target assays are missing; populate workspace/data/targets/gdpa1_competition_targets.csv")
integrator.save_feature_matrix(combined_df)

if args.skip_feature_refresh:
targets_df = integrator.load_targets()
else:
targets_df = integrator.load_targets()

if targets_df is None:
raise SystemExit("Target assays are missing; populate workspace/data/targets/gdpa1_competition_targets.csv")

timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

model_runs: List[Tuple[str, Dict[str, Dict]]] = []

results_rf = integrator.create_baseline_models(feature_matrix, targets_df)
model_runs.append(("random_forest", results_rf))

try:
results_xgb = integrator.train_xgboost_models(feature_matrix, targets_df)
model_runs.append(("xgboost", results_xgb))
except RuntimeError as exc:
print(f"Skipping XGBoost training: {exc}")

submissions_dir = integrator.workspace_root / "data" / "submissions"
submissions_dir.mkdir(parents=True, exist_ok=True)

research_dir = integrator.workspace_root / "research_outputs" / "model"
research_dir.mkdir(parents=True, exist_ok=True)

print("Competition modeling artifacts generated:")
print(f"- Feature matrix shape: {feature_matrix.shape}")

for model_name, results in model_runs:
cv_predictions = integrator.generate_cv_predictions(results)
cv_path = submissions_dir / f"{model_name}_cv_predictions_{timestamp}.csv"
cv_predictions.to_csv(cv_path, index=False)

training_preds, holdout_preds = _generate_predictions(feature_matrix, targets_df, results, integrator)

training_path = None
if not training_preds.empty:
training_path = submissions_dir / f"{model_name}_training_predictions_{timestamp}.csv"
training_preds.to_csv(training_path, index=False)

holdout_path = None
if not holdout_preds.empty:
holdout_path = submissions_dir / f"{model_name}_heldout_predictions_{timestamp}.csv"
holdout_preds.to_csv(holdout_path, index=False)

if training_path:
print(f"- Training predictions: {training_path}")
if holdout_path:
print(f"- Holdout predictions: {holdout_path}")
if cv_path.exists():
print(f"- Cross-validation predictions: {cv_path}")

return True

if __name__ == "__main__":
success = main()
if success:
print("\nCompetition modeling pipeline completed successfully!")
else:
print("\nCompetition modeling pipeline failed!")
sys.exit(0 if success else 1)
