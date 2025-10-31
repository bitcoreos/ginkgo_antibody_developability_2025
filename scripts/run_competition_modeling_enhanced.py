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
sys.path.append("/a0/bitcore/workspace")
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

    # Explicitly convert subset to numpy array
    values = subset.values.astype(np.float64)

    if imputer is not None:
        values = imputer.transform(values)
    elif np.isnan(values).any():
        mask = ~np.isnan(values).any(axis=1)
        subset = subset.loc[mask]
        values = subset.values.astype(np.float64)
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


    # Handle case where targets_df is None by extracting from feature_matrix
    if targets_df is None:
        print("Warning: Separate targets file not found, extracting targets from feature matrix")
        # Extract target columns from feature_matrix
        target_columns = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
        existing_target_columns = [col for col in target_columns if col in feature_matrix.columns]

        if existing_target_columns:
            # Create targets_df from feature_matrix
            targets_df = feature_matrix[['antibody_id'] + existing_target_columns].copy()
        else:
            print("Warning: No target columns found in feature matrix")
            targets_df = pd.DataFrame(columns=['antibody_id'])

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
        cv_predictions = integrator.generate_cv_predictions(results, feature_matrix, targets_df)
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


def main():
    import argparse
    from datetime import datetime
    from pathlib import Path
    import sys

    # Add workspace to Python path
    workspace_root = Path(__file__).parent.parent
    sys.path.insert(0, str(workspace_root))

    from bioinformatics.feature_integration import FeatureIntegration

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-feature-refresh', action='store_true')
    parser.add_argument('--timestamp', type=str, default=None)
    args = parser.parse_args()

    # Initialize integrator
    integrator = FeatureIntegration(workspace_root)

    # Load data
    if args.skip_feature_refresh:
        combined_df = pd.read_csv(integrator.features_dir / "combined_feature_matrix.csv")
        feature_matrix = pd.read_csv(integrator.features_dir / "modeling_feature_matrix_with_enhanced_cdr.csv")
    else:
        combined_df = integrator.load_and_combine_features()
        feature_matrix = integrator.prepare_features_for_modeling(combined_df)
        integrator.save_feature_matrix(combined_df)

    # Load targets
    targets_df = integrator.load_targets()

    # Handle case where targets_df is None by extracting from feature_matrix
    if targets_df is None:
        print("Warning: Separate targets file not found, extracting targets from feature matrix")
        # Extract target columns from feature_matrix
        target_columns = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
        existing_target_columns = [col for col in target_columns if col in feature_matrix.columns]

        if existing_target_columns:
            # Create targets_df from feature_matrix
            targets_df = feature_matrix[['antibody_id'] + existing_target_columns].copy()
        else:
            print("Warning: No target columns found in feature matrix")
            targets_df = pd.DataFrame(columns=['antibody_id'])

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Train models
    results_rf = integrator.create_baseline_models(feature_matrix, targets_df)
    results_xgb = integrator.train_xgboost_models(feature_matrix, targets_df)

    # Generate predictions
    cv_predictions_rf = integrator.generate_cv_predictions(results_rf, feature_matrix, targets_df)
    cv_predictions_xgb = integrator.generate_cv_predictions(results_xgb, feature_matrix, targets_df)

    training_preds_rf, holdout_preds_rf = _generate_predictions(feature_matrix, targets_df, results_rf, integrator)
    training_preds_xgb, holdout_preds_xgb = _generate_predictions(feature_matrix, targets_df, results_xgb, integrator)

    # Save predictions
    submissions_dir = integrator.workspace_root / "data" / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    cv_predictions_rf.to_csv(submissions_dir / f"random_forest_cv_predictions_{timestamp}.csv", index=False)
    cv_predictions_xgb.to_csv(submissions_dir / f"xgboost_cv_predictions_{timestamp}.csv", index=False)

    

def main():
    import argparse
    from datetime import datetime
    from pathlib import Path
    import sys

    # Add workspace to Python path
    workspace_root = Path(__file__).parent.parent
    sys.path.insert(0, str(workspace_root))

    from bioinformatics.feature_integration import FeatureIntegration

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-feature-refresh', action='store_true')
    parser.add_argument('--timestamp', type=str, default=None)
    args = parser.parse_args()

    # Initialize integrator
    integrator = FeatureIntegration(workspace_root)

    # Load data
    if args.skip_feature_refresh:
        combined_df = pd.read_csv(integrator.features_dir / "combined_feature_matrix.csv")
        feature_matrix = pd.read_csv(integrator.features_dir / "modeling_feature_matrix_with_enhanced_cdr.csv")
    else:
        combined_df = integrator.load_and_combine_features()
        feature_matrix = integrator.prepare_features_for_modeling(combined_df)
        integrator.save_feature_matrix(combined_df)

    # Load targets
    targets_df = integrator.load_targets()

    # Handle case where targets_df is None by extracting from feature_matrix
    if targets_df is None:
        print("Warning: Separate targets file not found, extracting targets from feature matrix")
        # Extract target columns from feature_matrix
        target_columns = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
        existing_target_columns = [col for col in target_columns if col in feature_matrix.columns]

        if existing_target_columns:
            # Create targets_df from feature_matrix
            targets_df = feature_matrix[['antibody_id'] + existing_target_columns].copy()
        else:
            print("Warning: No target columns found in feature matrix")
            targets_df = pd.DataFrame(columns=['antibody_id'])

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Train models
    results_rf = integrator.create_baseline_models(feature_matrix, targets_df)
    results_xgb = integrator.train_xgboost_models(feature_matrix, targets_df)

    # Generate predictions
    cv_predictions_rf = integrator.generate_cv_predictions(results_rf, feature_matrix, targets_df)
    cv_predictions_xgb = integrator.generate_cv_predictions(results_xgb, feature_matrix, targets_df)

    training_preds_rf, holdout_preds_rf = _generate_predictions(feature_matrix, targets_df, results_rf, integrator)
    training_preds_xgb, holdout_preds_xgb = _generate_predictions(feature_matrix, targets_df, results_xgb, integrator)

    # Save predictions
    submissions_dir = integrator.workspace_root / "data" / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    cv_predictions_rf.to_csv(submissions_dir / f"random_forest_cv_predictions_{timestamp}.csv", index=False)
    cv_predictions_xgb.to_csv(submissions_dir / f"xgboost_cv_predictions_{timestamp}.csv", index=False)

    if not training_preds_rf.empty:
        training_preds_rf.to_csv(submissions_dir / f"random_forest_training_predictions_{timestamp}.csv", index=False)

    if not training_preds_xgb.empty:
        training_preds_xgb.to_csv(submissions_dir / f"xgboost_training_predictions_{timestamp}.csv", index=False)

    if not holdout_preds_rf.empty:
        holdout_preds_rf.to_csv(submissions_dir / f"random_forest_holdout_predictions_{timestamp}.csv", index=False)

    if not holdout_preds_xgb.empty:
        holdout_preds_xgb.to_csv(submissions_dir / f"xgboost_holdout_predictions_{timestamp}.csv", index=False)

    print("Competition modeling pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
