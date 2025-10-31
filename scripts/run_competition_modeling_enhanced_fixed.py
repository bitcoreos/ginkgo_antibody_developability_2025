"""
Enhanced competition modeling script that handles missing targets file
and correctly generates predictions for both training and holdout samples.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# Add workspace to Python path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from bioinformatics.feature_integration import FeatureIntegration


def _predict_subset(
    feature_lookup: pd.DataFrame,
    subset_ids: List[str],
    model,
    imputer=None,
    scaler=None,
    integrator=None,
) -> List[Tuple[str, float]]:
    """Generate predictions for a subset of samples."""
    subset = feature_lookup.loc[subset_ids]

    # Apply preprocessing if provided
    if imputer is not None:
        subset = pd.DataFrame(imputer.transform(subset), columns=subset.columns, index=subset.index)

    if scaler is not None:
        subset = pd.DataFrame(scaler.transform(subset), columns=subset.columns, index=subset.index)

    # Generate predictions
    predictions = model.predict(subset)

    return list(zip(subset.index.tolist(), predictions))


def _generate_predictions(
    feature_matrix: pd.DataFrame,
    targets_df: pd.DataFrame,
    results: Dict,
    integrator: FeatureIntegration,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate training and holdout predictions for all models."""
    feature_cols = integrator.get_feature_columns(feature_matrix)
    feature_lookup = feature_matrix.set_index('antibody_id')
    feature_lookup = feature_lookup[feature_cols]

    # Handle case where targets_df is None by extracting from feature_matrix
    if targets_df is None or targets_df.empty:
        # Extract target columns from feature_matrix
        target_columns = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
        existing_target_columns = [col for col in target_columns if col in feature_matrix.columns]

        if existing_target_columns:
            # Create targets_df from feature_matrix
            targets_subset = feature_matrix[['antibody_id'] + existing_target_columns].copy()
            # Identify training samples (those with at least one non-NaN target value)
            has_target_data = targets_subset[existing_target_columns].notna().any(axis=1)
            targets_df = targets_subset[has_target_data].copy()
            target_lookup = targets_df.set_index('antibody_id')
        else:
            # No target columns found
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
        imputer = model_info.get('imputer')
        scaler = model_info.get('scaler')

        # Training predictions
        if training_ids:
            train_preds = _predict_subset(
                feature_lookup,
                sorted(training_ids),
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

    try:
        results_xgb = integrator.train_xgboost_models(feature_matrix, targets_df)
    except RuntimeError as exc:
        print(f"Skipping XGBoost training: {exc}")
        results_xgb = {}

    # Generate predictions
    cv_predictions_rf = integrator.generate_cv_predictions(results_rf, feature_matrix, targets_df)

    training_preds_rf, holdout_preds_rf = _generate_predictions(feature_matrix, targets_df, results_rf, integrator)

    if results_xgb:
        cv_predictions_xgb = integrator.generate_cv_predictions(results_xgb, feature_matrix, targets_df)
        training_preds_xgb, holdout_preds_xgb = _generate_predictions(feature_matrix, targets_df, results_xgb, integrator)

    # Save predictions
    submissions_dir = integrator.workspace_root / "data" / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    cv_predictions_rf.to_csv(submissions_dir / f"random_forest_cv_predictions_{timestamp}.csv", index=False)

    if not training_preds_rf.empty:
        training_preds_rf.to_csv(submissions_dir / f"random_forest_training_predictions_{timestamp}.csv", index=False)

    if not holdout_preds_rf.empty:
        holdout_preds_rf.to_csv(submissions_dir / f"random_forest_holdout_predictions_{timestamp}.csv", index=False)

    if results_xgb:
        cv_predictions_xgb.to_csv(submissions_dir / f"xgboost_cv_predictions_{timestamp}.csv", index=False)

        if not training_preds_xgb.empty:
            training_preds_xgb.to_csv(submissions_dir / f"xgboost_training_predictions_{timestamp}.csv", index=False)

        if not holdout_preds_xgb.empty:
            holdout_preds_xgb.to_csv(submissions_dir / f"xgboost_holdout_predictions_{timestamp}.csv", index=False)

    print("Competition modeling pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
