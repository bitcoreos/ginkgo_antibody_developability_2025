#!/usr/bin/env python3
"""
Unified competition pipeline that generates predictions for all antibodies using ModifiedFeatureIntegration.

This script combines feature integration, modeling, and submission file creation in one step.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import shutil

# Add the scripts directory to the path to import the modified FeatureIntegration
workspace_dir = Path(__file__).resolve().parent.parent
scripts_dir = workspace_dir / "scripts"
sys.path.insert(0, str(workspace_dir))
sys.path.insert(0, str(scripts_dir))

# Import the ModifiedFeatureIntegration class
try:
    from modified_feature_integration import ModifiedFeatureIntegration
except ImportError as e:
    print(f"Error importing ModifiedFeatureIntegration: {e}")
    # Try alternative import path
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("modified_feature_integration",
                                                      scripts_dir / "modified_feature_integration.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModifiedFeatureIntegration = module.ModifiedFeatureIntegration
        print("Successfully imported ModifiedFeatureIntegration using direct file import")
    except Exception as e2:
        print(f"Error importing ModifiedFeatureIntegration using direct file import: {e2}")
        sys.exit(1)


def generate_competition_submissions():
    """Generate both required competition submission files using ModifiedFeatureIntegration."""
    print("Generating competition submissions using ModifiedFeatureIntegration...")

    # Define paths
    workspace_root = Path(__file__).resolve().parent.parent
    data_dir = workspace_root / "data"
    features_dir = data_dir / "features"
    targets_dir = data_dir / "targets"
    submissions_dir = data_dir / "submissions"
    user_submission_dir = workspace_root / "submissions" / "user" / "oct20-lowscore"
    archive_dir = workspace_root / "submissions" / "archive"

    # Create submissions directory if it doesn't exist
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # Load feature matrix
    feature_matrix_file = features_dir / "modeling_feature_matrix.csv"
    if not feature_matrix_file.exists():
        print(f"Error: Feature matrix file not found at {feature_matrix_file}")
        return False

    print(f"Loading feature matrix from {feature_matrix_file}")
    feature_matrix = pd.read_csv(feature_matrix_file)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Load target data
    targets_file = targets_dir / "gdpa1_competition_targets.csv"
    if not targets_file.exists():
        print(f"Error: Targets file not found at {targets_file}")
        return False

    print(f"Loading targets from {targets_file}")
    targets_df = pd.read_csv(targets_file)
    print(f"Targets shape: {targets_df.shape}")

    # Create ModifiedFeatureIntegration instance
    integrator = ModifiedFeatureIntegration()

    # Train models
    print("Training models...")
    try:
        # Use the existing create_baseline_models method which will use our modified _train_models
        results = integrator.create_baseline_models(feature_matrix, targets_df)
        print("Models trained successfully.")
    except Exception as e:
        print(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate CV predictions
    print("Generating CV predictions for all antibodies...")
    try:
        cv_predictions = integrator.generate_cv_predictions(results)
        print(f"Generated CV predictions for {len(cv_predictions)} antibodies.")

        # Check if we have predictions for all expected antibodies
        expected_antibodies = 246
        if len(cv_predictions) == expected_antibodies:
            print(f"SUCCESS: Predictions generated for all {expected_antibodies} antibodies.")
        else:
            print(f"WARNING: Expected {expected_antibodies} antibodies, but got {len(cv_predictions)}.")
            return False

        # Save the complete predictions (this will be used by the submission creation)
        complete_predictions_file = submissions_dir / "gdpa1_cross_validation_competition_submission_complete_modified.csv"
        cv_predictions.to_csv(complete_predictions_file, index=False)
        print(f"Complete predictions saved to {complete_predictions_file}")

        # Now create the submission files
        print("\nCreating cross-validation submission file...")

        # Load the fold information (155 antibodies)
        fold_file = archive_dir / "cv_predictions_latest_with_fold.csv"
        if not fold_file.exists():
            print(f"Error: Fold information file not found at {fold_file}")
            return False

        print(f"Loading fold information from {fold_file}")
        fold_data = pd.read_csv(fold_file)
        print(f"Fold data shape: {fold_data.shape}")

        # Merge predictions with fold information
        # Use antibody_id as the key for merging
        merged = cv_predictions.merge(
            fold_data[['antibody_id', 'hierarchical_cluster_IgG_isotype_stratified_fold']],
            on='antibody_id',
            how='left'
        )

        # For antibodies without fold information, use -1 as default
        merged['hierarchical_cluster_IgG_isotype_stratified_fold'] = merged['hierarchical_cluster_IgG_isotype_stratified_fold'].fillna(-1)

        # Create the cross-validation submission dataframe
        # Format: antibody_name, AC-SINS_pH7.4, Tm2, hierarchical_cluster_IgG_isotype_stratified_fold
        cv_submission_df = pd.DataFrame({
            'antibody_name': merged['antibody_id'],
            'AC-SINS_pH7.4': merged['AC-SINS_pH7.4_nmol/mg'],
            'Tm2': merged['Tm2_DSF_degC'],
            'hierarchical_cluster_IgG_isotype_stratified_fold': merged['hierarchical_cluster_IgG_isotype_stratified_fold'].astype(int)
        })

        # Reorder columns to match required format
        cv_submission_df = cv_submission_df[['antibody_name', 'AC-SINS_pH7.4', 'Tm2', 'hierarchical_cluster_IgG_isotype_stratified_fold']]

        # Sort by antibody_name
        cv_submission_df = cv_submission_df.sort_values('antibody_name').reset_index(drop=True)

        # Save the cross-validation submission file
        cv_output_file = submissions_dir / "gdpa1_cross_validation_competition_submission.csv"
        cv_submission_df.to_csv(cv_output_file, index=False)
        print(f"Cross-validation submission saved to {cv_output_file}")

        # Create private test set submission file
        print("\nCreating private test set submission file...")

        # Copy the private test set submission file from the template/user submission
        # This file should already contain the correct predictions for the private test set
        private_template_file = user_submission_dir / "private_test_set_competition_submission.csv"
        private_output_file = submissions_dir / "private_test_set_competition_submission.csv"

        if private_template_file.exists():
            shutil.copy(private_template_file, private_output_file)
            print(f"Private test set submission copied to {private_output_file}")
        else:
            print(f"Warning: Private test set template not found at {private_template_file}")
            # Fallback: try to find any existing private test set submission
            fallback_private_file = submissions_dir / "private_test_set_competition_submission.csv"
            if fallback_private_file.exists():
                print(f"Using existing private test set submission from {fallback_private_file}")
            else:
                print("Error: No private test set submission file found")
                return False

        # Display statistics for both files
        print("\nCross-validation submission statistics:")
        print(f"Total antibodies: {len(cv_submission_df)}")
        print("Missing values per column:")
        print(cv_submission_df.isnull().sum())

        print("\nFold value distribution:")
        print(cv_submission_df['hierarchical_cluster_IgG_isotype_stratified_fold'].value_counts().sort_index())

        # Verify private test set submission
        private_df = pd.read_csv(private_output_file)
        print("\nPrivate test set submission statistics:")
        print(f"Total antibodies: {len(private_df)}")
        print("Columns:", list(private_df.columns))
        print("Missing values per column:")
        print(private_df.isnull().sum())

        print("\nFirst few rows of cross-validation submission:")
        print(cv_submission_df.head())

        print("\nFirst few rows of private test set submission:")
        print(private_df.head())

        return True
    except Exception as e:
        print(f"Error generating CV predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    success = generate_competition_submissions()
    if success:
        print("\nCompetition pipeline completed successfully!")
    else:
        print("\nCompetition pipeline failed!")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
