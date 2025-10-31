"""
Unified script to create both competition submission files from complete predictions.

This script generates both required submission files:
1. Cross-validation submission (246 training antibodies)
2. Private test set submission (80 private test antibodies)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil


def create_unified_competition_submission():
    """Create both required competition submission files."""
    print("Creating unified competition submission files...")

    # Define paths
    workspace_root = Path("/a0/bitcore/workspace")
    submissions_dir = workspace_root / "data" / "submissions"
    user_submission_dir = workspace_root / "submissions" / "user" / "oct20-lowscore"

    # Load the complete predictions (246 antibodies)
    input_file = submissions_dir / "gdpa1_cross_validation_competition_submission_complete_modified.csv"
    if not input_file.exists():
        print(f"Error: Complete predictions file not found at {input_file}")
        return False

    print(f"Loading predictions from {input_file}")
    predictions = pd.read_csv(input_file)
    print(f"Predictions shape: {predictions.shape}")

    # Check if we have predictions for all expected antibodies
    expected_antibodies = 246
    if len(predictions) != expected_antibodies:
        print(f"Error: Expected {expected_antibodies} antibodies, but got {len(predictions)}.")
        return False

    # Load the fold information (155 antibodies)
    fold_file = workspace_root / "submissions" / "archive" / "cv_predictions_latest_with_fold.csv"
    if not fold_file.exists():
        print(f"Error: Fold information file not found at {fold_file}")
        return False

    print(f"Loading fold information from {fold_file}")
    fold_data = pd.read_csv(fold_file)
    print(f"Fold data shape: {fold_data.shape}")

    # Create cross-validation submission file
    print("
Creating cross-validation submission file...")

    # Merge predictions with fold information
    # Use antibody_id as the key for merging
    merged = predictions.merge(
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
    print("
Creating private test set submission file...")

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
    print("
Cross-validation submission statistics:")
    print(f"Total antibodies: {len(cv_submission_df)}")
    print("Missing values per column:")
    print(cv_submission_df.isnull().sum())

    print("
Fold value distribution:")
    print(cv_submission_df['hierarchical_cluster_IgG_isotype_stratified_fold'].value_counts().sort_index())

    # Verify private test set submission
    private_df = pd.read_csv(private_output_file)
    print("
Private test set submission statistics:")
    print(f"Total antibodies: {len(private_df)}")
    print("Columns:", list(private_df.columns))
    print("Missing values per column:")
    print(private_df.isnull().sum())

    print("
First few rows of cross-validation submission:")
    print(cv_submission_df.head())

    print("
First few rows of private test set submission:")
    print(private_df.head())

    return True


if __name__ == "__main__":
    success = create_unified_competition_submission()
    if success:
        print("
Both competition submission files created successfully!")
    else:
        print("
Failed to create competition submission files.")
        sys.exit(1)
