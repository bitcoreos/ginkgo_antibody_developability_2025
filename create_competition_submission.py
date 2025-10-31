
"""
Script to create a competition-ready submission file from the complete predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

def create_competition_submission():
    """Create a competition-ready submission file."""
    print("Creating competition-ready submission file...")

    # Define paths
    workspace_root = Path("/a0/bitcore/workspace")
    submissions_dir = workspace_root / "data" / "submissions"
    user_submission_dir = workspace_root / "user-submission"

    # Load the complete predictions (246 antibodies)
    input_file = submissions_dir / "gdpa1_cross_validation_competition_submission_complete.csv"
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
    fold_file = user_submission_dir / "cv_predictions_latest_with_fold.csv"
    if not fold_file.exists():
        print(f"Error: Fold information file not found at {fold_file}")
        return False

    print(f"Loading fold information from {fold_file}")
    fold_data = pd.read_csv(fold_file)
    print(f"Fold data shape: {fold_data.shape}")

    # Merge predictions with fold information
    # Use antibody_id as the key for merging
    merged = predictions.merge(
        fold_data[['antibody_id', 'hierarchical_cluster_IgG_isotype_stratified_fold']], 
        on='antibody_id', 
        how='left'
    )

    # For antibodies without fold information, use -1 as default
    merged['hierarchical_cluster_IgG_isotype_stratified_fold'] = merged['hierarchical_cluster_IgG_isotype_stratified_fold'].fillna(-1)

    # Create the competition submission dataframe
    # According to the schema, we need: antibody_name, fold, AC-SINS_pH7.4, Tm2
    submission_df = pd.DataFrame({
        'antibody_name': merged['antibody_id'],  # Changed from 'sequence_id' to 'antibody_name'
        'fold': merged['hierarchical_cluster_IgG_isotype_stratified_fold'].astype(int),
        'AC-SINS_pH7.4': merged['AC-SINS_pH7.4_nmol/mg'],
        'Tm2': merged['Tm2_DSF_degC']
    })

    # Sort by antibody_name
    submission_df = submission_df.sort_values('antibody_name').reset_index(drop=True)

    # Save the competition submission file
    output_file = submissions_dir / "gdpa1_cross_validation_competition_submission.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Competition submission saved to {output_file}")

    # Display some statistics
    print("Submission statistics:")
    print(f"Total antibodies: {len(submission_df)}")
    print("Missing values per column:")
    print(submission_df.isnull().sum())

    # Display fold value distribution
    print("Fold value distribution:")
    print(submission_df['fold'].value_counts().sort_index())

    # Display first few rows
    print("First few rows:")
    print(submission_df.head())

    # Display last few rows
    print("Last few rows:")
    print(submission_df.tail())

    return True

if __name__ == "__main__":
    success = create_competition_submission()
    if success:
        print("Competition submission created successfully!")
    else:
        print("Failed to create competition submission.")
        sys.exit(1)
