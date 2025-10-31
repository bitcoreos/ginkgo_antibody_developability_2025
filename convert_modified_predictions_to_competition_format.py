"""
Script to convert modified predictions to competition submission format.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_modified_predictions():
    """Convert modified predictions to competition submission format."""
    print("Converting modified predictions to competition format...")
    
    # Define paths
    workspace_root = Path("/a0/bitcore/workspace")
    data_dir = workspace_root / "data"
    submissions_dir = data_dir / "submissions"
    archive_dir = workspace_root / "submissions" / "archive"
    
    # Load the modified predictions (246 antibodies)
    input_file = submissions_dir / "gdpa1_cross_validation_competition_submission_complete_modified.csv"
    if not input_file.exists():
        print(f"Error: Modified predictions file not found at {input_file}")
        return False
    
    print(f"Loading modified predictions from {input_file}")
    predictions = pd.read_csv(input_file)
    print(f"Predictions shape: {predictions.shape}")
    
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
        'antibody_name': merged['antibody_id'],
        'fold': merged['hierarchical_cluster_IgG_isotype_stratified_fold'].astype(int),
        'AC-SINS_pH7.4': merged['AC-SINS_pH7.4_nmol/mg'],
        'Tm2': merged['Tm2_DSF_degC']
    })
    
    # Sort by antibody_name
    submission_df = submission_df.sort_values('antibody_name').reset_index(drop=True)
    
    # Save the competition submission file
    output_file = submissions_dir / "gdpa1_cross_validation_competition_submission_modified.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Competition submission saved to {output_file}")
    
    # Display some statistics
    print("Submission statistics:")
    print(f"Total antibodies: {len(submission_df)}")
    print(f"Antibodies with fold -1 (no fold info): {len(submission_df[submission_df['fold'] == -1])}")
    print(f"Columns: {list(submission_df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(submission_df.head())
    
    return True

if __name__ == "__main__":
    convert_modified_predictions()
