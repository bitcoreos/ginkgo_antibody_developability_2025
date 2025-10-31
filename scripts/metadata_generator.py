import pandas as pd
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_metadata():
    """Generate metadata for the submission."""
    print("Generating metadata file for the submission...")
    
    # Define paths
    workspace_root = Path("/a0/bitcore/workspace")
    submissions_dir = workspace_root / "data" / "submissions"
    metadata_dir = workspace_root / "data" / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    # Submission file
    submission_file = submissions_dir / "gdpa1_cross_validation_competition_submission_reduced_features.csv"
    if not submission_file.exists():
        print(f"Error: Submission file not found at {submission_file}")
        return False
    
    # Compute SHA256 hash of the submission file
    submission_hash = compute_file_hash(submission_file)
    print(f"Submission file hash: {submission_hash}")
    
    # Load the submission file to get information about it
    submission_df = pd.read_csv(submission_file)
    print(f"Submission shape: {submission_df.shape}")
    
    # Get the current timestamp
    timestamp = datetime.now().isoformat()
    
    # Create the manifest
    manifest = {
        "submission_file": str(submission_file),
        "submission_hash": submission_hash,
        "model_ids": [
            "ridge_model_HIC_reduced.pkl",
            "ridge_model_PR_CHO_reduced.pkl",
            "ridge_model_AC-SINS_pH7.4_reduced.pkl",
            "ridge_model_Tm2_reduced.pkl",
            "ridge_model_Titer_reduced.pkl"
        ],
        "feature_hash": "reduced_87_features_hash",  # This would need to be computed from the actual features
        "validation_metrics": {
            "HIC": {
                "mean_spearman": 0.392,
                "std_spearman": 0.046
            },
            "PR_CHO": {
                "mean_spearman": 0.450,
                "std_spearman": 0.074
            },
            "AC-SINS_pH7.4": {
                "mean_spearman": 0.412,
                "std_spearman": 0.086
            },
            "Tm2": {
                "mean_spearman": -0.001,
                "std_spearman": 0.204
            },
            "Titer": {
                "mean_spearman": 0.199,
                "std_spearman": 0.155
            }
        },
        "dataset_hash": "GDPa1_v1.2_20250814_clean.csv_hash",  # This would need to be computed from the actual dataset
        "timestamp": timestamp,
        "git_commit": "N/A",  # This would need to be obtained from the actual git repository
        "validation_report_path": "/a0/bitcore/workspace/results/reduced_model_results_20251021_004204.json"
    }
    
    # Save the manifest
    manifest_file = metadata_dir / "gdpa1_cross_validation_competition_submission_reduced_features_metadata.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to {manifest_file}")
    
    # Display the manifest
    print("Manifest contents:")
    print(json.dumps(manifest, indent=2))
    
    return True

if __name__ == "__main__":
    success = generate_metadata()
    if success:
        print("Metadata file created successfully!")
    else:
        print("Failed to generate metadata file.")
