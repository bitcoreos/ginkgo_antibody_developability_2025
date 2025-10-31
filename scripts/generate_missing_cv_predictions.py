
"""
Generate predictions for missing antibodies in cross-validation submission

This script uses the structural propensity features for the 91 missing antibodies
and uses the trained Ridge models to predict their target values.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Define paths
workspace_root = Path("/a0/bitcore/workspace")
data_dir = workspace_root / "data"
features_dir = data_dir / "features"
models_dir = workspace_root / "ml_algorithms" / "models" / "final"
submission_file = workspace_root / "user-submission" / "huggingface_submission" / "gdpa1_cross_validation_submission.csv"
missing_antibodies_file = "/tmp/missing_antibodies.txt"

# Load the list of missing antibodies
with open(missing_antibodies_file, "r") as f:
    missing_antibodies = [line.strip() for line in f.readlines()]

print(f"Generating predictions for {len(missing_antibodies)} missing antibodies")

# Load the structural propensity features for all antibodies
structural_features_file = features_dir / "structural_propensity_features.csv"
all_structural_features = pd.read_csv(structural_features_file)

# Filter to only the missing antibodies
missing_structural_features = all_structural_features[
    all_structural_features["antibody_name"].isin(missing_antibodies)
].copy()

print(f"Found structural features for {len(missing_structural_features)} missing antibodies")

# Load the trained Ridge models
models = {}
assays = ["HIC_delta_G_ML", "PR_CHO", "AC-SINS_pH7.4_nmol/mg", "Tm2_DSF_degC", "Titer_g/L"]

for assay in assays:
    # Map assay names to model file names
    model_name_map = {
        "HIC_delta_G_ML": "HIC",
        "PR_CHO": "PR_CHO",
        "AC-SINS_pH7.4_nmol/mg": "AC-SINS_pH7.4",
        "Tm2_DSF_degC": "Tm2",
        "Titer_g/L": "Titer"
    }

    model_filename = f"ridge_model_{model_name_map[assay]}.pkl"
    model_path = models_dir / model_filename

    if model_path.exists():
        models[assay] = joblib.load(model_path)
        print(f"Loaded model for {assay}")
    else:
        print(f"Model not found: {model_path}")
        models[assay] = None

# Generate predictions for each assay
predictions = {
    "antibody_name": missing_structural_features["antibody_name"].values,
    "hierarchical_cluster_IgG_isotype_stratified_fold": missing_structural_features["hierarchical_cluster_IgG_isotype_stratified_fold"].values
}

# Use the three structural features for prediction
feature_cols = ["electrostatic_potential", "hydrophobicity_moment", "aromatic_cluster_density"]
X = missing_structural_features[feature_cols].values

for assay in assays:
    if models[assay] is not None:
        # Generate predictions
        pred = models[assay].predict(X)

        # Reverse for 'lower is better' assays (HIC, PR_CHO)
        if assay in ['HIC_delta_G_ML', 'PR_CHO']:
            pred = -pred

        predictions[assay] = pred
        print(f"Generated predictions for {assay}")
    else:
        predictions[assay] = np.nan
        print(f"Skipping predictions for {assay} (no model)")

# Create predictions dataframe
predictions_df = pd.DataFrame(predictions)

# Rename columns to match submission format
predictions_df = predictions_df.rename(columns={
    "HIC_delta_G_ML": "HIC_delta_G_ML",
    "PR_CHO": "PR_CHO",
    "AC-SINS_pH7.4_nmol/mg": "AC-SINS_pH7.4_nmol/mg",
    "Tm2_DSF_degC": "Tm2_DSF_degC",
    "Titer_g/L": "Titer_g/L"
})

print(f"Generated predictions for {len(predictions_df)} antibodies")

# Load the existing submission file
existing_submission = pd.read_csv(submission_file)

# Append the new predictions to the existing submission
updated_submission = pd.concat([existing_submission, predictions_df], ignore_index=True)

# Sort by antibody_name to maintain consistent order
updated_submission = updated_submission.sort_values("antibody_name").reset_index(drop=True)

# Save the updated submission
updated_submission.to_csv(submission_file, index=False)

print(f"Updated cross-validation submission file with {len(predictions_df)} new predictions")
print(f"Total predictions in submission: {len(updated_submission)}")

# Verify that we now have predictions for all required antibodies
all_antibodies = set(all_structural_features["antibody_name"].values)
predicted_antibodies = set(updated_submission["antibody_name"].values)
missing = all_antibodies - predicted_antibodies

if len(missing) == 0:
    print("All antibodies now have predictions in the submission file")
else:
    print(f"Still missing predictions for {len(missing)} antibodies")
    print(f"Missing antibodies: {list(missing)[:10]}...")
