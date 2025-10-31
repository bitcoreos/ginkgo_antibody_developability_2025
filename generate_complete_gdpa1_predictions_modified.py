"""
Script to generate predictions for all GDPa1 antibodies using the modified FeatureIntegration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the scripts directory to the path to import the modified FeatureIntegration
workspace_dir = "/a0/bitcore/workspace"
scripts_dir = os.path.join(workspace_dir, "scripts")
sys.path.insert(0, workspace_dir)
sys.path.insert(0, scripts_dir)

# Import the ModifiedFeatureIntegration class
try:
    from modified_feature_integration import ModifiedFeatureIntegration
except ImportError as e:
    print(f"Error importing ModifiedFeatureIntegration: {e}")
    # Try alternative import path
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("modified_feature_integration", 
                                                      os.path.join(scripts_dir, "modified_feature_integration.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModifiedFeatureIntegration = module.ModifiedFeatureIntegration
        print("Successfully imported ModifiedFeatureIntegration using direct file import")
    except Exception as e2:
        print(f"Error importing ModifiedFeatureIntegration using direct file import: {e2}")
        sys.exit(1)

def generate_complete_predictions():
    """Generate predictions for all GDPa1 antibodies."""
    print("Generating predictions for all GDPa1 antibodies...")

    # Define paths
    workspace_root = Path("/a0/bitcore/workspace")
    data_dir = workspace_root / "data"
    features_dir = data_dir / "features"
    targets_dir = data_dir / "targets"
    submissions_dir = data_dir / "submissions"

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
    print("Generating CV predictions...")
    try:
        cv_predictions = integrator.generate_cv_predictions(results)
        print(f"Generated CV predictions for {len(cv_predictions)} antibodies.")

        # Check if we have predictions for all expected antibodies
        expected_antibodies = 246
        if len(cv_predictions) == expected_antibodies:
            print(f"SUCCESS: Predictions generated for all {expected_antibodies} antibodies.")
        else:
            print(f"WARNING: Expected {expected_antibodies} antibodies, but got {len(cv_predictions)}.")

        # Save predictions
        output_file = submissions_dir / "gdpa1_cross_validation_competition_submission_complete_modified.csv"
        cv_predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

        # Display some statistics
        print("Prediction statistics:")
        print(f"Total antibodies: {len(cv_predictions)}")
        print("Missing values per column:")
        print(cv_predictions.isnull().sum())

        # Display first few predictions
        print("First few predictions:")
        print(cv_predictions.head())

        return True
    except Exception as e:
        print(f"Error generating CV predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_complete_predictions()
    if success:
        print("All predictions generated successfully!")
    else:
        print("Failed to generate predictions.")
    sys.exit(1)
