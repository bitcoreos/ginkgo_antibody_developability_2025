
import pandas as pd
from feature_integration import FeatureIntegration

def test_feature_integration():
    print("Starting test_feature_integration")
    # Create an instance of FeatureIntegration
    integrator = FeatureIntegration()

    # Load the combined feature matrix
    combined_df = integrator.load_and_combine_features()

    # Prepare features for modeling
    feature_matrix = integrator.prepare_features_for_modeling(combined_df)

    # Load targets correctly
    targets_file = integrator.data_dir / "targets" / "gdpa1_competition_targets.csv"
    targets_df = pd.read_csv(targets_file)

    # Save feature matrix
    integrator.save_feature_matrix(combined_df, feature_matrix, targets_df)

    print("Finished test_feature_integration")

test_feature_integration()
