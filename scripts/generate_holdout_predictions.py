
import pandas as pd
import numpy as np
import joblib
import os

# Load the structural propensity features for the holdout set
holdout_features = pd.read_csv('/a0/bitcore/workspace/data/features/structural_propensity_features_heldout.csv')

# Load the trained models
models_dir = '/a0/bitcore/workspace/ml_algorithms/models/final'

# Define the assays to predict
assays = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']

# Load Ridge models for all assays
models = {}
for assay in assays:
    model_path = f'{models_dir}/ridge_model_{assay}.pkl'
    models[assay] = joblib.load(model_path)

# Prepare features for prediction
X_holdout = holdout_features[['electrostatic_potential', 'hydrophobicity_moment', 'aromatic_cluster_density']].values

# Generate predictions for each assay
predictions = {'antibody_id': holdout_features['antibody_id'].values}
for assay in assays:
    model = models[assay]
    pred = model.predict(X_holdout)

    # Reverse for 'lower is better' assays (HIC, PR_CHO)
    if assay in ['HIC', 'PR_CHO']:
        pred = -pred

    predictions[assay] = pred

# Create predictions dataframe
predictions_df = pd.DataFrame(predictions)

# Reorder columns to match submission format
submission_columns = ['antibody_id', 'HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']
predictions_df = predictions_df[submission_columns]

# Save predictions
results_dir = '/a0/bitcore/workspace/results'
os.makedirs(results_dir, exist_ok=True)
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
predictions_df.to_csv(f'{results_dir}/holdout_predictions_structural_{timestamp}.csv', index=False)

# Also save as predictions.csv (current submission format)
predictions_df.to_csv('/a0/bitcore/workspace/data/submissions/predictions.csv', index=False)
