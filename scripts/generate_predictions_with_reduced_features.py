import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib
import os

# Load the reduced modeling feature matrix
reduced_features = pd.read_csv('/a0/bitcore/workspace/data/features/reduced_modeling_feature_matrix_87.csv')

# Load the combined GDPa1 dataset which contains both sequences and assays
# Using the processed clean version
df_data = pd.read_csv('/a0/bitcore/workspace/data/processed/original/GDPa1_v1.2_20250814_clean.csv')

# Extract the assay data from the combined dataset
assay_columns = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']
assay_data = df_data[['antibody_id'] + assay_columns].copy()

# Define the assays to predict
assays = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']

# Load pre-trained models and generate predictions
predictions = {}

for assay in assays:
    print(f"\nGenerating predictions for {assay}...")
    
    # Load pre-trained model
    model_path = f'/a0/bitcore/workspace/models/final/ridge_model_{assay}_reduced.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        continue
    
    model = joblib.load(model_path)
    
    # Get features (all columns except antibody_id)
    feature_cols = [col for col in reduced_features.columns if col != 'antibody_id']
    X = reduced_features[feature_cols].values
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Reverse for 'lower is better' assays if needed
    if assay in ['HIC', 'PR_CHO']:
        y_pred = -y_pred  # Reverse for 'lower is better' assays
    
    # Store predictions
    predictions[assay] = y_pred
    print(f"Generated predictions for {assay}: {len(y_pred)} values")

# Create a DataFrame with predictions in the competition format
submission_df = pd.DataFrame()
submission_df['antibody_id'] = reduced_features['antibody_id']

for assay in assays:
    if assay in predictions:
        submission_df[assay] = predictions[assay]
    else:
        submission_df[assay] = np.nan

# Save the predictions
submissions_dir = '/a0/bitcore/workspace/data/submissions'
os.makedirs(submissions_dir, exist_ok=True)

submission_file = f'{submissions_dir}/gdpa1_competition_submission_reduced_features.csv'
submission_df.to_csv(submission_file, index=False)

print(f"\nPredictions saved to {submission_file}")
print("Submission DataFrame shape:", submission_df.shape)
print("First few rows:")
print(submission_df.head())
