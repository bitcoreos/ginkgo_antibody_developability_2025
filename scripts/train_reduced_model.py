import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
import joblib
import os

# Custom scorer for Spearman correlation
def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

# Load the reduced modeling feature matrix
reduced_features = pd.read_csv('/a0/bitcore/workspace/data/features/reduced_modeling_feature_matrix_87.csv')

# Load the combined GDPa1 dataset which contains both sequences and assays
# Using the processed clean version
df_data = pd.read_csv('/a0/bitcore/workspace/data/processed/original/GDPa1_v1.2_20250814_clean.csv')

# Extract the assay data from the combined dataset
assay_columns = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']
assay_data = df_data[['antibody_id'] + assay_columns].copy()

# Merge reduced features with assay data
# Using antibody_id as key
df_merged = pd.merge(reduced_features, assay_data, on='antibody_id')

# Define the assays to predict
assays = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']

# Initialize model
model = Ridge(alpha=1.0)

# Store results
results = {}

# Train and evaluate model for each assay
for assay in assays:
    print(f"\nTraining model for {assay}...")
    
    # Get features (all columns except antibody_id)
    feature_cols = [col for col in reduced_features.columns if col != 'antibody_id']
    X = df_merged[feature_cols].values
    y = df_merged[assay].values
    
    # Check if assay is 'lower is better' and reverse if needed
    if assay in ['HIC', 'PR_CHO']:
        y = -y  # Reverse for 'lower is better' assays
    
    # Use regular KFold for regression
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate model
    scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(spearman_scorer))
    results[assay] = {
        'mean_spearman': scores.mean(),
        'std_spearman': scores.std()
    }
    print(f"Ridge: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Save the results
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
results_dir = '/a0/bitcore/workspace/results'
os.makedirs(results_dir, exist_ok=True)

# Save results to JSON
import json
with open(f'{results_dir}/reduced_model_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)

# Train final models on all data and save
final_models_dir = '/a0/bitcore/workspace/models/final'
os.makedirs(final_models_dir, exist_ok=True)

for assay in assays:
    # Get features and target
    feature_cols = [col for col in reduced_features.columns if col != 'antibody_id']
    X = df_merged[feature_cols].values
    y = df_merged[assay].values
    
    # Reverse for 'lower is better' assays
    if assay in ['HIC', 'PR_CHO']:
        y = -y
    
    # Train final model
    final_model = Ridge(alpha=1.0)
    final_model.fit(X, y)
    
    # Save model
    joblib.dump(final_model, f'{final_models_dir}/ridge_model_{assay}_reduced.pkl')

print(f"\nFinal models saved to {final_models_dir}")
print("Results saved to", results_dir)
