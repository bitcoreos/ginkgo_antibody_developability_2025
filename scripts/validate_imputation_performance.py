
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load data
targets_original = pd.read_csv("/a0/bitcore/workspace/data/targets/gdpa1_competition_targets.csv")
targets_imputed = pd.read_csv("/a0/bitcore/workspace/data/targets/gdpa1_competition_targets_imputed.csv")

# Load features (using the structural features as an example)
features_path = "/a0/bitcore/workspace/data/features/structural_propensity_features.csv"
features_df = pd.read_csv(features_path)

# Merge features with targets
original_data = pd.merge(features_df, targets_original, left_on="antibody_id", right_on="antibody_id", how="inner")
imputed_data = pd.merge(features_df, targets_imputed, left_on="antibody_id", right_on="antibody_id", how="inner")

print("Original data shape: " + str(original_data.shape))
print("Imputed data shape: " + str(imputed_data.shape))

# Define target columns
target_columns = ["HIC_delta_G_ML", "PR_CHO", "AC-SINS_pH7.4_nmol/mg", "Tm2_DSF_degC", "Titer_g/L"]

# Define feature columns (structural features)
feature_columns = ["electrostatic_potential", "hydrophobicity_moment", "aromatic_cluster_density"]

# Function to evaluate model performance
def evaluate_model(X, y, model, cv_folds=5):
    # Remove any rows with NaN in features
    mask = ~X.isnull().any(axis=1) & ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]

    # Reset index to ensure proper indexing
    X_clean = X_clean.reset_index(drop=True)
    y_clean = y_clean.reset_index(drop=True)

    if len(y_clean) < cv_folds:
        cv_folds = max(2, len(y_clean) // 2)

    # Cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []

    for train_idx, test_idx in kf.split(X_clean):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    return np.mean(rmse_scores), np.std(rmse_scores), np.mean(mae_scores), np.std(mae_scores)

# Evaluate models for each target
results = []

for target in target_columns:
    print("Evaluating models for " + target + "...")

    # Prepare data
    X_original = original_data[feature_columns]
    y_original = original_data[target]

    X_imputed = imputed_data[feature_columns]
    y_imputed = imputed_data[target]

    # Create models
    model_original = Ridge(alpha=1.0, random_state=42)
    model_imputed = Ridge(alpha=1.0, random_state=42)

    # Evaluate on original data (excluding missing values)
    rmse_orig_mean, rmse_orig_std, mae_orig_mean, mae_orig_std = evaluate_model(X_original, y_original, model_original)

    # Evaluate on imputed data
    rmse_imp_mean, rmse_imp_std, mae_imp_mean, mae_imp_std = evaluate_model(X_imputed, y_imputed, model_imputed)

    # Store results
    results.append({
        "target": target,
        "original_rmse_mean": rmse_orig_mean,
        "original_rmse_std": rmse_orig_std,
        "original_mae_mean": mae_orig_mean,
        "original_mae_std": mae_orig_std,
        "imputed_rmse_mean": rmse_imp_mean,
        "imputed_rmse_std": rmse_imp_std,
        "imputed_mae_mean": mae_imp_mean,
        "imputed_mae_std": mae_imp_std,
        "rmse_improvement": rmse_orig_mean - rmse_imp_mean,
        "mae_improvement": mae_orig_mean - mae_imp_mean
    })

    # Print results
    print("Original data - RMSE: " + str(round(rmse_orig_mean, 4)) + " (+/- " + str(round(rmse_orig_std, 4)) + "), MAE: " + str(round(mae_orig_mean, 4)) + " (+/- " + str(round(mae_orig_std, 4)) + ")")
    print("Imputed data - RMSE: " + str(round(rmse_imp_mean, 4)) + " (+/- " + str(round(rmse_imp_std, 4)) + "), MAE: " + str(round(mae_imp_mean, 4)) + " (+/- " + str(round(mae_imp_std, 4)) + ")")
    print("RMSE improvement: " + str(round(rmse_orig_mean - rmse_imp_mean, 4)))
    print("MAE improvement: " + str(round(mae_orig_mean - mae_imp_mean, 4)))

# Save results to CSV
results_df = pd.DataFrame(results)
results_path = "/a0/bitcore/workspace/data/targets/imputation_performance_comparison.csv"
results_df.to_csv(results_path, index=False)
print("Performance comparison saved to: " + results_path)

# Print summary
print("=== PERFORMANCE COMPARISON SUMMARY ===")
for _, row in results_df.iterrows():
    print(row["target"] + ": RMSE improvement = " + str(round(row["rmse_improvement"], 4)) + ", MAE improvement = " + str(round(row["mae_improvement"], 4)))
