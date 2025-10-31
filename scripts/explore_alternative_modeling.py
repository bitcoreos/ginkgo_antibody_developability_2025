#!/usr/bin/env python3
"""
Explore Alternative Modeling Approaches

Explores alternative modeling approaches for handling missing data in the antibody developability dataset.

Author: BITCORE Team
Date: 20251016
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(workspace_root: str = "/a0/bitcore/workspace") -> pd.DataFrame:
    """Load the modeling feature matrix"""
    workspace_path = Path(workspace_root)
    matrix_path = workspace_path / "data" / "features" / "modeling_feature_matrix.csv"
    
    if not matrix_path.exists():
        logger.error(f"Modeling feature matrix not found at {matrix_path}")
        return None
        
    df = pd.read_csv(matrix_path)
    logger.info(f"Loaded modeling feature matrix: {df.shape[0]} samples × {df.shape[1]} features")
    return df

def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns (all columns except antibody_id and fold)"""
    exclude_cols = ['antibody_id', 'fold']
    # Get all columns except excluded ones
    candidate_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter out non-numeric columns
    feature_cols = []
    for col in candidate_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    # Remove target columns from features
    target_cols = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    feature_cols = [col for col in feature_cols if col not in target_cols]
    
    logger.info(f"Identified {len(feature_cols)} feature columns")
    return feature_cols

def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare data for modeling"""
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Prepare feature matrix
    X = df[feature_cols].copy()
    
    # Define target columns
    target_cols = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    
    # Prepare target data
    y = df[target_cols].copy()
    
    return X, y

def evaluate_approach(X: pd.DataFrame, y: pd.DataFrame, approach_name: str, model_factory, imputer=None) -> dict:
    """Evaluate a modeling approach using cross-validation"""
    print(f"\nEvaluating {approach_name} approach:")
    
    results = {}
    
    # Get target columns
    target_cols = y.columns
    
    # Evaluate each target separately
    for target in target_cols:
        if target not in y.columns:
            logger.warning(f"Target {target} not found in data")
            continue
            
        # Get target values
        y_target = y[target].values
        
        # Check if we have any valid data for this target
        valid_mask = ~np.isnan(y_target)
        if not np.any(valid_mask):
            logger.warning(f"No valid data for target {target}")
            continue
            
        # Filter data to only include valid samples for this target
        X_valid = X.loc[valid_mask]
        y_valid = y_target[valid_mask]
        
        # Apply imputation if specified
        if imputer is not None:
            X_imputed = pd.DataFrame(imputer.fit_transform(X_valid), columns=X_valid.columns)
            X_valid = X_imputed
        
        # Create model
        model = model_factory()
        
        # Perform cross-validation
        try:
            # Use KFold with shuffling and fewer folds for speed
            kf = KFold(n_splits=2, shuffle=True, random_state=42)  # Reduced to 2 folds
            cv_scores = cross_val_score(model, X_valid, y_valid, cv=kf, scoring='r2')
            
            # Calculate mean and std of CV scores
            mean_r2 = np.mean(cv_scores)
            std_r2 = np.std(cv_scores)
            
            results[target] = {
                'mean_r2': mean_r2,
                'std_r2': std_r2
            }
            
            print(f"  {target}: R² = {mean_r2:.4f} (±{std_r2:.4f})")
            
        except Exception as e:
            logger.error(f"Error evaluating {approach_name} for target {target}: {e}")
            results[target] = {
                'mean_r2': np.nan,
                'std_r2': np.nan
            }
    
    return results

def main():
    """Main function to explore alternative modeling approaches"""
    print("Exploring alternative modeling approaches for handling missing data...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Define model factories
    def rf_factory():
        return RandomForestRegressor(n_estimators=25, random_state=42)  # Reduced n_estimators for speed
    
    def xgb_factory():
        return XGBRegressor(n_estimators=25, random_state=42)  # Reduced n_estimators for speed
    
    def gb_factory():
        return GradientBoostingRegressor(n_estimators=25, random_state=42)  # Reduced n_estimators for speed
    
    # Evaluate different approaches
    
    # 1. Random Forest with KNN imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    rf_results = evaluate_approach(X, y, "Random Forest with KNN Imputation", rf_factory, knn_imputer)
    
    # 2. XGBoost with native missing value handling
    xgb_results = evaluate_approach(X, y, "XGBoost with Native Missing Value Handling", xgb_factory)
    
    # 3. Gradient Boosting with KNN imputation
    gb_results = evaluate_approach(X, y, "Gradient Boosting with KNN Imputation", gb_factory, knn_imputer)
    
    # 4. XGBoost with KNN imputation
    xgb_knn_results = evaluate_approach(X, y, "XGBoost with KNN Imputation", xgb_factory, knn_imputer)
    
    # Print summary
    print("\nSummary of Alternative Modeling Approaches:")
    print("==========================================")
    
    approaches = [
        ("Random Forest with KNN Imputation", rf_results),
        ("XGBoost with Native Missing Value Handling", xgb_results),
        ("Gradient Boosting with KNN Imputation", gb_results),
        ("XGBoost with KNN Imputation", xgb_knn_results)
    ]
    
    target_cols = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    
    for target in target_cols:
        print(f"\n{target}:")
        for approach_name, results in approaches:
            if target in results and not np.isnan(results[target]['mean_r2']):
                mean_r2 = results[target]['mean_r2']
                std_r2 = results[target]['std_r2']
                print(f"  {approach_name}: R² = {mean_r2:.4f} (±{std_r2:.4f})")
            else:
                print(f"  {approach_name}: R² = N/A")
    
    print("\nAlternative modeling exploration completed!")

if __name__ == "__main__":
    main()
