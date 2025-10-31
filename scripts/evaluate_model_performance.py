#!/usr/bin/env python3
"""
Evaluate Model Performance

Evaluates the performance of the trained models using cross-validation predictions.

Author: BITCORE Team
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_targets(workspace_root: str = "/a0/bitcore/workspace") -> pd.DataFrame:
    """Load competition targets from file"""
    workspace_path = Path(workspace_root)
    targets_path = workspace_path / "data" / "targets" / "gdpa1_competition_targets.csv"
    
    if not targets_path.exists():
        logger.error(f"Targets file not found at {targets_path}")
        return None
        
    targets_df = pd.read_csv(targets_path)
    logger.info(f"Loaded targets from file: {targets_df.shape[0]} samples × {targets_df.shape[1]} targets")
    return targets_df

def evaluate_model_performance(predictions_file: str, targets_df: pd.DataFrame) -> dict:
    """Evaluate model performance using predictions and targets"""
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    logger.info(f"Loaded predictions from {predictions_file}: {predictions_df.shape[0]} predictions")
    
    # Merge predictions with targets
    merged_df = predictions_df.merge(targets_df, on='antibody_id', how='left')
    
    # Calculate performance metrics for each target
    results = {}
    target_columns = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    
    for target in target_columns:
        if target not in merged_df.columns:
            logger.warning(f"Target {target} not found in targets file")
            continue
            
        # Filter out rows with missing target values
        valid_df = merged_df.dropna(subset=[target])
        
        if len(valid_df) == 0:
            logger.warning(f"No valid data for target {target}")
            continue
            
        # Calculate metrics
        r2 = r2_score(valid_df[target], valid_df['prediction'])
        rmse = np.sqrt(mean_squared_error(valid_df[target], valid_df['prediction']))
        
        results[target] = {
            'R2': r2,
            'RMSE': rmse,
            'n_samples': len(valid_df)
        }
        
        logger.info(f"{target} - R²: {r2:.4f}, RMSE: {rmse:.4f}, n_samples: {len(valid_df)}")
    
    return results

def main():
    """Main function to evaluate model performance"""
    print("Evaluating model performance...")
    
    # Load targets
    targets_df = load_targets()
    if targets_df is None:
        return
    
    # Evaluate Random Forest model
    rf_predictions_file = "/a0/bitcore/workspace/data/submissions/random_forest_cv_predictions_20251016_020045.csv"
    if Path(rf_predictions_file).exists():
        print("\nEvaluating Random Forest model:")
        rf_results = evaluate_model_performance(rf_predictions_file, targets_df)
        print("\nRandom Forest Results:")
        for target, metrics in rf_results.items():
            print(f"  {target}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    else:
        print(f"Random Forest predictions file not found: {rf_predictions_file}")
    
    # Evaluate XGBoost model
    xgb_predictions_file = "/a0/bitcore/workspace/data/submissions/xgboost_cv_predictions_20251016_020045.csv"
    if Path(xgb_predictions_file).exists():
        print("\nEvaluating XGBoost model:")
        xgb_results = evaluate_model_performance(xgb_predictions_file, targets_df)
        print("\nXGBoost Results:")
        for target, metrics in xgb_results.items():
            print(f"  {target}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    else:
        print(f"XGBoost predictions file not found: {xgb_predictions_file}")
    
    print("\nModel performance evaluation completed!")

if __name__ == "__main__":
    main()
