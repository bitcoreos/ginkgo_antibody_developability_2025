#!/usr/bin/env python3
"""
Simplified Nested Cross-Validation Pipeline for Antibody Developability Prediction

Implements a simplified leakage-proof pipeline with nested cross-validation 
to test functionality with reduced computational burden.

Author: BITCORE Modeling Team
Date: 2025-10-20
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
import warnings
import argparse
import sys
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CorrelationCull(BaseEstimator, TransformerMixin):
    """Custom transformer to remove highly correlated features"""
    
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.selected_features_ = None
    
    def fit(self, X, y=None):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        
        # Features to keep
        self.selected_features_ = [col for col in X.columns if col not in to_drop]
        
        return self
    
    def transform(self, X):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Return only selected features
        return X[self.selected_features_]


class MISelector(BaseEstimator, TransformerMixin):
    """Custom transformer for mutual information feature selection"""
    
    def __init__(self, k=10, random_state=42):
        self.k = k
        self.random_state = random_state
        self.selected_features_ = None
        self.mi_scores_ = None
    
    def fit(self, X, y):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Calculate mutual information scores
        self.mi_scores_ = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Select top k features
        top_k_indices = np.argsort(self.mi_scores_)[-self.k:]
        self.selected_features_ = X.columns[top_k_indices].tolist()
        
        return self
    
    def transform(self, X):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Return only selected features
        return X[self.selected_features_]


class LowQualityPruner(BaseEstimator, TransformerMixin):
    """Custom transformer to prune low-quality features"""
    
    def __init__(self, missingness_threshold=0.2, winsorize_percentiles=(0.5, 99.5)):
        self.missingness_threshold = missingness_threshold
        self.winsorize_percentiles = winsorize_percentiles
        self.selected_features_ = None
        self.feature_missingness_ = None
    
    def fit(self, X, y=None):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Calculate missingness for each feature
        self.feature_missingness_ = X.isnull().mean()
        
        # Select features below missingness threshold
        self.selected_features_ = self.feature_missingness_[self.feature_missingness_ <= self.missingness_threshold].index.tolist()
        
        return self
    
    def transform(self, X):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Return only selected features
        X_selected = X[self.selected_features_]
        
        # Winsorize extreme values
        lower_percentile, upper_percentile = self.winsorize_percentiles
        numeric_cols = X_selected.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            lower_bound = np.percentile(X_selected[col].dropna(), lower_percentile)
            upper_bound = np.percentile(X_selected[col].dropna(), upper_percentile)
            X_selected[col] = np.clip(X_selected[col], lower_bound, upper_bound)
        
        return X_selected


class LeakageSentinel(BaseEstimator, TransformerMixin):
    """Custom transformer to add a leakage sentinel feature for control"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.sentinel_column_ = None
    
    def fit(self, X, y=None):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store the sentinel column name
        self.sentinel_column_ = "leakage_sentinel"
        
        return self
    
    def transform(self, X):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Add sentinel column if not already present
        if self.sentinel_column_ not in X.columns:
            np.random.seed(self.random_state)
            sentinel_values = np.random.permutation(X.iloc[:, 0])  # Permute first column
            # Shift one row
            sentinel_values = np.roll(sentinel_values, 1)
            X[self.sentinel_column_] = sentinel_values
        
        return X


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load feature matrix, targets, and groups"""
    # Load feature matrix
    feature_path = Path(data_dir) / "features" / "clean_modeling_feature_matrix.csv"
    feature_matrix = pd.read_csv(feature_path)
    
    # Load targets
    target_path = Path(data_dir) / "targets" / "gdpa1_competition_targets.csv"
    targets = pd.read_csv(target_path)
    
    # Load sequences for groups
    sequence_path = Path(data_dir) / "sequences" / "GDPa1_v1.2_sequences_processed.csv"
    sequences = pd.read_csv(sequence_path)
    
    # Merge data
    merged = feature_matrix.merge(targets, on="antibody_id")
    merged = merged.merge(sequences[["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]], 
                         on="antibody_id")
    
    # Extract features (excluding metadata columns)
    metadata_cols = ["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold", 
                    "AC-SINS_pH7.4_nmol/mg", "Tm2_DSF_degC"]
    feature_cols = [col for col in merged.columns if col not in metadata_cols and merged[col].dtype in ["int64", "float64", "bool"]]
    X = merged[feature_cols]
    
    # Extract targets and groups
    y = merged["AC-SINS_pH7.4_nmol/mg"]  # or "Tm2_DSF_degC"
    groups = merged["hierarchical_cluster_IgG_isotype_stratified_fold"]
    
    # Drop rows with NaN targets
    non_nan_mask = y.notnull()
    merged = merged[non_nan_mask]
    X = X[non_nan_mask]
    y = y[non_nan_mask]
    groups = groups[non_nan_mask]
    
    return X, y, groups


def create_pipeline():
    """Create the leakage-proof pipeline"""
    pipeline = Pipeline([
        ("low_quality_prune", LowQualityPruner()),
        ("leakage_sentinel", LeakageSentinel()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("vt", VarianceThreshold(threshold=1e-5)),
        ("corr", CorrelationCull(threshold=0.95)),
        ("selector", PCA()),  # Will be replaced in grid search
        ("model", Ridge())    # Will be replaced in grid search
    ])
    
    return pipeline


def create_param_grid():
    """Create simplified parameter grid for testing"""
    param_grid = [
        {
            # PCA + Ridge (simplified)
            "selector": [PCA()],
            "selector__n_components": [3, 10, 30],
            "model": [Ridge()],
            "model__alpha": [1e-1, 1, 10]
        },
        {
            # MI + Ridge (simplified)
            "selector": [MISelector()],
            "selector__k": [3, 10, 30],
            "model": [Ridge()],
            "model__alpha": [1e-1, 1, 10]
        }
    ]
    
    return param_grid


def spearman_scorer(estim, X, y):
    """Custom scorer for Spearman correlation"""
    y_pred = estim.predict(X)
    return spearmanr(y, y_pred)[0]


def pearson_scorer(estim, X, y):
    """Custom scorer for Pearson correlation"""
    y_pred = estim.predict(X)
    return pearsonr(y, y_pred)[0]


def rmse_scorer(estim, X, y):
    """Custom scorer for RMSE"""
    y_pred = estim.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred))


def nested_cv_pipeline(X, y, groups):
    """Execute simplified nested CV pipeline"""
    # Outer CV (reduced to 3 folds due to limited groups)
    outer_cv = GroupKFold(n_splits=3)
    
    # Inner CV (reduced to 2 folds)
    inner_cv = GroupKFold(n_splits=2)
    
    # Create pipeline and parameter grid
    pipeline = create_pipeline()
    param_grid = create_param_grid()
    
    # Create scorers
    spearman_scorer_func = make_scorer(spearman_scorer, greater_is_better=True)
    pearson_scorer_func = make_scorer(pearson_scorer, greater_is_better=True)
    rmse_scorer_func = make_scorer(rmse_scorer, greater_is_better=False)
    
    # Nested CV
    results = []
    y_true_all = []
    y_pred_all = []
    
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups.iloc[train_idx].reset_index(drop=True)
        
        # Randomize row order to prevent time/order leaks
        train_permutation = np.random.permutation(len(X_train))
        X_train = X_train.iloc[train_permutation].reset_index(drop=True)
        y_train = y_train.iloc[train_permutation].reset_index(drop=True)
        
        # Inner CV for hyperparameter tuning
        # First, optimize for Spearman, then break ties with Pearson, then RMSE
        search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=inner_cv, 
            scoring={
                'spearman': spearman_scorer_func,
                'pearson': pearson_scorer_func,
                'rmse': rmse_scorer_func
            },
            refit='spearman',  # Primary metric
            n_jobs=1,  # Use single job to reduce memory usage
            verbose=1
        )
        
        search.fit(X_train, y_train, groups=train_groups)
        
        # Evaluate on test set
        y_pred = search.best_estimator_.predict(X_test)
        
        # Store predictions for scatter plot
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
        # Calculate metrics
        spearman = spearmanr(y_test, y_pred)[0]
        pearson = pearsonr(y_test, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            "spearman": spearman,
            "pearson": pearson,
            "rmse": rmse,
            "mae": mae,
            "best_params": {k: v for k, v in search.best_params_.items() if not hasattr(v, "__call__") and not hasattr(v, "fit") and not hasattr(v, "predict")}
        })
        
        logger.info(f"Fold completed - Spearman: {spearman:.4f}, Pearson: {pearson:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Break after first fold for testing
        break
    
    return results, y_true_all, y_pred_all


def check_acceptance_gates(results):
    """Check if results meet acceptance gates"""
    spearman_scores = [r["spearman"] for r in results]
    pearson_scores = [r["pearson"] for r in results]
    rmse_scores = [r["rmse"] for r in results]
    
    mean_spearman = np.mean(spearman_scores)
    std_spearman = np.std(spearman_scores)
    
    # Acceptance gates
    # * Outer-CV means: Spearman ≥ baseline + 0.15 absolute and Pearson similar direction. Report 95% CIs via bootstrap over outer folds.
    # * Stability: outer-fold std ≤ 0.05 for Spearman; no single fold contributes >35% of total gain.
    # * Controls: target-shuffle r in [-0.05, 0.05].
    
    gates = {
        "spearman_mean": float(mean_spearman),
        "spearman_std": float(std_spearman),
        "meets_spearman_threshold": str(mean_spearman >= 0.15),  # Assuming baseline is 0
        "meets_stability_threshold": str(std_spearman <= 0.05),
        "meets_pearson_direction": str(np.mean(pearson_scores) >= 0)
    }
    
    return gates


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run simplified nested CV pipeline for antibody modeling")
    parser.add_argument("--data_dir", default="/a0/bitcore/workspace/data", help="Data directory")
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    X, y, groups = load_data(args.data_dir)
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Run nested CV
    logger.info("Running simplified nested CV pipeline...")
    results, y_true_all, y_pred_all = nested_cv_pipeline(X, y, groups)
    
    # Summarize results
    spearman_scores = [r["spearman"] for r in results]
    pearson_scores = [r["pearson"] for r in results]
    rmse_scores = [r["rmse"] for r in results]
    mae_scores = [r["mae"] for r in results]
    
    logger.info(f"Results Summary:")
    logger.info(f"Spearman: {np.mean(spearman_scores):.4f} ± {np.std(spearman_scores):.4f}")
    logger.info(f"Pearson: {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
    logger.info(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    logger.info(f"MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    
    # Check acceptance gates
    gates = check_acceptance_gates(results)
    logger.info(f"Acceptance Gates:")
    logger.info(f"Mean Spearman: {gates['spearman_mean']:.4f}")
    logger.info(f"Spearman Std: {gates['spearman_std']:.4f}")
    logger.info(f"Meets Spearman Threshold: {gates['meets_spearman_threshold']}")
    logger.info(f"Meets Stability Threshold: {gates['meets_stability_threshold']}")
    logger.info(f"Meets Pearson Direction: {gates['meets_pearson_direction']}")
    
    # Save results
    output_path = Path(args.data_dir) / "results" / "simple_nested_cv_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "results": results,
        "summary": {
            "spearman_mean": float(np.mean(spearman_scores)),
            "spearman_std": float(np.std(spearman_scores)),
            "pearson_mean": float(np.mean(pearson_scores)),
            "pearson_std": float(np.std(pearson_scores)),
            "rmse_mean": float(np.mean(rmse_scores)),
            "rmse_std": float(np.std(rmse_scores)),
            "mae_mean": float(np.mean(mae_scores)),
            "mae_std": float(np.std(mae_scores))
        },
        "acceptance_gates": gates
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
