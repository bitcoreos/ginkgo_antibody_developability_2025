#!/usr/bin/env python3
"""
Investigate Data Quality

Investigates data quality issues in the antibody developability dataset.

Author: BITCORE Team
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

def analyze_missing_data(df: pd.DataFrame) -> None:
    """Analyze missing data patterns"""
    print("\nMissing Data Analysis:")
    
    # Define target columns
    target_cols = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    
    # Calculate missing data for each target
    for col in target_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count} missing ({missing_pct:.1f}%)")
    
    # Calculate samples with complete target data
    complete_samples = df[target_cols].dropna().shape[0]
    complete_pct = (complete_samples / len(df)) * 100
    print(f"\n  Samples with complete target data: {complete_samples}/{len(df)} ({complete_pct:.1f}%)")
    
    # Check for missing data patterns
    print("\nMissing Data Patterns:")
    missing_patterns = df[target_cols].isnull().groupby(target_cols).size().reset_index(name='count')
    missing_patterns['pattern'] = missing_patterns[target_cols].apply(
        lambda row: ''.join(['M' if pd.isnull(x) else 'P' for x in row]), axis=1)
    missing_patterns = missing_patterns[['pattern', 'count']].sort_values('count', ascending=False)
    print(missing_patterns.head(10))

def analyze_target_distributions(df: pd.DataFrame) -> None:
    """Analyze target variable distributions"""
    print("\nTarget Variable Distributions:")
    
    # Define target columns
    target_cols = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    
    # Calculate basic statistics for each target
    for col in target_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            print(f"\n  {col}:")
            print(f"    Count: {len(valid_data)}")
            print(f"    Mean: {valid_data.mean():.4f}")
            print(f"    Std: {valid_data.std():.4f}")
            print(f"    Min: {valid_data.min():.4f}")
            print(f"    Max: {valid_data.max():.4f}")
            print(f"    Median: {valid_data.median():.4f}")

def check_for_outliers(df: pd.DataFrame) -> None:
    """Check for outliers in target variables"""
    print("\nOutlier Analysis:")
    
    # Define target columns
    target_cols = ['Titer_g/L', 'AC-SINS_pH7.4_nmol/mg', 'HIC_delta_G_ML', 'PR_CHO', 'Tm2_DSF_degC']
    
    # Check for outliers using IQR method
    for col in target_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
            outlier_pct = (len(outliers) / len(valid_data)) * 100
            
            print(f"  {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")

def main():
    """Main function to investigate data quality"""
    print("Investigating data quality issues...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Analyze missing data
    analyze_missing_data(df)
    
    # Analyze target distributions
    analyze_target_distributions(df)
    
    # Check for outliers
    check_for_outliers(df)
    
    print("\nData quality investigation completed!")

if __name__ == "__main__":
    main()
