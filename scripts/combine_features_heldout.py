#!/usr/bin/env python3
"""
Combine Feature Matrices for Heldout Set

Combines CDR, aggregation propensity, and thermal stability features for the heldout set
into a single feature matrix for model training.

Author: BITCORE Feature Engineering Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import json

# Add workspace to path
WORKSPACE_ROOT = Path("/a0/bitcore/workspace")
sys.path.insert(0, str(WORKSPACE_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    """Main execution function for combining feature matrices for heldout set"""
    # Define input paths
    cdr_file = WORKSPACE_ROOT / "data" / "features" / "cdr_features_evidence_based_heldout.csv"
    aggregation_file = WORKSPACE_ROOT / "data" / "features" / "aggregation_propensity_features_heldout.csv"
    thermal_file = WORKSPACE_ROOT / "data" / "features" / "thermal_stability_features_heldout.csv"
    
    # Define output path
    output_file = WORKSPACE_ROOT / "data" / "features" / "combined_features_heldout.csv"
    report_file = WORKSPACE_ROOT / "data" / "features" / "combined_features_heldout_report.json"
    
    logger.info("Loading feature matrices")
    logger.info(f"CDR features: {cdr_file}")
    logger.info(f"Aggregation features: {aggregation_file}")
    logger.info(f"Thermal stability features: {thermal_file}")
    
    # Load feature matrices
    cdr_df = pd.read_csv(cdr_file)
    aggregation_df = pd.read_csv(aggregation_file)
    thermal_df = pd.read_csv(thermal_file)
    
    logger.info(f"CDR features shape: {cdr_df.shape}")
    logger.info(f"Aggregation features shape: {aggregation_df.shape}")
    logger.info(f"Thermal stability features shape: {thermal_df.shape}")
    
    # Merge on antibody_id
    # First merge CDR and aggregation features
    combined_df = pd.merge(cdr_df, aggregation_df, on='antibody_id', how='outer')
    logger.info(f"After merging CDR and aggregation features: {combined_df.shape}")
    
    # Then merge with thermal stability features
    combined_df = pd.merge(combined_df, thermal_df, on='antibody_id', how='outer')
    logger.info(f"After merging all features: {combined_df.shape}")
    
    # Save combined features
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined features saved to {output_file}")
    
    # Generate report
    missing_counts = combined_df.isnull().sum()
    report = {
        "total_antibodies": len(combined_df),
        "total_features": len(combined_df.columns) - 1,  # -1 for antibody_id column
        "feature_sources": {
            "cdr_features": cdr_df.shape[1] - 1,
            "aggregation_features": aggregation_df.shape[1] - 1,
            "thermal_stability_features": thermal_df.shape[1] - 1
        },
        "missing_data_summary": {
            "total_missing_values": int(missing_counts.sum()),
            "missing_values_per_column": {col: int(count) for col, count in missing_counts.items()}
        }
    }
    
    # Convert numpy types to native Python types
    report = convert_numpy_types(report)
    
    # Save report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Combined features report saved to {report_file}")
    
    print(f"Feature Combination for Heldout Set Complete:")
    print(f"- Total antibodies: {report['total_antibodies']}")
    print(f"- Total features: {report['total_features']}")
    print(f"- CDR features: {report['feature_sources']['cdr_features']}")
    print(f"- Aggregation features: {report['feature_sources']['aggregation_features']}")
    print(f"- Thermal stability features: {report['feature_sources']['thermal_stability_features']}")
    print(f"- Output files: {output_file}, {report_file}")

if __name__ == "__main__":
    main()
