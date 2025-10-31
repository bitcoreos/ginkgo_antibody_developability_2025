#!/usr/bin/env python3
"""
Aggregation Propensity Feature Engineering for Heldout Set

Generates aggregation propensity features for the heldout set using the same methodology
as the evidence-based aggregation feature extraction for the training set.

Author: BITCORE Feature Engineering Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add workspace to path
WORKSPACE_ROOT = Path("/a0/bitcore/workspace")
sys.path.insert(0, str(WORKSPACE_ROOT))

from research.bioinformatics.aggregation_propensity_features import AggregationPropensityFeatures

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main execution function for aggregation propensity feature generation for heldout set"""
    # Initialize feature extractor
    logger.info("Initializing aggregation propensity feature extractor")
    agg_features = AggregationPropensityFeatures(workspace_root=str(WORKSPACE_ROOT))

    # Define input and output paths
    input_file = WORKSPACE_ROOT / "data" / "sequences" / "heldout-set-sequences.csv"
    output_file = WORKSPACE_ROOT / "data" / "features" / "aggregation_propensity_features_heldout.csv"

    logger.info(f"Generating aggregation propensity features for heldout set from {input_file}")
    logger.info(f"Output will be saved to {output_file}")

    # Extract features
    features_df = agg_features.extract_aggregation_features_dataset(str(input_file), str(output_file))

    # Generate report
    report = agg_features.generate_aggregation_feature_report(features_df)

    # Save report
    report_file = WORKSPACE_ROOT / "data" / "features" / "aggregation_features_heldout_report.json"
    with open(report_file, 'w') as f:
        import json
        json.dump(report, f, indent=2)

    logger.info(f"Aggregation propensity feature report for heldout set saved to {report_file}")

    print(f"Aggregation Propensity Feature Engineering for Heldout Set Complete:")
    print(f"- Features extracted: {len(features_df.columns)}")
    print(f"- Antibodies processed: {len(features_df)}")
    print(f"- Output files: {output_file}, {report_file}")

if __name__ == "__main__":
    main()
