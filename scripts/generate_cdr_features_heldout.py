#!/usr/bin/env python3
"""
CDR Feature Engineering for Heldout Set

Generates CDR features for the heldout set using the same methodology
as the evidence-based CDR feature extraction for the training set.

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

from research.bioinformatics.evidence_based_cdr_features import EvidenceBasedCDRFeatures

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main execution function for CDR feature generation for heldout set"""
    # Initialize feature extractor
    logger.info("Initializing CDR feature extractor")
    cdr_features = EvidenceBasedCDRFeatures(workspace_root=str(WORKSPACE_ROOT))

    # Define input and output paths
    input_file = WORKSPACE_ROOT / "data" / "sequences" / "heldout-set-sequences.csv"
    output_file = WORKSPACE_ROOT / "data" / "features" / "cdr_features_evidence_based_heldout.csv"

    logger.info(f"Generating CDR features for heldout set from {input_file}")
    logger.info(f"Output will be saved to {output_file}")

    # Extract features
    features_df = cdr_features.extract_cdr_features_dataset(str(input_file), str(output_file))

    # Generate report
    report = cdr_features.generate_cdr_feature_report(features_df)

    # Save report
    report_file = WORKSPACE_ROOT / "data" / "features" / "cdr_features_heldout_report.json"
    with open(report_file, 'w') as f:
        import json
        json.dump(report, f, indent=2)

    logger.info(f"CDR feature report for heldout set saved to {report_file}")

    print(f"CDR Feature Engineering for Heldout Set Complete:")
    print(f"- Features extracted: {len(features_df.columns)}")
    print(f"- Antibodies processed: {len(features_df)}")
    print(f"- Output files: {output_file}, {report_file}")

if __name__ == "__main__":
    main()
