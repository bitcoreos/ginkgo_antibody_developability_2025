# Data Loading and Initial Exploration

## Overview
This document describes the process of loading and initially exploring the antibody developability dataset for the nested cross-validation implementation.

## Dataset Location
The dataset is located at:
`/a0/bitcore/workspace/data/features/modeling_feature_matrix_with_enhanced_cdr.csv`

## Dataset Characteristics
- Number of samples: 246 (plus header row)
- Number of features: Approximately 390
- Target variables: AC-SINS_pH7.4 and Tm2_DSF_degC
- Grouping variable: antibody_id (for GroupKFold splits)
- Fold assignment column: fold

## Data Loading Process
The dataset was loaded using pandas with the following approach:

```python
import pandas as pd

df = pd.read_csv('/a0/bitcore/workspace/data/features/modeling_feature_matrix_with_enhanced_cdr.csv')
```

## Initial Exploration

### Column Structure
The dataset contains a wide variety of features related to:
- CDR (Complementarity Determining Regions) sequence characteristics
- Hydrophobic properties
- Charge properties
- Structural features
- Thermal stability metrics
- Isotype information
- Mutual information features

### Target Variables
- `AC-SINS_pH7.4_nmol/mg`: Antibody concentration at a specific pH
- `Tm2_DSF_degC`: Melting temperature from differential scanning fluorimetry

### Grouping Variable
- `antibody_id`: Unique identifier for each antibody sample

### Fold Assignment
- `fold`: Pre-assigned fold for cross-validation (values 0-4)

## Data Quality Assessment
Initial inspection shows:
- Proper CSV format with headers
- No immediately apparent formatting issues
- Appropriate data types for numerical features
- Presence of all expected columns

## Next Steps
1. Implement data preprocessing steps
2. Handle missing values if any
3. Perform feature pruning
4. Implement scaling within CV framework
5. Verify group structure for GroupKFold
