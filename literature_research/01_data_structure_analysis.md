# Data Structure Analysis

## Overview
This document analyzes the grouping structure of the antibody developability dataset to inform the nested cross-validation implementation.

## Key Findings

### Unique Identifiers
- Total samples: 246
- Unique antibody_id values: 246 (all unique)
- No repeated antibodies in the dataset

### Pre-assigned Folds
- The 'fold' column contains 5 values (0-4)
- Distribution is relatively even:
  - Fold 0: 54 samples
  - Fold 1: 49 samples
  - Fold 2: 48 samples
  - Fold 3: 46 samples
  - Fold 4: 49 samples

### Potential Grouping Variables
After examining all columns, no meaningful grouping variables were found:
- All antibody_id values are unique
- No columns representing antibody families or clusters
- No columns representing experimental batches or conditions

## Implications for Nested CV

### GroupKFold Limitations
- Using GroupKFold with antibody_id provides no actual grouping
- Each group contains exactly one sample
- No protection against data leakage from related samples

### Alternative Approaches

#### 1. Use Pre-assigned Folds
- Use the 'fold' column for outer CV splits
- This maintains the original CV structure
- Still need to address inner CV grouping

#### 2. Create Artificial Groupings
- Group samples by similarity in feature space
- Use clustering to create meaningful groups
- Would provide some protection against leakage

#### 3. Modify CV Strategy
- Use regular KFold instead of GroupKFold
- Accept that some leakage may occur
- Document this limitation clearly

## Recommendations

### For Outer CV
- Use the pre-assigned 'fold' column for 5-fold CV
- This maintains consistency with the original data structure

### For Inner CV
- Consider using regular KFold instead of GroupKFold
- Document the limitation in leakage prevention
- Alternatively, explore clustering to create artificial groups

### Data Leakage Considerations
- Without meaningful groups, some data leakage is likely
- Results should be interpreted with caution
- Consider this a limitation of the available data rather than methodology

## Next Steps
1. Modify nested CV implementation to use pre-assigned folds
2. Decide on approach for inner CV grouping
3. Document limitations clearly
4. Proceed with implementation
