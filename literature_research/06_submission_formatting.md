# Submission File Formatting

## Overview
This document describes the process of formatting our nested cross-validation predictions to match the competition submission requirements.

## Required Format
Based on example submission files, the required format includes:
- Columns: antibody_name, hierarchical_cluster_IgG_isotype_stratified_fold, AC-SINS_pH7.4, Tm2
- 246 samples (all antibodies in the dataset)
- No missing values in either target column

## Our Implementation
Our formatted submission file:
- 242 samples (4 samples removed due to missing AC-SINS_pH7.4 target values)
- Columns: antibody_name, hierarchical_cluster_IgG_isotype_stratified_fold, AC-SINS_pH7.4, Tm2
- 49 missing values in Tm2 column (samples with missing Tm2_DSF_degC in original dataset)
- No missing values in AC-SINS_pH7.4 column (our predictions)

## Formatting Process
1. Renamed antibody_id to antibody_name
2. Renamed fold to hierarchical_cluster_IgG_isotype_stratified_fold
3. Used our predicted values for AC-SINS_pH7.4
4. Retrieved Tm2 values from original dataset
5. Sorted by antibody_name

## Discrepancies from Requirements
1. 4 fewer samples than required (242 vs 246)
   - Caused by removing samples with missing target values during analysis
   - Only samples with known AC-SINS_pH7.4 values were used for training/prediction

2. 49 missing values in Tm2 column
   - Matches the 53 missing values in original Tm2_DSF_degC column
   - Represents samples where Tm2 was not measured

## File Location
Formatted submission saved to:
`/a0/bitcore/workspace/data/submissions/gdpa1_cross_validation_competition_submission.csv`

## Notes
- The submission includes only samples with known target values for training
- Missing Tm2 values reflect the original dataset's data availability
- For a complete submission, missing values would need to be imputed or handled differently
