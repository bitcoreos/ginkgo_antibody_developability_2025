# BITCORE Workflow and Execution

## Overview

This document describes the complete workflow for generating predictions and submissions in the BITCORE antibody developability system. The workflow is executed through the `execution_script.sh` which orchestrates all steps from feature generation to final validation.

## Complete Workflow

### 1. Feature Generation for Heldout Set

The first step involves generating features for the heldout set using specialized scripts:

#### 1.1 Structural Features
- Script: `/a0/bitcore/workspace/scripts/generate_structural_features_heldout.py`
- Input: `/a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv`
- Output: `/a0/bitcore/workspace/data/features/structural_propensity_features_heldout.csv`
- Process: Extracts structural properties from antibody sequences using amino acid property dictionaries

#### 1.2 CDR Features
- Script: `/a0/bitcore/workspace/research/bioinformatics/evidence_based_cdr_features.py`
- Input: `/a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv`
- Output: `/a0/bitcore/workspace/data/features/cdr_features_evidence_based_heldout.csv`
- Process: Extracts CDR regions using regex patterns and computes evidence-based features

#### 1.3 Aggregation Propensity Features
- Script: `/a0/bitcore/workspace/research/bioinformatics/aggregation_propensity_features.py`
- Input: `/a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv`
- Output: `/a0/bitcore/workspace/data/features/aggregation_propensity_features_heldout.csv`
- Process: Computes aggregation propensity scores based on sequence analysis

#### 1.4 Thermal Stability Features
- Script: `/a0/bitcore/workspace/research/bioinformatics/thermal_stability_features.py`
- Input: `/a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv`
- Output: `/a0/bitcore/workspace/data/features/thermal_stability_features_heldout.csv`
- Process: Calculates thermal stability predictions using sequence-based models

#### 1.5 Feature Integration
- Script: `/a0/bitcore/workspace/research/bioinformatics/feature_integration.py`
- Process: Combines all generated features into a unified feature matrix

### 2. Model Retraining

After feature generation, models are retrained on the complete dataset:

#### 2.1 Random Forest Model
- Script: `/a0/bitcore/workspace/scripts/run_competition_modeling.py --model_type random_forest`
- Process: Trains a Random Forest regressor with optimized hyperparameters

#### 2.2 XGBoost Model
- Script: `/a0/bitcore/workspace/scripts/run_competition_modeling.py --model_type xgboost`
- Process: Trains an XGBoost gradient boosting model

### 3. Prediction Generation

Once models are trained, predictions are generated for both cross-validation and heldout sets:

#### 3.1 Missing CV Predictions
- Script: `/a0/bitcore/workspace/scripts/generate_missing_cv_predictions.py`
- Process: Generates predictions for missing cross-validation folds

#### 3.2 Holdout Predictions
- Script: `/a0/bitcore/workspace/scripts/generate_holdout_predictions.py`
- Process: Generates predictions for the heldout set using trained models

### 4. Submission File Creation

Final predictions are formatted into submission files for the competition:

#### 4.1 CV Submission
- Script: `/a0/bitcore/workspace/scripts/run_competition_modeling.py --create_submission cv`
- Process: Creates submission file for cross-validation predictions

#### 4.2 Holdout Submission
- Script: `/a0/bitcore/workspace/scripts/run_competition_modeling.py --create_submission holdout`
- Process: Creates submission file for heldout predictions

### 5. Validation

The final step involves validating all outputs to ensure correctness:

#### 5.1 Workspace Validation
- Script: `/a0/bitcore/workspace/scripts/validate_workspace.sh`
- Process: Validates data file integrity using hash verification

#### 5.2 Model Performance Evaluation
- Script: `/a0/bitcore/workspace/scripts/evaluate_model_performance.py`
- Process: Evaluates model performance metrics

#### 5.3 Simple Validation
- Script: `/a0/bitcore/workspace/scripts/validation/simple_check.sh`
- Process: Performs basic validation checks

## Execution

To execute the complete workflow, run:

```bash
/a0/bitcore/workspace/scripts/execution_script.sh
```

This script orchestrates all the steps in the correct order, ensuring proper data flow between components.
