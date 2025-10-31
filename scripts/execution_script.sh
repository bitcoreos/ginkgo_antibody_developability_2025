#!/bin/bash

# Execution script for next steps

echo "=== Regenerating features for heldout set ==="
python /a0/bitcore/workspace/scripts/generate_structural_features_heldout.py
python /a0/bitcore/workspace/research/bioinformatics/evidence_based_cdr_features.py --input /a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv --output /a0/bitcore/workspace/data/features/cdr_features_evidence_based_heldout.csv
python /a0/bitcore/workspace/research/bioinformatics/aggregation_propensity_features.py --input /a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv --output /a0/bitcore/workspace/data/features/aggregation_propensity_features_heldout.csv
python /a0/bitcore/workspace/research/bioinformatics/thermal_stability_features.py --input /a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv --output /a0/bitcore/workspace/data/features/thermal_stability_features_heldout.csv
python /a0/bitcore/workspace/research/bioinformatics/feature_integration.py --heldout

echo "=== Retraining models ==="
python /a0/bitcore/workspace/scripts/run_competition_modeling.py --model_type random_forest
python /a0/bitcore/workspace/scripts/run_competition_modeling.py --model_type xgboost

echo "=== Generating predictions ==="
python /a0/bitcore/workspace/scripts/generate_missing_cv_predictions.py
python /a0/bitcore/workspace/scripts/generate_holdout_predictions.py

echo "=== Creating submission files ==="
python /a0/bitcore/workspace/scripts/run_competition_modeling.py --create_submission cv
python /a0/bitcore/workspace/scripts/run_competition_modeling.py --create_submission holdout

echo "=== Validating outputs ==="
bash /a0/bitcore/workspace/scripts/validation/validate_workspace.sh
python /a0/bitcore/workspace/scripts/evaluate_model_performance.py
bash /a0/bitcore/workspace/scripts/validation/simple_check.sh

echo "=== Next steps execution completed ==="
