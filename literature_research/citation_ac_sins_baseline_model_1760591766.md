# Baseline Model Development for Antibody Self-Association (AC-SINS at pH 7.4)

## Authors
BITCORE Research Team

## Publication Date
2025-10-16 05:16:06

## Abstract
This report documents the development of baseline models for predicting antibody self-association (AC-SINS at pH 7.4) using the GDPa1 dataset. Two models were developed: a simple numerical feature-based linear regression model and an enhanced model incorporating both numerical features and sequence-based features derived from amino acid composition. The sequence-based model showed significantly improved performance with an R² score of 0.328 compared to 0.077 for the baseline model.

## Key Findings
1. Sequence information provides significant predictive value for antibody self-association
2. Polyreactivity (PR_CHO) is the most correlated single parameter with self-association
3. Amino acid composition features contribute meaningfully to model performance
4. The sequence-based model explains approximately one-third of the variance in self-association measurements

## Methods
Linear regression models with feature scaling using scikit-learn. Sequence features extracted using amino acid composition analysis.

## Dataset
GDPa1 v1.2 dataset (246 antibodies)

## File Location
/a0/bitcore/workspace/research_outputs/ac_sins_baseline_model_report.md

## Related Citations
See citation_antibody_self_association_1760590956.md for related work on AC-SINS.
