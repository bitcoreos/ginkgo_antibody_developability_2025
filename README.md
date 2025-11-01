# Ginkgo Antibody Developability Competition 2025 Submission

This repository contains our final submission for the Ginkgo Antibody Developability Competition 2025, including our code, data, and documentation.

## Overview

Our approach focuses on improving feature quality through advanced feature engineering techniques and developing a hybrid model framework for predicting antibody developability. We implemented the FLAb (Fragment Library for Antibody) framework to enhance prediction accuracy.

## Repository Structure

This repository is organized as follows:

- **Root Directory**: Contains key documentation and summary files
- **code/**: Implementation of our FLAb framework and supporting code
- **data/**: Dataset files including competition targets and predictions
- **dataset/**: Additional dataset files
- **deprecated/**: Outdated components moved for reference
- **documentation/**: Technical documentation
- **features/**: Feature engineering components
- **flab_framework/**: Complete FLAb framework implementation
- **framework/**: Informational theory framework implementation
- **ml_algorithms/**: Machine learning algorithms and models

## Key Files

### Documentation
- `SUMMARY.md`: High-level summary of our approach and results
- `technical_implementation_report.md`: Detailed technical implementation report
- `FEATURES.md`: Description of our feature engineering approach
- `MODELS.md`: Overview of our modeling approach
- `COMPLETED_WORK_SUMMARY.md`: Summary of completed work

### Competition Overview
- `2025 AbDev Competition Overview.md`: Overview of the competition
- `GDPa1 Dataset Overview.md`: Overview of the competition dataset
- `How to Train an Antibody Developability Model.md`: Guide to training models

### Data Files
- `gdpa1_competition_targets_imputed.csv`: Imputed competition targets
- `cv_predictions_latest_corrected.csv`: Cross-validation predictions
- `gdpa1_holdout_predictions_corrected_ids_20251017_014953.csv`: Holdout set predictions

### Reports
- `polyreactivity_research_report.md`: Polyreactivity analysis
- `hic_baseline_model_report.md`: HIC baseline model report
- `ac_sins_baseline_model_report.md`: AC-SINS baseline model report

## Approach

Our solution improves feature quality and model performance by:

1. Generating sequential embeddings using ESM-2
2. Extracting structural features from ABodyBuilder3 predictions
3. Applying advanced feature engineering techniques
4. Using hierarchical cluster IgG isotype-stratified fold for cross-validation
5. Including IgG subclass as a categorical feature

## Technical Implementation

Our technical implementation includes:

- FLAb (Fragment Library for Antibody) Framework
- Multi-Channel Information Theory Framework
- AbLEF (Antibody Language Ensemble Fusion)
- Neural-ODEs for temporal dynamics modeling
- Cross-Attention Mechanisms for multi-modal fusion
- Graph Neural Networks for structure-based predictions
- Heavy-Light Coupling Analysis
- Uncertainty Quantification
- Ensemble Methods
- Polyreactivity Analysis

## Usage

To use our code:

1. Navigate to the desired directory (flab_framework, ml_algorithms, etc.)
2. Install required dependencies
3. Run the desired modules or scripts

Detailed usage instructions are included in each module.

## Citations

If you use our code or data, please cite our work and the competition organizers.
- **[citations.md](citations.md)**: Key citations relevant to our implementation organized by component

## License

This work is licensed under the MIT License.
