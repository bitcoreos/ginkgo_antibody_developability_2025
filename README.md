# Ginkgo Antibody Developability Competition Submission

This repository contains our supplementary data and code for the Ginkgo Antibody Developability Competition 2025.

## Overview

Our approach focuses on improving feature quality through advanced feature engineering techniques and developing a hybrid model framework for predicting antibody developability. We implemented the FLAb (Feature Learning and Analysis baseline) framework to enhance prediction accuracy.

## Contents

### Code

The `code/` directory contains our FLAb framework implementation, including:

- Multi-channel information theory integration
- Heavy-light chain coupling analysis
- Developability prediction models
- Ensemble methods

The `flab_framework/` directory contains the complete FLAb framework implementation.

The `ml_algorithms/` directory contains implementations of various machine learning algorithms used in our approach.

The `scripts/` directory contains various scripts used for data processing, model training, and result generation.

### Results

The `results/` directory contains our prediction results:

- Cross-validation predictions
- Holdout set predictions
- Imputed competition targets

### Documentation

The `documentation/` directory contains our research reports:

- HIC baseline model report
- Polyreactivity research report
- AC-SINS baseline model report

### Literature Research

The `literature_research/` directory contains our literature research and citations:

- Antibody trends analysis (2024-2025)
- Analysis of AbDev target assays
- Biophysical characterization analysis
- HIC baseline model report
- Polyreactivity research report
- AC-SINS baseline model report

### Framework

The `framework/` directory contains our informational theory framework implementation.

### Markov Models

The `markov_models/` directory contains our Markov model implementations, including surprisal analysis.

## Approach

Our solution improves feature quality and model performance by:

1. Generating sequential embeddings using p-IgGen or ESM-2
2. Extracting structural features from ABodyBuilder3 predictions
3. Applying advanced feature engineering techniques
4. Using hierarchical cluster IgG isotype-stratified fold for cross-validation
5. Including IgG subclass as a categorical feature

## Usage

To use our code:

1. Navigate to the desired directory (code/flab_framework, ml_algorithms, scripts, etc.)
2. Install required dependencies
3. Run the desired modules or scripts

Detailed usage instructions are included in each module.

## Citations

If you use our code or data, please cite our work and the competition organizers.

## License

This work is licensed under the MIT License.
