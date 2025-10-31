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

\n## Literature Research\n\nThis repository includes comprehensive literature research conducted during the development of our antibody developability prediction models. The research encompasses:\n\n- Project memory files documenting key research findings\n- User research files with analysis of antibody trends and assays\n- Technical documentation of implemented models and frameworks\n- Implementation plans and gap analysis\n- Investigation reports and validation documentation\n\nAll literature research is now available in the `literature_research` directory, containing 274 files organized by source directory. A comprehensive index of all files is provided in `literature_research/citations.md`, which includes:\n\n- Files from `/a0/bitcore/project_memories`\n- Files from `/a0/bitcore/workspace/data/user_research/antibody_research_chatgpt5`\n- Files from `/a0/bitcore/workspace/docs/documentation`\n- Files from `/a0/bitcore/workspace/docs/other`\n- Files from `/a0/bitcore/workspace/docs`\n- Files from `/a0/bitcore/workspace/investigation`\n- Files from `/a0/bitcore/workspace/implementation_plans`\n- Files from `/a0/bitcore/workspace/implementation_plans/gap_analysis`\n\nThe citations.md file also includes a separate section for competition resources:\n\n- Ginkgo Bioworks Datapoints - 2025 AbDev Competition\n- GDPa1 Dataset\n- Hugging Face Blog - Making Antibody Embeddings and Predictions\n\nIf you use our code or data, please cite our work, the competition organizers, and any relevant literature as indicated in the citations.md file.
