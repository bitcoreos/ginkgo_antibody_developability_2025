# BITCORE Terminology Glossary

This document provides standardized definitions for key terms used in the BITCORE framework to ensure clear communication and understanding.

## Model Terminology

### Machine Learning Algorithms
**Definition**: Classical statistical and machine learning methods used for prediction and classification.
**Examples**: Random Forest, XGBoost, Support Vector Machines, Neural Networks (traditional)
**Usage Context**: Refers to specific algorithms used in the predictive modeling pipeline for antibody developability.

### AI Models
**Definition**: Large language models and other artificial intelligence systems that process and generate information.
**Examples**: qwen3, llamas, BioLangFusion, PoET-2
**Usage Context**: Refers to pre-trained AI systems used for reasoning, feature extraction, or other AI-based tasks.

### AI Fine-Tuned Models
**Definition**: AI models that have been specifically adapted or fine-tuned for particular tasks within the BITCORE framework.
**Examples**: Fine-tuned versions of qwen3 or llamas for antibody sequence analysis
**Usage Context**: Refers to specialized versions of AI models that have been adapted for specific BITCORE tasks.

## Competition Terminology

### Antibody Developability
**Definition**: The likelihood that an antibody candidate will successfully progress through development and manufacturing processes without encountering stability, solubility, or other biophysical issues.
**Usage Context**: The primary prediction target in the BITCORE framework and the Ginkgo Bio 2025 AbDev competition.

### GDPa1 Dataset
**Definition**: The primary dataset provided by Ginkgo Bioworks for the AbDev competition, containing antibody sequences and developability metrics for 246 IgGs across 10 assays.
**Usage Context**: The main training and evaluation dataset for all BITCORE models.

### AC-SINS
**Definition**: A key target variable in the competition representing a composite developability score.
**Usage Context**: The primary prediction target for BITCORE models.

## Technical Terminology

### FLAb Framework
**Definition**: The Feature Learning for Antibodies framework that forms the foundation of BITCORE.
**Usage Context**: The underlying architecture for all feature engineering and modeling approaches.

### CDR Features
**Definition**: Features extracted from Complementarity Determining Regions of antibody sequences.
**Usage Context**: Key input features for developability prediction models.

### Thermal Stability (Tm2)
**Definition**: A specific biophysical property measured in the competition representing thermal stability.
**Usage Context**: One of the key prediction targets in the BITCORE framework.

This glossary ensures consistent understanding of key terms throughout the BITCORE project.
