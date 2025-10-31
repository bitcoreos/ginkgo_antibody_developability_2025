# BITCORE Feature Engineering

## Overview

BITCORE employs a comprehensive feature engineering pipeline that extracts meaningful signals from antibody sequences to predict developability properties. The system generates four primary types of features: structural properties, CDR features, aggregation propensity, and thermal stability metrics.

## Feature Types

### 1. Structural Features

Structural features capture the physicochemical properties of amino acids in antibody sequences.

#### Generation Process
- Script: `scripts/generate_structural_features_heldout.py`
- Method: Dictionary-based lookup of amino acid properties
- Properties: Hydrophobicity, charge, aromaticity

#### Key Components
- Amino acid property dictionary with 20 standard amino acids
- Regex-based CDR region extraction (CDR-H1, CDR-H2, CDR-H3)
- Simplified approach using common motifs for CDR identification

#### Output
- File: `structural_propensity_features_heldout.csv`
- Features: Physicochemical properties of antibody sequences

### 2. CDR Features

Complementarity Determining Region (CDR) features capture the structural characteristics of antibody binding regions.

#### Generation Process
- Script: `scripts/evidence_based_cdr_features.py`
- Method: Evidence-based extraction using regex patterns
- Regions: CDR-H1, CDR-H2, CDR-H3

#### Key Components
- CDR-H1 identification using [GSTNA] motifs
- CDR-H2 identification using [WYF] patterns
- CDR-H3 identification using conserved W and WG residues

#### Output
- File: `cdr_features_evidence_based_heldout.csv`
- Features: Structural characteristics of antibody binding regions

### 3. Aggregation Propensity Features

Aggregation propensity features predict the likelihood of antibody aggregation, which is critical for developability.

#### Generation Process
- Script: `scripts/aggregation_propensity_features.py`
- Target Correlation: r=0.91 with experimental aggregation data
- Method: Sequence-based aggregation prediction

#### Key Components
- Aggregation scoring algorithms
- Sequence pattern recognition
- Physicochemical property analysis

#### Output
- File: `aggregation_propensity_features_heldout.csv`
- Features: Aggregation likelihood scores

### 4. Thermal Stability Features

Thermal stability features predict the thermal robustness of antibodies, which affects shelf life and handling.

#### Generation Process
- Script: `scripts/thermal_stability_features.py`
- Target Correlation: Spearman 0.4-0.52 with experimental thermal stability data
- Method: Sequence-based thermal stability prediction

#### Key Components
- Thermal stability prediction algorithms
- Sequence analysis for stability motifs
- Physicochemical property integration

#### Output
- File: `thermal_stability_features_heldout.csv`
- Features: Thermal stability predictions

## Feature Integration

The feature integration process combines all individual feature sets into a unified matrix for modeling.

#### Process
- Script: `scripts/feature_integration.py`
- Method: DataFrame merging and preprocessing
- Components: Imputation, scaling, feature selection

#### Output
- File: Integrated feature matrix
- Features: Combined set of all engineered features

## Data Flow

1. Input sequences are processed by individual feature generation scripts
2. Each script produces a CSV file with specialized features
3. Feature integration script combines all features into a unified matrix
4. Integrated features are used for model training and prediction

This consolidated approach ensures that all relevant signals are captured while maintaining modularity in the feature engineering pipeline.
