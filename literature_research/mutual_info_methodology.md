# Antibody Developability Research - Mutual Information Analysis

## Objective
Reproduce and extend mutual information analysis from HIC dataset to Tm dataset using information-theoretic framework.

## Methodology
1. **Data Processing**: Chunked loading of Tm dataset with float32 precision to stay within 4GB RAM limit
2. **Feature Extraction**: CDR regions extracted using simplified Kabat numbering:
   - Heavy Chain: CDR-H1 (31-35), CDR-H2 (50-65), CDR-H3 (95-102)
   - Light Chain: CDR-L1 (24-34), CDR-L2 (50-56), CDR-L3 (89-97)
3. **Encoding**: Categorical amino acid sequences converted to numerical features using LabelEncoder
4. **Analysis**: Mutual information calculated using scikit-learn's mutual_info_regression

## Parameters
- Dataset: jain2017biophysical_Tm.csv (137 clinical-stage therapeutic antibodies)
- Chunk size: 1000 rows
- Data type: float32
- Random state: 42
- Number of jobs: 1 (single-threaded for memory efficiency)

## Expected Outputs
1. CSV file with mutual information scores for each CDR region
2. JSON file with detailed feature importance
3. Validation against HIC dataset results (l_cdr_h2 expected to show highest MI)

## Success Metrics
- Complete analysis within 24 hours
- Memory usage below 3.5GB
- Reproduce l_cdr_h2 as highest information feature (MI > 0.05)
- Identify at least 3 CDR positions with MI > 0.05

## Four-Layer Signaling System
- YAML: Configuration parameters
- XML: Feature extraction rules
- JSON: Runtime state and intermediate results
- Markdown: This documentation
