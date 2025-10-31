#!/usr/bin/env python3
"""
Memory-efficient mutual information analysis for antibody developability prediction.
Processes data in chunks to stay within 4GB RAM limit.
Uses information-theoretic framework to identify high-information CDR positions.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error
import yaml
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple
import sys

class AntibodyDevelopabilityAnalyzer:
    """Memory-efficient analyzer for antibody developability prediction using information-theoretic framework."""

    def __init__(self, config_path: str, rules_path: str):
        """Initialize analyzer with configuration and rules."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load feature extraction rules
        self.rules = ET.parse(rules_path)

        # Setup logging
        logging.basicConfig(
            filename=self.config['output']['logs_path'],
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize encoders for categorical features
        self.encoders = {}

    def extract_cdr_features(self, sequence: str, chain_type: str) -> Dict[str, str]:
        """Extract CDR regions from antibody sequence based on Kabat numbering."""
        features = {}

        # Get CDR boundaries from XML rules
        rule_set = self.rules.find(f".//RuleSet[@name='CDR_Extraction']")
        if chain_type == 'heavy':
            cdr1_start = int(rule_set.find(".//HeavyChain/CDR1").get('start'))
            cdr1_end = int(rule_set.find(".//HeavyChain/CDR1").get('end'))
            cdr2_start = int(rule_set.find(".//HeavyChain/CDR2").get('start'))
            cdr2_end = int(rule_set.find(".//HeavyChain/CDR2").get('end'))
            cdr3_start = int(rule_set.find(".//HeavyChain/CDR3").get('start'))
            cdr3_end = int(rule_set.find(".//HeavyChain/CDR3").get('end'))

            features['h_cdr1'] = sequence[cdr1_start-1:cdr1_end]  # Convert to 0-based indexing
            features['h_cdr2'] = sequence[cdr2_start-1:cdr2_end]
            features['h_cdr3'] = sequence[cdr3_start-1:cdr3_end]

        else:  # light chain
            cdr1_start = int(rule_set.find(".//LightChain/CDR1").get('start'))
            cdr1_end = int(rule_set.find(".//LightChain/CDR1").get('end'))
            cdr2_start = int(rule_set.find(".//LightChain/CDR2").get('start'))
            cdr2_end = int(rule_set.find(".//LightChain/CDR2").get('end'))
            cdr3_start = int(rule_set.find(".//LightChain/CDR3").get('start'))
            cdr3_end = int(rule_set.find(".//LightChain/CDR3").get('end'))

            features['l_cdr1'] = sequence[cdr1_start-1:cdr1_end]
            features['l_cdr2'] = sequence[cdr2_start-1:cdr2_end]
            features['l_cdr3'] = sequence[cdr3_start-1:cdr3_end]

        return features

    def calculate_physchem_properties(self, sequence: str) -> Dict[str, float]:
        """Calculate physicochemical properties of amino acid sequence."""
        properties = {}

        # Kyte-Doolittle hydrophobicity scale
        kd_scale = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
            'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
            'H': -3.2, 'E': -3.5, 'N': -3.5, 'Q': -3.5, 'D': -3.5, 'K': -3.9, 'R': -4.5
        }

        # Calculate average hydrophobicity
        if len(sequence) > 0:
            hydrophobicity = np.mean([kd_scale.get(aa, 0) for aa in sequence])
        else:
            hydrophobicity = 0

        # Calculate net charge at pH 7.4
        acidic = ['D', 'E']  # Negative charge
        basic = ['K', 'R', 'H']  # Positive charge
        charge = len([aa for aa in sequence if aa in basic]) - len([aa for aa in sequence if aa in acidic])

        properties['hydrophobicity'] = float(hydrophobicity)
        properties['charge'] = float(charge)
        properties['length'] = len(sequence)

        return properties

    def extract_features(self, row: pd.Series) -> Dict:
        """Extract all features from antibody sequence pair."""
        features = {}

        # Extract heavy chain CDR features
        hc_cdr_features = self.extract_cdr_features(row['heavy'], 'heavy')
        features.update(hc_cdr_features)

        # Extract light chain CDR features
        lc_cdr_features = self.extract_cdr_features(row['light'], 'light')
        features.update(lc_cdr_features)

        # Calculate physicochemical properties for CDR regions
        for cdr_name, cdr_seq in hc_cdr_features.items():
            props = self.calculate_physchem_properties(cdr_seq)
            for prop_name, prop_value in props.items():
                features[f'{cdr_name}_{prop_name}'] = prop_value

        for cdr_name, cdr_seq in lc_cdr_features.items():
            props = self.calculate_physchem_properties(cdr_seq)
            for prop_name, prop_value in props.items():
                features[f'{cdr_name}_{prop_name}'] = prop_value

        return features

    def encode_features(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Encode categorical features using LabelEncoder."""
        # Handle categorical features (CDR sequences)
        categorical_features = [col for col in feature_df.columns if feature_df[col].dtype == 'object']

        X_encoded = feature_df.copy()

        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                # Fit on non-null values
                non_null_values = feature_df[feature].dropna()
                if len(non_null_values) > 0:
                    self.encoders[feature].fit(non_null_values)

            # Transform all values
            if feature in self.encoders:
                # Handle unseen labels by setting to 0
                X_encoded[feature] = feature_df[feature].apply(
                    lambda x: self.encoders[feature].transform([x])[0] 
                    if pd.notna(x) and x in self.encoders[feature].classes_ 
                    else 0
                )

        # Ensure float32 for memory efficiency
        X_encoded = X_encoded.astype(np.float32)

        return X_encoded.values

    def process_chunk(self, chunk: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process a chunk of data and extract features."""
        # Extract features for all rows in chunk
        feature_list = []
        for idx, row in chunk.iterrows():
            features = self.extract_features(row)
            feature_list.append(features)

        # Create DataFrame from features
        feature_df = pd.DataFrame(feature_list)

        # Encode features
        X = self.encode_features(feature_df)

        # Extract target variable (Tm)
        y = chunk['Fab Tm by DSF (°C)'].values.astype(np.float32)

        return X, y

    def calculate_mutual_information(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Calculate mutual information between features and target."""
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': feature_names,
            'mutual_information': mi_scores
        })

        # Sort by mutual information score
        results_df = results_df.sort_values('mutual_information', ascending=False)

        return results_df

    def run_analysis(self):
        """Run complete analysis on dataset."""
        self.logger.info("Starting antibody developability analysis")

        # Initialize variables to store results
        all_X = []
        all_y = []
        feature_names = None

        # Process data in chunks
        chunk_iter = pd.read_csv(
            self.config['dataset']['path'],
            sep=self.config['dataset']['separator'],
            chunksize=self.config['processing']['chunk_size']
        )

        chunk_count = 0
        for chunk in chunk_iter:
            chunk_count += 1
            self.logger.info(f"Processing chunk {chunk_count}")

            # Process chunk
            X_chunk, y_chunk = self.process_chunk(chunk)

            # Store results
            all_X.append(X_chunk)
            all_y.append(y_chunk)

            # Store feature names from first chunk
            if feature_names is None:
                feature_names = [col for col in chunk.columns if col not in ['heavy', 'light', 'Fab Tm by DSF (°C)', 'fitness']]
                # Add CDR feature names
                cdr_features = []
                for chain in ['h', 'l']:
                    for cdr in ['cdr1', 'cdr2', 'cdr3']:
                        cdr_features.append(f'{chain}_{cdr}')
                        cdr_features.append(f'{chain}_{cdr}_hydrophobicity')
                        cdr_features.append(f'{chain}_{cdr}_charge')
                        cdr_features.append(f'{chain}_{cdr}_length')
                feature_names.extend(cdr_features)

        # Combine all chunks
        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        self.logger.info(f"Combined data shape: X={X.shape}, y={y.shape}")

        # Calculate mutual information
        mi_results = self.calculate_mutual_information(X, y, feature_names)

        # Save results
        mi_results.to_csv(self.config['output']['results_path'], index=False)
        self.logger.info(f"Results saved to {self.config['output']['results_path']}")

        # Log top features
        top_features = mi_results.head(10)
        self.logger.info("Top 10 features by mutual information:")
        for _, row in top_features.iterrows():
            self.logger.info(f"{row['feature']}: {row['mutual_information']:.3f}")

        return mi_results

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AntibodyDevelopabilityAnalyzer(
        config_path='/a0/bitcore/workspace/antibody_developability/config/tm_analysis.yaml',
        rules_path='/a0/bitcore/workspace/antibody_developability/rules/cdr_feature_rules.xml'
    )

    # Run analysis
    results = analyzer.run_analysis()

    print("Analysis completed successfully")
