#!/usr/bin/env python3
"""
Evidence-Based CDR Feature Engineering Module

Implements CDR feature extraction based on:
- citation_cdr_analysis_1760067088.md: Benchmarking inverse folding models
- citation_cdr_analysis_1760106901.md: Molecular dynamics studies  
- citation_vhh_research_1760017144.md: Nanobody CDR3 length trends

Features implemented:
1. CDR region extraction (H1, H2, H3, L1, L2, L3)
2. CDR-H3 flexibility motifs and length analysis
3. Sequence composition and physicochemical properties
4. Structural prediction scores
5. Nanobody-specific features (VHH compatibility)

Author: BITCORE Feature Engineering Team
Date: 2025-10-14
Evidence: Based on operational_feature_evidence_map.md citations
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re
from collections import Counter

# Import existing CDR extractor
import sys
sys.path.append('/workspaces/bitcore/workspace')
from research.bioinformatics.modules.cdr_extraction import CDRExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvidenceBasedCDRFeatures:
    """
    Evidence-based CDR feature engineering for antibody developability prediction.
    
    Based on scientific literature analysis and competition requirements.
    """
    
    def __init__(self, workspace_root: str = "/workspaces/bitcore/workspace"):
        self.workspace_root = Path(workspace_root)
        self.data_dir = self.workspace_root / "data"
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Initialize CDR extractor
        self.cdr_extractor = CDRExtractor()
        
        # Physicochemical properties from literature
        self.aa_properties = {
            'hydrophobic': set('AILMFPWV'),
            'polar': set('NQST'), 
            'positive': set('KRH'),
            'negative': set('DE'),
            'aromatic': set('FWY'),
            'small': set('AGCS'),
            'proline': set('P'),
            'glycine': set('G'),
            'cysteine': set('C')
        }
        
        # CDR-H3 flexibility motifs from citation evidence
        self.h3_flexibility_motifs = {
            'high_flexibility': ['GG', 'DG', 'NG', 'SG', 'GD', 'GN', 'GS'],
            'medium_flexibility': ['DT', 'DS', 'NS', 'ST', 'TS'],
            'low_flexibility': ['PP', 'WW', 'FF', 'YY']
        }
        
        # CDR-H3 length categories based on structural studies
        self.h3_length_categories = {
            'short': (4, 8),      # 4-8 residues
            'medium': (9, 13),    # 9-13 residues  
            'long': (14, 20),     # 14-20 residues
            'ultralong': (21, 35) # >20 residues (bovine-like)
        }
        
        # Nanobody-specific features (VHH recognition patterns)
        self.vhh_signatures = {
            'framework2_vhh': ['GLEW', 'GLER', 'GLES'],  # VHH-specific FR2
            'framework4_vhh': ['WGQG', 'WGKG', 'WGRG']   # VHH-specific FR4
        }
        
    def calculate_aa_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate amino acid composition percentages"""
        if not sequence:
            return {}
            
        length = len(sequence)
        composition = {}
        
        # Individual amino acid frequencies
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            composition[f'{aa}_freq'] = sequence.count(aa) / length
        
        # Property-based compositions
        for prop_name, aa_set in self.aa_properties.items():
            count = sum(sequence.count(aa) for aa in aa_set)
            composition[f'{prop_name}_freq'] = count / length
            
        return composition
    
    def analyze_h3_flexibility(self, h3_sequence: str) -> Dict[str, Union[float, int]]:
        """
        Analyze CDR-H3 flexibility based on evidence from citation_cdr_analysis_1760106901.md
        Bovine antibody MD study shows flexibility motifs correlate with developability.
        """
        if not h3_sequence:
            return {'h3_flexibility_score': 0.0, 'h3_motif_count': 0}
            
        features = {}
        
        # Count flexibility motifs
        flexibility_scores = []
        motif_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for i in range(len(h3_sequence) - 1):
            dipeptide = h3_sequence[i:i+2]
            
            if dipeptide in self.h3_flexibility_motifs['high_flexibility']:
                flexibility_scores.append(3.0)
                motif_counts['high'] += 1
            elif dipeptide in self.h3_flexibility_motifs['medium_flexibility']:
                flexibility_scores.append(2.0)
                motif_counts['medium'] += 1
            elif dipeptide in self.h3_flexibility_motifs['low_flexibility']:
                flexibility_scores.append(1.0)
                motif_counts['low'] += 1
            else:
                flexibility_scores.append(1.5)  # baseline flexibility
        
        # Calculate overall flexibility score
        avg_flexibility = np.mean(flexibility_scores) if flexibility_scores else 1.5
        
        features.update({
            'h3_flexibility_score': avg_flexibility,
            'h3_high_flex_motifs': motif_counts['high'],
            'h3_medium_flex_motifs': motif_counts['medium'], 
            'h3_low_flex_motifs': motif_counts['low'],
            'h3_total_motifs': sum(motif_counts.values())
        })
        
        return features
    
    def analyze_h3_length_category(self, h3_sequence: str) -> Dict[str, Union[str, int, bool]]:
        """
        Categorize CDR-H3 length based on structural evidence.
        Different length categories have different developability profiles.
        """
        length = len(h3_sequence) if h3_sequence else 0
        
        features = {
            'h3_length': length,
            'h3_category': 'invalid'
        }
        
        # Categorize based on length
        for category, (min_len, max_len) in self.h3_length_categories.items():
            if min_len <= length <= max_len:
                features['h3_category'] = category
                features[f'h3_is_{category}'] = True
            else:
                features[f'h3_is_{category}'] = False
        
        # Risk flags based on literature
        features['h3_length_risk'] = 'low'
        if length < 4:
            features['h3_length_risk'] = 'very_high'  # Too short
        elif length > 25:
            features['h3_length_risk'] = 'high'       # Ultralong
        elif length > 18:
            features['h3_length_risk'] = 'medium'     # Long
            
        return features
    
    def detect_vhh_signatures(self, vh_sequence: str) -> Dict[str, bool]:
        """
        Detect VHH (nanobody) signatures from citation_vhh_research_1760017144.md
        Important for identifying single-domain antibodies.
        """
        features = {}
        
        # Check for VHH-specific framework patterns
        fr2_match = any(motif in vh_sequence for motif in self.vhh_signatures['framework2_vhh'])
        fr4_match = any(motif in vh_sequence for motif in self.vhh_signatures['framework4_vhh'])
        
        features.update({
            'is_vhh_candidate': fr2_match and fr4_match,
            'has_vhh_fr2_signature': fr2_match,
            'has_vhh_fr4_signature': fr4_match,
            'vhh_signature_count': sum([fr2_match, fr4_match])
        })
        
        # VHH-specific sequence characteristics
        if len(vh_sequence) > 0:
            # VHH typically shorter than conventional VH
            features['vh_length_vhh_compatible'] = len(vh_sequence) < 130
            
            # Hydrophobicity patterns (VHH often more hydrophilic)
            hydrophobic_count = sum(vh_sequence.count(aa) for aa in self.aa_properties['hydrophobic'])
            features['vh_hydrophobic_ratio'] = hydrophobic_count / len(vh_sequence)
            features['vh_low_hydrophobic'] = features['vh_hydrophobic_ratio'] < 0.4
        
        return features
    
    def calculate_cdr_consensus_score(self, cdr_sequence: str, cdr_type: str) -> float:
        """
        Calculate consensus score based on common CDR patterns.
        Higher scores indicate more canonical/stable CDR conformations.
        """
        if not cdr_sequence:
            return 0.0
            
        # Simplified consensus scoring (would use databases in production)
        canonical_patterns = {
            'CDR1': {'canonical_motifs': ['GFT', 'GFS', 'DYY', 'SYG']},
            'CDR2': {'canonical_motifs': ['ISG', 'ING', 'IDG', 'VRG']},
            'CDR3': {'canonical_motifs': ['DY', 'FD', 'YY', 'MD']}
        }
        
        if cdr_type not in canonical_patterns:
            return 0.5  # baseline score
            
        motifs = canonical_patterns[cdr_type]['canonical_motifs']
        score = sum(1 for motif in motifs if motif in cdr_sequence)
        normalized_score = min(score / len(motifs), 1.0)
        
        return normalized_score
    
    def extract_cdr_features_single(self, antibody_id: str, vh_seq: str, vl_seq: str) -> Dict[str, Union[float, int, bool, str]]:
        """Extract comprehensive CDR features for a single antibody"""
        features = {'antibody_id': antibody_id}
        
        try:
            # Extract CDR regions
            vh_cdrs = self.cdr_extractor.extract_cdr(vh_seq, 'H')
            vl_cdrs = self.cdr_extractor.extract_cdr(vl_seq, 'L')
            
            # Store CDR sequences
            for cdr_name, cdr_seq in vh_cdrs.items():
                features[f'vh_{cdr_name.lower()}_sequence'] = cdr_seq
                features[f'vh_{cdr_name.lower()}_length'] = len(cdr_seq)
                
                # Amino acid composition for each CDR
                composition = self.calculate_aa_composition(cdr_seq)
                for comp_name, comp_val in composition.items():
                    features[f'vh_{cdr_name.lower()}_{comp_name}'] = comp_val
                
                # Consensus scores
                features[f'vh_{cdr_name.lower()}_consensus'] = self.calculate_cdr_consensus_score(cdr_seq, cdr_name)
            
            for cdr_name, cdr_seq in vl_cdrs.items():
                features[f'vl_{cdr_name.lower()}_sequence'] = cdr_seq
                features[f'vl_{cdr_name.lower()}_length'] = len(cdr_seq)
                
                # Amino acid composition for each CDR
                composition = self.calculate_aa_composition(cdr_seq)
                for comp_name, comp_val in composition.items():
                    features[f'vl_{cdr_name.lower()}_{comp_name}'] = comp_val
                
                # Consensus scores  
                features[f'vl_{cdr_name.lower()}_consensus'] = self.calculate_cdr_consensus_score(cdr_seq, cdr_name)
            
            # CDR-H3 specific analysis (most important for developability)
            h3_features = self.analyze_h3_flexibility(vh_cdrs['CDR3'])
            features.update({f'cdr_h3_{k}': v for k, v in h3_features.items()})
            
            h3_length_features = self.analyze_h3_length_category(vh_cdrs['CDR3'])
            features.update({f'cdr_{k}': v for k, v in h3_length_features.items()})
            
            # VHH detection
            vhh_features = self.detect_vhh_signatures(vh_seq)
            features.update(vhh_features)
            
            # Overall CDR characteristics
            total_cdr_length = sum(len(cdr) for cdr in list(vh_cdrs.values()) + list(vl_cdrs.values()))
            features['total_cdr_length'] = total_cdr_length
            features['cdr_length_ratio'] = total_cdr_length / (len(vh_seq) + len(vl_seq))
            
            # CDR charge and hydrophobicity
            all_cdr_seq = ''.join(list(vh_cdrs.values()) + list(vl_cdrs.values()))
            cdr_composition = self.calculate_aa_composition(all_cdr_seq)
            features.update({f'total_cdr_{k}': v for k, v in cdr_composition.items()})
            
            logger.debug(f"CDR features extracted for {antibody_id}")
            
        except Exception as e:
            logger.error(f"Failed to extract CDR features for {antibody_id}: {str(e)}")
            # Return minimal features on error
            features.update({
                'extraction_error': True,
                'error_message': str(e)
            })
        
        return features
    
    def extract_cdr_features_dataset(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Extract CDR features for entire dataset"""
        logger.info(f"Extracting CDR features from {input_file}")
        
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} antibodies")
        
        # Extract features for each antibody
        all_features = []
        
        for idx, row in df.iterrows():
            antibody_id = row['antibody_id'] if 'antibody_id' in row else row['antibody_name']
            vh_seq = row['vh_protein_sequence']
            vl_seq = row['vl_protein_sequence']
            
            features = self.extract_cdr_features_single(antibody_id, vh_seq, vl_seq)
            all_features.append(features)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} antibodies")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save if output file specified
        if output_file:
            features_df.to_csv(output_file, index=False)
            logger.info(f"CDR features saved to {output_file}")
        
        logger.info(f"CDR feature extraction complete: {len(features_df)} antibodies, {len(features_df.columns)} features")
        
        return features_df
    
    def generate_cdr_feature_report(self, features_df: pd.DataFrame) -> Dict:
        """Generate comprehensive report on CDR features"""
        report = {
            'generation_time': datetime.now().isoformat(),
            'evidence_basis': [
                'citation_cdr_analysis_1760067088.md',
                'citation_cdr_analysis_1760106901.md', 
                'citation_vhh_research_1760017144.md'
            ],
            'dataset_summary': {
                'total_antibodies': len(features_df),
                'successful_extractions': len(features_df[~features_df.get('extraction_error', False)]),
                'extraction_success_rate': (len(features_df[~features_df.get('extraction_error', False)]) / len(features_df)) * 100
            },
            'cdr_h3_analysis': {},
            'vhh_analysis': {},
            'feature_statistics': {}
        }
        
        # CDR-H3 analysis
        if 'cdr_h3_length' in features_df.columns:
            h3_lengths = features_df['cdr_h3_length'].dropna()
            report['cdr_h3_analysis'] = {
                'length_distribution': {
                    'mean': float(h3_lengths.mean()),
                    'std': float(h3_lengths.std()),
                    'min': int(h3_lengths.min()),
                    'max': int(h3_lengths.max()),
                    'median': float(h3_lengths.median())
                }
            }
            
            # Length categories
            if 'cdr_h3_category' in features_df.columns:
                category_counts = features_df['cdr_h3_category'].value_counts()
                report['cdr_h3_analysis']['category_distribution'] = category_counts.to_dict()
        
        # VHH analysis
        if 'is_vhh_candidate' in features_df.columns:
            vhh_count = features_df['is_vhh_candidate'].sum()
            report['vhh_analysis'] = {
                'vhh_candidates': int(vhh_count),
                'vhh_percentage': float((vhh_count / len(features_df)) * 100)
            }
        
        # Feature statistics
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        report['feature_statistics'] = {
            'total_features': len(features_df.columns),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(features_df.columns) - len(numeric_cols)
        }
        
        return report

def main():
    """Main execution function for CDR feature engineering"""
    # Initialize feature extractor
    cdr_features = EvidenceBasedCDRFeatures()
    
    # Extract features from primary dataset
    input_file = "/workspaces/bitcore/workspace/data/sequences/GDPa1_v1.2_sequences.csv"
    output_file = "/workspaces/bitcore/workspace/data/features/cdr_features_evidence_based.csv"
    
    logger.info("Starting evidence-based CDR feature extraction")
    
    # Extract features
    features_df = cdr_features.extract_cdr_features_dataset(input_file, output_file)
    
    # Generate report
    report = cdr_features.generate_cdr_feature_report(features_df)
    
    # Save report
    report_file = "/workspaces/bitcore/workspace/data/features/cdr_features_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"CDR feature report saved to {report_file}")
    
    print(f"CDR Feature Engineering Complete:")
    print(f"- Features extracted: {len(features_df.columns)}")
    print(f"- Antibodies processed: {len(features_df)}")
    print(f"- Success rate: {report['dataset_summary']['extraction_success_rate']:.1f}%")
    print(f"- Output files: {output_file}, {report_file}")
    
    return features_df, report

if __name__ == "__main__":
    main()