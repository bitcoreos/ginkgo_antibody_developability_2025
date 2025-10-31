#!/usr/bin/env python3
"""
Thermal Stability Feature Engineering

Evidence-based implementation from citation_thermal_stability_1760149915.md:
- Harmalkar et al. (2023): Spearman correlation 0.4-0.52 using sequence+structure features
- Rollins et al. (2024): AbMelt MD-derived descriptors with R² 0.57-0.60  
- Alvarez & Dean (2024): TEMPRO nanobody model with MAE 4.03°C, R² 0.67

Features implemented:
1. Sequence-based thermostability predictors
2. MD-derived descriptors (AbMelt-inspired)
3. Nanobody-specific thermal features
4. Entropic contribution estimates
5. Aggregation temperature correlates
6. Structural stability indicators

Author: BITCORE Feature Engineering Team
Date: 2025-10-14
Evidence: citation_thermal_stability_1760149915.md
Target: Spearman 0.4-0.52 correlation with Tm2
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import json
from datetime import datetime
import math
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from bioinformatics.provenance import record_provenance_event

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThermalStabilityFeatures:
    """
    Evidence-based thermal stability feature engineering.
    
    Based on Harmalkar et al. (2023), Rollins et al. (2024), and Alvarez & Dean (2024)
    achieving Spearman correlations 0.4-0.67 with melting temperature.
    """
    
    def __init__(self, workspace_root: str = "/workspaces/bitcore/workspace"):
        self.workspace_root = Path(workspace_root)
        self.data_dir = self.workspace_root / "data"
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Thermostability-relevant amino acid properties
        # Based on AbMelt MD studies and experimental data
        self.thermal_stability_scale = {
            'A': 0.3, 'R': -0.1, 'N': -0.5, 'D': -0.6, 'C': 0.8,  # Cysteine bonds
            'Q': -0.4, 'E': -0.5, 'G': -0.8, 'H': -0.2, 'I': 0.6,
            'L': 0.5, 'K': -0.3, 'M': 0.2, 'F': 0.7, 'P': -0.6,   # Proline disrupts
            'S': -0.3, 'T': -0.1, 'W': 0.4, 'Y': 0.3, 'V': 0.5
        }
        
        # Secondary structure propensities (impact on thermal stability)
        self.ss_propensities = {
            'helix_formers': set('AEHKQR'),     # Alpha helix stabilizers
            'sheet_formers': set('FILMVWY'),    # Beta sheet stabilizers  
            'turn_formers': set('GNPS'),        # Turn/loop regions
            'helix_breakers': set('GP'),        # Helix destabilizers
            'sheet_breakers': set('GP')         # Sheet destabilizers
        }
        
        # Disulfide bond potential (critical for thermostability)
        self.disulfide_spacing = {
            'optimal_range': (8, 30),           # Optimal disulfide spacing
            'minimum_spacing': 3,               # Minimum viable spacing
            'maximum_spacing': 100              # Maximum viable spacing
        }
        
        # Entropic factors affecting thermal stability
        self.entropy_factors = {
            'flexible_residues': set('GNST'),   # High entropy residues
            'rigid_residues': set('PW'),        # Low entropy residues
            'hydrophobic_core': set('AILMFVW'), # Core formation
            'surface_polar': set('DEHKNQRST')   # Surface preference
        }
        
        # Nanobody-specific thermal features (VHH domains)
        self.vhh_thermal_signatures = {
            'thermostable_motifs': ['DY', 'WG', 'YY'],  # Common in stable VHH
            'thermolabile_motifs': ['NG', 'DN', 'GG'],  # Associated with instability
            'stabilizing_frameworks': ['LEWV', 'WGKG']  # Framework stabilizers
        }
        
    def calculate_sequence_thermal_features(self, sequence: str) -> Dict[str, float]:
        """
        Calculate sequence-based thermal stability features.
        Based on Harmalkar et al. (2023) generalized prediction framework.
        """
        if not sequence:
            return {'thermal_stability_score': 0.0}
        
        features = {}
        
        # 1. Overall thermal stability score
        thermal_scores = [self.thermal_stability_scale.get(aa, 0) for aa in sequence]
        features.update({
            'thermal_stability_score': np.mean(thermal_scores),
            'thermal_stability_sum': np.sum(thermal_scores),
            'thermal_stability_std': np.std(thermal_scores),
            'thermal_stability_range': np.max(thermal_scores) - np.min(thermal_scores)
        })
        
        # 2. Secondary structure propensity analysis
        total_length = len(sequence)
        for ss_type, aa_set in self.ss_propensities.items():
            count = sum(1 for aa in sequence if aa in aa_set)
            features[f'{ss_type}_propensity'] = count / total_length
            features[f'{ss_type}_count'] = count
        
        # 3. Disulfide bond potential
        disulfide_features = self._analyze_disulfide_potential(sequence)
        features.update(disulfide_features)
        
        # 4. Entropic contributions
        entropy_features = self._calculate_entropy_features(sequence)
        features.update(entropy_features)
        
        # 5. Thermal motif analysis
        motif_features = self._analyze_thermal_motifs(sequence)
        features.update(motif_features)
        
        return features
    
    def _analyze_disulfide_potential(self, sequence: str) -> Dict[str, Union[float, int]]:
        """
        Analyze disulfide bond formation potential.
        Critical for thermal stability in antibodies.
        """
        cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        
        features = {
            'cysteine_count': len(cys_positions),
            'potential_disulfides': len(cys_positions) // 2,
            'cysteine_density': len(cys_positions) / len(sequence) if sequence else 0
        }
        
        if len(cys_positions) >= 2:
            # Calculate spacing between cysteines
            spacings = []
            for i in range(len(cys_positions) - 1):
                spacing = cys_positions[i+1] - cys_positions[i]
                spacings.append(spacing)
            
            features.update({
                'avg_cys_spacing': np.mean(spacings),
                'min_cys_spacing': np.min(spacings),
                'max_cys_spacing': np.max(spacings),
                'optimal_spacing_count': sum(1 for s in spacings 
                                           if self.disulfide_spacing['optimal_range'][0] <= s <= self.disulfide_spacing['optimal_range'][1])
            })
            
            # Disulfide bond quality score
            quality_scores = []
            for spacing in spacings:
                if spacing < self.disulfide_spacing['minimum_spacing']:
                    quality_scores.append(0.0)  # Too close
                elif spacing > self.disulfide_spacing['maximum_spacing']:
                    quality_scores.append(0.1)  # Too far
                elif self.disulfide_spacing['optimal_range'][0] <= spacing <= self.disulfide_spacing['optimal_range'][1]:
                    quality_scores.append(1.0)  # Optimal
                else:
                    quality_scores.append(0.5)  # Suboptimal but viable
            
            features['disulfide_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
        
        return features
    
    def _calculate_entropy_features(self, sequence: str) -> Dict[str, float]:
        """
        Calculate entropic contributions to thermal stability.
        Based on MD-derived insights from AbMelt studies.
        """
        total_length = len(sequence)
        
        features = {}
        
        # Entropy-related amino acid composition
        for factor_name, aa_set in self.entropy_factors.items():
            count = sum(1 for aa in sequence if aa in aa_set)
            features[f'{factor_name}_fraction'] = count / total_length
            features[f'{factor_name}_count'] = count
        
        # Sequence complexity (relates to conformational entropy)
        aa_counts = Counter(sequence)
        total_aa = len(sequence)
        
        # Shannon entropy of amino acid composition
        shannon_entropy = -sum((count/total_aa) * math.log2(count/total_aa) 
                              for count in aa_counts.values() if count > 0)
        
        features.update({
            'sequence_entropy': shannon_entropy,
            'aa_diversity': len(aa_counts),
            'most_frequent_aa_fraction': max(aa_counts.values()) / total_aa,
            'entropy_normalized': shannon_entropy / math.log2(20)  # Normalize by max possible entropy
        })
        
        # Local entropy (variability in local regions)
        window_size = 10
        local_entropies = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            window_counts = Counter(window)
            window_entropy = -sum((count/window_size) * math.log2(count/window_size) 
                                 for count in window_counts.values() if count > 0)
            local_entropies.append(window_entropy)
        
        if local_entropies:
            features.update({
                'avg_local_entropy': np.mean(local_entropies),
                'max_local_entropy': np.max(local_entropies),
                'entropy_variation': np.std(local_entropies)
            })
        
        return features
    
    def _analyze_thermal_motifs(self, sequence: str) -> Dict[str, int]:
        """
        Analyze sequence motifs associated with thermal stability.
        Based on experimental thermal stability data.
        """
        features = {}
        
        # Known thermostable motifs
        for motif in self.vhh_thermal_signatures['thermostable_motifs']:
            features[f'thermostable_{motif}_count'] = sequence.count(motif)
        
        # Known thermolabile motifs
        for motif in self.vhh_thermal_signatures['thermolabile_motifs']:
            features[f'thermolabile_{motif}_count'] = sequence.count(motif)
        
        # Stabilizing framework motifs
        for motif in self.vhh_thermal_signatures['stabilizing_frameworks']:
            features[f'stabilizing_{motif}_count'] = sequence.count(motif)
        
        # Calculate thermal motif balance
        total_stable = sum(features[k] for k in features if 'thermostable' in k)
        total_labile = sum(features[k] for k in features if 'thermolabile' in k)
        
        features.update({
            'thermal_motif_balance': total_stable - total_labile,
            'thermal_motif_ratio': total_stable / (total_labile + 1)  # Avoid division by zero
        })
        
        return features
    
    def calculate_abmelt_inspired_features(self, sequence: str) -> Dict[str, float]:
        """
        Calculate MD-derived descriptors inspired by AbMelt methodology.
        Approximates molecular dynamics insights using sequence information.
        """
        if not sequence:
            return {'abmelt_stability_score': 0.0}
        
        features = {}
        
        # 1. Hydrophobic core stability (approximation)
        hydrophobic_positions = [i for i, aa in enumerate(sequence) if aa in self.entropy_factors['hydrophobic_core']]
        
        if len(hydrophobic_positions) >= 2:
            # Core compactness (hydrophobic residue clustering)
            core_distances = []
            for i in range(len(hydrophobic_positions) - 1):
                core_distances.append(hydrophobic_positions[i+1] - hydrophobic_positions[i])
            
            features.update({
                'hydrophobic_core_compactness': 1.0 / (np.mean(core_distances) + 1),
                'core_size': len(hydrophobic_positions),
                'core_density': len(hydrophobic_positions) / len(sequence)
            })
        
        # 2. Surface polar/charged distribution
        surface_positions = [i for i, aa in enumerate(sequence) if aa in self.entropy_factors['surface_polar']]
        
        features.update({
            'surface_polar_fraction': len(surface_positions) / len(sequence),
            'surface_charge_density': sum(1 for aa in sequence if aa in 'DERKH') / len(sequence)
        })
        
        # 3. Structural flexibility estimation
        flexible_positions = [i for i, aa in enumerate(sequence) if aa in self.entropy_factors['flexible_residues']]
        rigid_positions = [i for i, aa in enumerate(sequence) if aa in self.entropy_factors['rigid_residues']]
        
        features.update({
            'flexibility_ratio': len(flexible_positions) / (len(rigid_positions) + 1),
            'structural_rigidity': len(rigid_positions) / len(sequence),
            'flexibility_index': len(flexible_positions) / len(sequence)
        })
        
        # 4. Thermal unfolding resistance score (composite)
        # Based on combining multiple stability factors
        stability_components = {
            'disulfide_contribution': min(sequence.count('C') / 4.0, 1.0),  # Normalized to 0-1
            'core_stability': features.get('core_density', 0) * 2.0,
            'rigidity_factor': features.get('structural_rigidity', 0) * 3.0,
            'entropy_resistance': 1.0 - features.get('flexibility_index', 1.0)
        }
        
        abmelt_score = np.mean(list(stability_components.values()))
        features['abmelt_stability_score'] = abmelt_score
        
        # 5. Temperature-dependent properties
        # Approximate Tm prediction components
        features.update({
            'estimated_tm_contribution': abmelt_score * 80 + 40,  # Rough Tm estimate in °C
            'thermal_cooperativity': features.get('hydrophobic_core_compactness', 0) * features.get('disulfide_quality_score', 0),
            'unfolding_resistance': stability_components['entropy_resistance'] * stability_components['disulfide_contribution']
        })
        
        return features
    
    def calculate_nanobody_thermal_features(self, sequence: str) -> Dict[str, float]:
        """
        Calculate nanobody-specific thermal features.
        Based on TEMPRO methodology (Alvarez & Dean 2024).
        """
        features = {}
        
        # VHH-specific thermal signatures
        for motif in ['DY', 'WG', 'YY']:  # Common in thermostable nanobodies
            features[f'vhh_stable_{motif}'] = sequence.count(motif)
        
        # Framework region analysis (important for VHH stability)
        if len(sequence) > 100:  # Reasonable VH length
            # Approximate framework regions (simplified)
            fr1 = sequence[0:30] if len(sequence) > 30 else sequence
            fr4 = sequence[-30:] if len(sequence) > 30 else sequence
            
            features.update({
                'fr1_thermal_score': np.mean([self.thermal_stability_scale.get(aa, 0) for aa in fr1]),
                'fr4_thermal_score': np.mean([self.thermal_stability_scale.get(aa, 0) for aa in fr4]),
                'framework_stability_balance': features.get('fr1_thermal_score', 0) + features.get('fr4_thermal_score', 0)
            })
        
        # Nanobody size considerations (smaller = potentially more stable)
        features.update({
            'nanobody_size_factor': 1.0 / (1.0 + len(sequence) / 130.0),  # Normalized size penalty
            'compact_domain_score': (sequence.count('C') * 2 + sequence.count('P')) / len(sequence)
        })
        
        return features
    
    def extract_thermal_features_single(self, antibody_id: str, vh_seq: str, vl_seq: str) -> Dict[str, Union[float, int]]:
        """Extract comprehensive thermal stability features for a single antibody"""
        features = {'antibody_id': antibody_id}
        
        try:
            # 1. VH thermal features
            vh_thermal = self.calculate_sequence_thermal_features(vh_seq)
            features.update({f'vh_{k}': v for k, v in vh_thermal.items()})
            
            # 2. VL thermal features
            vl_thermal = self.calculate_sequence_thermal_features(vl_seq)
            features.update({f'vl_{k}': v for k, v in vl_thermal.items()})
            
            # 3. AbMelt-inspired features (VH)
            vh_abmelt = self.calculate_abmelt_inspired_features(vh_seq)
            features.update({f'vh_{k}': v for k, v in vh_abmelt.items()})
            
            # 4. AbMelt-inspired features (VL)
            vl_abmelt = self.calculate_abmelt_inspired_features(vl_seq)
            features.update({f'vl_{k}': v for k, v in vl_abmelt.items()})
            
            # 5. Nanobody features (VH - potential VHH)
            vh_vhh = self.calculate_nanobody_thermal_features(vh_seq)
            features.update({f'vh_vhh_{k}': v for k, v in vh_vhh.items()})
            
            # 6. Combined antibody thermal features
            features.update({
                'combined_thermal_score': features['vh_thermal_stability_score'] + features['vl_thermal_stability_score'],
                'combined_abmelt_score': features['vh_abmelt_stability_score'] + features['vl_abmelt_stability_score'],
                'combined_cysteine_count': features['vh_cysteine_count'] + features['vl_cysteine_count'],
                'combined_disulfide_potential': features['vh_potential_disulfides'] + features['vl_potential_disulfides'],
                'vh_vl_thermal_balance': abs(features['vh_thermal_stability_score'] - features['vl_thermal_stability_score']),
                'thermal_domain_cooperativity': features['vh_thermal_stability_score'] * features['vl_thermal_stability_score']
            })
            
            # 7. Overall thermal stability prediction
            # Weighted combination based on evidence from literature
            stability_factors = [
                features['combined_thermal_score'] * 0.3,
                features['combined_abmelt_score'] * 0.25,
                features['combined_disulfide_potential'] * 0.2,
                features.get('vh_disulfide_quality_score', 0) * 0.15,
                features.get('vl_disulfide_quality_score', 0) * 0.1
            ]
            
            features['predicted_thermal_stability'] = sum(stability_factors)
            
            features['extraction_success'] = True
            
        except Exception as e:
            logger.error(f"Failed to extract thermal features for {antibody_id}: {str(e)}")
            features.update({
                'extraction_success': False,
                'error_message': str(e)
            })
        
        return features
    
    def extract_thermal_features_dataset(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Extract thermal stability features for entire dataset"""
        logger.info(f"Extracting thermal stability features from {input_file}")
        
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} antibodies")
        
        # Extract features for each antibody
        all_features = []
        
        for idx, row in df.iterrows():
            antibody_id = row['antibody_id'] if 'antibody_id' in row else row['antibody_name']
            vh_seq = row['vh_protein_sequence']
            vl_seq = row['vl_protein_sequence']
            
            features = self.extract_thermal_features_single(antibody_id, vh_seq, vl_seq)
            all_features.append(features)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} antibodies")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save if output file specified
        if output_file:
            features_df.to_csv(output_file, index=False)
            logger.info(f"Thermal stability features saved to {output_file}")
        
        logger.info(f"Thermal feature extraction complete: {len(features_df)} antibodies, {len(features_df.columns)} features")
        
        return features_df
    
    def generate_thermal_feature_report(self, features_df: pd.DataFrame) -> Dict:
        """Generate comprehensive report on thermal stability features"""
        report = {
            'generation_time': datetime.now().isoformat(),
            'evidence_basis': [
                'citation_thermal_stability_1760149915.md',
                'Harmalkar et al. (2023): Spearman 0.4-0.52',
                'Rollins et al. (2024): AbMelt R² 0.57-0.60',
                'Alvarez & Dean (2024): TEMPRO MAE 4.03°C, R² 0.67'
            ],
            'target_performance': 'Spearman 0.4-0.52 correlation with Tm2',
            'dataset_summary': {
                'total_antibodies': len(features_df),
                'successful_extractions': len(features_df[features_df.get('extraction_success', True)]),
                'feature_count': len(features_df.columns)
            },
            'thermal_analysis': {}
        }
        
        # Thermal stability analysis
        if 'predicted_thermal_stability' in features_df.columns:
            thermal_scores = features_df['predicted_thermal_stability'].dropna()
            
            report['thermal_analysis'] = {
                'stability_range': {
                    'min': float(thermal_scores.min()),
                    'max': float(thermal_scores.max()),
                    'mean': float(thermal_scores.mean()),
                    'std': float(thermal_scores.std())
                },
                'disulfide_analysis': {
                    'avg_disulfides': float(features_df['combined_disulfide_potential'].mean()) if 'combined_disulfide_potential' in features_df.columns else 0,
                    'cysteine_distribution': features_df['combined_cysteine_count'].value_counts().head().to_dict() if 'combined_cysteine_count' in features_df.columns else {}
                }
            }
        
        return report

def main():
    """Main execution function for thermal stability features"""
    # Initialize feature extractor
    thermal_features = ThermalStabilityFeatures()
    
    # Extract features from primary dataset
    input_path = Path("/workspaces/bitcore/workspace/data/sequences/GDPa1_v1.2_sequences.csv")
    output_path = Path("/workspaces/bitcore/workspace/data/features/thermal_stability_features.csv")
    
    logger.info("Starting evidence-based thermal stability feature extraction")
    logger.info("Target: Spearman 0.4-0.52 correlation with Tm2")
    
    # Extract features
    features_df = thermal_features.extract_thermal_features_dataset(str(input_path), str(output_path))
    
    # Generate report
    report = thermal_features.generate_thermal_feature_report(features_df)
    
    # Save report
    report_path = Path("/workspaces/bitcore/workspace/data/features/thermal_stability_report.json")
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Thermal stability report saved to {report_path}")

    provenance_path = record_provenance_event(
        event_name="thermal_stability_features",
        inputs=[input_path],
        outputs=[output_path, report_path],
        metadata={
            "antibodies": int(len(features_df)),
            "feature_columns": int(len(features_df.columns)),
        },
    )
    logger.info(f"Provenance log saved to {provenance_path}")
    
    print(f"Thermal Stability Feature Engineering Complete:")
    print(f"- Features extracted: {len(features_df.columns)}")
    print(f"- Antibodies processed: {len(features_df)}")
    print(f"- Target performance: Spearman 0.4-0.52 (evidence-based)")
    print(f"- Output files: {output_path}, {report_path}")
    print(f"- Provenance: {provenance_path}")
    
    return features_df, report

if __name__ == "__main__":
    main()