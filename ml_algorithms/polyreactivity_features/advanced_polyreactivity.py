"""
Advanced Polyreactivity Features Implementation

This module implements advanced polyreactivity features including VH/VL charge imbalance,
residue clustering patterns, hydrophobic patch analysis, paratope dynamics proxies,
and PSR/PSP assay mapping.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Amino acid properties
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
CHARGED_AA = {'positive': 'KR', 'negative': 'DE'}
HYDROPHOBIC_AA = 'AILMFWV'
POLAR_AA = 'NQST'


class ChargeImbalanceAnalyzer:
    """
    VH/VL charge imbalance analysis beyond basic net charge.
    """
    
    def __init__(self):
        """
        Initialize the charge imbalance analyzer.
        """
        self.positive_aa = set(CHARGED_AA['positive'])
        self.negative_aa = set(CHARGED_AA['negative'])
    
    def calculate_charge_distribution(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[float, Dict]]:
        """
        Calculate charge distribution for VH and VL domains.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain domain sequence
            
        Returns:
            Dict[str, Union[float, Dict]]: Charge distribution analysis
        """
        # Calculate charges for VH
        vh_pos_count = sum(1 for aa in vh_sequence if aa in self.positive_aa)
        vh_neg_count = sum(1 for aa in vh_sequence if aa in self.negative_aa)
        vh_net_charge = vh_pos_count - vh_neg_count
        
        # Calculate charges for VL
        vl_pos_count = sum(1 for aa in vl_sequence if aa in self.positive_aa)
        vl_neg_count = sum(1 for aa in vl_sequence if aa in self.negative_aa)
        vl_net_charge = vl_pos_count - vl_neg_count
        
        # Charge imbalance metrics
        charge_imbalance = abs(vh_net_charge - vl_net_charge)
        charge_ratio = (vh_pos_count + vh_neg_count + 1) / (vl_pos_count + vl_neg_count + 1)
        
        return {
            'vh_domain': {
                'positive_charged': vh_pos_count,
                'negative_charged': vh_neg_count,
                'net_charge': vh_net_charge,
                'sequence_length': len(vh_sequence)
            },
            'vl_domain': {
                'positive_charged': vl_pos_count,
                'negative_charged': vl_neg_count,
                'net_charge': vl_net_charge,
                'sequence_length': len(vl_sequence)
            },
            'charge_imbalance_metrics': {
                'absolute_imbalance': float(charge_imbalance),
                'charge_ratio': float(charge_ratio),
                'total_charge_difference': float(abs(vh_pos_count + vh_neg_count - vl_pos_count - vl_neg_count))
            }
        }
    
    def analyze_charge_clustering(self, sequence: str, window_size: int = 5) -> Dict[str, Union[float, List]]:
        """
        Analyze clustering of charged residues.
        
        Args:
            sequence (str): Protein sequence
            window_size (int): Size of sliding window for clustering analysis
            
        Returns:
            Dict[str, Union[float, List]]: Charge clustering analysis
        """
        charged_positions = []
        for i, aa in enumerate(sequence):
            if aa in self.positive_aa or aa in self.negative_aa:
                charged_positions.append(i)
        
        if len(charged_positions) < 2:
            return {
                'clustering_score': 0.0,
                'clustered_regions': [],
                'max_cluster_size': 0
            }
        
        # Calculate clustering score based on distances between charged residues
        distances = [charged_positions[i+1] - charged_positions[i] for i in range(len(charged_positions)-1)]
        avg_distance = np.mean(distances)
        clustering_score = 1.0 / (avg_distance + 1)  # Higher score for closer clustering
        
        # Identify clustered regions
        clustered_regions = []
        current_cluster = [charged_positions[0]]
        
        for i in range(1, len(charged_positions)):
            if charged_positions[i] - charged_positions[i-1] <= window_size:
                current_cluster.append(charged_positions[i])
            else:
                if len(current_cluster) > 1:
                    clustered_regions.append({
                        'start': current_cluster[0],
                        'end': current_cluster[-1],
                        'size': len(current_cluster),
                        'positions': current_cluster
                    })
                current_cluster = [charged_positions[i]]
        
        # Add last cluster if it has more than one residue
        if len(current_cluster) > 1:
            clustered_regions.append({
                'start': current_cluster[0],
                'end': current_cluster[-1],
                'size': len(current_cluster),
                'positions': current_cluster
            })
        
        max_cluster_size = max([region['size'] for region in clustered_regions], default=0)
        
        return {
            'clustering_score': float(clustering_score),
            'clustered_regions': clustered_regions,
            'max_cluster_size': max_cluster_size
        }


class ResidueClusteringAnalyzer:
    """
    Basic residue clustering patterns analysis (not just density).
    """
    
    def __init__(self):
        """
        Initialize the residue clustering analyzer.
        """
        pass
    
    def analyze_residue_clustering(self, sequence: str, residue_types: List[str] = None, 
                                 window_size: int = 5) -> Dict[str, Union[float, Dict]]:
        """
        Analyze clustering patterns of specific residue types.
        
        Args:
            sequence (str): Protein sequence
            residue_types (List[str]): List of residue types to analyze (default: hydrophobic)
            window_size (int): Size of sliding window for clustering analysis
            
        Returns:
            Dict[str, Union[float, Dict]]: Residue clustering analysis
        """
        if residue_types is None:
            residue_types = list(HYDROPHOBIC_AA)
        
        # Convert to set for faster lookup
        residue_set = set(residue_types)
        
        # Find positions of specified residues
        residue_positions = defaultdict(list)
        for i, aa in enumerate(sequence):
            if aa in residue_set:
                residue_positions[aa].append(i)
        
        # Analyze clustering for each residue type
        clustering_results = {}
        overall_clustering_score = 0.0
        
        for residue, positions in residue_positions.items():
            if len(positions) < 2:
                clustering_results[residue] = {
                    'clustering_score': 0.0,
                    'clustered_regions': [],
                    'max_cluster_size': 0
                }
                continue
            
            # Calculate clustering score based on distances between residues
            distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_distance = np.mean(distances) if distances else 0
            clustering_score = 1.0 / (avg_distance + 1)  # Higher score for closer clustering
            
            # Identify clustered regions
            clustered_regions = []
            current_cluster = [positions[0]]
            
            for i in range(1, len(positions)):
                if positions[i] - positions[i-1] <= window_size:
                    current_cluster.append(positions[i])
                else:
                    if len(current_cluster) > 1:
                        clustered_regions.append({
                            'start': current_cluster[0],
                            'end': current_cluster[-1],
                            'size': len(current_cluster),
                            'positions': current_cluster
                        })
                    current_cluster = [positions[i]]
            
            # Add last cluster if it has more than one residue
            if len(current_cluster) > 1:
                clustered_regions.append({
                    'start': current_cluster[0],
                    'end': current_cluster[-1],
                    'size': len(current_cluster),
                    'positions': current_cluster
                })
            
            max_cluster_size = max([region['size'] for region in clustered_regions], default=0)
            
            clustering_results[residue] = {
                'clustering_score': float(clustering_score),
                'clustered_regions': clustered_regions,
                'max_cluster_size': max_cluster_size
            }
            
            overall_clustering_score += clustering_score
        
        # Calculate overall clustering score
        overall_clustering_score = overall_clustering_score / len(residue_positions) if residue_positions else 0.0
        
        return {
            'overall_clustering_score': float(overall_clustering_score),
            'residue_clustering': clustering_results,
            'analyzed_residues': residue_types
        }


class HydrophobicPatchAnalyzer:
    """
    Hydrophobic patch analysis for surface binding prediction.
    """
    
    def __init__(self):
        ""
        Initialize the hydrophobic patch analyzer.
        """
        self.hydrophobic_aa = set(HYDROPHOBIC_AA)
    
    def identify_hydrophobic_patches(self, sequence: str, min_patch_size: int = 3, 
                                   window_size: int = 10) -> Dict[str, Union[int, List]]:
        """
        Identify hydrophobic patches in a sequence.
        
        Args:
            sequence (str): Protein sequence
            min_patch_size (int): Minimum size of hydrophobic patch
            window_size (int): Size of sliding window for patch analysis
            
        Returns:
            Dict[str, Union[int, List]]: Hydrophobic patch analysis
        """
        # Find hydrophobic patches
        patches = []
        current_patch = []
        
        for i, aa in enumerate(sequence):
            if aa in self.hydrophobic_aa:
                current_patch.append((i, aa))
            else:
                # End of current patch
                if len(current_patch) >= min_patch_size:
                    patches.append({
                        'start': current_patch[0][0],
                        'end': current_patch[-1][0],
                        'size': len(current_patch),
                        'residues': [aa for _, aa in current_patch],
                        'positions': [pos for pos, _ in current_patch]
                    })
                current_patch = []
        
        # Add last patch if it meets minimum size
        if len(current_patch) >= min_patch_size:
            patches.append({
                'start': current_patch[0][0],
                'end': current_patch[-1][0],
                'size': len(current_patch),
                'residues': [aa for _, aa in current_patch],
                'positions': [pos for pos, _ in current_patch]
            })
        
        # Calculate patch density
        total_hydrophobic = sum(1 for aa in sequence if aa in self.hydrophobic_aa)
        patch_density = total_hydrophobic / len(sequence) if sequence else 0.0
        
        # Calculate largest patch
        largest_patch = max([patch['size'] for patch in patches], default=0)
        
        return {
            'total_patches': len(patches),
            'patches': patches,
            'patch_density': float(patch_density),
            'largest_patch_size': largest_patch,
            'total_hydrophobic_residues': total_hydrophobic
        }
    
    def predict_surface_binding_potential(self, patches: List[Dict]) -> Dict[str, Union[float, str]]:
        """
        Predict surface binding potential based on hydrophobic patches.
        
        Args:
            patches (List[Dict]): List of identified hydrophobic patches
            
        Returns:
            Dict[str, Union[float, str]]: Surface binding potential prediction
        """
        if not patches:
            return {
                'binding_potential': 0.0,
                'risk_level': 'low',
                'recommendation': 'No significant hydrophobic patches identified'
            }
        
        # Calculate binding potential based on patch characteristics
        total_patch_size = sum(patch['size'] for patch in patches)
        largest_patch = max(patch['size'] for patch in patches)
        patch_count = len(patches)
        
        # Simple scoring system
        size_score = min(total_patch_size / 10.0, 1.0)  # Normalize by 10
        largest_score = min(largest_patch / 5.0, 1.0)   # Normalize by 5
        count_score = min(patch_count / 3.0, 1.0)       # Normalize by 3
        
        # Combined score
        binding_potential = (size_score + largest_score + count_score) / 3.0
        
        # Determine risk level
        if binding_potential < 0.3:
            risk_level = 'low'
            recommendation = 'Low surface binding potential'
        elif binding_potential < 0.7:
            risk_level = 'medium'
            recommendation = 'Moderate surface binding potential - consider optimization'
        else:
            risk_level = 'high'
            recommendation = 'High surface binding potential - significant optimization recommended'
        
        return {
            'binding_potential': float(binding_potential),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'patch_metrics': {
                'total_size': total_patch_size,
                'largest_patch': largest_patch,
                'patch_count': patch_count
            }
        }


class ParatopeDynamicsAnalyzer:
    """
    Paratope dynamics proxies (entropy of predicted paratope states).
    """
    
    def __init__(self):
        """
        Initialize the paratope dynamics analyzer.
        """
        pass
    
    def calculate_paratope_entropy(self, paratope_predictions: List[float], 
                                bins: int = 10) -> Dict[str, Union[float, Dict]]:
        """
        Calculate entropy of predicted paratope states as a proxy for dynamics.
        
        Args:
            paratope_predictions (List[float]): List of paratope prediction probabilities
            bins (int): Number of bins for histogram calculation
            
        Returns:
            Dict[str, Union[float, Dict]]: Paratope dynamics analysis
        """
        if not paratope_predictions:
            return {
                'entropy': 0.0,
                'dynamics_score': 0.0,
                'distribution_metrics': {}
            }
        
        # Convert to numpy array
        predictions = np.array(paratope_predictions)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(predictions, bins=bins, range=(0, 1), density=True)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        hist = hist + epsilon
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist))
        
        # Normalize entropy (maximum entropy for uniform distribution)
        max_entropy = np.log(bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Dynamics score (higher entropy = higher dynamics)
        dynamics_score = normalized_entropy
        
        # Distribution metrics
        distribution_metrics = {
            'mean_probability': float(np.mean(predictions)),
            'std_probability': float(np.std(predictions)),
            'min_probability': float(np.min(predictions)),
            'max_probability': float(np.max(predictions)),
            'median_probability': float(np.median(predictions))
        }
        
        return {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'dynamics_score': float(dynamics_score),
            'distribution_metrics': distribution_metrics
        }
    
    def analyze_paratope_stability(self, paratope_predictions: List[float], 
                                 stability_threshold: float = 0.7) -> Dict[str, Union[float, str]]:
        """
        Analyze paratope stability based on prediction consistency.
        
        Args:
            paratope_predictions (List[float]): List of paratope prediction probabilities
            stability_threshold (float): Threshold for considering a residue as stable
            
        Returns:
            Dict[str, Union[float, str]]: Paratope stability analysis
        """
        if not paratope_predictions:
            return {
                'stability_score': 0.0,
                'stability_level': 'low',
                'recommendation': 'No paratope predictions available'
            }
        
        # Calculate stability metrics
        predictions = np.array(paratope_predictions)
        stable_residues = np.sum(predictions >= stability_threshold)
        unstable_residues = len(predictions) - stable_residues
        
        # Stability score
        stability_score = stable_residues / len(predictions) if predictions.size > 0 else 0.0
        
        # Determine stability level
        if stability_score < 0.3:
            stability_level = 'low'
            recommendation = 'Low paratope stability - consider optimization'
        elif stability_score < 0.7:
            stability_level = 'medium'
            recommendation = 'Moderate paratope stability'
        else:
            stability_level = 'high'
            recommendation = 'High paratope stability'
        
        return {
            'stability_score': float(stability_score),
            'stability_level': stability_level,
            'recommendation': recommendation,
            'stability_metrics': {
                'stable_residues': int(stable_residues),
                'unstable_residues': int(unstable_residues),
                'total_residues': len(paratope_predictions)
            }
        }


class PSR_PSP_Analyzer:
    """
    Comprehensive PSR/PSP assay mapping and decision rules.
    """
    
    def __init__(self):
        """
        Initialize the PSR/PSP analyzer.
        """
        pass
    
    def map_psr_psp_assay_results(self, assay_data: Dict[str, Any]) -> Dict[str, Union[float, str, Dict]]:
        """
        Map PSR/PSP assay results to developability predictions.
        
        Args:
            assay_data (Dict[str, Any]): Dictionary containing assay results
            
        Returns:
            Dict[str, Union[float, str, Dict]]: PSR/PSP mapping analysis
        """
        # Extract relevant metrics
        psr_score = assay_data.get('psr_score', 0.0)
        psp_score = assay_data.get('psp_score', 0.0)
        binding_affinity = assay_data.get('binding_affinity', 0.0)
        specificity_ratio = assay_data.get('specificity_ratio', 1.0)
        
        # Calculate composite scores
        polyreactivity_risk = (psr_score + psp_score) / 2.0
        
        # Decision rules
        if polyreactivity_risk < 0.3 and specificity_ratio > 3.0:
            risk_level = 'low'
            recommendation = 'Low polyreactivity risk - suitable for development'
            developability_score = 0.9
        elif polyreactivity_risk < 0.6 and specificity_ratio > 2.0:
            risk_level = 'medium'
            recommendation = 'Moderate polyreactivity risk - consider optimization'
            developability_score = 0.6
        else:
            risk_level = 'high'
            recommendation = 'High polyreactivity risk - significant optimization recommended'
            developability_score = 0.3
        
        return {
            'polyreactivity_risk': float(polyreactivity_risk),
            'risk_level': risk_level,
            'developability_score': float(developability_score),
            'recommendation': recommendation,
            'assay_metrics': {
                'psr_score': float(psr_score),
                'psp_score': float(psp_score),
                'binding_affinity': float(binding_affinity),
                'specificity_ratio': float(specificity_ratio)
            }
        }
    
    def generate_developability_profile(self, charge_analysis: Dict, 
                                      clustering_analysis: Dict,
                                      hydrophobic_analysis: Dict,
                                      paratope_analysis: Dict,
                                      psr_psp_analysis: Dict) -> Dict[str, Union[float, str, Dict]]:
        """
        Generate comprehensive developability profile.
        
        Args:
            charge_analysis (Dict): Charge imbalance analysis results
            clustering_analysis (Dict): Residue clustering analysis results
            hydrophobic_analysis (Dict): Hydrophobic patch analysis results
            paratope_analysis (Dict): Paratope dynamics analysis results
            psr_psp_analysis (Dict): PSR/PSP assay analysis results
            
        Returns:
            Dict[str, Union[float, str, Dict]]: Comprehensive developability profile
        """
        # Extract key metrics
        charge_imbalance = charge_analysis.get('charge_imbalance_metrics', {}).get('absolute_imbalance', 0.0)
        clustering_score = clustering_analysis.get('overall_clustering_score', 0.0)
        binding_potential = hydrophobic_analysis.get('binding_potential', 0.0)
        dynamics_score = paratope_analysis.get('dynamics_score', 0.0)
        polyreactivity_risk = psr_psp_analysis.get('polyreactivity_risk', 0.0)
        developability_score = psr_psp_analysis.get('developability_score', 0.0)
        
        # Calculate composite developability score
        # Weighted average (adjust weights as needed)
        composite_score = (
            (1.0 - min(charge_imbalance / 10.0, 1.0)) * 0.2 +  # Lower imbalance is better
            (1.0 - min(clustering_score, 1.0)) * 0.1 +  # Lower clustering is better
            (1.0 - min(binding_potential, 1.0)) * 0.2 +  # Lower binding potential is better
            (1.0 - min(dynamics_score, 1.0)) * 0.1 +  # Lower dynamics is better
            (1.0 - min(polyreactivity_risk, 1.0)) * 0.3 +  # Lower polyreactivity is better
            developability_score * 0.1  # Direct developability score
        )
        
        # Determine overall risk level
        if composite_score > 0.7:
            overall_risk = 'low'
            recommendation = 'Good developability profile'
        elif composite_score > 0.4:
            overall_risk = 'medium'
            recommendation = 'Moderate developability concerns - consider optimization'
        else:
            overall_risk = 'high'
            recommendation = 'Significant developability concerns - substantial optimization recommended'
        
        return {
            'composite_developability_score': float(composite_score),
            'overall_risk_level': overall_risk,
            'recommendation': recommendation,
            'component_scores': {
                'charge_imbalance': float(charge_imbalance),
                'clustering_score': float(clustering_score),
                'binding_potential': float(binding_potential),
                'dynamics_score': float(dynamics_score),
                'polyreactivity_risk': float(polyreactivity_risk),
                'developability_score': float(developability_score)
            }
        }


def main():
    """
    Example usage of the advanced polyreactivity features implementation.
    """
    # Example VH and VL sequences
    vh_sequence = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSVSSSYLAWYQQKPGKAPKLLIYDASNRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQRSNWPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Charge imbalance analysis
    charge_analyzer = ChargeImbalanceAnalyzer()
    charge_analysis = charge_analyzer.calculate_charge_distribution(vh_sequence, vl_sequence)
    
    print("Charge Imbalance Analysis Results:")
    print(f"  VH Net Charge: {charge_analysis['vh_domain']['net_charge']}")
    print(f"  VL Net Charge: {charge_analysis['vl_domain']['net_charge']}")
    print(f"  Absolute Imbalance: {charge_analysis['charge_imbalance_metrics']['absolute_imbalance']}")
    
    # Charge clustering analysis
    vh_clustering = charge_analyzer.analyze_charge_clustering(vh_sequence)
    vl_clustering = charge_analyzer.analyze_charge_clustering(vl_sequence)
    
    print(f"  VH Clustering Score: {vh_clustering['clustering_score']:.4f}")
    print(f"  VL Clustering Score: {vl_clustering['clustering_score']:.4f}")
    
    # Residue clustering analysis
    clustering_analyzer = ResidueClusteringAnalyzer()
    vh_hydrophobic_clustering = clustering_analyzer.analyze_residue_clustering(vh_sequence, list(HYDROPHOBIC_AA))
    vl_hydrophobic_clustering = clustering_analyzer.analyze_residue_clustering(vl_sequence, list(HYDROPHOBIC_AA))
    
    print("\nResidue Clustering Analysis Results:")
    print(f"  VH Overall Clustering Score: {vh_hydrophobic_clustering['overall_clustering_score']:.4f}")
    print(f"  VL Overall Clustering Score: {vl_hydrophobic_clustering['overall_clustering_score']:.4f}")
    
    # Hydrophobic patch analysis
    hydrophobic_analyzer = HydrophobicPatchAnalyzer()
    vh_patches = hydrophobic_analyzer.identify_hydrophobic_patches(vh_sequence)
    vl_patches = hydrophobic_analyzer.identify_hydrophobic_patches(vl_sequence)
    
    print("\nHydrophobic Patch Analysis Results:")
    print(f"  VH Total Patches: {vh_patches['total_patches']}")
    print(f"  VL Total Patches: {vl_patches['total_patches']}")
    print(f"  VH Largest Patch: {vh_patches['largest_patch_size']}")
    print(f"  VL Largest Patch: {vl_patches['largest_patch_size']}")
    
    # Surface binding potential prediction
    vh_binding = hydrophobic_analyzer.predict_surface_binding_potential(vh_patches['patches'])
    vl_binding = hydrophobic_analyzer.predict_surface_binding_potential(vl_patches['patches'])
    
    print(f"  VH Binding Potential: {vh_binding['binding_potential']:.4f} ({vh_binding['risk_level']})")
    print(f"  VL Binding Potential: {vl_binding['binding_potential']:.4f} ({vl_binding['risk_level']})")
    
    # Paratope dynamics analysis
    paratope_analyzer = ParatopeDynamicsAnalyzer()
    
    # Simulated paratope predictions (in practice, these would come from a model)
    np.random.seed(42)
    vh_paratope_predictions = np.random.beta(2, 5, len(vh_sequence)).tolist()  # Skewed toward lower probabilities
    vl_paratope_predictions = np.random.beta(3, 3, len(vl_sequence)).tolist()  # More uniform distribution
    
    vh_paratope_entropy = paratope_analyzer.calculate_paratope_entropy(vh_paratope_predictions)
    vl_paratope_entropy = paratope_analyzer.calculate_paratope_entropy(vl_paratope_predictions)
    
    print("\nParatope Dynamics Analysis Results:")
    print(f"  VH Dynamics Score: {vh_paratope_entropy['dynamics_score']:.4f}")
    print(f"  VL Dynamics Score: {vl_paratope_entropy['dynamics_score']:.4f}")
    
    # Paratope stability analysis
    vh_paratope_stability = paratope_analyzer.analyze_paratope_stability(vh_paratope_predictions)
    vl_paratope_stability = paratope_analyzer.analyze_paratope_stability(vl_paratope_predictions)
    
    print(f"  VH Stability Score: {vh_paratope_stability['stability_score']:.4f} ({vh_paratope_stability['stability_level']})")
    print(f"  VL Stability Score: {vl_paratope_stability['stability_score']:.4f} ({vl_paratope_stability['stability_level']})")
    
    # PSR/PSP assay mapping
    psr_psp_analyzer = PSR_PSP_Analyzer()
    
    # Simulated assay data
    assay_data = {
        'psr_score': 0.4,
        'psp_score': 0.3,
        'binding_affinity': 8.5,
        'specificity_ratio': 2.8
    }
    
    psr_psp_analysis = psr_psp_analyzer.map_psr_psp_assay_results(assay_data)
    
    print("\nPSR/PSP Assay Mapping Results:")
    print(f"  Polyreactivity Risk: {psr_psp_analysis['polyreactivity_risk']:.4f} ({psr_psp_analysis['risk_level']})")
    print(f"  Developability Score: {psr_psp_analysis['developability_score']:.4f}")
    print(f"  Recommendation: {psr_psp_analysis['recommendation']}")
    
    # Comprehensive developability profile
    developability_profile = psr_psp_analyzer.generate_developability_profile(
        charge_analysis, 
        vh_hydrophobic_clustering,  # Using VH clustering for simplicity
        vh_patches,  # Using VH patches for simplicity
        vh_paratope_entropy,  # Using VH paratope for simplicity
        psr_psp_analysis
    )
    
    print("\nComprehensive Developability Profile:")
    print(f"  Composite Score: {developability_profile['composite_developability_score']:.4f} ({developability_profile['overall_risk_level']})")
    print(f"  Recommendation: {developability_profile['recommendation']}")


if __name__ == "__main__":
    main()
