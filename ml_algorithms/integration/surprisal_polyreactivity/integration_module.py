"""
Integration Module for Surprisal and Polyreactivity Features

This module implements the integration of Markov surprisal features with advanced polyreactivity features
for comprehensive antibody developability prediction.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import sys
import os

# Add paths to import the required modules
sys.path.append('/a0/bitcore/workspace/research/semantic_mesh/markov/src')
sys.path.append('/a0/bitcore/workspace/research/polyreactivity/src')

# Import required modules
try:
    from markov_model_enhanced import MarkovModel, compute_sequence_surprisal_features
    from charge_imbalance import ChargeImbalanceAnalyzer
    from residue_clustering import ResidueClusteringAnalyzer
    from hydrophobic_patch import HydrophobicPatchAnalyzer
    from paratope_dynamics import ParatopeDynamicsAnalyzer
    from assay_mapping import AssayMapper
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Create dummy classes for testing
    class MarkovModel:
        pass
    
    def compute_sequence_surprisal_features(*args, **kwargs):
        return {}
    
    class ChargeImbalanceAnalyzer:
        def calculate_charge_imbalance_score(self, *args, **kwargs):
            return {'imbalance_score': 0.1}
    
    class ResidueClusteringAnalyzer:
        def calculate_clustering_risk_score(self, *args, **kwargs):
            return {'clustering_risk_score': 0.1}
    
    class HydrophobicPatchAnalyzer:
        def calculate_binding_potential(self, *args, **kwargs):
            return {'binding_potential': 0.1}
    
    class ParatopeDynamicsAnalyzer:
        def calculate_dynamics_risk_score(self, *args, **kwargs):
            return {'dynamics_risk_score': 0.1}
    
    class AssayMapper:
        def generate_assay_report(self, *args, **kwargs):
            return {'psr_score': 0.1, 'psp_score': 0.1}


class SurprisalPolyreactivityIntegrator:
    """
    Integrator for combining Markov surprisal features with advanced polyreactivity features.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrator.
        
        Args:
            config_path (str): Path to configuration file (optional)
        """
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'integration_weights': {
                    'weight_surprisal': 0.2,
                    'weight_charge': 0.2,
                    'weight_clustering': 0.2,
                    'weight_hydrophobic': 0.2,
                    'weight_dynamics': 0.2
                },
                'markov_model': {
                    'order': 1,
                    'smoothing_method': 'additive',
                    'epsilon': 1e-10,
                    'log_base': 'natural'
                },
                'polyreactivity_analysis': {
                    'hydrophobic_patch_window_size': 5,
                    'cdr_regions': [(23, 33), (49, 55), (94, 105)]
                }
            }
        
        # Initialize analyzers
        self.markov_model = None
        self.charge_analyzer = ChargeImbalanceAnalyzer()
        self.clustering_analyzer = ResidueClusteringAnalyzer()
        self.hydrophobic_analyzer = HydrophobicPatchAnalyzer()
        self.dynamics_analyzer = ParatopeDynamicsAnalyzer()
        self.assay_mapper = AssayMapper()
    
    def initialize_markov_model(self, training_sequences: List[str]) -> None:
        """
        Initialize and train the Markov model.
        
        Args:
            training_sequences (List[str]): List of training sequences
        """
        markov_config = self.config.get('markov_model', {})
        self.markov_model = MarkovModel(
            order=markov_config.get('order', 1),
            smoothing_method=markov_config.get('smoothing_method', 'additive'),
            epsilon=markov_config.get('epsilon', 1e-10),
            log_base=markov_config.get('log_base', 'natural')
        )
        self.markov_model.train(training_sequences)
    
    def compute_comprehensive_features(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[float, Dict]]:
        """
        Compute comprehensive features combining surprisal and polyreactivity features.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict: Comprehensive feature analysis
        """
        # Compute surprisal features if Markov model is initialized
        surprisal_features = {}
        if self.markov_model:
            # Combine VH and VL sequences for surprisal analysis
            combined_sequence = vh_sequence + vl_sequence
            surprisal_features = compute_sequence_surprisal_features(combined_sequence, self.markov_model)
        
        # Compute charge imbalance features
        charge_analysis = self.charge_analyzer.calculate_charge_imbalance_score(vh_sequence, vl_sequence)
        
        # Compute clustering features (on combined sequence)
        combined_sequence = vh_sequence + vl_sequence
        clustering_analysis = self.clustering_analyzer.calculate_clustering_risk_score(combined_sequence)
        
        # Compute hydrophobic patch features
        hydrophobic_analysis = self.hydrophobic_analyzer.calculate_binding_potential(
            combined_sequence, 
            self.config.get('polyreactivity_analysis', {}).get('hydrophobic_patch_window_size', 5)
        )
        
        # Compute paratope dynamics features
        dynamics_analysis = self.dynamics_analyzer.calculate_dynamics_risk_score(
            combined_sequence,
            self.config.get('polyreactivity_analysis', {}).get('cdr_regions', [(23, 33), (49, 55), (94, 105)])
        )
        
        # Combine all features
        comprehensive_features = {
            'vh_sequence': vh_sequence,
            'vl_sequence': vl_sequence,
            'surprisal_features': surprisal_features,
            'charge_features': charge_analysis,
            'clustering_features': clustering_analysis,
            'hydrophobic_features': hydrophobic_analysis,
            'dynamics_features': dynamics_analysis
        }
        
        return comprehensive_features
    
    def compute_integrated_risk_score(self, comprehensive_features: Dict) -> Dict[str, Union[float, str]]:
        """
        Compute integrated risk score combining all features with weighted integration.
        
        Args:
            comprehensive_features (Dict): Comprehensive feature analysis
            
        Returns:
            Dict: Integrated risk score and interpretation
        """
        # Extract individual scores
        # Surprisal features
        burden_q = comprehensive_features['surprisal_features'].get('burden_q', 0.0)
        
        # Polyreactivity features
        charge_score = comprehensive_features['charge_features'].get('imbalance_score', 0.0)
        clustering_score = comprehensive_features['clustering_features'].get('clustering_risk_score', 0.0)
        hydrophobic_score = comprehensive_features['hydrophobic_features'].get('binding_potential', 0.0)
        dynamics_score = comprehensive_features['dynamics_features'].get('dynamics_risk_score', 0.0)
        
        # Get weights
        weights = self.config.get('integration_weights', {})
        weight_surprisal = weights.get('weight_surprisal', 0.2)
        weight_charge = weights.get('weight_charge', 0.2)
        weight_clustering = weights.get('weight_clustering', 0.2)
        weight_hydrophobic = weights.get('weight_hydrophobic', 0.2)
        weight_dynamics = weights.get('weight_dynamics', 0.2)
        
        # Compute integrated risk score
        integrated_risk = (
            weight_surprisal * burden_q +
            weight_charge * charge_score +
            weight_clustering * clustering_score +
            weight_hydrophobic * hydrophobic_score +
            weight_dynamics * dynamics_score
        )
        
        # Normalize to 0-1 range
        integrated_risk = min(1.0, max(0.0, integrated_risk))
        
        # Interpret the score
        if integrated_risk < 0.2:
            interpretation = "Low integrated risk - favorable for developability"
        elif integrated_risk < 0.4:
            interpretation = "Moderate integrated risk - generally acceptable"
        elif integrated_risk < 0.6:
            interpretation = "High integrated risk - may affect developability"
        else:
            interpretation = "Very high integrated risk - likely to cause developability issues"
        
        return {
            'integrated_risk_score': integrated_risk,
            'burden_q': burden_q,
            'charge_score': charge_score,
            'clustering_score': clustering_score,
            'hydrophobic_score': hydrophobic_score,
            'dynamics_score': dynamics_score,
            'weights': weights,
            'interpretation': interpretation,
            'scoring_complete': True
        }
    
    def generate_comprehensive_report(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Generate a comprehensive report combining all analyses.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict: Comprehensive report
        """
        # Compute comprehensive features
        comprehensive_features = self.compute_comprehensive_features(vh_sequence, vl_sequence)
        
        # Compute integrated risk score
        integrated_risk = self.compute_integrated_risk_score(comprehensive_features)
        
        # Generate assay report
        assay_report = self.assay_mapper.generate_assay_report(
            vh_sequence + vl_sequence,
            comprehensive_features['charge_features'].get('imbalance_score', 0.0),
            comprehensive_features['clustering_features'].get('clustering_risk_score', 0.0),
            comprehensive_features['hydrophobic_features'].get('binding_potential', 0.0),
            comprehensive_features['dynamics_features'].get('dynamics_risk_score', 0.0)
        )
        
        # Combine all information
        comprehensive_report = {
            'vh_sequence': vh_sequence,
            'vl_sequence': vl_sequence,
            'comprehensive_features': comprehensive_features,
            'integrated_risk': integrated_risk,
            'assay_report': assay_report,
            'report_complete': True
        }
        
        return comprehensive_report


def main():
    """
    Example usage of the surprisal-polyreactivity integrator.
    """
    # Example sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYGSSPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Example training sequences for Markov model
    training_sequences = [
        "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMSWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGYYAMDYWGQGTMVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGNTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGKGNYYAMDYWGQGTLVTVSS",
        "EVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNISWVRQAPGQGLEWMGWISSSGNTIYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGKGNYYAMDYWGQGTMVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMSWVRQAPGQGLEWMGWISAGSGNTIYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGKGNYYAMDYWGQGTLVTVSS"
    ]
    
    # Create integrator
    integrator = SurprisalPolyreactivityIntegrator()
    
    # Initialize Markov model
    integrator.initialize_markov_model(training_sequences)
    
    # Generate comprehensive report
    comprehensive_report = integrator.generate_comprehensive_report(vh_sequence, vl_sequence)
    
    # Print results
    print("Comprehensive Antibody Developability Analysis")
    print("=" * 50)
    print(f"Integrated Risk Score: {comprehensive_report['integrated_risk']['integrated_risk_score']:.3f}")
    print(f"Interpretation: {comprehensive_report['integrated_risk']['interpretation']}")
    
    print("\nComponent Scores:")
    print(f"  Burden Q (Surprisal): {comprehensive_report['integrated_risk']['burden_q']:.3f}")
    print(f"  Charge Imbalance: {comprehensive_report['integrated_risk']['charge_score']:.3f}")
    print(f"  Clustering Risk: {comprehensive_report['integrated_risk']['clustering_score']:.3f}")
    print(f"  Hydrophobic Binding Potential: {comprehensive_report['integrated_risk']['hydrophobic_score']:.3f}")
    print(f"  Dynamics Risk: {comprehensive_report['integrated_risk']['dynamics_score']:.3f}")
    
    print("\nWeights:")
    weights = comprehensive_report['integrated_risk']['weights']
    print(f"  Surprisal Weight: {weights['weight_surprisal']:.2f}")
    print(f"  Charge Weight: {weights['weight_charge']:.2f}")
    print(f"  Clustering Weight: {weights['weight_clustering']:.2f}")
    print(f"  Hydrophobic Weight: {weights['weight_hydrophobic']:.2f}")
    print(f"  Dynamics Weight: {weights['weight_dynamics']:.2f}")
    
    # Print assay report summary
    print("\nAssay Report Summary:")
    print(comprehensive_report['assay_report']['summary'])


if __name__ == "__main__":
    main()
