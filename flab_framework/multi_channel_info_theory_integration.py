"""
FLAb Multi-Channel Information Theory Integration

This module integrates the Multi-Channel Information Theory Framework with the FLAb framework.
"""

import sys
import os
import numpy as np

# Add the research directory to the path to import the MultiChannelInfoTheory class

try:
    from multi_channel_info_theory import MultiChannelInfoTheory
except ImportError as e:
    print(f"Error importing MultiChannelInfoTheory: {e}")
    MultiChannelInfoTheory = object  # Fallback to avoid breaking the class definition


class FLAbMultiChannelInfoTheory(MultiChannelInfoTheory):
    """
    FLAb-specific Multi-Channel Information Theory Framework.
    """
    
    def __init__(self):
        """
        Initialize the FLAb Multi-Channel Information Theory Framework.
        """
        super().__init__()
        
    def extract_channel_data_from_fragment_analysis(self, fragment_analysis):
        """
        Extract channel data from fragment analysis for multi-channel information theory analysis.
        
        Args:
        fragment_analysis (dict): Fragment analysis results
        """
        # Extract sequence channel data (amino acid composition)
        composition = fragment_analysis.get('composition', {})
        if composition:
            # Use amino acid frequencies as sequence channel data
            aa_counts = composition.get('amino_acids', {})
            if aa_counts:
                # Convert to numpy array
                aa_names = sorted(aa_counts.keys())
                sequence_data = np.array([aa_counts[aa] for aa in aa_names])
                self.add_channel_data('sequence', sequence_data)
        
        # Extract structure channel data (physicochemical properties and stability)
        phys_props = fragment_analysis.get('physicochemical_properties', {})
        stability = fragment_analysis.get('stability', {})
        
        if phys_props or stability:
            # Combine physicochemical properties and stability metrics
            structure_features = []
            
            # Add physicochemical properties
            if phys_props:
                structure_features.extend([
                    phys_props.get('hydrophobicity', 0),
                    phys_props.get('isoelectric_point', 7.0),
                    phys_props.get('molecular_weight', 0),
                    phys_props.get('charge_distribution', {}).get('net_charge', 0)
                ])
            
            # Add stability metrics
            if stability:
                structure_features.extend([
                    stability.get('aggregation_propensity', 0),
                    stability.get('thermal_stability', 0)
                ])
            
            # Convert to numpy array
            structure_data = np.array(structure_features)
            self.add_channel_data('structure', structure_data)
        
        # Extract temporal channel data (for this implementation, we'll use derived features)
        # In a more advanced implementation, this could be actual temporal data
        complexity = fragment_analysis.get('complexity', {})
        
        if complexity:
            # Use complexity measures and derived temporal features
            temporal_features = [
                complexity.get('overall_complexity', 0),
                complexity.get('shannon_entropy', 0),
                complexity.get('simpson_diversity', 0)
            ]
            
            # Convert to numpy array
            temporal_data = np.array(temporal_features)
            self.add_channel_data('temporal', temporal_data)
    
    def analyze_fragment_developability(self, fragment_analysis):
        """
        Analyze fragment developability using multi-channel information theory.
        
        Args:
        fragment_analysis (dict): Fragment analysis results
        
        Returns:
        dict: Information theory analysis results
        """
        # Extract channel data from fragment analysis
        self.extract_channel_data_from_fragment_analysis(fragment_analysis)
        
        # Analyze all channels
        analysis_results = self.analyze_all_channels()
        
        # Generate report
        report = self.generate_report()
        
        return {
            'analysis_results': analysis_results,
            'report': report
        }
    
    def get_information_theoretic_features(self):
        """
        Get information-theoretic features for use in developability prediction.
        
        Returns:
        dict: Information-theoretic features
        """
        features = {}
        
        # Get channel metrics
        channel_metrics = self.get_information_metrics()
        for channel, metrics in channel_metrics.items():
            for metric, value in metrics.items():
                features[f'{channel}_{metric}'] = value
        
        # Get cross-channel metrics
        cross_channel_metrics = self.get_cross_channel_metrics()
        for pair, metrics in cross_channel_metrics.items():
            for metric, value in metrics.items():
                features[f'{pair}_{metric}'] = value
        
        return features

# Example usage function
def main():
    """
    Example usage of the FLAbMultiChannelInfoTheory class.
    """
    print("FLAb Multi-Channel Information Theory Integration Example")
    print("=====================================================")
    
    # Create example fragment analysis data
    fragment_analysis = {
        'composition': {
            'amino_acids': {
                'A': 10, 'C': 5, 'D': 8, 'E': 7, 'F': 4,
                'G': 9, 'H': 3, 'I': 6, 'K': 7, 'L': 12,
                'M': 2, 'N': 6, 'P': 5, 'Q': 4, 'R': 8,
                'S': 9, 'T': 7, 'V': 8, 'W': 2, 'Y': 3
            },
            'total_amino_acids': 100
        },
        'physicochemical_properties': {
            'hydrophobicity': 0.45,
            'isoelectric_point': 6.8,
            'molecular_weight': 12000,
            'charge_distribution': {
                'net_charge': -2.0
            }
        },
        'stability': {
            'aggregation_propensity': 0.3,
            'thermal_stability': 0.75
        },
        'complexity': {
            'overall_complexity': 0.65,
            'shannon_entropy': 2.1,
            'simpson_diversity': 0.85
        }
    }
    
    # Create FLAbMultiChannelInfoTheory instance
    info_theory = FLAbMultiChannelInfoTheory()
    
    # Analyze fragment developability
    results = info_theory.analyze_fragment_developability(fragment_analysis)
    
    # Get information-theoretic features
    features = info_theory.get_information_theoretic_features()
    
    print(f"Information-theoretic features extracted: {len(features)}")
    for feature, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {feature}: {value:.4f}")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
