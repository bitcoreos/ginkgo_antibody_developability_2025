"""
FLAb Framework Integration for Polyreactivity Analysis
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add paths for FLAb framework and polyreactivity analysis
sys.path.append('/a0/bitcore/workspace/flab_framework')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis')

from vhvl_charge_analysis import VHVLChargeAnalyzer


class FLAbPolyreactivityAnalyzer:
    """
    FLAb Framework Integration for Polyreactivity Analysis
    """
    
    def __init__(self):
        """
        Initialize FLAb Polyreactivity Analyzer
        """
        self.analyzer = VHVLChargeAnalyzer()
    
    def analyze_antibody(self, antibody_id, vh_sequence, vl_sequence):
        """
        Analyze an antibody's polyreactivity features
        
        Parameters:
        antibody_id (str): Identifier for the antibody
        vh_sequence (str): VH chain sequence
        vl_sequence (str): VL chain sequence
        
        Returns:
        dict: Analysis results
        """
        return self.analyzer.analyze_antibody(antibody_id, vh_sequence, vl_sequence)
    
    def analyze_antibodies(self, antibodies_dict):
        """
        Analyze multiple antibodies
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        
        Returns:
        dict: Analysis results for all antibodies
        """
        return self.analyzer.analyze_antibodies(antibodies_dict)
    
    def export_features(self, antibodies_dict, output_path):
        """
        Export polyreactivity features to a CSV file
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        output_path (str): Path to output CSV file
        
        Returns:
        dict: Export summary
        """
        return self.analyzer.export_features(antibodies_dict, output_path)
    
    def get_feature_dataframe(self, antibodies_dict):
        """
        Get polyreactivity features as a pandas DataFrame
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        
        Returns:
        pd.DataFrame: DataFrame with polyreactivity features
        """
        # Analyze antibodies
        results = self.analyze_antibodies(antibodies_dict)
        
        # Convert to DataFrame
        df_data = []
        for ab_id, result in results.items():
            row = {
                'antibody_id': ab_id,
                'vh_sequence_length': result['vh_sequence_length'],
                'vl_sequence_length': result['vl_sequence_length'],
                'vh_net_charge': result['charge_imbalance']['vh_net_charge'],
                'vl_net_charge': result['charge_imbalance']['vl_net_charge'],
                'charge_difference': result['charge_imbalance']['charge_difference'],
                'charge_ratio': result['charge_imbalance']['charge_ratio'],
                'positive_fraction_difference': result['charge_imbalance']['positive_fraction_difference'],
                'negative_fraction_difference': result['charge_imbalance']['negative_fraction_difference'],
                'vh_max_hydrophobicity': result['vh_hydrophobic_patches']['max_hydrophobicity'],
                'vl_max_hydrophobicity': result['vl_hydrophobic_patches']['max_hydrophobicity'],
                'vh_hydrophobic_patch_count': result['vh_hydrophobic_patches']['hydrophobic_patch_count'],
                'vl_hydrophobic_patch_count': result['vl_hydrophobic_patches']['hydrophobic_patch_count'],
                'total_hydrophobic_patches': result['total_hydrophobic_patches']
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)


def main():
    """
    Main function for testing FLAb Polyreactivity Analyzer
    """
    print("Testing FLAb Polyreactivity Analyzer Integration...")
    
    # Example antibodies
    antibodies = {
        'ab1': ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCARDESGYYYYYYMDVWGQGTTVTVSS', 
                'DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIK'),
        'ab2': ('QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGNTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGEGGFYYYYYMDVWGQGTTVTVSS', 
                'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPRTFGQGTKVEIK'),
        'ab3': ('EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYYSGSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGRYYYYYMDVWGQGTTVTVSS', 
                'DIVMTQSPDSLAVSLGERATINCKSSQSLVHSNGNTYLQWFLQRPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPRTFGGGTKLEIK')
    }
    
    # Create analyzer
    analyzer = FLAbPolyreactivityAnalyzer()
    
    # Analyze antibodies
    results = analyzer.analyze_antibodies(antibodies)
    print(f"Analysis results for ab1: {results['ab1']}")
    
    # Get feature DataFrame
    df = analyzer.get_feature_dataframe(antibodies)
    print(f"Feature DataFrame shape: {df.shape}")
    print(f"Feature DataFrame columns: {list(df.columns)}")
    
    # Export features
    export_result = analyzer.export_features(antibodies, '/a0/bitcore/workspace/polyreactivity_analysis/test_features_flab.csv')
    print(f"Export result: {export_result}")
    
    print("\nFLAb Polyreactivity Analyzer test completed successfully!")


if __name__ == '__main__':
    main()
