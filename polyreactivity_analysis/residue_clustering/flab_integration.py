"""
FLAb Framework Integration for Residue Clustering Analysis
"""

import sys
import os

# Add paths for FLAb framework and residue clustering analysis
sys.path.append('/a0/bitcore/workspace/flab_framework')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis/residue_clustering')

from residue_clustering import ResidueClusteringAnalyzer


class FLAbResidueClusteringIntegrator:
    """
    FLAb Framework Integration for Residue Clustering Analysis
    """
    
    def __init__(self):
        """
        Initialize FLAb Residue Clustering Integrator
        """
        self.analyzer = ResidueClusteringAnalyzer()
    
    def analyze_antibody(self, antibody_id, vh_sequence, vl_sequence):
        """
        Analyze an antibody's residue clustering patterns
        
        Parameters:
        antibody_id (str): Identifier for the antibody
        vh_sequence (str): VH chain sequence
        vl_sequence (str): VL chain sequence
        
        Returns:
        dict: Analysis results
        """
        # Analyze VH chain
        vh_clustering = self.analyzer.calculate_clustering_risk_score(vh_sequence)
        
        # Analyze VL chain
        vl_clustering = self.analyzer.calculate_clustering_risk_score(vl_sequence)
        
        # Combine results
        results = {
            'antibody_id': antibody_id,
            'vh_sequence_length': len(vh_sequence),
            'vl_sequence_length': len(vl_sequence),
            'vh_clustering_risk_score': vh_clustering['clustering_risk_score'],
            'vl_clustering_risk_score': vl_clustering['clustering_risk_score'],
            'vh_charged_clustering_score': vh_clustering['charged_clustering_score'],
            'vh_hydrophobic_clustering_score': vh_clustering['hydrophobic_clustering_score'],
            'vh_aromatic_clustering_score': vh_clustering['aromatic_clustering_score'],
            'vl_charged_clustering_score': vl_clustering['charged_clustering_score'],
            'vl_hydrophobic_clustering_score': vl_clustering['hydrophobic_clustering_score'],
            'vl_aromatic_clustering_score': vl_clustering['aromatic_clustering_score'],
            'vh_clustering_interpretation': vh_clustering['interpretation'],
            'vl_clustering_interpretation': vl_clustering['interpretation']
        }
        
        return results
    
    def analyze_antibodies(self, antibodies_dict):
        """
        Analyze multiple antibodies
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        
        Returns:
        dict: Analysis results for all antibodies
        """
        results = {}
        for ab_id, (vh_seq, vl_seq) in antibodies_dict.items():
            results[ab_id] = self.analyze_antibody(ab_id, vh_seq, vl_seq)
        
        return results
    
    def get_feature_dataframe(self, antibodies_dict):
        """
        Get residue clustering features as a pandas DataFrame
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        
        Returns:
        pd.DataFrame: DataFrame with residue clustering features
        """
        import pandas as pd
        
        # Analyze antibodies
        results = self.analyze_antibodies(antibodies_dict)
        
        # Convert to DataFrame
        df_data = []
        for ab_id, result in results.items():
            row = {
                'antibody_id': ab_id,
                'vh_sequence_length': result['vh_sequence_length'],
                'vl_sequence_length': result['vl_sequence_length'],
                'vh_clustering_risk_score': result['vh_clustering_risk_score'],
                'vl_clustering_risk_score': result['vl_clustering_risk_score'],
                'vh_charged_clustering_score': result['vh_charged_clustering_score'],
                'vh_hydrophobic_clustering_score': result['vh_hydrophobic_clustering_score'],
                'vh_aromatic_clustering_score': result['vh_aromatic_clustering_score'],
                'vl_charged_clustering_score': result['vl_charged_clustering_score'],
                'vl_hydrophobic_clustering_score': result['vl_hydrophobic_clustering_score'],
                'vl_aromatic_clustering_score': result['vl_aromatic_clustering_score']
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def export_features(self, antibodies_dict, output_path):
        """
        Export residue clustering features to a CSV file
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        output_path (str): Path to output CSV file
        
        Returns:
        dict: Export summary
        """
        # Get feature DataFrame
        df = self.get_feature_dataframe(antibodies_dict)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'antibodies_processed': len(df),
            'output_file': output_path
        }

def main():
    """
    Main function for testing FLAb Residue Clustering Integrator
    """
    print("Testing FLAb Residue Clustering Integrator...")
    
    # Example antibodies
    antibodies = {
        'ab1': ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCARDESGYYYYYYMDVWGQGTTVTVSS', 
                'DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIK'),
        'ab2': ('QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGNTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGEGGFYYYYYMDVWGQGTTVTVSS', 
                'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPRTFGQGTKVEIK'),
        'ab3': ('EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYYSGSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGRYYYYYMDVWGQGTTVTVSS', 
                'DIVMTQSPDSLAVSLGERATINCKSSQSLVHSNGNTYLQWFLQRPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPRTFGGGTKLEIK')
    }
    
    # Create integrator
    integrator = FLAbResidueClusteringIntegrator()
    
    # Analyze antibodies
    results = integrator.analyze_antibodies(antibodies)
    print(f"Analysis results for ab1: {results['ab1']}")
    
    # Get feature DataFrame
    df = integrator.get_feature_dataframe(antibodies)
    print(f"Feature DataFrame shape: {df.shape}")
    print(f"Feature DataFrame columns: {list(df.columns)}")
    
    # Export features
    export_result = integrator.export_features(antibodies, '/a0/bitcore/workspace/polyreactivity_analysis/residue_clustering/test_features.csv')
    print(f"Export result: {export_result}")
    
    print("\nFLAb Residue Clustering Integrator test completed successfully!")


if __name__ == '__main__':
    main()
