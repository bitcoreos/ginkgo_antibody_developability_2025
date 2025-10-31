""
FLAb Framework Integration for Paratope Dynamics Analysis
"""

import sys
import os

# Add paths for FLAb framework and paratope dynamics analysis
sys.path.append('/a0/bitcore/workspace/flab_framework')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis/paratope_dynamics')

from paratope_dynamics import ParatopeDynamicsAnalyzer


class FLAbParatopeDynamicsIntegrator:
    """
    FLAb Framework Integration for Paratope Dynamics Analysis
    """
    
    def __init__(self):
        """
        Initialize FLAb Paratope Dynamics Integrator
        """
        self.analyzer = ParatopeDynamicsAnalyzer()
    
    def analyze_antibody(self, antibody_id, vh_sequence, vl_sequence, vh_cdr_regions=None, vl_cdr_regions=None):
        """
        Analyze an antibody's paratope dynamics
        
        Parameters:
        antibody_id (str): Identifier for the antibody
        vh_sequence (str): VH chain sequence
        vl_sequence (str): VL chain sequence
        vh_cdr_regions (list): List of tuples (start, end) for VH CDR regions
        vl_cdr_regions (list): List of tuples (start, end) for VL CDR regions
        
        Returns:
        dict: Analysis results
        """
        # Analyze VH chain
        vh_dynamics = self.analyzer.calculate_dynamics_risk_score(vh_sequence, vh_cdr_regions)
        
        # Analyze VL chain
        vl_dynamics = self.analyzer.calculate_dynamics_risk_score(vl_sequence, vl_cdr_regions)
        
        # Combine results
        results = {
            'antibody_id': antibody_id,
            'vh_sequence_length': len(vh_sequence),
            'vl_sequence_length': len(vl_sequence),
            'vh_dynamics_risk_score': vh_dynamics['dynamics_risk_score'],
            'vl_dynamics_risk_score': vl_dynamics['dynamics_risk_score'],
            'vh_avg_flexibility_score': vh_dynamics['avg_flexibility_score'],
            'vh_avg_rigidity_score': vh_dynamics['avg_rigidity_score'],
            'vh_avg_paratope_score': vh_dynamics['avg_paratope_score'],
            'vh_avg_entropy_proxy': vh_dynamics['avg_entropy_proxy'],
            'vl_avg_flexibility_score': vl_dynamics['avg_flexibility_score'],
            'vl_avg_rigidity_score': vl_dynamics['avg_rigidity_score'],
            'vl_avg_paratope_score': vl_dynamics['avg_paratope_score'],
            'vl_avg_entropy_proxy': vl_dynamics['avg_entropy_proxy'],
            'vh_dynamics_interpretation': vh_dynamics['interpretation'],
            'vl_dynamics_interpretation': vl_dynamics['interpretation']
        }
        
        return results
    
    def analyze_antibodies(self, antibodies_dict, vh_cdr_regions=None, vl_cdr_regions=None):
        """
        Analyze multiple antibodies
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        vh_cdr_regions (list): List of tuples (start, end) for VH CDR regions
        vl_cdr_regions (list): List of tuples (start, end) for VL CDR regions
        
        Returns:
        dict: Analysis results for all antibodies
        """
        results = {}
        for ab_id, (vh_seq, vl_seq) in antibodies_dict.items():
            results[ab_id] = self.analyze_antibody(ab_id, vh_seq, vl_seq, vh_cdr_regions, vl_cdr_regions)
        
        return results
    
    def get_feature_dataframe(self, antibodies_dict, vh_cdr_regions=None, vl_cdr_regions=None):
        """
        Get paratope dynamics features as a pandas DataFrame
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        vh_cdr_regions (list): List of tuples (start, end) for VH CDR regions
        vl_cdr_regions (list): List of tuples (start, end) for VL CDR regions
        
        Returns:
        pd.DataFrame: DataFrame with paratope dynamics features
        """
        import pandas as pd
        
        # Analyze antibodies
        results = self.analyze_antibodies(antibodies_dict, vh_cdr_regions, vl_cdr_regions)
        
        # Convert to DataFrame
        df_data = []
        for ab_id, result in results.items():
            row = {
                'antibody_id': ab_id,
                'vh_sequence_length': result['vh_sequence_length'],
                'vl_sequence_length': result['vl_sequence_length'],
                'vh_dynamics_risk_score': result['vh_dynamics_risk_score'],
                'vl_dynamics_risk_score': result['vl_dynamics_risk_score'],
                'vh_avg_flexibility_score': result['vh_avg_flexibility_score'],
                'vh_avg_rigidity_score': result['vh_avg_rigidity_score'],
                'vh_avg_paratope_score': result['vh_avg_paratope_score'],
                'vh_avg_entropy_proxy': result['vh_avg_entropy_proxy'],
                'vl_avg_flexibility_score': result['vl_avg_flexibility_score'],
                'vl_avg_rigidity_score': result['vl_avg_rigidity_score'],
                'vl_avg_paratope_score': result['vl_avg_paratope_score'],
                'vl_avg_entropy_proxy': result['vl_avg_entropy_proxy']
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def export_features(self, antibodies_dict, output_path, vh_cdr_regions=None, vl_cdr_regions=None):
        """
        Export paratope dynamics features to a CSV file
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        output_path (str): Path to output CSV file
        vh_cdr_regions (list): List of tuples (start, end) for VH CDR regions
        vl_cdr_regions (list): List of tuples (start, end) for VL CDR regions
        
        Returns:
        dict: Export summary
        """
        # Get feature DataFrame
        df = self.get_feature_dataframe(antibodies_dict, vh_cdr_regions, vl_cdr_regions)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'antibodies_processed': len(df),
            'output_file': output_path
        }

def main():
    """
    Main function for testing FLAb Paratope Dynamics Integrator
    """
    print("Testing FLAb Paratope Dynamics Integrator...")
    
    # Example antibodies
    antibodies = {
        'ab1': ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCARDESGYYYYYYMDVWGQGTTVTVSS', 
                'DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIK'),
        'ab2': ('QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGNTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGEGGFYYYYYMDVWGQGTTVTVSS', 
                'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPRTFGQGTKVEIK'),
        'ab3': ('EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYYSGSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGRYYYYYMDVWGQGTTVTVSS', 
                'DIVMTQSPDSLAVSLGERATINCKSSQSLVHSNGNTYLQWFLQRPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPRTFGGGTKLEIK')
    }
    
    # Define CDR regions (approximate)
    vh_cdr_regions = [(23, 33), (49, 55), (94, 105)]
    vl_cdr_regions = [(23, 33), (49, 55), (94, 105)]
    
    # Create integrator
    integrator = FLAbParatopeDynamicsIntegrator()
    
    # Analyze antibodies
    results = integrator.analyze_antibodies(antibodies, vh_cdr_regions, vl_cdr_regions)
    print(f"Analysis results for ab1: {results['ab1']}")
    
    # Get feature DataFrame
    df = integrator.get_feature_dataframe(antibodies, vh_cdr_regions, vl_cdr_regions)
    print(f"Feature DataFrame shape: {df.shape}")
    print(f"Feature DataFrame columns: {list(df.columns)}")
    
    # Export features
    export_result = integrator.export_features(antibodies, '/a0/bitcore/workspace/polyreactivity_analysis/paratope_dynamics/test_features.csv', vh_cdr_regions, vl_cdr_regions)
    print(f"Export result: {export_result}")
    
    print("\nFLAb Paratope Dynamics Integrator test completed successfully!")


if __name__ == '__main__':
    main()
