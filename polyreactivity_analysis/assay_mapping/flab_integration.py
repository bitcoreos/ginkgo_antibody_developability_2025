"""
FLAb Framework Integration for Assay Mapping
"""

import sys
import os

# Add paths for FLAb framework and assay mapping
sys.path.append('/a0/bitcore/workspace/flab_framework')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis')
sys.path.append('/a0/bitcore/workspace/polyreactivity_analysis/assay_mapping')

from assay_mapping import AssayMapper


class FLAbAssayMappingIntegrator:
    """
    FLAb Framework Integration for Assay Mapping
    """
    
    def __init__(self):
        """
        Initialize FLAb Assay Mapping Integrator
        """
        self.mapper = AssayMapper()
    
    def map_antibody_features(self, antibody_id, sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score):
        """
        Map polyreactivity features for an antibody to assay data
        
        Parameters:
        antibody_id (str): Identifier for the antibody
        sequence (str): Antibody sequence
        charge_imbalance_score (float): Charge imbalance score
        clustering_risk_score (float): Clustering risk score
        binding_potential (float): Binding potential score
        dynamics_risk_score (float): Dynamics risk score
        
        Returns:
        dict: Assay mapping results
        """
        # Map polyreactivity features
        assay_results = self.mapper.map_polyreactivity_features(
            sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score
        )
        
        # Combine results
        results = {
            'antibody_id': antibody_id,
            'psr': assay_results['psr'],
            'psp': assay_results['psp'],
            'polyreactivity_assessment': assay_results['polyreactivity_assessment'],
            'developability_risk': assay_results['developability_risk'],
            'recommendations': assay_results['recommendations'],
            'priority_ranking': assay_results['priority_ranking']
        }
        
        return results
    
    def map_antibodies_features(self, antibodies_features_dict):
        """
        Map polyreactivity features for multiple antibodies to assay data
        
        Parameters:
        antibodies_features_dict (dict): Dictionary mapping antibody IDs to feature dictionaries
        
        Returns:
        dict: Assay mapping results for all antibodies
        """
        results = {}
        for ab_id, features in antibodies_features_dict.items():
            results[ab_id] = self.map_antibody_features(
                ab_id, features['sequence'], features['charge_imbalance_score'], 
                features['clustering_risk_score'], features['binding_potential'], 
                features['dynamics_risk_score']
            )
        
        return results
    
    def get_feature_dataframe(self, antibodies_features_dict):
        """
        Get assay mapping features as a pandas DataFrame
        
        Parameters:
        antibodies_features_dict (dict): Dictionary mapping antibody IDs to feature dictionaries
        
        Returns:
        pd.DataFrame: DataFrame with assay mapping features
        """
        import pandas as pd
        
        # Map antibodies features
        results = self.map_antibodies_features(antibodies_features_dict)
        
        # Convert to DataFrame
        df_data = []
        for ab_id, result in results.items():
            row = {
                'antibody_id': ab_id,
                'psr': result['psr'],
                'psp': result['psp'],
                'polyreactivity_assessment': result['polyreactivity_assessment'],
                'developability_risk': result['developability_risk'],
                'priority_ranking': result['priority_ranking']
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def export_features(self, antibodies_features_dict, output_path):
        """
        Export assay mapping features to a CSV file
        
        Parameters:
        antibodies_features_dict (dict): Dictionary mapping antibody IDs to feature dictionaries
        output_path (str): Path to output CSV file
        
        Returns:
        dict: Export summary
        """
        # Get feature DataFrame
        df = self.get_feature_dataframe(antibodies_features_dict)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'antibodies_processed': len(df),
            'output_file': output_path
        }

def main():
    """
    Main function for testing FLAb Assay Mapping Integrator
    """
    print("Testing FLAb Assay Mapping Integrator...")
    
    # Example antibodies with features
    antibodies_features = {
        'ab1': {
            'sequence': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCARDESGYYYYYYMDVWGQGTTVTVSSDIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIK',
            'charge_imbalance_score': 0.013,
            'clustering_risk_score': 0.760,
            'binding_potential': 0.311,
            'dynamics_risk_score': 0.361
        },
        'ab2': {
            'sequence': 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGNTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGEGGFYYYYYMDVWGQGTTVTVSSEIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPRTFGQGTKVEIK',
            'charge_imbalance_score': 0.025,
            'clustering_risk_score': 0.620,
            'binding_potential': 0.285,
            'dynamics_risk_score': 0.412
        },
        'ab3': {
            'sequence': 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYYSGSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGRYYYYYMDVWGQGTTVTVSSDIVMTQSPDSLAVSLGERATINCKSSQSLVHSNGNTYLQWFLQRPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPRTFGGGTKLEIK',
            'charge_imbalance_score': 0.008,
            'clustering_risk_score': 0.810,
            'binding_potential': 0.356,
            'dynamics_risk_score': 0.298
        }
    }
    
    # Create integrator
    integrator = FLAbAssayMappingIntegrator()
    
    # Map antibodies features
    results = integrator.map_antibodies_features(antibodies_features)
    print(f"Assay mapping results for ab1: {results['ab1']}")
    
    # Get feature DataFrame
    df = integrator.get_feature_dataframe(antibodies_features)
    print(f"Feature DataFrame shape: {df.shape}")
    print(f"Feature DataFrame columns: {list(df.columns)}")
    
    # Export features
    export_result = integrator.export_features(antibodies_features, '/a0/bitcore/workspace/polyreactivity_analysis/assay_mapping/test_features.csv')
    print(f"Export result: {export_result}")
    
    print("\nFLAb Assay Mapping Integrator test completed successfully!")


if __name__ == '__main__':
    main()
