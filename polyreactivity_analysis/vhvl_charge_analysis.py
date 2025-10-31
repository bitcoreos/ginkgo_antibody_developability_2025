"""
VH/VL Charge Imbalance Analysis for Antibody Polyreactivity Prediction
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# Amino acid charges at physiological pH (approximate)
# Positive: K, R, H
# Negative: D, E
# Neutral: all others
AMINO_ACID_CHARGES = {
    'A': 0,  # Alanine
    'C': 0,  # Cysteine
    'D': -1, # Aspartic acid
    'E': -1, # Glutamic acid
    'F': 0,  # Phenylalanine
    'G': 0,  # Glycine
    'H': 1,  # Histidine (partially protonated at pH 7.4)
    'I': 0,  # Isoleucine
    'K': 1,  # Lysine
    'L': 0,  # Leucine
    'M': 0,  # Methionine
    'N': 0,  # Asparagine
    'P': 0,  # Proline
    'Q': 0,  # Glutamine
    'R': 1,  # Arginine
    'S': 0,  # Serine
    'T': 0,  # Threonine
    'V': 0,  # Valine
    'W': 0,  # Tryptophan
    'Y': 0,  # Tyrosine
}

# Hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8,   # Alanine
    'C': 2.5,   # Cysteine
    'D': -3.5,  # Aspartic acid
    'E': -3.5,  # Glutamic acid
    'F': 2.8,   # Phenylalanine
    'G': -0.4,  # Glycine
    'H': -3.2,  # Histidine
    'I': 4.5,   # Isoleucine
    'K': -3.9,  # Lysine
    'L': 3.8,   # Leucine
    'M': 1.9,   # Methionine
    'N': -3.5,  # Asparagine
    'P': -1.6,  # Proline
    'Q': -3.5,  # Glutamine
    'R': -4.5,  # Arginine
    'S': -0.8,  # Serine
    'T': -0.7,  # Threonine
    'V': 4.2,   # Valine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
}


class VHVLChargeAnalyzer:
    """
    Analyzer for VH/VL charge imbalance in antibodies
    """
    
    def __init__(self):
        """
        Initialize VH/VL Charge Analyzer
        """
        self.aa_charges = AMINO_ACID_CHARGES
        self.aa_hydrophobicity = HYDROPHOBICITY
    
    def calculate_net_charge(self, sequence):
        """
        Calculate net charge of a sequence
        
        Parameters:
        sequence (str): Amino acid sequence
        
        Returns:
        float: Net charge
        """
        net_charge = 0
        for aa in sequence:
            if aa in self.aa_charges:
                net_charge += self.aa_charges[aa]
        return net_charge
    
    def calculate_charge_distribution(self, sequence):
        """
        Calculate charge distribution metrics
        
        Parameters:
        sequence (str): Amino acid sequence
        
        Returns:
        dict: Charge distribution metrics
        """
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_count = len(sequence)
        
        for aa in sequence:
            if aa in self.aa_charges:
                charge = self.aa_charges[aa]
                if charge > 0:
                    positive_count += 1
                elif charge < 0:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_fraction': positive_count / total_count if total_count > 0 else 0,
            'negative_fraction': negative_count / total_count if total_count > 0 else 0,
            'neutral_fraction': neutral_count / total_count if total_count > 0 else 0
        }
    
    def calculate_charge_imbalance(self, vh_sequence, vl_sequence):
        """
        Calculate charge imbalance between VH and VL chains
        
        Parameters:
        vh_sequence (str): VH chain sequence
        vl_sequence (str): VL chain sequence
        
        Returns:
        dict: Charge imbalance metrics
        """
        # Calculate net charges
        vh_net_charge = self.calculate_net_charge(vh_sequence)
        vl_net_charge = self.calculate_net_charge(vl_sequence)
        
        # Calculate charge distributions
        vh_charge_dist = self.calculate_charge_distribution(vh_sequence)
        vl_charge_dist = self.calculate_charge_distribution(vl_sequence)
        
        # Calculate imbalance metrics
        charge_difference = abs(vh_net_charge - vl_net_charge)
        charge_ratio = max(abs(vh_net_charge), abs(vl_net_charge)) / min(abs(vh_net_charge), abs(vl_net_charge)) if min(abs(vh_net_charge), abs(vl_net_charge)) > 0 else float('inf')
        
        # Calculate distribution differences
        pos_diff = abs(vh_charge_dist['positive_fraction'] - vl_charge_dist['positive_fraction'])
        neg_diff = abs(vh_charge_dist['negative_fraction'] - vl_charge_dist['negative_fraction'])
        
        return {
            'vh_net_charge': vh_net_charge,
            'vl_net_charge': vl_net_charge,
            'charge_difference': charge_difference,
            'charge_ratio': charge_ratio,
            'positive_fraction_difference': pos_diff,
            'negative_fraction_difference': neg_diff,
            'vh_charge_distribution': vh_charge_dist,
            'vl_charge_distribution': vl_charge_dist
        }
    
    def calculate_hydrophobic_patch_score(self, sequence, window_size=5):
        """
        Calculate hydrophobic patch score as a proxy for surface binding potential
        
        Parameters:
        sequence (str): Amino acid sequence
        window_size (int): Size of sliding window for patch analysis
        
        Returns:
        dict: Hydrophobic patch metrics
        """
        if len(sequence) < window_size:
            return {
                'max_hydrophobicity': 0,
                'mean_hydrophobicity': 0,
                'hydrophobic_patch_count': 0,
                'max_patch_score': 0
            }
        
        # Calculate hydrophobicity for each window
        window_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            window_hydrophobicity = sum(self.aa_hydrophobicity.get(aa, 0) for aa in window) / window_size
            window_scores.append(window_hydrophobicity)
        
        # Identify hydrophobic patches (above threshold)
        threshold = 2.0  # Threshold for hydrophobic patches
        patch_count = sum(1 for score in window_scores if score > threshold)
        
        return {
            'max_hydrophobicity': max(window_scores) if window_scores else 0,
            'mean_hydrophobicity': np.mean(window_scores) if window_scores else 0,
            'hydrophobic_patch_count': patch_count,
            'max_patch_score': max(window_scores) if window_scores else 0
        }
    
    def analyze_antibody(self, antibody_id, vh_sequence, vl_sequence):
        """
        Comprehensive analysis of an antibody's charge properties
        
        Parameters:
        antibody_id (str): Identifier for the antibody
        vh_sequence (str): VH chain sequence
        vl_sequence (str): VL chain sequence
        
        Returns:
        dict: Comprehensive analysis results
        """
        # Calculate charge imbalance
        charge_imbalance = self.calculate_charge_imbalance(vh_sequence, vl_sequence)
        
        # Calculate hydrophobic patches
        vh_hydrophobic = self.calculate_hydrophobic_patch_score(vh_sequence)
        vl_hydrophobic = self.calculate_hydrophobic_patch_score(vl_sequence)
        
        # Combine results
        results = {
            'antibody_id': antibody_id,
            'vh_sequence_length': len(vh_sequence),
            'vl_sequence_length': len(vl_sequence),
            'charge_imbalance': charge_imbalance,
            'vh_hydrophobic_patches': vh_hydrophobic,
            'vl_hydrophobic_patches': vl_hydrophobic,
            'total_hydrophobic_patches': vh_hydrophobic['hydrophobic_patch_count'] + vl_hydrophobic['hydrophobic_patch_count']
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
    
    def export_features(self, antibodies_dict, output_path):
        """
        Export polyreactivity features to a CSV file
        
        Parameters:
        antibodies_dict (dict): Dictionary mapping antibody IDs to (vh_sequence, vl_sequence) tuples
        output_path (str): Path to output CSV file
        
        Returns:
        dict: Export summary
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
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'antibodies_processed': len(df_data),
            'output_file': output_path
        }


def main():
    """
    Main function for testing VH/VL Charge Analyzer
    """
    print("Testing VH/VL Charge Analyzer...")
    
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
    analyzer = VHVLChargeAnalyzer()
    
    # Analyze antibodies
    results = analyzer.analyze_antibodies(antibodies)
    print(f"Analysis results for ab1: {results['ab1']}")
    
    # Export features
    export_result = analyzer.export_features(antibodies, '/a0/bitcore/workspace/polyreactivity_analysis/test_features.csv')
    print(f"Export result: {export_result}")
    
    print("\nVH/VL Charge Analyzer test completed successfully!")


if __name__ == '__main__':
    main()
