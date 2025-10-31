"""
Heavy-Light Coupling & Isotype Systematics Implementation

This module implements detailed VH-VL pairing analysis, isotype-specific feature engineering,
and heavy-light chain interaction modeling for antibody developability prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class VHPairingAnalyzer:
    """
    Detailed VH-VL pairing analysis.
    """
    
    def __init__(self):
        """
        Initialize the VH-VL pairing analyzer.
        """
        # Common VH gene families
        self.vh_families = ['IGHV1', 'IGHV2', 'IGHV3', 'IGHV4', 'IGHV5', 'IGHV6', 'IGHV7']
        
        # Common VL gene families
        self.vl_families = ['IGKV1', 'IGKV2', 'IGKV3', 'IGLV1', 'IGLV2', 'IGLV3']
        
        # Known pairing preferences (simplified)
        self.pairing_preferences = {
            'IGHV1': ['IGKV1', 'IGLV1'],
            'IGHV2': ['IGKV2', 'IGLV2'],
            'IGHV3': ['IGKV3', 'IGLV3'],
            'IGHV4': ['IGKV1', 'IGLV1'],
            'IGHV5': ['IGKV2', 'IGLV2'],
            'IGHV6': ['IGKV3', 'IGLV3'],
            'IGHV7': ['IGKV1', 'IGLV1']
        }
    
    def analyze_pairing_frequency(self, vh_sequences: List[str], vl_sequences: List[str], 
                                vh_genes: List[str], vl_genes: List[str]) -> Dict[str, Union[Dict, float]]:
        """
        Analyze pairing frequency between VH and VL gene families.
        
        Args:
            vh_sequences (List[str]): List of VH sequences
            vl_sequences (List[str]): List of VL sequences
            vh_genes (List[str]): List of VH gene assignments
            vl_genes (List[str]): List of VL gene assignments
            
        Returns:
            Dict[str, Union[Dict, float]]: Pairing frequency analysis results
        """
        if len(vh_sequences) != len(vl_sequences) or len(vh_genes) != len(vl_genes):
            raise ValueError("All input lists must have the same length")
        
        # Extract gene families
        vh_families = [gene.split('-')[0] if '-' in gene else gene for gene in vh_genes]
        vl_families = [gene.split('-')[0] if '-' in gene else gene for gene in vl_genes]
        
        # Count pairings
        pairing_counts = defaultdict(int)
        vh_counts = defaultdict(int)
        vl_counts = defaultdict(int)
        
        for vh_fam, vl_fam in zip(vh_families, vl_families):
            pairing_counts[(vh_fam, vl_fam)] += 1
            vh_counts[vh_fam] += 1
            vl_counts[vl_fam] += 1
        
        # Calculate pairing frequencies
        total_pairs = len(vh_sequences)
        pairing_frequencies = {}
        for (vh_fam, vl_fam), count in pairing_counts.items():
            pairing_frequencies[(vh_fam, vl_fam)] = count / total_pairs
        
        # Calculate expected frequencies (assuming independence)
        expected_frequencies = {}
        for vh_fam in vh_counts:
            for vl_fam in vl_counts:
                expected_freq = (vh_counts[vh_fam] / total_pairs) * (vl_counts[vl_fam] / total_pairs)
                expected_frequencies[(vh_fam, vl_fam)] = expected_freq
        
        # Calculate pairing bias (observed/expected)
        pairing_bias = {}
        for (vh_fam, vl_fam), observed_freq in pairing_frequencies.items():
            expected_freq = expected_frequencies.get((vh_fam, vl_fam), 0.0)
            if expected_freq > 0:
                pairing_bias[(vh_fam, vl_fam)] = observed_freq / expected_freq
            else:
                pairing_bias[(vh_fam, vl_fam)] = float('inf')
        
        # Identify preferred pairings (bias > 1.5)
        preferred_pairings = [pair for pair, bias in pairing_bias.items() if bias > 1.5]
        
        # Identify disfavored pairings (bias < 0.67)
        disfavored_pairings = [pair for pair, bias in pairing_bias.items() if bias < 0.67]
        
        return {
            'pairing_counts': dict(pairing_counts),
            'pairing_frequencies': pairing_frequencies,
            'pairing_bias': pairing_bias,
            'preferred_pairings': preferred_pairings,
            'disfavored_pairings': disfavored_pairings,
            'total_pairs': total_pairs
        }
    
    def calculate_pairing_stability(self, vh_sequences: List[str], vl_sequences: List[str]) -> Dict[str, float]:
        """
        Calculate pairing stability metrics based on sequence compatibility.
        
        Args:
            vh_sequences (List[str]): List of VH sequences
            vl_sequences (List[str]): List of VL sequences
            
        Returns:
            Dict[str, float]: Pairing stability metrics
        """
        if len(vh_sequences) != len(vl_sequences):
            raise ValueError("VH and VL sequence lists must have the same length")
        
        stability_metrics = []
        
        for vh_seq, vl_seq in zip(vh_sequences, vl_sequences):
            # Calculate length compatibility
            length_diff = abs(len(vh_seq) - len(vl_seq))
            length_compatibility = max(0, 1 - length_diff / 100)  # Normalize by 100
            
            # Calculate charge compatibility (net charge difference)
            vh_charge = self._calculate_net_charge(vh_seq)
            vl_charge = self._calculate_net_charge(vl_seq)
            charge_diff = abs(vh_charge - vl_charge)
            charge_compatibility = max(0, 1 - charge_diff / 10)  # Normalize by 10
            
            # Calculate hydrophobicity compatibility
            vh_hydro = self._calculate_hydrophobicity(vh_seq)
            vl_hydro = self._calculate_hydrophobicity(vl_seq)
            hydro_diff = abs(vh_hydro - vl_hydro)
            hydro_compatibility = max(0, 1 - hydro_diff / 0.5)  # Normalize by 0.5
            
            # Calculate overall stability score
            stability_score = (length_compatibility + charge_compatibility + hydro_compatibility) / 3
            stability_metrics.append(stability_score)
        
        return {
            'mean_stability': float(np.mean(stability_metrics)),
            'std_stability': float(np.std(stability_metrics)),
            'min_stability': float(np.min(stability_metrics)),
            'max_stability': float(np.max(stability_metrics)),
            'all_stability_scores': stability_metrics
        }
    
    def _calculate_net_charge(self, sequence: str) -> float:
        """
        Calculate net charge of a protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            float: Net charge
        """
        # Basic charge calculation (simplified)
        positive_aas = set('KR')  # Lysine, Arginine
        negative_aas = set('DE')  # Aspartic acid, Glutamic acid
        
        positive_count = sum(1 for aa in sequence.upper() if aa in positive_aas)
        negative_count = sum(1 for aa in sequence.upper() if aa in negative_aas)
        
        return positive_count - negative_count
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """
        Calculate hydrophobicity of a protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            float: Hydrophobicity (fraction of hydrophobic residues)
        """
        hydrophobic_aas = set('AILMFWV')
        if len(sequence) == 0:
            return 0.0
        
        hydrophobic_count = sum(1 for aa in sequence.upper() if aa in hydrophobic_aas)
        return hydrophobic_count / len(sequence)
    
    def comprehensive_pairing_analysis(self, vh_sequences: List[str], vl_sequences: List[str],
                                     vh_genes: List[str], vl_genes: List[str]) -> Dict[str, Union[Dict, float]]:
        """
        Perform comprehensive VH-VL pairing analysis.
        
        Args:
            vh_sequences (List[str]): List of VH sequences
            vl_sequences (List[str]): List of VL sequences
            vh_genes (List[str]): List of VH gene assignments
            vl_genes (List[str]): List of VL gene assignments
            
        Returns:
            Dict[str, Union[Dict, float]]: Comprehensive pairing analysis results
        """
        # Analyze pairing frequency
        pairing_frequency_results = self.analyze_pairing_frequency(vh_sequences, vl_sequences, vh_genes, vl_genes)
        
        # Calculate pairing stability
        pairing_stability_results = self.calculate_pairing_stability(vh_sequences, vl_sequences)
        
        # Compile results
        comprehensive_results = {
            'pairing_frequency_analysis': pairing_frequency_results,
            'pairing_stability_analysis': pairing_stability_results
        }
        
        return comprehensive_results


class IsotypeFeatureEngineer:
    """
    Isotype-specific feature engineering.
    """
    
    def __init__(self):
        """
        Initialize the isotype feature engineer.
        """
        # Isotype information
        self.isotypes = ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'IgA1', 'IgA2', 'IgM', 'IgE', 'IgD']
        
        # Isotype-specific properties (simplified)
        self.isotype_properties = {
            'IgG1': {
                'molecular_weight': 140000,
                'flexibility': 0.8,
                'effector_functions': ['ADCC', 'CDC', 'opsonization'],
                'serum_half_life': 21  # days
            },
            'IgG2': {
                'molecular_weight': 140000,
                'flexibility': 0.6,
                'effector_functions': ['opsonization'],
                'serum_half_life': 21  # days
            },
            'IgG3': {
                'molecular_weight': 140000,
                'flexibility': 1.0,
                'effector_functions': ['ADCC', 'CDC', 'opsonization'],
                'serum_half_life': 7  # days
            },
            'IgG4': {
                'molecular_weight': 140000,
                'flexibility': 0.9,
                'effector_functions': ['minimal'],
                'serum_half_life': 21  # days
            },
            'IgA1': {
                'molecular_weight': 160000,
                'flexibility': 0.7,
                'effector_functions': ['mucosal_immunity'],
                'serum_half_life': 6  # days
            },
            'IgA2': {
                'molecular_weight': 160000,
                'flexibility': 0.7,
                'effector_functions': ['mucosal_immunity'],
                'serum_half_life': 6  # days
            },
            'IgM': {
                'molecular_weight': 900000,
                'flexibility': 0.5,
                'effector_functions': ['complement_activation'],
                'serum_half_life': 10  # days
            },
            'IgE': {
                'molecular_weight': 190000,
                'flexibility': 0.4,
                'effector_functions': ['allergy', 'parasite_immunity'],
                'serum_half_life': 2  # days
            },
            'IgD': {
                'molecular_weight': 180000,
                'flexibility': 0.6,
                'effector_functions': ['B_cell_activation'],
                'serum_half_life': 3  # days
            }
        }
    
    def generate_isotype_features(self, isotypes: List[str]) -> pd.DataFrame:
        """
        Generate isotype-specific features.
        
        Args:
            isotypes (List[str]): List of isotype assignments
            
        Returns:
            pd.DataFrame: DataFrame with isotype-specific features
        """
        features = []
        
        for isotype in isotypes:
            # Get isotype properties
            props = self.isotype_properties.get(isotype, {})
            
            # Extract features
            feature_dict = {
                'isotype': isotype,
                'molecular_weight': props.get('molecular_weight', 0),
                'flexibility': props.get('flexibility', 0),
                'serum_half_life': props.get('serum_half_life', 0),
                'num_effector_functions': len(props.get('effector_functions', []))
            }
            
            # One-hot encode isotype
            for iso in self.isotypes:
                feature_dict[f'is_{iso}'] = 1 if isotype == iso else 0
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def calculate_isotype_compatibility(self, vh_isotypes: List[str], vl_isotypes: List[str]) -> Dict[str, float]:
        """
        Calculate isotype compatibility between VH and VL chains.
        
        Args:
            vh_isotypes (List[str]): List of VH isotype assignments
            vl_isotypes (List[str]): List of VL isotype assignments
            
        Returns:
            Dict[str, float]: Isotype compatibility metrics
        """
        if len(vh_isotypes) != len(vl_isotypes):
            raise ValueError("VH and VL isotype lists must have the same length")
        
        compatibility_scores = []
        
        for vh_iso, vl_iso in zip(vh_isotypes, vl_isotypes):
            # For most antibodies, VH and VL have the same isotype
            # Compatibility score is 1.0 for matching isotypes, 0.0 for mismatching
            compatibility = 1.0 if vh_iso == vl_iso else 0.0
            compatibility_scores.append(compatibility)
        
        return {
            'mean_compatibility': float(np.mean(compatibility_scores)),
            'std_compatibility': float(np.std(compatibility_scores)),
            'compatible_pairs': sum(1 for score in compatibility_scores if score == 1.0),
            'incompatible_pairs': sum(1 for score in compatibility_scores if score == 0.0),
            'all_compatibility_scores': compatibility_scores
        }
    
    def comprehensive_isotype_analysis(self, vh_isotypes: List[str], vl_isotypes: List[str]) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Perform comprehensive isotype analysis.
        
        Args:
            vh_isotypes (List[str]): List of VH isotype assignments
            vl_isotypes (List[str]): List of VL isotype assignments
            
        Returns:
            Dict[str, Union[pd.DataFrame, Dict]]: Comprehensive isotype analysis results
        """
        # Generate isotype features for VH chains
        vh_isotype_features = self.generate_isotype_features(vh_isotypes)
        
        # Generate isotype features for VL chains
        vl_isotype_features = self.generate_isotype_features(vl_isotypes)
        
        # Calculate isotype compatibility
        compatibility_results = self.calculate_isotype_compatibility(vh_isotypes, vl_isotypes)
        
        # Compile results
        comprehensive_results = {
            'vh_isotype_features': vh_isotype_features,
            'vl_isotype_features': vl_isotype_features,
            'isotype_compatibility_analysis': compatibility_results
        }
        
        return comprehensive_results


class HeavyLightInteractionModeler:
    """
    Heavy-light chain interaction modeling.
    """
    
    def __init__(self):
        """
        Initialize the heavy-light interaction modeler.
        """
        # Interface residues (simplified)
        self.interface_positions = {
            'VH': [35, 37, 39, 41, 42, 44, 46, 48, 66, 68, 70, 72, 74, 76, 78, 80],
            'VL': [35, 37, 39, 41, 42, 44, 46, 48, 66, 68, 70, 72, 74, 76, 78, 80]
        }
        
        # Complementarity-determining regions (CDRs)
        self.cdr_positions = {
            'VH': {
                'CDR1': (26, 32),
                'CDR2': (52, 56),
                'CDR3': (95, 102)
            },
            'VL': {
                'CDR1': (24, 34),
                'CDR2': (50, 56),
                'CDR3': (89, 97)
            }
        }
    
    def extract_interface_residues(self, vh_sequence: str, vl_sequence: str) -> Dict[str, str]:
        """
        Extract interface residues from VH and VL sequences.
        
        Args:
            vh_sequence (str): VH sequence
            vl_sequence (str): VL sequence
            
        Returns:
            Dict[str, str]: Interface residues for VH and VL
        """
        vh_interface = ''
        vl_interface = ''
        
        # Extract VH interface residues
        for pos in self.interface_positions['VH']:
            if pos <= len(vh_sequence):
                vh_interface += vh_sequence[pos - 1]  # Convert to 0-based indexing
            else:
                vh_interface += 'X'  # Unknown residue
        
        # Extract VL interface residues
        for pos in self.interface_positions['VL']:
            if pos <= len(vl_sequence):
                vl_interface += vl_sequence[pos - 1]  # Convert to 0-based indexing
            else:
                vl_interface += 'X'  # Unknown residue
        
        return {
            'vh_interface': vh_interface,
            'vl_interface': vl_interface
        }
    
    def calculate_interface_energy(self, vh_sequence: str, vl_sequence: str) -> Dict[str, float]:
        """
        Calculate interface energy based on residue interactions.
        
        Args:
            vh_sequence (str): VH sequence
            vl_sequence (str): VL sequence
            
        Returns:
            Dict[str, float]: Interface energy metrics
        """
        # Extract interface residues
        interface_residues = self.extract_interface_residues(vh_sequence, vl_sequence)
        vh_interface = interface_residues['vh_interface']
        vl_interface = interface_residues['vl_interface']
        
        # Calculate interface energy (simplified)
        # This is a very simplified model based on hydrophobic interactions
        # and charge complementarity
        
        # Hydrophobic interaction score
        hydrophobic_aas = set('AILMFWV')
        vh_hydrophobic = sum(1 for aa in vh_interface if aa in hydrophobic_aas)
        vl_hydrophobic = sum(1 for aa in vl_interface if aa in hydrophobic_aas)
        hydrophobic_interaction = vh_hydrophobic * vl_hydrophobic
        
        # Charge complementarity score
        positive_aas = set('KR')
        negative_aas = set('DE')
        
        vh_positive = sum(1 for aa in vh_interface if aa in positive_aas)
        vh_negative = sum(1 for aa in vh_interface if aa in negative_aas)
        vl_positive = sum(1 for aa in vl_interface if aa in positive_aas)
        vl_negative = sum(1 for aa in vl_interface if aa in negative_aas)
        
        # Charge complementarity (positive-negative interactions)
        charge_complementarity = vh_positive * vl_negative + vh_negative * vl_positive
        
        # Total interface energy (simplified)
        interface_energy = hydrophobic_interaction + charge_complementarity
        
        return {
            'hydrophobic_interaction': float(hydrophobic_interaction),
            'charge_complementarity': float(charge_complementarity),
            'total_interface_energy': float(interface_energy)
        }
    
    def analyze_cdr_interactions(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[str, float]]:
        """
        Analyze CDR interactions between VH and VL chains.
        
        Args:
            vh_sequence (str): VH sequence
            vl_sequence (str): VL sequence
            
        Returns:
            Dict[str, Union[str, float]]: CDR interaction analysis
        """
        # Extract CDR sequences
        vh_cdrs = {}
        vl_cdrs = {}
        
        for cdr, (start, end) in self.cdr_positions['VH'].items():
            if end <= len(vh_sequence):
                vh_cdrs[cdr] = vh_sequence[start - 1:end]  # Convert to 0-based indexing
            else:
                vh_cdrs[cdr] = vh_sequence[start - 1:] if start <= len(vh_sequence) else ''
        
        for cdr, (start, end) in self.cdr_positions['VL'].items():
            if end <= len(vl_sequence):
                vl_cdrs[cdr] = vl_sequence[start - 1:end]  # Convert to 0-based indexing
            else:
                vl_cdrs[cdr] = vl_sequence[start - 1:] if start <= len(vl_sequence) else ''
        
        # Calculate CDR length compatibility
        length_compatibility = {}
        for cdr in vh_cdrs:
            vh_length = len(vh_cdrs[cdr])
            vl_length = len(vl_cdrs.get(cdr, ''))
            length_diff = abs(vh_length - vl_length)
            # Normalize by maximum possible difference (assuming max CDR length of 20)
            length_compatibility[cdr] = max(0, 1 - length_diff / 20)
        
        # Calculate CDR charge compatibility
        charge_compatibility = {}
        for cdr in vh_cdrs:
            vh_charge = self._calculate_net_charge(vh_cdrs[cdr])
            vl_charge = self._calculate_net_charge(vl_cdrs.get(cdr, ''))
            charge_diff = abs(vh_charge - vl_charge)
            # Normalize by maximum possible difference (assuming max charge difference of 10)
            charge_compatibility[cdr] = max(0, 1 - charge_diff / 10)
        
        # Calculate CDR hydrophobicity compatibility
        hydrophobicity_compatibility = {}
        for cdr in vh_cdrs:
            vh_hydro = self._calculate_hydrophobicity(vh_cdrs[cdr])
            vl_hydro = self._calculate_hydrophobicity(vl_cdrs.get(cdr, ''))
            hydro_diff = abs(vh_hydro - vl_hydro)
            # Normalize by maximum possible difference (0.5)
            hydrophobicity_compatibility[cdr] = max(0, 1 - hydro_diff / 0.5)
        
        # Calculate overall CDR compatibility
        overall_compatibility = {}
        for cdr in vh_cdrs:
            length_comp = length_compatibility.get(cdr, 0)
            charge_comp = charge_compatibility.get(cdr, 0)
            hydro_comp = hydrophobicity_compatibility.get(cdr, 0)
            overall_compatibility[cdr] = (length_comp + charge_comp + hydro_comp) / 3
        
        return {
            'vh_cdr_sequences': vh_cdrs,
            'vl_cdr_sequences': vl_cdrs,
            'cdr_length_compatibility': length_compatibility,
            'cdr_charge_compatibility': charge_compatibility,
            'cdr_hydrophobicity_compatibility': hydrophobicity_compatibility,
            'cdr_overall_compatibility': overall_compatibility
        }
    
    def _calculate_net_charge(self, sequence: str) -> float:
        """
        Calculate net charge of a protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            float: Net charge
        """
        # Basic charge calculation (simplified)
        positive_aas = set('KR')  # Lysine, Arginine
        negative_aas = set('DE')  # Aspartic acid, Glutamic acid
        
        positive_count = sum(1 for aa in sequence.upper() if aa in positive_aas)
        negative_count = sum(1 for aa in sequence.upper() if aa in negative_aas)
        
        return positive_count - negative_count
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """
        Calculate hydrophobicity of a protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            float: Hydrophobicity (fraction of hydrophobic residues)
        """
        hydrophobic_aas = set('AILMFWV')
        if len(sequence) == 0:
            return 0.0
        
        hydrophobic_count = sum(1 for aa in sequence.upper() if aa in hydrophobic_aas)
        return hydrophobic_count / len(sequence)
    
    def comprehensive_interaction_modeling(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[Dict, float]]:
        """
        Perform comprehensive heavy-light chain interaction modeling.
        
        Args:
            vh_sequence (str): VH sequence
            vl_sequence (str): VL sequence
            
        Returns:
            Dict[str, Union[Dict, float]]: Comprehensive interaction modeling results
        """
        # Extract interface residues
        interface_residues = self.extract_interface_residues(vh_sequence, vl_sequence)
        
        # Calculate interface energy
        interface_energy = self.calculate_interface_energy(vh_sequence, vl_sequence)
        
        # Analyze CDR interactions
        cdr_interactions = self.analyze_cdr_interactions(vh_sequence, vl_sequence)
        
        # Compile results
        comprehensive_results = {
            'interface_residues': interface_residues,
            'interface_energy': interface_energy,
            'cdr_interactions': cdr_interactions
        }
        
        return comprehensive_results


def main():
    """
    Example usage of the heavy-light coupling & isotype systematics implementation.
    """
    # Example antibody data
    vh_sequences = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS",
        "QVQLQQSGAELVRPGSSVKISCKASGYTFTSYGMNWVKQTPGRGLEWIGYINPSRGYTNYNQKFKDKATLTVDKSSSTAYMQLSSLTSEDSAVYYCARYYDYYAMDYWGQGTLVTVSS"
    ]
    
    vl_sequences = [
        "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNYWPLTFGQGTKVEIK",
        "DIVMTQSQKFMSTSVGDRVSITCRASQNVGTAVAWYQQKPGQSPKLLIYSASFLYSGVPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIK"
    ]
    
    vh_genes = ['IGHV1-69', 'IGHV3-23']
    vl_genes = ['IGKV1-39', 'IGKV2-30']
    
    vh_isotypes = ['IgG1', 'IgG1']
    vl_isotypes = ['IgG1', 'IgG1']
    
    # VH-VL pairing analysis
    pairing_analyzer = VHPairingAnalyzer()
    pairing_analysis = pairing_analyzer.comprehensive_pairing_analysis(vh_sequences, vl_sequences, vh_genes, vl_genes)
    
    print("VH-VL Pairing Analysis Results:")
    print(f"Total pairs analyzed: {pairing_analysis['pairing_frequency_analysis']['total_pairs']}")
    print(f"Preferred pairings: {len(pairing_analysis['pairing_frequency_analysis']['preferred_pairings'])}")
    print(f"Disfavored pairings: {len(pairing_analysis['pairing_frequency_analysis']['disfavored_pairings'])}")
    print(f"Mean pairing stability: {pairing_analysis['pairing_stability_analysis']['mean_stability']:.4f}")
    
    # Isotype feature engineering
    isotype_engineer = IsotypeFeatureEngineer()
    isotype_analysis = isotype_engineer.comprehensive_isotype_analysis(vh_isotypes, vl_isotypes)
    
    print("\nIsotype Analysis Results:")
    print(f"Mean isotype compatibility: {isotype_analysis['isotype_compatibility_analysis']['mean_compatibility']:.4f}")
    print(f"Compatible pairs: {isotype_analysis['isotype_compatibility_analysis']['compatible_pairs']}")
    
    # Heavy-light interaction modeling
    interaction_modeler = HeavyLightInteractionModeler()
    interaction_analysis = interaction_modeler.comprehensive_interaction_modeling(vh_sequences[0], vl_sequences[0])
    
    print("\nHeavy-Light Interaction Modeling Results:")
    print(f"Total interface energy: {interaction_analysis['interface_energy']['total_interface_energy']:.4f}")
    print(f"CDR3 overall compatibility: {interaction_analysis['cdr_interactions']['cdr_overall_compatibility']['CDR3']:.4f}")


if __name__ == "__main__":
    main()
