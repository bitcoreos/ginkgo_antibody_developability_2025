"""
Heavy-Light Coupling Analysis Module

This module implements heavy-light chain coupling analysis for antibody developability.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Isotype information
ISOTYPES = {
    'IgG1': {'heavy_chain': 'IGHG1', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.7, 'effector_function': 0.8}},
    'IgG2': {'heavy_chain': 'IGHG2', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.5, 'effector_function': 0.6}},
    'IgG3': {'heavy_chain': 'IGHG3', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.9, 'effector_function': 0.7}},
    'IgG4': {'heavy_chain': 'IGHG4', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.6, 'effector_function': 0.3}},
    'IgA1': {'heavy_chain': 'IGHA1', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.8, 'effector_function': 0.5}},
    'IgA2': {'heavy_chain': 'IGHA2', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.8, 'effector_function': 0.5}},
    'IgM': {'heavy_chain': 'IGHM', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.4, 'effector_function': 0.9}},
    'IgE': {'heavy_chain': 'IGHE', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.7, 'effector_function': 0.9}},
    'IgD': {'heavy_chain': 'IGHD', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.6, 'effector_function': 0.4}}
}

# VH gene families
VH_GENE_FAMILIES = [
    'IGHV1', 'IGHV2', 'IGHV3', 'IGHV4', 'IGHV5', 'IGHV6', 'IGHV7'
]

# VL gene families
VL_GENE_FAMILIES = [
    'IGKV1', 'IGKV2', 'IGKV3', 'IGKV4', 'IGKV5', 'IGKV6', 'IGKV7',
    'IGLV1', 'IGLV2', 'IGLV3', 'IGLV4', 'IGLV5', 'IGLV6', 'IGLV7'
]

# Charge pairing rules
CHARGE_PAIRING_RULES = {
    'IGHV1': {'IGKV1': 0.8, 'IGKV2': 0.6, 'IGKV3': 0.7, 'IGLV1': 0.7, 'IGLV2': 0.5, 'IGLV3': 0.6},
    'IGHV2': {'IGKV1': 0.7, 'IGKV2': 0.8, 'IGKV3': 0.6, 'IGLV1': 0.6, 'IGLV2': 0.7, 'IGLV3': 0.5},
    'IGHV3': {'IGKV1': 0.6, 'IGKV2': 0.7, 'IGKV3': 0.8, 'IGLV1': 0.5, 'IGLV2': 0.6, 'IGLV3': 0.7}
}

# Hydrophobicity pairing rules
HYDROPHOBICITY_PAIRING_RULES = {
    'IGHV1': {'IGKV1': 0.7, 'IGKV2': 0.6, 'IGKV3': 0.8, 'IGLV1': 0.6, 'IGLV2': 0.5, 'IGLV3': 0.7},
    'IGHV2': {'IGKV1': 0.6, 'IGKV2': 0.8, 'IGKV3': 0.7, 'IGLV1': 0.5, 'IGLV2': 0.7, 'IGLV3': 0.6},
    'IGHV3': {'IGKV1': 0.8, 'IGKV2': 0.7, 'IGKV3': 0.9, 'IGLV1': 0.7, 'IGLV2': 0.6, 'IGLV3': 0.8}
}


class HeavyLightAnalyzer:
    """
    Analyzer for heavy-light chain coupling.
    """
    
    def __init__(self):
        """
        Initialize the heavy-light analyzer.
        """
        pass
    
    def analyze_vh_vl_pairing(self, vh_sequence: str, vl_sequence: str, vh_gene: str = None, vl_gene: str = None) -> Dict[str, Union[float, str, bool]]:
        """
        Analyze VH-VL pairing compatibility.
        
        Args:
            vh_sequence (str): VH chain amino acid sequence
            vl_sequence (str): VL chain amino acid sequence
            vh_gene (str): VH gene family (optional)
            vl_gene (str): VL gene family (optional)
            
        Returns:
            Dict: Pairing analysis results
        """
        vh_sequence = vh_sequence.upper()
        vl_sequence = vl_sequence.upper()
        
        # Determine gene families if not provided
        if vh_gene is None:
            vh_gene = self._predict_gene_family(vh_sequence, 'VH')
        
        if vl_gene is None:
            vl_gene = self._predict_gene_family(vl_sequence, 'VL')
        
        # Calculate pairing scores
        charge_compatibility = self._calculate_charge_compatibility(vh_gene, vl_gene)
        hydrophobicity_compatibility = self._calculate_hydrophobicity_compatibility(vh_gene, vl_gene)
        length_compatibility = self._calculate_length_compatibility(len(vh_sequence), len(vl_sequence))
        
        # Calculate overall pairing score
        pairing_score = (
            charge_compatibility * 0.4 +
            hydrophobicity_compatibility * 0.4 +
            length_compatibility * 0.2
        )
        
        return {
            'vh_sequence': vh_sequence,
            'vl_sequence': vl_sequence,
            'vh_gene': vh_gene,
            'vl_gene': vl_gene,
            'vh_length': len(vh_sequence),
            'vl_length': len(vl_sequence),
            'charge_compatibility': charge_compatibility,
            'hydrophobicity_compatibility': hydrophobicity_compatibility,
            'length_compatibility': length_compatibility,
            'pairing_score': pairing_score,
            'analysis_complete': True
        }
    
    def _predict_gene_family(self, sequence: str, chain_type: str) -> str:
        """
        Predict gene family based on sequence properties.
        
        Args:
            sequence (str): Amino acid sequence
            chain_type (str): Chain type ('VH' or 'VL')
            
        Returns:
            str: Predicted gene family
        """
        # Simple prediction based on length and composition
        length = len(sequence)
        
        if chain_type == 'VH':
            if length > 120:
                return 'IGHV1'
            elif length > 110:
                return 'IGHV3'
            else:
                return 'IGHV2'
        else:  # VL
            if 'K' in sequence[-10:]:  # Check for kappa in CDR3 region
                return 'IGKV3'
            elif 'L' in sequence[-10:]:  # Check for lambda in CDR3 region
                return 'IGLV3'
            else:
                return 'IGKV1'
    
    def _calculate_charge_compatibility(self, vh_gene: str, vl_gene: str) -> float:
        """
        Calculate charge compatibility between VH and VL genes.
        
        Args:
            vh_gene (str): VH gene family
            vl_gene (str): VL gene family
            
        Returns:
            float: Charge compatibility score (0-1)
        """
        # Check if pairing rules exist
        if vh_gene in CHARGE_PAIRING_RULES and vl_gene in CHARGE_PAIRING_RULES[vh_gene]:
            return CHARGE_PAIRING_RULES[vh_gene][vl_gene]
        
        # Default compatibility score
        return 0.5
    
    def _calculate_hydrophobicity_compatibility(self, vh_gene: str, vl_gene: str) -> float:
        """
        Calculate hydrophobicity compatibility between VH and VL genes.
        
        Args:
            vh_gene (str): VH gene family
            vl_gene (str): VL gene family
            
        Returns:
            float: Hydrophobicity compatibility score (0-1)
        """
        # Check if pairing rules exist
        if vh_gene in HYDROPHOBICITY_PAIRING_RULES and vl_gene in HYDROPHOBICITY_PAIRING_RULES[vh_gene]:
            return HYDROPHOBICITY_PAIRING_RULES[vh_gene][vl_gene]
        
        # Default compatibility score
        return 0.5
    
    def _calculate_length_compatibility(self, vh_length: int, vl_length: int) -> float:
        """
        Calculate length compatibility between VH and VL chains.
        
        Args:
            vh_length (int): VH chain length
            vl_length (int): VL chain length
            
        Returns:
            float: Length compatibility score (0-1)
        """
        # Ideal length difference is 0
        length_diff = abs(vh_length - vl_length)
        
        # Convert to compatibility score (0-1)
        # Perfect match = 1.0, large difference = 0.0
        compatibility = max(0.0, 1.0 - (length_diff / 50.0))
        
        return compatibility
    
    def analyze_isotype_features(self, heavy_chain_sequence: str, light_chain_sequence: str, isotype: str = 'IgG1') -> Dict[str, Union[float, str, bool]]:
        """
        Analyze isotype-specific features.
        
        Args:
            heavy_chain_sequence (str): Heavy chain amino acid sequence
            light_chain_sequence (str): Light chain amino acid sequence
            isotype (str): Antibody isotype
            
        Returns:
            Dict: Isotype analysis results
        """
        heavy_chain_sequence = heavy_chain_sequence.upper()
        light_chain_sequence = light_chain_sequence.upper()
        
        # Get isotype properties
        if isotype in ISOTYPES:
            isotype_properties = ISOTYPES[isotype]['properties']
        else:
            isotype_properties = {'flexibility': 0.5, 'effector_function': 0.5}
        
        # Calculate isotype-specific features
        flexibility_score = self._calculate_flexibility_score(heavy_chain_sequence, isotype)
        effector_function_score = self._calculate_effector_function_score(heavy_chain_sequence, isotype)
        
        # Calculate overall isotype score
        isotype_score = (
            flexibility_score * 0.5 +
            effector_function_score * 0.5
        )
        
        return {
            'heavy_chain_sequence': heavy_chain_sequence,
            'light_chain_sequence': light_chain_sequence,
            'isotype': isotype,
            'isotype_properties': isotype_properties,
            'flexibility_score': flexibility_score,
            'effector_function_score': effector_function_score,
            'isotype_score': isotype_score,
            'analysis_complete': True
        }
    
    def _calculate_flexibility_score(self, heavy_chain_sequence: str, isotype: str) -> float:
        """
        Calculate flexibility score based on heavy chain sequence and isotype.
        
        Args:
            heavy_chain_sequence (str): Heavy chain amino acid sequence
            isotype (str): Antibody isotype
            
        Returns:
            float: Flexibility score (0-1)
        """
        # Base flexibility from isotype
        if isotype in ISOTYPES:
            base_flexibility = ISOTYPES[isotype]['properties']['flexibility']
        else:
            base_flexibility = 0.5
        
        # Adjust based on sequence properties
        # High proline content increases flexibility
        proline_count = heavy_chain_sequence.count('P')
        proline_ratio = proline_count / len(heavy_chain_sequence) if len(heavy_chain_sequence) > 0 else 0
        
        # Adjust flexibility score
        flexibility_score = base_flexibility + (proline_ratio * 0.3)
        flexibility_score = min(1.0, flexibility_score)  # Clamp to 0-1
        
        return flexibility_score
    
    def _calculate_effector_function_score(self, heavy_chain_sequence: str, isotype: str) -> float:
        """
        Calculate effector function score based on heavy chain sequence and isotype.
        
        Args:
            heavy_chain_sequence (str): Heavy chain amino acid sequence
            isotype (str): Antibody isotype
            
        Returns:
            float: Effector function score (0-1)
        """
        # Base effector function from isotype
        if isotype in ISOTYPES:
            base_effector_function = ISOTYPES[isotype]['properties']['effector_function']
        else:
            base_effector_function = 0.5
        
        # Adjust based on sequence properties
        # High glycine content may affect effector function
        glycine_count = heavy_chain_sequence.count('G')
        glycine_ratio = glycine_count / len(heavy_chain_sequence) if len(heavy_chain_sequence) > 0 else 0
        
        # Adjust effector function score
        effector_function_score = base_effector_function - (glycine_ratio * 0.2)
        effector_function_score = max(0.0, effector_function_score)  # Clamp to 0-1
        
        return effector_function_score
    
    def generate_coupling_report(self, vh_sequence: str, vl_sequence: str, heavy_chain_sequence: str, light_chain_sequence: str, isotype: str = 'IgG1') -> Dict[str, Union[str, float, bool]]:
        """
        Generate a comprehensive heavy-light coupling report.
        
        Args:
            vh_sequence (str): VH chain amino acid sequence
            vl_sequence (str): VL chain amino acid sequence
            heavy_chain_sequence (str): Heavy chain amino acid sequence
            light_chain_sequence (str): Light chain amino acid sequence
            isotype (str): Antibody isotype
            
        Returns:
            Dict: Comprehensive coupling report
        """
        # Analyze VH-VL pairing
        pairing_analysis = self.analyze_vh_vl_pairing(vh_sequence, vl_sequence)
        
        # Analyze isotype features
        isotype_analysis = self.analyze_isotype_features(heavy_chain_sequence, light_chain_sequence, isotype)
        
        # Generate summary
        summary = f"""
Heavy-Light Coupling Report
==========================

VH-VL Pairing Analysis:
- VH Gene: {pairing_analysis['vh_gene']}
- VL Gene: {pairing_analysis['vl_gene']}
- Pairing Score: {pairing_analysis['pairing_score']:.3f}

Isotype Analysis:
- Isotype: {isotype}
- Isotype Score: {isotype_analysis['isotype_score']:.3f}

Overall Compatibility: {(pairing_analysis['pairing_score'] + isotype_analysis['isotype_score']) / 2:.3f}
"""
        
        return {
            'vh_sequence': vh_sequence,
            'vl_sequence': vl_sequence,
            'heavy_chain_sequence': heavy_chain_sequence,
            'light_chain_sequence': light_chain_sequence,
            'isotype': isotype,
            'pairing_analysis': pairing_analysis,
            'isotype_analysis': isotype_analysis,
            'overall_compatibility': (pairing_analysis['pairing_score'] + isotype_analysis['isotype_score']) / 2,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the heavy-light analyzer.
    """
    # Example sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    heavy_chain_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    light_chain_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Create analyzer
    analyzer = HeavyLightAnalyzer()
    
    # Analyze VH-VL pairing
    pairing_analysis = analyzer.analyze_vh_vl_pairing(vh_sequence, vl_sequence)
    print("VH-VL Pairing Analysis:")
    print(f"  VH Gene: {pairing_analysis['vh_gene']}")
    print(f"  VL Gene: {pairing_analysis['vl_gene']}")
    print(f"  Charge Compatibility: {pairing_analysis['charge_compatibility']:.3f}")
    print(f"  Hydrophobicity Compatibility: {pairing_analysis['hydrophobicity_compatibility']:.3f}")
    print(f"  Length Compatibility: {pairing_analysis['length_compatibility']:.3f}")
    print(f"  Pairing Score: {pairing_analysis['pairing_score']:.3f}")
    
    # Analyze isotype features
    isotype_analysis = analyzer.analyze_isotype_features(heavy_chain_sequence, light_chain_sequence, 'IgG1')
    print("\nIsotype Analysis:")
    print(f"  Isotype: {isotype_analysis['isotype']}")
    print(f"  Flexibility Score: {isotype_analysis['flexibility_score']:.3f}")
    print(f"  Effector Function Score: {isotype_analysis['effector_function_score']:.3f}")
    print(f"  Isotype Score: {isotype_analysis['isotype_score']:.3f}")
    
    # Generate comprehensive coupling report
    coupling_report = analyzer.generate_coupling_report(vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1')
    print("\nCoupling Report Summary:")
    print(coupling_report['summary'])


if __name__ == "__main__":
    main()
