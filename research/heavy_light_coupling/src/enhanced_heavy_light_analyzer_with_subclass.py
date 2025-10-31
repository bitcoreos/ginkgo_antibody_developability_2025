"""
Enhanced Heavy-Light Coupling Analysis Module with Subclass-Specific Developability Prediction

This module implements enhanced heavy-light chain coupling analysis for antibody developability,
including subclass-specific developability prediction.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Isotype information with subclass-specific properties
ISOTYPES = {
'IgG1': {'heavy_chain': 'IGHG1', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.7, 'effector_function': 0.8, 'complement_binding': 0.6, 'fc_gamma_receptor_binding': 0.7}},
'IgG2': {'heavy_chain': 'IGHG2', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.5, 'effector_function': 0.6, 'complement_binding': 0.3, 'fc_gamma_receptor_binding': 0.4}},
'IgG3': {'heavy_chain': 'IGHG3', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.9, 'effector_function': 0.7, 'complement_binding': 0.8, 'fc_gamma_receptor_binding': 0.6}},
'IgG4': {'heavy_chain': 'IGHG4', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.6, 'effector_function': 0.3, 'complement_binding': 0.2, 'fc_gamma_receptor_binding': 0.5}},
'IgA1': {'heavy_chain': 'IGHA1', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.8, 'effector_function': 0.5, 'complement_binding': 0.4, 'fc_gamma_receptor_binding': 0.3}},
'IgA2': {'heavy_chain': 'IGHA2', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.8, 'effector_function': 0.5, 'complement_binding': 0.4, 'fc_gamma_receptor_binding': 0.3}},
'IgM': {'heavy_chain': 'IGHM', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.4, 'effector_function': 0.9, 'complement_binding': 0.9, 'fc_gamma_receptor_binding': 0.8}},
'IgE': {'heavy_chain': 'IGHE', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.7, 'effector_function': 0.9, 'complement_binding': 0.2, 'fc_gamma_receptor_binding': 0.9}},
'IgD': {'heavy_chain': 'IGHD', 'light_chain': 'IGLC/K', 'properties': {'flexibility': 0.6, 'effector_function': 0.4, 'complement_binding': 0.3, 'fc_gamma_receptor_binding': 0.2}}
}

# Extended VH gene families with characteristic motifs
VH_GENE_FAMILIES = {
'IGHV1': {'length_range': (110, 130), 'motifs': ['GFTFSSYGM', 'YATS', 'KAKIK']},
'IGHV2': {'length_range': (100, 120), 'motifs': ['GFTFSSYAM', 'YATS', 'KAKIK']},
'IGHV3': {'length_range': (115, 135), 'motifs': ['GFTFSSYWM', 'YATS', 'KAKIK']},
'IGHV4': {'length_range': (105, 125), 'motifs': ['GFTFSSYSM', 'YATS', 'KAKIK']},
'IGHV5': {'length_range': (110, 130), 'motifs': ['GFTFSSYGM', 'YATS', 'KAKIK']},
'IGHV6': {'length_range': (100, 120), 'motifs': ['GFTFSSYAM', 'YATS', 'KAKIK']},
'IGHV7': {'length_range': (115, 135), 'motifs': ['GFTFSSYWM', 'YATS', 'KAKIK']}
}

# Extended VL gene families with characteristic motifs
VL_GENE_FAMILIES = {
'IGKV1': {'length_range': (105, 115), 'motifs': ['RASQSVSSSYLAWYQQKPG', 'KLLIY']},
'IGKV2': {'length_range': (100, 110), 'motifs': ['RASQSISDSYLHWYQQKPG', 'KLLIY']},
'IGKV3': {'length_range': (110, 120), 'motifs': ['RASQSISSYLAWYQQKPG', 'KLLIY']},
'IGKV4': {'length_range': (105, 115), 'motifs': ['RASQSVSSSYLAWYQQKPG', 'KLLIY']},
'IGKV5': {'length_range': (100, 110), 'motifs': ['RASQSISDSYLHWYQQKPG', 'KLLIY']},
'IGKV6': {'length_range': (110, 120), 'motifs': ['RASQSISSYLAWYQQKPG', 'KLLIY']},
'IGKV7': {'length_range': (105, 115), 'motifs': ['RASQSVSSSYLAWYQQKPG', 'KLLIY']},
'IGLV1': {'length_range': (105, 115), 'motifs': ['RASQSVSSSYLAWYQQKPG', 'KLLIY']},
'IGLV2': {'length_range': (100, 110), 'motifs': ['RASQSISDSYLHWYQQKPG', 'KLLIY']},
'IGLV3': {'length_range': (110, 120), 'motifs': ['RASQSISSYLAWYQQKPG', 'KLLIY']},
'IGLV4': {'length_range': (105, 115), 'motifs': ['RASQSVSSSYLAWYQQKPG', 'KLLIY']},
'IGLV5': {'length_range': (100, 110), 'motifs': ['RASQSISDSYLHWYQQKPG', 'KLLIY']},
'IGLV6': {'length_range': (110, 120), 'motifs': ['RASQSISSYLAWYQQKPG', 'KLLIY']},
'IGLV7': {'length_range': (105, 115), 'motifs': ['RASQSVSSSYLAWYQQKPG', 'KLLIY']}
}

# Extended charge pairing rules
CHARGE_PAIRING_RULES = {
'IGHV1': {'IGKV1': 0.8, 'IGKV2': 0.6, 'IGKV3': 0.7, 'IGLV1': 0.7, 'IGLV2': 0.5, 'IGLV3': 0.6, 'IGKV4': 0.75, 'IGKV5': 0.55, 'IGKV6': 0.75, 'IGLV4': 0.75, 'IGLV5': 0.55, 'IGLV6': 0.75},
'IGHV2': {'IGKV1': 0.7, 'IGKV2': 0.8, 'IGKV3': 0.6, 'IGLV1': 0.6, 'IGLV2': 0.7, 'IGLV3': 0.5, 'IGKV4': 0.65, 'IGKV5': 0.75, 'IGKV6': 0.65, 'IGLV4': 0.65, 'IGLV5': 0.75, 'IGLV6': 0.65},
'IGHV3': {'IGKV1': 0.6, 'IGKV2': 0.7, 'IGKV3': 0.8, 'IGLV1': 0.5, 'IGLV2': 0.6, 'IGLV3': 0.7, 'IGKV4': 0.55, 'IGKV5': 0.65, 'IGKV6': 0.55, 'IGLV4': 0.55, 'IGLV5': 0.65, 'IGLV6': 0.55},
'IGHV4': {'IGKV1': 0.75, 'IGKV2': 0.55, 'IGKV3': 0.65, 'IGLV1': 0.65, 'IGLV2': 0.45, 'IGLV3': 0.55, 'IGKV4': 0.8, 'IGKV5': 0.6, 'IGKV6': 0.8, 'IGLV4': 0.8, 'IGLV5': 0.6, 'IGLV6': 0.8},
'IGHV5': {'IGKV1': 0.65, 'IGKV2': 0.45, 'IGKV3': 0.55, 'IGLV1': 0.55, 'IGLV2': 0.35, 'IGLV3': 0.45, 'IGKV4': 0.7, 'IGKV5': 0.5, 'IGKV6': 0.7, 'IGLV4': 0.7, 'IGLV5': 0.5, 'IGLV6': 0.7},
'IGHV6': {'IGKV1': 0.55, 'IGKV2': 0.75, 'IGKV3': 0.65, 'IGLV1': 0.65, 'IGLV2': 0.55, 'IGLV3': 0.65, 'IGKV4': 0.6, 'IGKV5': 0.8, 'IGKV6': 0.6, 'IGLV4': 0.6, 'IGLV5': 0.8, 'IGLV6': 0.6},
'IGHV7': {'IGKV1': 0.65, 'IGKV2': 0.65, 'IGKV3': 0.75, 'IGLV1': 0.55, 'IGLV2': 0.65, 'IGLV3': 0.75, 'IGKV4': 0.7, 'IGKV5': 0.7, 'IGKV6': 0.7, 'IGLV4': 0.7, 'IGLV5': 0.7, 'IGLV6': 0.7}
}

# Extended hydrophobicity pairing rules
HYDROPHOBICITY_PAIRING_RULES = {
'IGHV1': {'IGKV1': 0.7, 'IGKV2': 0.6, 'IGKV3': 0.8, 'IGLV1': 0.6, 'IGLV2': 0.5, 'IGLV3': 0.7, 'IGKV4': 0.75, 'IGKV5': 0.55, 'IGKV6': 0.75, 'IGLV4': 0.75, 'IGLV5': 0.55, 'IGLV6': 0.75},
'IGHV2': {'IGKV1': 0.6, 'IGKV2': 0.8, 'IGKV3': 0.7, 'IGLV1': 0.5, 'IGLV2': 0.7, 'IGLV3': 0.6, 'IGKV4': 0.65, 'IGKV5': 0.75, 'IGKV6': 0.65, 'IGLV4': 0.65, 'IGLV5': 0.75, 'IGLV6': 0.65},
'IGHV3': {'IGKV1': 0.8, 'IGKV2': 0.7, 'IGKV3': 0.9, 'IGLV1': 0.7, 'IGLV2': 0.6, 'IGLV3': 0.8, 'IGKV4': 0.85, 'IGKV5': 0.65, 'IGKV6': 0.85, 'IGLV4': 0.85, 'IGLV5': 0.65, 'IGLV6': 0.85},
'IGHV4': {'IGKV1': 0.65, 'IGKV2': 0.55, 'IGKV3': 0.75, 'IGLV1': 0.55, 'IGLV2': 0.45, 'IGLV3': 0.65, 'IGKV4': 0.7, 'IGKV5': 0.5, 'IGKV6': 0.7, 'IGLV4': 0.7, 'IGLV5': 0.5, 'IGLV6': 0.7},
'IGHV5': {'IGKV1': 0.55, 'IGKV2': 0.45, 'IGKV3': 0.65, 'IGLV1': 0.45, 'IGLV2': 0.35, 'IGLV3': 0.55, 'IGKV4': 0.6, 'IGKV5': 0.4, 'IGKV6': 0.6, 'IGLV4': 0.6, 'IGLV5': 0.4, 'IGLV6': 0.6},
'IGHV6': {'IGKV1': 0.65, 'IGKV2': 0.75, 'IGKV3': 0.65, 'IGLV1': 0.55, 'IGLV2': 0.65, 'IGLV3': 0.65, 'IGKV4': 0.7, 'IGKV5': 0.7, 'IGKV6': 0.7, 'IGLV4': 0.7, 'IGLV5': 0.7, 'IGLV6': 0.7},
'IGHV7': {'IGKV1': 0.75, 'IGKV2': 0.65, 'IGKV3': 0.85, 'IGLV1': 0.65, 'IGLV2': 0.55, 'IGLV3': 0.75, 'IGKV4': 0.8, 'IGKV5': 0.6, 'IGKV6': 0.8, 'IGLV4': 0.8, 'IGLV5': 0.6, 'IGLV6': 0.8}
}

# Subclass-specific developability properties
SUBCLASS_DEVELOPABILITY_PROPERTIES = {
'IgG1': {
'solubility': 0.7,
'expression_level': 0.8,
'aggregation_propensity': 0.3,
'thermal_stability': 0.7,
'immunogenicity': 0.4
},
'IgG2': {
'solubility': 0.6,
'expression_level': 0.7,
'aggregation_propensity': 0.4,
'thermal_stability': 0.6,
'immunogenicity': 0.5
},
'IgG3': {
'solubility': 0.8,
'expression_level': 0.6,
'aggregation_propensity': 0.5,
'thermal_stability': 0.5,
'immunogenicity': 0.6
},
'IgG4': {
'solubility': 0.7,
'expression_level': 0.7,
'aggregation_propensity': 0.2,
'thermal_stability': 0.8,
'immunogenicity': 0.3
},
'IgA1': {
'solubility': 0.6,
'expression_level': 0.5,
'aggregation_propensity': 0.6,
'thermal_stability': 0.4,
'immunogenicity': 0.7
},
'IgA2': {
'solubility': 0.6,
'expression_level': 0.5,
'aggregation_propensity': 0.6,
'thermal_stability': 0.4,
'immunogenicity': 0.7
},
'IgM': {
'solubility': 0.4,
'expression_level': 0.3,
'aggregation_propensity': 0.8,
'thermal_stability': 0.3,
'immunogenicity': 0.9
},
'IgE': {
'solubility': 0.5,
'expression_level': 0.4,
'aggregation_propensity': 0.7,
'thermal_stability': 0.4,
'immunogenicity': 0.8
},
'IgD': {
'solubility': 0.5,
'expression_level': 0.4,
'aggregation_propensity': 0.7,
'thermal_stability': 0.4,
'immunogenicity': 0.8
}
}

class EnhancedHeavyLightAnalyzer:
    """
    Enhanced analyzer for heavy-light chain coupling with subclass-specific developability prediction.
    """

    def __init__(self):
        """
        Initialize the enhanced heavy-light analyzer.
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
            vh_gene = self._enhanced_predict_gene_family(vh_sequence, 'VH')

        if vl_gene is None:
            vl_gene = self._enhanced_predict_gene_family(vl_sequence, 'VL')

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

    def _enhanced_predict_gene_family(self, sequence: str, chain_type: str) -> str:
        """
        Enhanced prediction of gene family based on sequence properties and motifs.

        Args:
            sequence (str): Amino acid sequence
            chain_type (str): Chain type ('VH' or 'VL')

        Returns:
            str: Predicted gene family
        """
        length = len(sequence)

        # Get the appropriate gene family dictionary
        if chain_type == 'VH':
            gene_families = VH_GENE_FAMILIES
        else:
            gene_families = VL_GENE_FAMILIES

        # Initialize scores for each gene family
        scores = {}

        # Score based on length
        for gene_family, properties in gene_families.items():
            min_len, max_len = properties['length_range']
            if min_len <= length <= max_len:
                scores[gene_family] = 1.0
            else:
                # Calculate distance from range and convert to score
                if length < min_len:
                    distance = min_len - length
                elif length > max_len:
                    distance = length - max_len
                else:
                    distance = 0
                scores[gene_family] = max(0.0, 1.0 - (distance / 50.0))  # Normalize distance

        # Score based on motifs
        motif_scores = {}
        for gene_family, properties in gene_families.items():
            motifs = properties['motifs']
            motif_score = 0.0
            for motif in motifs:
                if motif in sequence:
                    motif_score += 1.0 / len(motifs)  # Equal weight for each motif
            motif_scores[gene_family] = motif_score

        # Combine length and motif scores
        combined_scores = {}
        for gene_family in gene_families:
            combined_scores[gene_family] = (scores[gene_family] * 0.6) + (motif_scores[gene_family] * 0.4)

        # Return the gene family with the highest combined score
        predicted_gene_family = max(combined_scores, key=combined_scores.get)

        return predicted_gene_family

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

    def _calculate_length_compatibility(self, vh_length: int, vl_length: int, vl_domain_length: int = None) -> float:
        """
        Calculate length compatibility between VH and VL chains.

        Args:
            vh_length (int): VH chain length
            vl_length (int): VL chain length (full chain)
            vl_domain_length (int): VL domain length (optional)

        Returns:
            float: Length compatibility score (0-1)
        """
        # If VL domain length is provided, use it for comparison
        if vl_domain_length is not None:
            length_diff = abs(vh_length - vl_domain_length)
        else:
            # Estimate VL domain length if full chain is provided
            # Typical VL domain length is 100-120 residues
            if vl_length > 150:
                # This is likely a full light chain, estimate VL domain length
                vl_domain_length = min(vl_length, 110)  # Assume 110 residues for VL domain
                length_diff = abs(vh_length - vl_domain_length)
            else:
                # This is likely already the VL domain
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
            isotype_properties = {'flexibility': 0.5, 'effector_function': 0.5, 'complement_binding': 0.5, 'fc_gamma_receptor_binding': 0.5}

        # Calculate isotype-specific features
        flexibility_score = self._calculate_flexibility_score(heavy_chain_sequence, isotype)
        effector_function_score = self._calculate_effector_function_score(heavy_chain_sequence, isotype)
        complement_binding_score = self._calculate_complement_binding_score(heavy_chain_sequence, isotype)
        fc_gamma_receptor_binding_score = self._calculate_fc_gamma_receptor_binding_score(heavy_chain_sequence, isotype)

        # Calculate overall isotype score
        isotype_score = (
            flexibility_score * 0.25 +
            effector_function_score * 0.25 +
            complement_binding_score * 0.25 +
            fc_gamma_receptor_binding_score * 0.25
        )

        return {
            'heavy_chain_sequence': heavy_chain_sequence,
            'light_chain_sequence': light_chain_sequence,
            'isotype': isotype,
            'isotype_properties': isotype_properties,
            'flexibility_score': flexibility_score,
            'effector_function_score': effector_function_score,
            'complement_binding_score': complement_binding_score,
            'fc_gamma_receptor_binding_score': fc_gamma_receptor_binding_score,
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

    def _calculate_complement_binding_score(self, heavy_chain_sequence: str, isotype: str) -> float:
        """
        Calculate complement binding score based on heavy chain sequence and isotype.

        Args:
            heavy_chain_sequence (str): Heavy chain amino acid sequence
            isotype (str): Antibody isotype

        Returns:
            float: Complement binding score (0-1)
        """
        # Base complement binding from isotype
        if isotype in ISOTYPES:
            base_complement_binding = ISOTYPES[isotype]['properties']['complement_binding']
        else:
            base_complement_binding = 0.5

        # Adjust based on sequence properties
        # High cysteine content may affect complement binding
        cysteine_count = heavy_chain_sequence.count('C')
        cysteine_ratio = cysteine_count / len(heavy_chain_sequence) if len(heavy_chain_sequence) > 0 else 0

        # Adjust complement binding score
        complement_binding_score = base_complement_binding - (cysteine_ratio * 0.1)
        complement_binding_score = max(0.0, complement_binding_score)  # Clamp to 0-1

        return complement_binding_score

    def _calculate_fc_gamma_receptor_binding_score(self, heavy_chain_sequence: str, isotype: str) -> float:
        """
        Calculate Fc-gamma receptor binding score based on heavy chain sequence and isotype.

        Args:
            heavy_chain_sequence (str): Heavy chain amino acid sequence
            isotype (str): Antibody isotype

        Returns:
            float: Fc-gamma receptor binding score (0-1)
        """
        # Base Fc-gamma receptor binding from isotype
        if isotype in ISOTYPES:
            base_fc_gamma_receptor_binding = ISOTYPES[isotype]['properties']['fc_gamma_receptor_binding']
        else:
            base_fc_gamma_receptor_binding = 0.5

        # Adjust based on sequence properties
        # High asparagine content may affect Fc-gamma receptor binding
        asparagine_count = heavy_chain_sequence.count('N')
        asparagine_ratio = asparagine_count / len(heavy_chain_sequence) if len(heavy_chain_sequence) > 0 else 0

        # Adjust Fc-gamma receptor binding score
        fc_gamma_receptor_binding_score = base_fc_gamma_receptor_binding + (asparagine_ratio * 0.2)
        fc_gamma_receptor_binding_score = min(1.0, fc_gamma_receptor_binding_score)  # Clamp to 0-1

        return fc_gamma_receptor_binding_score

    def predict_subclass_developability(self, isotype: str) -> Dict[str, float]:
        """
        Predict subclass-specific developability properties.

        Args:
            isotype (str): Antibody isotype

        Returns:
            Dict: Predicted developability properties for the subclass
        """
        # Get subclass-specific developability properties
        if isotype in SUBCLASS_DEVELOPABILITY_PROPERTIES:
            return SUBCLASS_DEVELOPABILITY_PROPERTIES[isotype].copy()
        else:
            # Return default properties
            return {
                'solubility': 0.5,
                'expression_level': 0.5,
                'aggregation_propensity': 0.5,
                'thermal_stability': 0.5,
                'immunogenicity': 0.5
            }

    def generate_coupling_report(self, vh_sequence: str, vl_sequence: str, heavy_chain_sequence: str, light_chain_sequence: str, isotype: str = 'IgG1') -> Dict[str, Union[str, float, bool, Dict]]:
        """
        Generate a comprehensive heavy-light coupling report with subclass-specific developability prediction.

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

        # Predict subclass-specific developability
        subclass_developability = self.predict_subclass_developability(isotype)

        # Generate summary
        summary = f"""
Enhanced Heavy-Light Coupling Report with Subclass-Specific Developability Prediction
========================================================================================

VH-VL Pairing Analysis:
- VH Gene: {pairing_analysis['vh_gene']}
- VL Gene: {pairing_analysis['vl_gene']}
- Pairing Score: {pairing_analysis['pairing_score']:.3f}

Isotype Analysis:
- Isotype: {isotype}
- Isotype Score: {isotype_analysis['isotype_score']:.3f}

Subclass-Specific Developability Prediction:
- Solubility: {subclass_developability['solubility']:.3f}
- Expression Level: {subclass_developability['expression_level']:.3f}
- Aggregation Propensity: {subclass_developability['aggregation_propensity']:.3f}
- Thermal Stability: {subclass_developability['thermal_stability']:.3f}
- Immunogenicity: {subclass_developability['immunogenicity']:.3f}

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
            'subclass_developability': subclass_developability,
            'overall_compatibility': (pairing_analysis['pairing_score'] + isotype_analysis['isotype_score']) / 2,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the enhanced heavy-light analyzer with subclass-specific developability prediction.
    """
    # Example sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    heavy_chain_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    light_chain_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

    # Create analyzer
    analyzer = EnhancedHeavyLightAnalyzer()

    # Analyze VH-VL pairing
    pairing_analysis = analyzer.analyze_vh_vl_pairing(vh_sequence, vl_sequence)
    print("Enhanced VH-VL Pairing Analysis:")
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
    print(f"  Complement Binding Score: {isotype_analysis['complement_binding_score']:.3f}")
    print(f"  Fc-Gamma Receptor Binding Score: {isotype_analysis['fc_gamma_receptor_binding_score']:.3f}")
    print(f"  Isotype Score: {isotype_analysis['isotype_score']:.3f}")

    # Predict subclass-specific developability
    subclass_developability = analyzer.predict_subclass_developability('IgG1')
    print("\nSubclass-Specific Developability Prediction:")
    print(f"  Solubility: {subclass_developability['solubility']:.3f}")
    print(f"  Expression Level: {subclass_developability['expression_level']:.3f}")
    print(f"  Aggregation Propensity: {subclass_developability['aggregation_propensity']:.3f}")
    print(f"  Thermal Stability: {subclass_developability['thermal_stability']:.3f}")
    print(f"  Immunogenicity: {subclass_developability['immunogenicity']:.3f}")

    # Generate comprehensive coupling report
    coupling_report = analyzer.generate_coupling_report(vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1')
    print("\nEnhanced Coupling Report Summary:")
    print(coupling_report['summary'])


if __name__ == "__main__":
    main()
