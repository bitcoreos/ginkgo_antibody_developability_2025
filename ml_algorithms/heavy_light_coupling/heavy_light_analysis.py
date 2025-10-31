"""
Heavy-Light Coupling & Isotype Systematics Implementation

This module implements detailed VH-VL pairing analysis, isotype-specific feature engineering,
and heavy-light chain interaction modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Amino acid alphabet
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Isotype information
ISOTYPES = {
    'IGHA1': {'heavy_chain': 'IGHA1', 'light_chain': 'IGLC', 'function': 'Mucosal immunity'},
    'IGHA2': {'heavy_chain': 'IGHA2', 'light_chain': 'IGLC', 'function': 'Mucosal immunity'},
    'IGHD': {'heavy_chain': 'IGHD', 'light_chain': 'IGLC/IGKC', 'function': 'Antigen recognition'},
    'IGHE': {'heavy_chain': 'IGHE', 'light_chain': 'IGLC/IGKC', 'function': 'Mast cell activation'},
    'IGHG1': {'heavy_chain': 'IGHG1', 'light_chain': 'IGLC/IGKC', 'function': 'Opsonization'},
    'IGHG2': {'heavy_chain': 'IGHG2', 'light_chain': 'IGLC/IGKC', 'function': 'Opsonization'},
    'IGHG3': {'heavy_chain': 'IGHG3', 'light_chain': 'IGLC/IGKC', 'function': 'Complement activation'},
    'IGHG4': {'heavy_chain': 'IGHG4', 'light_chain': 'IGLC/IGKC', 'function': 'Anti-inflammatory'},
    'IGHM': {'heavy_chain': 'IGHM', 'light_chain': 'IGLC/IGKC', 'function': 'Primary immune response'}
}


class VHPairingAnalyzer:
    """
    Detailed VH-VL pairing analysis.
    """
    
    def __init__(self):
        """
        Initialize the VH-VL pairing analyzer.
        """
        pass
    
    def analyze_pairing_compatibility(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[float, str, Dict]]:
        """
        Analyze compatibility of VH-VL pairing.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict[str, Union[float, str, Dict]]: Pairing compatibility analysis
        """
        # Calculate interface properties
        interface_properties = self._calculate_interface_properties(vh_sequence, vl_sequence)
        
        # Calculate pairing score
        pairing_score = self._calculate_pairing_score(interface_properties)
        
        # Determine compatibility level
        if pairing_score > 0.7:
            compatibility = 'high'
            recommendation = 'Excellent VH-VL pairing compatibility'
        elif pairing_score > 0.4:
            compatibility = 'medium'
            recommendation = 'Moderate VH-VL pairing compatibility'
        else:
            compatibility = 'low'
            recommendation = 'Poor VH-VL pairing compatibility - consider re-pairing'
        
        return {
            'pairing_score': float(pairing_score),
            'compatibility': compatibility,
            'recommendation': recommendation,
            'interface_properties': interface_properties
        }
    
    def _calculate_interface_properties(self, vh_sequence: str, vl_sequence: str) -> Dict[str, float]:
        """
        Calculate interface properties between VH and VL domains.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict[str, float]: Interface properties
        """
        # Simplified interface property calculations
        # In practice, this would involve more detailed structural analysis
        
        # Charge complementarity
        vh_positive = sum(1 for aa in vh_sequence if aa in 'KR')
        vh_negative = sum(1 for aa in vh_sequence if aa in 'DE')
        vl_positive = sum(1 for aa in vl_sequence if aa in 'KR')
        vl_negative = sum(1 for aa in vl_sequence if aa in 'DE')
        
        charge_complementarity = abs((vh_positive - vh_negative) + (vl_positive - vl_negative))
        charge_complementarity_score = 1.0 / (1.0 + charge_complementarity)
        
        # Hydrophobicity complementarity
        vh_hydrophobic = sum(1 for aa in vh_sequence if aa in 'AILMFWV')
        vl_hydrophobic = sum(1 for aa in vl_sequence if aa in 'AILMFWV')
        
        hydrophobicity_diff = abs(vh_hydrophobic - vl_hydrophobic)
        hydrophobicity_score = 1.0 / (1.0 + hydrophobicity_diff / 10.0)
        
        # Length compatibility
        length_diff = abs(len(vh_sequence) - len(vl_sequence))
        length_score = 1.0 / (1.0 + length_diff / 50.0)
        
        # Amino acid composition similarity
        vh_composition = Counter(vh_sequence)
        vl_composition = Counter(vl_sequence)
        
        # Normalize compositions
        vh_total = sum(vh_composition.values())
        vl_total = sum(vl_composition.values())
        
        if vh_total > 0 and vl_total > 0:
            vh_norm = {aa: count/vh_total for aa, count in vh_composition.items()}
            vl_norm = {aa: count/vl_total for aa, count in vl_composition.items()}
            
            # Calculate composition similarity (cosine similarity)
            dot_product = sum(vh_norm.get(aa, 0) * vl_norm.get(aa, 0) for aa in set(vh_norm.keys()) | set(vl_norm.keys()))
            vh_magnitude = np.sqrt(sum(count**2 for count in vh_norm.values()))
            vl_magnitude = np.sqrt(sum(count**2 for count in vl_norm.values()))
            
            if vh_magnitude > 0 and vl_magnitude > 0:
                composition_similarity = dot_product / (vh_magnitude * vl_magnitude)
            else:
                composition_similarity = 0.0
        else:
            composition_similarity = 0.0
        
        return {
            'charge_complementarity_score': float(charge_complementarity_score),
            'hydrophobicity_score': float(hydrophobicity_score),
            'length_score': float(length_score),
            'composition_similarity': float(composition_similarity)
        }
    
    def _calculate_pairing_score(self, interface_properties: Dict[str, float]) -> float:
        """
        Calculate overall pairing score from interface properties.
        
        Args:
            interface_properties (Dict[str, float]): Interface properties
            
        Returns:
            float: Pairing score (0-1)
        """
        # Weighted average of interface properties
        weights = {
            'charge_complementarity_score': 0.3,
            'hydrophobicity_score': 0.3,
            'length_score': 0.2,
            'composition_similarity': 0.2
        }
        
        pairing_score = sum(interface_properties.get(prop, 0) * weight for prop, weight in weights.items())
        
        return pairing_score


class IsotypeFeatureEngineer:
    """
    Isotype-specific feature engineering.
    """
    
    def __init__(self):
        """
        Initialize the isotype feature engineer.
        """
        self.isotypes = ISOTYPES
    
    def engineer_isotype_features(self, vh_sequence: str, vl_sequence: str, 
                                isotype: str) -> Dict[str, Union[float, str, Dict]]:
        """
        Engineer isotype-specific features.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            isotype (str): Isotype identifier
            
        Returns:
            Dict[str, Union[float, str, Dict]]: Isotype-specific features
        """
        # Get isotype information
        isotype_info = self.isotypes.get(isotype, {})
        
        # Calculate isotype-specific properties
        isotype_properties = self._calculate_isotype_properties(vh_sequence, vl_sequence, isotype)
        
        # Generate feature vector
        feature_vector = self._generate_feature_vector(isotype_properties, isotype_info)
        
        return {
            'isotype': isotype,
            'isotype_info': isotype_info,
            'isotype_properties': isotype_properties,
            'feature_vector': feature_vector
        }
    
    def _calculate_isotype_properties(self, vh_sequence: str, vl_sequence: str, 
                                   isotype: str) -> Dict[str, Union[float, str]]:
        """
        Calculate isotype-specific properties.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            isotype (str): Isotype identifier
            
        Returns:
            Dict[str, Union[float, str]]: Isotype-specific properties
        """
        # Isotype-specific calculations
        properties = {}
        
        # Heavy chain properties
        properties['vh_length'] = len(vh_sequence)
        properties['vh_charge'] = sum(1 for aa in vh_sequence if aa in 'KR') - sum(1 for aa in vh_sequence if aa in 'DE')
        properties['vh_hydrophobicity'] = sum(1 for aa in vh_sequence if aa in 'AILMFWV') / len(vh_sequence) if vh_sequence else 0
        
        # Light chain properties
        properties['vl_length'] = len(vl_sequence)
        properties['vl_charge'] = sum(1 for aa in vl_sequence if aa in 'KR') - sum(1 for aa in vl_sequence if aa in 'DE')
        properties['vl_hydrophobicity'] = sum(1 for aa in vl_sequence if aa in 'AILMFWV') / len(vl_sequence) if vl_sequence else 0
        
        # Pairing properties
        properties['vh_vl_length_ratio'] = properties['vh_length'] / properties['vl_length'] if properties['vl_length'] > 0 else 0
        properties['vh_vl_charge_balance'] = abs(properties['vh_charge'] - properties['vl_charge'])
        
        # Isotype-specific properties
        if isotype in ['IGHA1', 'IGHA2']:
            properties['isotype_class'] = 'IgA'
            properties['mucosal_function_score'] = 0.9
        elif isotype == 'IGHD':
            properties['isotype_class'] = 'IgD'
            properties['antigen_recognition_score'] = 0.8
        elif isotype == 'IGHE':
            properties['isotype_class'] = 'IgE'
            properties['mast_cell_activation_score'] = 0.95
        elif isotype in ['IGHG1', 'IGHG2', 'IGHG3', 'IGHG4']:
            properties['isotype_class'] = 'IgG'
            properties['opsonization_score'] = 0.85
        elif isotype == 'IGHM':
            properties['isotype_class'] = 'IgM'
            properties['primary_response_score'] = 0.9
        else:
            properties['isotype_class'] = 'Unknown'
            properties['function_score'] = 0.5
        
        return properties
    
    def _generate_feature_vector(self, isotype_properties: Dict[str, Union[float, str]], 
                               isotype_info: Dict[str, str]) -> List[float]:
        """
        Generate feature vector from isotype properties.
        
        Args:
            isotype_properties (Dict[str, Union[float, str]]): Isotype properties
            isotype_info (Dict[str, str]): Isotype information
            
        Returns:
            List[float]: Feature vector
        """
        # Extract numerical properties for feature vector
        feature_keys = [
            'vh_length', 'vh_charge', 'vh_hydrophobicity',
            'vl_length', 'vl_charge', 'vl_hydrophobicity',
            'vh_vl_length_ratio', 'vh_vl_charge_balance'
        ]
        
        # Add isotype-specific scores if available
        if isotype_properties.get('isotype_class') == 'IgA':
            feature_keys.append('mucosal_function_score')
        elif isotype_properties.get('isotype_class') == 'IgD':
            feature_keys.append('antigen_recognition_score')
        elif isotype_properties.get('isotype_class') == 'IgE':
            feature_keys.append('mast_cell_activation_score')
        elif isotype_properties.get('isotype_class') == 'IgG':
            feature_keys.append('opsonization_score')
        elif isotype_properties.get('isotype_class') == 'IgM':
            feature_keys.append('primary_response_score')
        
        # Generate feature vector
        feature_vector = []
        for key in feature_keys:
            value = isotype_properties.get(key, 0)
            if isinstance(value, (int, float)):
                feature_vector.append(float(value))
            else:
                # Convert categorical values to numerical (simplified)
                feature_vector.append(1.0 if value else 0.0)
        
        return feature_vector


class HeavyLightInteractionModel:
    """
    Heavy-light chain interaction modeling.
    """
    
    def __init__(self):
        """
        Initialize the heavy-light interaction model.
        """
        pass
    
    def model_interactions(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[float, str, Dict]]:
        """
        Model heavy-light chain interactions.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict[str, Union[float, str, Dict]]: Interaction modeling results
        """
        # Calculate interaction properties
        interaction_properties = self._calculate_interaction_properties(vh_sequence, vl_sequence)
        
        # Calculate interaction score
        interaction_score = self._calculate_interaction_score(interaction_properties)
        
        # Determine interaction quality
        if interaction_score > 0.7:
            quality = 'high'
            recommendation = 'Strong heavy-light chain interactions'
        elif interaction_score > 0.4:
            quality = 'medium'
            recommendation = 'Moderate heavy-light chain interactions'
        else:
            quality = 'low'
            recommendation = 'Weak heavy-light chain interactions - consider optimization'
        
        return {
            'interaction_score': float(interaction_score),
            'quality': quality,
            'recommendation': recommendation,
            'interaction_properties': interaction_properties
        }
    
    def _calculate_interaction_properties(self, vh_sequence: str, vl_sequence: str) -> Dict[str, float]:
        """
        Calculate heavy-light chain interaction properties.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict[str, float]: Interaction properties
        """
        # Simplified interaction property calculations
        # In practice, this would involve more detailed structural analysis
        
        # Interface complementarity
        vh_composition = Counter(vh_sequence)
        vl_composition = Counter(vl_sequence)
        
        # Calculate complementarity based on charge and hydrophobicity
        vh_positive = sum(1 for aa in vh_sequence if aa in 'KR')
        vh_negative = sum(1 for aa in vh_sequence if aa in 'DE')
        vl_positive = sum(1 for aa in vl_sequence if aa in 'KR')
        vl_negative = sum(1 for aa in vl_sequence if aa in 'DE')
        
        charge_complementarity = abs((vh_positive - vh_negative) + (vl_positive - vl_negative))
        charge_complementarity_score = 1.0 / (1.0 + charge_complementarity)
        
        # Hydrophobicity complementarity
        vh_hydrophobic = sum(1 for aa in vh_sequence if aa in 'AILMFWV')
        vl_hydrophobic = sum(1 for aa in vl_sequence if aa in 'AILMFWV')
        
        hydrophobicity_complementarity = abs(vh_hydrophobic - vl_hydrophobic)
        hydrophobicity_score = 1.0 / (1.0 + hydrophobicity_complementarity / 10.0)
        
        # Amino acid pairing preferences
        # Simplified model based on known pairing preferences
        vh_polar = sum(1 for aa in vh_sequence if aa in 'NQST')
        vl_polar = sum(1 for aa in vl_sequence if aa in 'NQST')
        
        polar_pairing_score = 1.0 - abs(vh_polar - vl_polar) / max(len(vh_sequence), len(vl_sequence), 1)
        
        # Length compatibility
        length_ratio = min(len(vh_sequence), len(vl_sequence)) / max(len(vh_sequence), len(vl_sequence), 1)
        
        return {
            'charge_complementarity_score': float(charge_complementarity_score),
            'hydrophobicity_score': float(hydrophobicity_score),
            'polar_pairing_score': float(polar_pairing_score),
            'length_compatibility': float(length_ratio)
        }
    
    def _calculate_interaction_score(self, interaction_properties: Dict[str, float]) -> float:
        """
        Calculate overall interaction score from interaction properties.
        
        Args:
            interaction_properties (Dict[str, float]): Interaction properties
            
        Returns:
            float: Interaction score (0-1)
        """
        # Weighted average of interaction properties
        weights = {
            'charge_complementarity_score': 0.3,
            'hydrophobicity_score': 0.3,
            'polar_pairing_score': 0.2,
            'length_compatibility': 0.2
        }
        
        interaction_score = sum(interaction_properties.get(prop, 0) * weight for prop, weight in weights.items())
        
        return interaction_score


def main():
    """
    Example usage of the heavy-light coupling and isotype systematics implementation.
    """
    # Example VH and VL sequences
    vh_sequence = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSVSSSYLAWYQQKPGKAPKLLIYDASNRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQRSNWPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # VH-VL pairing analysis
    vh_pairing_analyzer = VHPairingAnalyzer()
    pairing_analysis = vh_pairing_analyzer.analyze_pairing_compatibility(vh_sequence, vl_sequence)
    
    print("VH-VL Pairing Analysis Results:")
    print(f"  Pairing Score: {pairing_analysis['pairing_score']:.4f} ({pairing_analysis['compatibility']})")
    print(f"  Recommendation: {pairing_analysis['recommendation']}")
    
    # Interface properties
    interface_props = pairing_analysis['interface_properties']
    print("  Interface Properties:")
    print(f"    Charge Complementarity: {interface_props['charge_complementarity_score']:.4f}")
    print(f"    Hydrophobicity Score: {interface_props['hydrophobicity_score']:.4f}")
    print(f"    Length Score: {interface_props['length_score']:.4f}")
    print(f"    Composition Similarity: {interface_props['composition_similarity']:.4f}")
    
    # Isotype-specific feature engineering
    isotype_engineer = IsotypeFeatureEngineer()
    isotype_analysis = isotype_engineer.engineer_isotype_features(vh_sequence, vl_sequence, 'IGHG1')
    
    print("\nIsotype-Specific Feature Engineering Results:")
    print(f"  Isotype: {isotype_analysis['isotype']}")
    print(f"  Isotype Class: {isotype_analysis['isotype_properties'].get('isotype_class', 'Unknown')}")
    print(f"  Feature Vector Length: {len(isotype_analysis['feature_vector'])}")
    print(f"  First 5 Feature Values: {isotype_analysis['feature_vector'][:5]}")
    
    # Heavy-light chain interaction modeling
    interaction_model = HeavyLightInteractionModel()
    interaction_analysis = interaction_model.model_interactions(vh_sequence, vl_sequence)
    
    print("\nHeavy-Light Chain Interaction Modeling Results:")
    print(f"  Interaction Score: {interaction_analysis['interaction_score']:.4f} ({interaction_analysis['quality']})")
    print(f"  Recommendation: {interaction_analysis['recommendation']}")
    
    # Interaction properties
    interaction_props = interaction_analysis['interaction_properties']
    print("  Interaction Properties:")
    print(f"    Charge Complementarity: {interaction_props['charge_complementarity_score']:.4f}")
    print(f"    Hydrophobicity Score: {interaction_props['hydrophobicity_score']:.4f}")
    print(f"    Polar Pairing Score: {interaction_props['polar_pairing_score']:.4f}")
    print(f"    Length Compatibility: {interaction_props['length_compatibility']:.4f}")


if __name__ == "__main__":
    main()
