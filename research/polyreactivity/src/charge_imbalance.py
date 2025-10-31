"""
VH/VL Charge Imbalance Analysis Module

This module implements advanced charge imbalance analysis for antibody variable domains.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Amino acid charge properties
POSITIVE_CHARGE_AA = 'KRH'  # Lysine, Arginine, Histidine
NEGATIVE_CHARGE_AA = 'DE'   # Aspartic acid, Glutamic acid


class ChargeImbalanceAnalyzer:
    """
    Analyzer for VH/VL charge imbalance in antibody variable domains.
    """
    
    def __init__(self):
        """
        Initialize the charge imbalance analyzer.
        """
        pass
    
    def analyze_charge_distribution(self, sequence: str) -> Dict[str, Union[int, float]]:
        """
        Analyze detailed charge distribution in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Charge distribution analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'positive_count': 0,
                'negative_count': 0,
                'net_charge': 0,
                'charge_density': 0.0,
                'positive_density': 0.0,
                'negative_density': 0.0,
                'charge_balance': 0.0,
                'analysis_complete': True
            }
        
        # Count charged residues
        positive_count = sum(1 for aa in sequence if aa in POSITIVE_CHARGE_AA)
        negative_count = sum(1 for aa in sequence if aa in NEGATIVE_CHARGE_AA)
        net_charge = positive_count - negative_count
        
        # Calculate densities
        positive_density = positive_count / length
        negative_density = negative_count / length
        charge_density = (positive_count + negative_count) / length
        
        # Calculate charge balance (0 = perfectly balanced, 1 = completely unbalanced)
        charge_balance = abs(net_charge) / length if length > 0 else 0
        
        return {
            'sequence': sequence,
            'length': length,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'net_charge': net_charge,
            'charge_density': charge_density,
            'positive_density': positive_density,
            'negative_density': negative_density,
            'charge_balance': charge_balance,
            'analysis_complete': True
        }
    
    def analyze_charge_pairing(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze charge pairing between VH and VL domains.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict: Charge pairing analysis results
        """
        vh_sequence = vh_sequence.upper()
        vl_sequence = vl_sequence.upper()
        
        # Analyze each domain separately
        vh_charge_analysis = self.analyze_charge_distribution(vh_sequence)
        vl_charge_analysis = self.analyze_charge_distribution(vl_sequence)
        
        # Calculate pairing metrics
        vh_net_charge = vh_charge_analysis['net_charge']
        vl_net_charge = vl_charge_analysis['net_charge']
        
        # Charge pairing imbalance (absolute difference in net charge)
        pairing_imbalance = abs(vh_net_charge - vl_net_charge)
        
        # Charge pairing compatibility (0 = perfectly compatible, higher = less compatible)
        pairing_compatibility = abs(vh_net_charge + vl_net_charge)
        
        # Charge pairing score (0 = perfectly paired, 1 = completely unpaired)
        total_length = len(vh_sequence) + len(vl_sequence)
        pairing_score = pairing_imbalance / total_length if total_length > 0 else 0
        
        return {
            'vh_sequence': vh_sequence,
            'vl_sequence': vl_sequence,
            'vh_charge_analysis': vh_charge_analysis,
            'vl_charge_analysis': vl_charge_analysis,
            'pairing_imbalance': pairing_imbalance,
            'pairing_compatibility': pairing_compatibility,
            'pairing_score': pairing_score,
            'analysis_complete': True
        }
    
    def calculate_charge_imbalance_score(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Union[float, str]]:
        """
        Calculate a comprehensive charge imbalance score.
        
        Args:
            vh_sequence (str): VH domain sequence
            vl_sequence (str): VL domain sequence
            
        Returns:
            Dict: Charge imbalance score and interpretation
        """
        # Analyze charge pairing
        pairing_analysis = self.analyze_charge_pairing(vh_sequence, vl_sequence)
        
        # Extract relevant metrics
        pairing_score = pairing_analysis['pairing_score']
        vh_charge_balance = pairing_analysis['vh_charge_analysis']['charge_balance']
        vl_charge_balance = pairing_analysis['vl_charge_analysis']['charge_balance']
        
        # Calculate weighted charge imbalance score
        # Weight pairing score more heavily as it represents the interaction between domains
        imbalance_score = (
            0.5 * pairing_score +
            0.25 * vh_charge_balance +
            0.25 * vl_charge_balance
        )
        
        # Interpret the score
        if imbalance_score < 0.1:
            interpretation = "Low charge imbalance - favorable for stability and solubility"
        elif imbalance_score < 0.2:
            interpretation = "Moderate charge imbalance - generally acceptable"
        elif imbalance_score < 0.3:
            interpretation = "High charge imbalance - may affect stability and solubility"
        else:
            interpretation = "Very high charge imbalance - likely to cause developability issues"
        
        return {
            'vh_sequence': vh_sequence,
            'vl_sequence': vl_sequence,
            'imbalance_score': imbalance_score,
            'pairing_score': pairing_score,
            'vh_charge_balance': vh_charge_balance,
            'vl_charge_balance': vl_charge_balance,
            'interpretation': interpretation,
            'scoring_complete': True
        }


def main():
    """
    Example usage of the charge imbalance analyzer.
    """
    # Example VH and VL sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYGSSPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Create analyzer
    analyzer = ChargeImbalanceAnalyzer()
    
    # Analyze charge distribution in VH domain
    vh_charge_analysis = analyzer.analyze_charge_distribution(vh_sequence)
    print("VH Domain Charge Analysis:")
    print(f"  Length: {vh_charge_analysis['length']}")
    print(f"  Positive residues: {vh_charge_analysis['positive_count']}")
    print(f"  Negative residues: {vh_charge_analysis['negative_count']}")
    print(f"  Net charge: {vh_charge_analysis['net_charge']}")
    print(f"  Charge density: {vh_charge_analysis['charge_density']:.3f}")
    print(f"  Charge balance: {vh_charge_analysis['charge_balance']:.3f}")
    
    # Analyze charge distribution in VL domain
    vl_charge_analysis = analyzer.analyze_charge_distribution(vl_sequence)
    print("\nVL Domain Charge Analysis:")
    print(f"  Length: {vl_charge_analysis['length']}")
    print(f"  Positive residues: {vl_charge_analysis['positive_count']}")
    print(f"  Negative residues: {vl_charge_analysis['negative_count']}")
    print(f"  Net charge: {vl_charge_analysis['net_charge']}")
    print(f"  Charge density: {vl_charge_analysis['charge_density']:.3f}")
    print(f"  Charge balance: {vl_charge_analysis['charge_balance']:.3f}")
    
    # Analyze charge pairing
    pairing_analysis = analyzer.analyze_charge_pairing(vh_sequence, vl_sequence)
    print("\nCharge Pairing Analysis:")
    print(f"  Pairing imbalance: {pairing_analysis['pairing_imbalance']}")
    print(f"  Pairing compatibility: {pairing_analysis['pairing_compatibility']}")
    print(f"  Pairing score: {pairing_analysis['pairing_score']:.3f}")
    
    # Calculate comprehensive charge imbalance score
    imbalance_score = analyzer.calculate_charge_imbalance_score(vh_sequence, vl_sequence)
    print("\nComprehensive Charge Imbalance Score:")
    print(f"  Imbalance score: {imbalance_score['imbalance_score']:.3f}")
    print(f"  Pairing score: {imbalance_score['pairing_score']:.3f}")
    print(f"  VH charge balance: {imbalance_score['vh_charge_balance']:.3f}")
    print(f"  VL charge balance: {imbalance_score['vl_charge_balance']:.3f}")
    print(f"  Interpretation: {imbalance_score['interpretation']}")


if __name__ == "__main__":
    main()
